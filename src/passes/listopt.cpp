// Pass "listopt" — header caching (ListData/ElemLoad/CheckIndex) and check elision (DESIGN.md §5.2/§5.3).
//
// For every list local L of static type List(r) that is the base of some
// Index / SetIndex / Len, the pass may keep L's data pointer and length in two
// Cache locals dL : Ptr(r), nL : Int and lower
//     Index(L, i)        -> ElemLoad(dL, CheckIndex(i, nL))
//     Len(L)             -> nL
//     SetIndex(L, i, v)  -> ElemStore(dL, CheckIndex(i, nL), v)        (v pure)
//                        -> Comma[tv := v, ElemStore(dL, CheckIndex(i, nL), tv)]   (i simple)
//                        -> Comma[ti := i, tv := v, ElemStore(dL, CheckIndex(ti, nL), tv)]
// whenever the cache is provably current. Correctness rests on ONE forward
// abstract interpretation over the structured tree (visit / cond below), in
// exact evaluation order (kid order; And/Or/If/While/InlineBlock/Break/Return
// handled as control flow), whose state says per candidate L:
//   valid[L]  dL/nL hold L's current data pointer and length
//   ep[L]     an epoch, bumped whenever the cache is (re)loaded or dropped;
//             an access is lowered only if the epoch after evaluating its
//             index/value equals the epoch right after its base was read
//   asg[x]    local x is definitely assigned (needed before loading a cache)
//   facts     (i, L): 0 <= i < nL holds now, for a "simple" index expression
//             i (constants, Int locals, + - * neg) -> CheckIndex(i, nL) elided
// Invalidation (kills_node): assignment to L; Append/Pop/Free whose base may be
// L's list (same region; not if L is unique or the base is another unique
// local); CallDirect whose summary mutates r (or Any regions, for Any
// elements); CallDyn (Any regions + everything a generic entry mutates);
// DynAppend/DynNeg (Any regions); any op not explicitly classified.
// Append/Pop directly on a cached L re-load the cache right after (Pop also
// drops L's facts; Append keeps them: lists only grow).
// Loops: the head state is the greatest fixpoint of meet(entry, back-edge),
// found by dry runs (no rewriting) before the single rewriting run, so the
// cache/fact claims at the head hold on every iteration. A cache is loaded at
// a loop preheader when L is used in the loop, definitely assigned, and the
// loop never invalidates it; otherwise in a Seq before a statement when at
// least two uses follow before the next invalidating statement.
// Loop-guard elision: a fact (i, L) is created by an evaluated-true LtI32(i,
// Len L) / GtI32(Len L, i) condition when i is provably non-negative (a non-
// constant ConstInt >= 0, or an Int local whose every assignment is a
// non-negative constant or the `i := i + 1` step of a loop whose And-chain
// condition contains `i < X` and whose cond/body do not assign i).
#include <algorithm>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

using namespace util;

struct Fact {
    Node* idx;
    int k;  // candidate index
};

struct State {
    bool dead = false;
    std::vector<char> valid;
    std::vector<uint32_t> ep;
    std::vector<char> asg;
    std::vector<Fact> facts;
};

struct Cand {
    uint32_t L = 0, r = 0;
    int32_t dL = -1, nL = -1;
};

bool is_simple(const Node* n, int depth = 0) {
    if (depth > 6) return false;
    switch (n->op) {
    case Op::ConstInt: return true;
    case Op::Local: return n->type.k == TK::Int;
    case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::NegI32:
        for (uint32_t i = 0; i < n->nkids; ++i)
            if (!is_simple(n->kids[i], depth + 1)) return false;
        return true;
    default: return false;
    }
}

bool mentions_local(const Node* n, int32_t id) {
    if (n->op == Op::Local && n->a == id) return true;
    for (uint32_t i = 0; i < n->nkids; ++i)
        if (mentions_local(n->kids[i], id)) return true;
    return false;
}

bool sets_any_local_of(const Node* n, const Node* expr) {
    if (n->op == Op::SetLocal && mentions_local(expr, n->a)) return true;
    for (uint32_t i = 0; i < n->nkids; ++i)
        if (sets_any_local_of(n->kids[i], expr)) return true;
    return false;
}

struct LO {
    Module& m;
    Func& f;
    bool dry = false;
    uint32_t epc = 0;
    size_t nloc0 = 0;
    std::vector<int> ci;        // local id -> candidate index (-1)
    std::vector<Cand> cs;
    std::vector<char> uniq;     // local id -> unique (sole name of its list)
    std::vector<char> nonneg;   // local id -> provably >= 0 always
    std::vector<char> genmut;   // region -> mutated by some generic entry
    std::vector<std::pair<int32_t, State>> labels;
    bool loops_off = false;      // loop nest too deep for the dry-run fixpoint: no caches across loops

    LO(Module& m_, Func& f_) : m(m_), f(f_) {}

    // ---------------------------------------------------------------- analysis
    bool is_base_local(const Node* b, uint32_t L) const { return b->op == Op::Local && b->a == (int32_t)L; }

    bool kills_node(const Node* x, const Cand& c, bool count_direct) const {
        bool any = m.regions[c.r].elem.k == TK::Any;
        bool u = uniq[c.L];
        switch (x->op) {
        case Op::SetLocal: return x->a == (int32_t)c.L;
        case Op::Append: case Op::Pop: case Op::Free: {
            const Node* b = x->kids[0];
            if (b->type.k != TK::List || b->type.a != c.r) return false;
            if (is_base_local(b, c.L)) return x->op == Op::Free || count_direct;
            if (u) return false;
            if (b->op == Op::Local && (size_t)b->a < uniq.size() && uniq[(size_t)b->a]) return false;
            return true;
        }
        case Op::CallDirect: {
            if (u) return false;
            const Func& g = *m.funcs[(size_t)x->a];
            if (g.mutates_any && any) return true;
            return std::find(g.mutates.begin(), g.mutates.end(), c.r) != g.mutates.end();
        }
        case Op::CallDyn: return !u && (any || genmut[c.r]);
        case Op::DynAppend: case Op::DynNeg: return !u && any;
        case Op::Seq: case Op::If: case Op::While: case Op::Return: case Op::Print:
        case Op::ConstInt: case Op::ConstFn: case Op::Local: case Op::GetGlobal: case Op::Input:
        case Op::SetGlobal: case Op::Box: case Op::UnboxInt: case Op::Coerce: case Op::Comma: case Op::Fail:
        case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::DivI32: case Op::NegI32: case Op::AbsI32:
        case Op::NotI32: case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
        case Op::DynArith: case Op::DynAbs: case Op::DynNot: case Op::Eq: case Op::Ne: case Op::Cmp:
        case Op::ListLit: case Op::Len: case Op::Index: case Op::SetIndex: case Op::DynIndex: case Op::DynSetIndex:
        case Op::And: case Op::Or: case Op::Select: case Op::InlineBlock: case Op::Break:
        case Op::PairMake: case Op::PairFirst: case Op::PairSecond: case Op::PairIndex:
        case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
        case Op::ListData: case Op::ElemLoad: case Op::ElemStore: case Op::CheckIndex: case Op::SameList: case Op::Nop:
            return false;
        }
        return true;  // unclassified
    }
    // count_direct: also stop at Append/Pop directly on L (they re-load the cache
    // only when it is valid, so a straight-line cache gains nothing across them)
    bool kills(const Node* n, const Cand& c, bool count_direct = false) const {
        if (kills_node(n, c, count_direct)) return true;
        for (uint32_t i = 0; i < n->nkids; ++i)
            if (kills(n->kids[i], c, count_direct)) return true;
        return false;
    }
    static bool is_access(const Node* x, uint32_t L) {
        return (x->op == Op::Index || x->op == Op::SetIndex || x->op == Op::Len) && x->nkids > 0 &&
               x->kids[0]->op == Op::Local && x->kids[0]->a == (int32_t)L;
    }
    int uses(const Node* n, uint32_t L, int w) const {
        int s = is_access(n, L) ? w : 0;
        int kw = n->op == Op::While ? 2 : w;
        for (uint32_t i = 0; i < n->nkids && s < 2; ++i) s += uses(n->kids[i], L, kw);
        return s;
    }

    void setup() {
        nloc0 = f.locals.size();
        ci.assign(nloc0, -1);
        uniq.assign(nloc0, 0);
        nonneg.assign(nloc0, 0);
        genmut.assign(m.regions.size(), 0);
        for (auto& g : m.funcs)
            if (g->generic)
                for (uint32_t r : g->mutates) genmut[r] = 1;
        for (size_t l = 0; l < nloc0; ++l) uniq[l] = f.locals[l].unique && f.locals[l].type.k == TK::List;
        // candidates and Append-value uses (the analysis' `unique` ignores that
        // an Append's value is the list itself)
        // SetLocal nodes per local, with their innermost enclosing While (and whether
        // they sit in its condition)
        std::vector<std::vector<SetSite>> sets(nloc0);
        collect_sets(f.body, nullptr, false, sets);
        walk(f.body, [&](Node* x, Node* p, uint32_t idx) {
            if ((x->op == Op::Index || x->op == Op::SetIndex || x->op == Op::Len) && x->kids[0]->op == Op::Local) {
                uint32_t L = (uint32_t)x->kids[0]->a;
                const Local& lc = f.locals[L];
                if (L < nloc0 && ci[L] < 0 && lc.type.k == TK::List && lc.kind != LocalKind::Cache) {
                    ci[L] = (int)cs.size();
                    Cand c;
                    c.L = L;
                    c.r = lc.type.a;
                    cs.push_back(c);
                }
            }
            if (x->op == Op::Append && x->kids[0]->op == Op::Local && (size_t)x->kids[0]->a < nloc0) {
                bool discarded = p && (p->op == Op::Seq || (p->op == Op::Comma && idx + 1 < p->nkids) ||
                                       ((p->op == Op::If || p->op == Op::While) && idx >= 1) || p->op == Op::InlineBlock);
                if (!discarded) uniq[(size_t)x->kids[0]->a] = 0;
            }
            return true;
        });
        // non-negative induction locals
        for (size_t l = 0; l < nloc0; ++l) {
            const Local& lc = f.locals[l];
            if (lc.type.k != TK::Int || lc.kind == LocalKind::Param || lc.kind == LocalKind::Cache) continue;
            bool ok = true;
            for (auto& s : sets[l]) {
                Node* v = s.set->kids[0];
                Node* w = s.loop;
                if (v->op == Op::ConstInt && v->a >= 0) continue;
                // `i := i + 1` in the body/step of W (not in a loop nested in W), the only
                // assignment of i in W, and W's And-chain condition requires i < X: it runs
                // at most once per iteration, after the guard held with i unchanged, so
                // 0 <= i < X <= INT_MAX before it and no overflow.
                if (v->op == Op::AddI32 && v->kids[0]->op == Op::Local && v->kids[0]->a == (int32_t)l &&
                    v->kids[1]->op == Op::ConstInt && v->kids[1]->a == 1 && w && !s.in_cond &&
                    guard_in(w->kids[0], (int32_t)l) && count_sets(w, (int32_t)l) == 1)
                    continue;
                ok = false;
                break;
            }
            nonneg[l] = ok;
        }
    }
    struct SetSite {
        Node* set;
        Node* loop;
        bool in_cond;
    };
    void collect_sets(Node* n, Node* loop, bool in_cond, std::vector<std::vector<SetSite>>& out) {
        if (n->op == Op::SetLocal && (size_t)n->a < out.size()) out[(size_t)n->a].push_back({n, loop, in_cond});
        for (uint32_t i = 0; i < n->nkids; ++i) {
            if (n->op == Op::While) collect_sets(n->kids[i], n, i == 0, out);
            else collect_sets(n->kids[i], loop, in_cond, out);
        }
    }
    static int count_sets(const Node* n, int32_t id) {
        int c = n->op == Op::SetLocal && n->a == id;
        for (uint32_t i = 0; i < n->nkids; ++i) c += count_sets(n->kids[i], id);
        return c;
    }
    static bool guard_in(const Node* c, int32_t i) {
        if (c->op == Op::And && (c->flags & F_COND)) return guard_in(c->kids[0], i) || guard_in(c->kids[1], i);
        if (c->op == Op::LtI32 && c->kids[0]->op == Op::Local && c->kids[0]->a == i) return true;
        if (c->op == Op::GtI32 && c->kids[1]->op == Op::Local && c->kids[1]->a == i) return true;
        return false;
    }
    bool is_nonneg(const Node* x) const {
        if (x->op == Op::ConstInt) return x->a >= 0;
        if (x->op == Op::Local) return (size_t)x->a < nonneg.size() && nonneg[(size_t)x->a];
        return false;
    }

    // ---------------------------------------------------------------- states
    State empty() const {
        State s;
        s.valid.assign(cs.size(), 0);
        s.ep.assign(cs.size(), 0);
        s.asg.assign(nloc0, 0);
        for (uint32_t p : f.params)
            if (p < nloc0) s.asg[p] = 1;
        return s;
    }
    State dead_state() const {
        State s = empty();
        s.dead = true;
        return s;
    }
    State meet(const State& a, const State& b) {
        if (a.dead) return b;
        if (b.dead) return a;
        State r = a;
        for (size_t k = 0; k < cs.size(); ++k) {
            r.valid[k] = a.valid[k] && b.valid[k];
            r.ep[k] = (r.valid[k] && a.ep[k] == b.ep[k]) ? a.ep[k] : ++epc;
        }
        for (size_t l = 0; l < nloc0; ++l) r.asg[l] = a.asg[l] && b.asg[l];
        r.facts.clear();
        for (auto& fa : a.facts)
            if (r.valid[(size_t)fa.k] && has_fact(b, fa.idx, fa.k)) r.facts.push_back(fa);
        return r;
    }
    bool same(const State& a, const State& b) const {
        if (a.dead != b.dead || a.valid != b.valid || a.asg != b.asg || a.facts.size() != b.facts.size()) return false;
        for (auto& fa : a.facts)
            if (!has_fact(b, fa.idx, fa.k)) return false;
        return true;
    }
    static bool has_fact(const State& s, const Node* idx, int k) {
        for (auto& fa : s.facts)
            if (fa.k == k && same_expr(fa.idx, idx)) return true;
        return false;
    }
    void add_fact(State& s, Node* idx, int k) {
        if (!is_simple(idx) || has_fact(s, idx, k)) return;
        if (s.facts.size() >= 24) s.facts.erase(s.facts.begin());
        s.facts.push_back({idx, k});
    }
    void drop_facts(State& s, int k) {
        s.facts.erase(std::remove_if(s.facts.begin(), s.facts.end(), [&](const Fact& fa) { return fa.k == k; }),
                      s.facts.end());
    }
    void invalidate(State& s, int k) {
        s.valid[(size_t)k] = 0;
        s.ep[(size_t)k] = ++epc;
        drop_facts(s, k);
    }
    void establish(State& s, int k) {  // cache freshly loaded
        s.valid[(size_t)k] = 1;
        s.ep[(size_t)k] = ++epc;
    }

    // ---------------------------------------------------------------- nodes
    Cand& cache_of(int k) {
        Cand& c = cs[(size_t)k];
        if (c.dL < 0) {
            c.dL = (int32_t)new_cache_local(f, Type::ptr(c.r));
            c.nL = (int32_t)new_cache_local(f, Type::i32());
        }
        return c;
    }
    Node* lget(int32_t id) { return mk_local(m, f, (uint32_t)id); }
    std::vector<Node*> load_nodes(int k) {  // nL := Len(L); dL := ListData(L)
        Cand& c = cache_of(k);
        Node* len = m.node(Op::Len, Type::i32(), {lget((int32_t)c.L)});
        Node* dat = m.node(Op::ListData, Type::ptr(c.r), {lget((int32_t)c.L)});
        return {mk_set(m, f, (uint32_t)c.nL, len), mk_set(m, f, (uint32_t)c.dL, dat)};
    }
    int cand_of_base(const Node* b) const {
        if (b->op != Op::Local || b->a < 0 || (size_t)b->a >= ci.size()) return -1;
        return ci[(size_t)b->a];
    }

    void visit(Node*& slot, State& S, bool stmt) {
        if (S.dead) {
            State E = empty();
            visit_live(slot, E, stmt);
            return;  // S stays dead
        }
        visit_live(slot, S, stmt);
    }

    void cond(Node*& c, State& S, State& T, State& F) {
        if (S.dead) {
            State E = empty();
            State t, fl;
            cond_live(c, E, t, fl);
            T = F = S;
            return;
        }
        cond_live(c, S, T, F);
    }
    void cond_live(Node*& c, State& S, State& T, State& F) {
        Node* n = c;
        if (n->op == Op::And && (n->flags & F_COND)) {
            State At, Af, Bt, Bf;
            cond(n->kids[0], S, At, Af);
            cond(n->kids[1], At, Bt, Bf);
            T = Bt;
            F = meet(Af, Bf);
            return;
        }
        if (n->op == Op::Or && (n->flags & F_COND)) {
            State At, Af, Bt, Bf;
            cond(n->kids[0], S, At, Af);
            cond(n->kids[1], Af, Bt, Bf);
            T = meet(At, Bt);
            F = Bf;
            return;
        }
        // guard pattern  i < |L|  /  |L| > i
        Node* gi = nullptr;
        int gk = -1;
        if (n->op == Op::LtI32 && n->kids[1]->op == Op::Len) {
            gi = n->kids[0];
            gk = cand_of_base(n->kids[1]->kids[0]);
        } else if (n->op == Op::GtI32 && n->kids[0]->op == Op::Len) {
            gi = n->kids[1];
            gk = cand_of_base(n->kids[0]->kids[0]);
        }
        visit(c, S, false);
        T = S;
        F = S;
        if (gk >= 0 && !S.dead && T.valid[(size_t)gk] && is_simple(gi) && is_nonneg(gi)) add_fact(T, gi, gk);
    }

    bool stmt_kid(const Node* p, uint32_t i) const {
        switch (p->op) {
        case Op::Seq: return true;
        case Op::If: case Op::While: return i >= 1;
        case Op::InlineBlock: return true;
        default: return false;
        }
    }

    void visit_live(Node*& slot, State& S, bool stmt) {
        Node* n = slot;
        switch (n->op) {
        case Op::Seq:
            for (uint32_t k = 0; k < n->nkids; ++k) {
                std::vector<Node*> est;
                if (!S.dead) seq_establish(n, k, S, est);
                visit(n->kids[k], S, true);
                if (!dry && !est.empty()) {
                    est.push_back(n->kids[k]);
                    n->kids[k] = mk_seq(m, est);
                }
            }
            return;
        case Op::If: {
            State T, F;
            cond(n->kids[0], S, T, F);
            if (n->nkids > 1) visit(n->kids[1], T, true);
            if (n->nkids > 2) visit(n->kids[2], F, true);
            S = meet(T, F);
            return;
        }
        case Op::While:
            visit_while(slot, S);
            return;
        case Op::InlineBlock: {
            labels.push_back({n->a, dead_state()});
            visit(n->kids[0], S, true);
            State b = labels.back().second;
            labels.pop_back();
            S = meet(S, b);
            return;
        }
        case Op::Break:
            for (size_t i = labels.size(); i-- > 0;)
                if (labels[i].first == n->a) {
                    labels[i].second = meet(labels[i].second, S);
                    break;
                }
            S = dead_state();
            return;
        case Op::Return:
            for (uint32_t i = 0; i < n->nkids; ++i) visit(n->kids[i], S, false);
            S = dead_state();
            return;
        case Op::And: case Op::Or: {
            visit(n->kids[0], S, false);
            State S1 = S;
            visit(n->kids[1], S, false);
            S = meet(S1, S);
            return;
        }
        case Op::Len: {
            visit(n->kids[0], S, false);
            int k = cand_of_base(n->kids[0]);
            if (k >= 0 && !S.dead && S.valid[(size_t)k] && !dry) slot = lget(cache_of(k).nL);
            return;
        }
        case Op::Index: lower_index(slot, S); return;
        case Op::SetIndex: lower_set_index(slot, S); return;
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) visit(n->kids[i], S, stmt_kid(n, i));
            if (!S.dead) event(slot, S, stmt);
            return;
        }
    }

    void lower_index(Node*& slot, State& S) {
        Node* n = slot;
        visit(n->kids[0], S, false);
        int k = cand_of_base(n->kids[0]);
        bool v0 = k >= 0 && !S.dead && S.valid[(size_t)k];
        uint32_t ep0 = k >= 0 ? S.ep[(size_t)k] : 0;
        visit(n->kids[1], S, false);
        if (!v0 || S.dead || !S.valid[(size_t)k] || S.ep[(size_t)k] != ep0) return;
        Cand& c = cs[(size_t)k];
        Node* i = n->kids[1];
        Type et = m.regions[c.r].elem;
        if (i->type.k != TK::Int || et.k == TK::Bot || n->type != et) return;
        bool known = has_fact(S, i, k);
        if (!dry) {
            Cand& cc = cache_of(k);
            Node* ix = known ? i : m.node(Op::CheckIndex, Type::i32(), {i, lget(cc.nL)});
            Node* ld = m.node(Op::ElemLoad, et, {lget(cc.dL), ix});
            ld->line = n->line;
            slot = ld;
        }
        if (!known) add_fact(S, i, k);
    }

    void lower_set_index(Node*& slot, State& S) {
        Node* n = slot;
        visit(n->kids[0], S, false);
        int k = cand_of_base(n->kids[0]);
        bool v0 = k >= 0 && !S.dead && S.valid[(size_t)k];
        uint32_t ep0 = k >= 0 ? S.ep[(size_t)k] : 0;
        visit(n->kids[1], S, false);
        visit(n->kids[2], S, false);
        if (!v0 || S.dead || !S.valid[(size_t)k] || S.ep[(size_t)k] != ep0) return;
        Cand& c = cs[(size_t)k];
        Node* i = n->kids[1];
        Node* v = n->kids[2];
        Type et = m.regions[c.r].elem;
        if (i->type.k != TK::Int || et.k == TK::Bot || v->type != et || n->type != et) return;
        int form = is_pure(v) ? 0 : (is_simple(i) && !sets_any_local_of(v, i)) ? 1 : 2;
        bool known = form != 2 && has_fact(S, i, k);
        if (!dry) {
            Cand& cc = cache_of(k);
            auto chk = [&](Node* ix) { return known ? ix : m.node(Op::CheckIndex, Type::i32(), {ix, lget(cc.nL)}); };
            Node* r;
            if (form == 0) {
                r = m.node(Op::ElemStore, et, {lget(cc.dL), chk(i), v});
            } else if (form == 1) {
                uint32_t tv = m.new_temp(&f, et);
                Node* st = m.node(Op::ElemStore, et, {lget(cc.dL), chk(i), lget((int32_t)tv)});
                r = m.node(Op::Comma, et, {mk_set(m, f, tv, v), st});
            } else {
                uint32_t ti = m.new_temp(&f, Type::i32());
                uint32_t tv = m.new_temp(&f, et);
                Node* st = m.node(Op::ElemStore, et,
                                  {lget(cc.dL), m.node(Op::CheckIndex, Type::i32(), {lget((int32_t)ti), lget(cc.nL)}),
                                   lget((int32_t)tv)});
                r = m.node(Op::Comma, et, {mk_set(m, f, ti, i), mk_set(m, f, tv, v), st});
            }
            r->line = n->line;
            slot = r;
        }
        if (form != 2 && !known) add_fact(S, i, k);
    }

    // Post-kids effect of node *slot on the state (may rewrite direct Append/Pop).
    void event(Node*& slot, State& S, bool stmt) {
        Node* x = slot;
        if (x->op == Op::SetLocal) {
            if ((size_t)x->a < S.asg.size()) S.asg[(size_t)x->a] = 1;
            S.facts.erase(std::remove_if(S.facts.begin(), S.facts.end(),
                                         [&](const Fact& fa) { return mentions_local(fa.idx, x->a); }),
                          S.facts.end());
            if ((size_t)x->a < ci.size() && ci[(size_t)x->a] >= 0) invalidate(S, ci[(size_t)x->a]);
            return;
        }
        int dk = -1;
        if ((x->op == Op::Append || x->op == Op::Pop) && x->kids[0]->type.k == TK::List) dk = cand_of_base(x->kids[0]);
        bool was = dk >= 0 && S.valid[(size_t)dk];
        for (size_t k = 0; k < cs.size(); ++k)
            if ((int)k != dk && S.valid[k] && kills_node(x, cs[k], true)) invalidate(S, (int)k);
        if (dk < 0) return;
        bool can = was && (stmt || x->op == Op::Append || x->type.k != TK::Bot);
        if (!can) {
            invalidate(S, dk);
            return;
        }
        if (!dry) {
            std::vector<Node*> ld = load_nodes(dk);
            Node* r;
            if (stmt) {
                r = mk_seq(m, {x, ld[0], ld[1]});
            } else if (x->op == Op::Append) {
                r = m.node(Op::Comma, x->type, {x, ld[0], ld[1], lget(x->kids[0]->a)});
            } else {
                uint32_t t = m.new_temp(&f, x->type);
                r = m.node(Op::Comma, x->type, {mk_set(m, f, t, x), ld[0], ld[1], lget((int32_t)t)});
            }
            r->line = x->line;
            slot = r;
        }
        establish(S, dk);
        if (x->op == Op::Pop) drop_facts(S, dk);
    }

    void seq_establish(Node* seq, uint32_t k, State& S, std::vector<Node*>& est) {
        for (size_t c = 0; c < cs.size(); ++c) {
            if (S.valid[c] || !S.asg[cs[c].L]) continue;
            int cnt = 0;
            for (uint32_t j = k; j < seq->nkids && cnt < 2; ++j) {
                if (kills(seq->kids[j], cs[c], true)) break;
                cnt += uses(seq->kids[j], cs[c].L, 1);
            }
            if (cnt < 2) continue;
            if (!dry) {
                std::vector<Node*> ld = load_nodes((int)c);
                est.insert(est.end(), ld.begin(), ld.end());
            }
            establish(S, (int)c);
            drop_facts(S, (int)c);
        }
    }

    // One pass over the loop from head state H; returns the back-edge state, F = exit state.
    // dry_run: a fixpoint probe (no rewriting; Break states it records are discarded).
    State run_loop(Node* W, const State& H, bool dry_run, State& F) {
        bool saved_dry = dry;
        std::vector<std::pair<int32_t, State>> saved_labels;
        if (dry_run) {
            dry = true;
            saved_labels = labels;
        }
        State T;
        State h = H;
        cond(W->kids[0], h, T, F);
        if (W->nkids > 1) visit(W->kids[1], T, true);
        if (W->nkids > 2) visit(W->kids[2], T, true);
        if (dry_run) labels = saved_labels;
        dry = saved_dry;
        return T;
    }

    void visit_while(Node*& slot, State& S) {
        Node* W = slot;
        std::vector<int> pre;
        State H = S;
        for (size_t c = 0; c < cs.size(); ++c) {
            if (S.valid[c] || !S.asg[cs[c].L]) continue;
            if (uses(W, cs[c].L, 1) == 0 || kills(W, cs[c])) continue;
            establish(H, (int)c);
            drop_facts(H, (int)c);
            pre.push_back((int)c);
        }
        bool converged = false;
        for (int it = 0; it < 6 && !loops_off; ++it) {
            State F;
            State St = run_loop(W, H, true, F);
            State Hn = meet(H, St);
            if (same(Hn, H)) {
                converged = true;
                break;
            }
            H = Hn;
        }
        if (!converged) {  // give up on caches/facts for this loop: trivially a fixpoint
            for (size_t c = 0; c < cs.size(); ++c) invalidate(H, (int)c);
            H.facts.clear();
        }
        std::vector<Node*> est;
        if (!dry)
            for (int c : pre)
                if (H.valid[(size_t)c]) {
                    std::vector<Node*> ld = load_nodes(c);
                    est.insert(est.end(), ld.begin(), ld.end());
                }
        State F;
        run_loop(W, H, false, F);  // the real visit (still dry if an enclosing probe is)
        S = F;
        if (!est.empty()) {
            est.push_back(W);
            slot = mk_seq(m, est);
        }
    }

    void run() {
        setup();
        if (cs.empty()) return;
        int maxd = 0;
        walk_depth(f.body, [&](Node*, int d) { maxd = std::max(maxd, d); });
        loops_off = maxd > 9;
        State S = empty();
        visit(f.body, S, true);
    }
};

}  // namespace

void pass_listopt(Module& m) {
    pass_region_info(m);  // fresh summaries / uniqueness after the earlier passes
    for (auto& fp : m.funcs) {
        LO lo(m, *fp);
        lo.run();
    }
}

}  // namespace epq::tir
