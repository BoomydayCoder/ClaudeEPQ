// Pass "appendopt" — append-loop optimisation: capacity hints (Reserve) and bulk
// copies (AppendRange). Ops: src/tir.hpp; runtime: epq_list_reserve /
// epq_append_range in runtime/epq_rt.h. Runs after ifconv and before listopt, so
// it sees high-level Index / Append (listopt's lowering would hide the patterns).
//
// 1. Bulk copy. A loop (for, or while with the increment at the end of the body)
//        while (i < b) { L :+ M[i]; i := i + 1; }
//    where L, M are distinct List locals with the same element type, i an Int
//    local, b a pure Int expression that mentions neither i nor any list of L's
//    region other than M itself (so b cannot change while only L grows, given
//    L != M), becomes
//        If(AppendRange(L, M, i, b), i := Select(i < b, b, i), <original loop>)
//    AppendRange copies m[i..b) and yields 1 only when the original loop would
//    have done exactly that without failing (L != M at run time, the range inside
//    M); otherwise it does nothing and the original loop runs, so errors fire at
//    the same iteration after the same appends. Kids are pure, so evaluating i and
//    b once more is unobservable.
// 2. Capacity hints, inserted just before a loop: Reserve(L, estimate) for every
//    list local L (not assigned in the loop) appended to unconditionally at the
//    top statement level of the body:
//      counted loops  while (i < b | i <= b) {...; i := i + c}  (c >= 1 const, b pure,
//                     no i in b, the step is the only assignment of i):
//                     (b - i [+1]) / c x appends-per-iteration
//      for-each       |list| - idx  x appends-per-iteration (L not in the iterated
//                     list's region)
//      merge shape    while (i < |A| & j < |B|) { if (..) {L :+ ..} else {L :+ ..} }
//                     (|A| - i) + (|B| - j)   x min appends per branch (+ top-level appends)
//    Reserve never fails and never changes contents; the hint is capped at 2^26
//    elements in the runtime, and negative / wrapped estimates do nothing.
//
// Safe on any TIR: only exact shapes are touched; anything else is left alone.
#include "passes.hpp"
#include "util.hpp"

#include <map>

namespace epq::tir {

namespace {

using namespace util;

struct AppendOpt {
    Module& M;
    Func& F;
    int copies = 0, hints = 0;

    // Flatten nested Seqs (and skip Nops) into top-level statements.
    static void flat(Node* n, std::vector<Node*>& out) {
        if (n->op == Op::Seq) {
            for (uint32_t i = 0; i < n->nkids; ++i) flat(n->kids[i], out);
        } else if (n->op != Op::Nop) {
            out.push_back(n);
        }
    }
    bool is_local(const Node* n, TK k) const { return n->op == Op::Local && F.locals[(size_t)n->a].type.k == k; }
    bool int_local(const Node* n) const { return n->op == Op::Local && n->type.k == TK::Int && F.locals[(size_t)n->a].type.k == TK::Int; }
    // SetLocal(i, AddI32(Local i, ConstInt c)) with c >= 1: returns c, else 0.
    static int32_t incr_of(const Node* s, int32_t i) {
        if (s->op != Op::SetLocal || s->a != i) return 0;
        const Node* v = s->kids[0];
        if (v->op != Op::AddI32) return 0;
        if (v->kids[0]->op != Op::Local || v->kids[0]->a != i) return 0;
        if (v->kids[1]->op != Op::ConstInt || v->kids[1]->a < 1) return 0;
        return v->kids[1]->a;
    }
    static bool mentions_local(const Node* n, int32_t id) {
        bool r = false;
        walk(n, [&](Node* x, Node*, uint32_t) {
            if ((x->op == Op::Local || x->op == Op::SetLocal) && x->a == id) r = true;
            return !r;
        });
        return r;
    }

    // ---- 1. bulk copy ---------------------------------------------------------------
    Node* try_copy(Node* w) {
        if (w->flags & F_FOREACH) return nullptr;
        Node* cond = w->kids[0];
        if (cond->op != Op::LtI32 || !int_local(cond->kids[0])) return nullptr;
        int32_t i = cond->kids[0]->a;
        Node* b = cond->kids[1];
        if (b->type.k != TK::Int || !is_pure(b) || mentions_local(b, i)) return nullptr;
        std::vector<Node*> st;
        flat(w->kids[1], st);
        flat(w->kids[2], st);
        if (st.size() != 2) return nullptr;
        Node* ap = st[0];
        if (ap->op != Op::Append || incr_of(st[1], i) != 1) return nullptr;
        Node* L = ap->kids[0];
        Node* ix = ap->kids[1];
        if (!is_local(L, TK::List) || ix->op != Op::Index) return nullptr;
        Node* Ml = ix->kids[0];
        if (!is_local(Ml, TK::List) || ix->kids[1]->op != Op::Local || ix->kids[1]->a != i) return nullptr;
        if (L->a == Ml->a) return nullptr;
        uint32_t rl = L->type.a, rm = Ml->type.a;
        if (M.regions[rl].pair || M.regions[rm].pair) return nullptr;
        if (elem_of(M, L->type) != elem_of(M, Ml->type) || M.elem_is_i32(rl) != M.elem_is_i32(rm)) return nullptr;
        if (ap->kids[1]->type != elem_of(M, L->type)) return nullptr;
        // b must not observe L's growth: no value of L's region in b except M itself.
        bool bad = false;
        walk(b, [&](Node* x, Node*, uint32_t) {
            if ((x->type.k == TK::List || x->type.k == TK::Ptr) && x->type.a == rl && !(x->op == Op::Local && x->a == Ml->a)) bad = true;
            if (x->type.k == TK::Any) bad = true;  // be conservative about dynamic values
            return !bad;
        });
        if (bad) return nullptr;
        Node* ar = M.node(Op::AppendRange, Type::i32(),
                          {mk_local(M, F, (uint32_t)L->a, w->line), mk_local(M, F, (uint32_t)Ml->a, w->line),
                           mk_local(M, F, (uint32_t)i, w->line), clone(M, b)});
        ar->line = w->line;
        Node* lt = M.node(Op::LtI32, Type::i32(), {mk_local(M, F, (uint32_t)i), clone(M, b)});
        Node* sel = M.node(Op::Select, Type::i32(), {lt, clone(M, b), mk_local(M, F, (uint32_t)i)});
        Node* fin = mk_set(M, F, (uint32_t)i, sel);
        Node* iff = M.node(Op::If, Type::bot(), {ar, fin, w});
        iff->line = w->line;
        ++copies;
        return iff;
    }

    // ---- 2. capacity hints ------------------------------------------------------------
    // Appends per iteration to each List local, counted at the top statement level.
    void count_appends(const std::vector<Node*>& st, std::map<int32_t, int>& cnt) {
        for (Node* s : st)
            if (s->op == Op::Append && is_local(s->kids[0], TK::List)) cnt[s->kids[0]->a]++;
    }

    Node* try_hint(Node* w) {
        std::vector<Node*> body;
        flat(w->kids[1], body);
        std::map<int32_t, int> cnt;
        count_appends(body, cnt);
        Node* trips = nullptr;  // estimate of the number of iterations (pure Int)
        Node* cond = w->kids[0];
        auto len_of = [&](Node* x) -> int32_t {  // Len(Local list) -> local id, else -1
            if (x->op == Op::Len && is_local(x->kids[0], TK::List)) return x->kids[0]->a;
            return -1;
        };
        int32_t skip_region = -1;
        if (w->flags & F_FOREACH) {
            int32_t idx = w->a, lst = w->b;
            if (idx < 0 || lst < 0 || F.locals[(size_t)lst].type.k != TK::List || F.locals[(size_t)idx].type.k != TK::Int) return nullptr;
            skip_region = (int32_t)F.locals[(size_t)lst].type.a;
            Node* len = M.node(Op::Len, Type::i32(), {mk_local(M, F, (uint32_t)lst)});
            trips = M.node(Op::SubI32, Type::i32(), {len, mk_local(M, F, (uint32_t)idx)});
        } else if ((cond->op == Op::LtI32 || cond->op == Op::LeI32) && int_local(cond->kids[0])) {
            int32_t i = cond->kids[0]->a;
            Node* b = cond->kids[1];
            if (b->type.k != TK::Int || !is_pure(b) || mentions_local(b, i)) return nullptr;
            // exactly one assignment of i in the loop: the increment, at top level
            std::vector<Node*> all = body;
            flat(w->kids[2], all);
            int32_t c = 0;
            int nset = 0;
            for (Node* s : all)
                if ((c = incr_of(s, i)) != 0) break;
            // count SetLocal(i) nodes anywhere
            walk(w, [&](Node* x, Node*, uint32_t) {
                if (x->op == Op::SetLocal && x->a == i) nset++;
                return true;
            });
            if (c < 1 || nset != 1) return nullptr;
            Node* t = M.node(Op::SubI32, Type::i32(), {clone(M, b), mk_local(M, F, (uint32_t)i)});
            if (cond->op == Op::LeI32) t = M.node(Op::AddI32, Type::i32(), {t, mk_int(M, 1)});
            if (c > 1) t = M.node(Op::DivI32, Type::i32(), {t, mk_int(M, c)});  // constant divisor >= 2: cannot fail
            trips = t;
        } else if (cond->op == Op::And && cond->nkids == 2 && cond->kids[0]->op == Op::LtI32 && cond->kids[1]->op == Op::LtI32 &&
                   int_local(cond->kids[0]->kids[0]) && int_local(cond->kids[1]->kids[0]) &&
                   cond->kids[0]->kids[0]->a != cond->kids[1]->kids[0]->a) {
            int32_t a = len_of(cond->kids[0]->kids[1]), b = len_of(cond->kids[1]->kids[1]);
            if (a < 0 || b < 0) return nullptr;
            // merge shape: one If at top level whose both branches append to L
            Node* iff = nullptr;
            for (Node* s : body)
                if (s->op == Op::If) {
                    if (iff) return nullptr;
                    iff = s;
                }
            if (!iff) return nullptr;
            std::vector<Node*> tb, eb;
            flat(iff->kids[1], tb);
            flat(iff->kids[2], eb);
            std::map<int32_t, int> ct, ce;
            count_appends(tb, ct);
            count_appends(eb, ce);
            for (auto& [l, k] : ct)
                if (ce.count(l)) cnt[l] += std::min(k, ce[l]);
            Node* ra = M.node(Op::SubI32, Type::i32(),
                              {clone(M, cond->kids[0]->kids[1]), mk_local(M, F, (uint32_t)cond->kids[0]->kids[0]->a)});
            Node* rb = M.node(Op::SubI32, Type::i32(),
                              {clone(M, cond->kids[1]->kids[1]), mk_local(M, F, (uint32_t)cond->kids[1]->kids[0]->a)});
            trips = M.node(Op::AddI32, Type::i32(), {ra, rb});
        } else {
            return nullptr;
        }
        if (cnt.empty()) return nullptr;
        std::vector<bool> asg;
        locals_assigned(w, asg);
        std::vector<Node*> out;
        for (auto& [l, k] : cnt) {
            if ((size_t)l < asg.size() && asg[(size_t)l]) continue;  // L names different lists over the loop
            uint32_t rl = F.locals[(size_t)l].type.a;
            if ((int32_t)rl == skip_region) continue;
            if (M.regions[rl].pair) continue;
            Node* h = out.empty() ? trips : clone(M, trips);
            if (k > 1) h = M.node(Op::MulI32, Type::i32(), {h, mk_int(M, k)});
            Node* r = M.node(Op::Reserve, Type::bot(), {mk_local(M, F, (uint32_t)l, w->line), h});
            r->line = w->line;
            out.push_back(r);
            ++hints;
        }
        if (out.empty()) return nullptr;
        out.push_back(w);
        return mk_seq(M, out);
    }

    // Post-order over statements: inner loops first; replacements are not revisited.
    void stmt(Node*& slot) {
        Node* n = slot;
        switch (n->op) {
        case Op::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) stmt(n->kids[i]);
            return;
        case Op::If:
            stmt(n->kids[1]);
            stmt(n->kids[2]);
            return;
        case Op::InlineBlock:
            stmt(n->kids[0]);
            return;
        case Op::While: {
            stmt(n->kids[1]);
            stmt(n->kids[2]);
            if (Node* r = try_copy(n)) {
                slot = r;
                return;
            }
            if (Node* r = try_hint(n)) slot = r;
            return;
        }
        default:
            return;  // expressions (incl. InlineBlocks inside expressions) are left alone
        }
    }
};

}  // namespace

void pass_appendopt(Module& m) {
    for (auto& f : m.funcs) {
        AppendOpt A{m, *f};
        A.stmt(f->body);
    }
}

}  // namespace epq::tir
