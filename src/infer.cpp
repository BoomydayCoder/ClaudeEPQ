#include "infer.hpp"

#include <algorithm>

#include <string>

namespace epq::infer {

bool completes(const Node* n) {
    switch (n->kind) {
    case NK::Seq:
        for (uint32_t i = 0; i < n->nkids; ++i)
            if (!completes(n->kids[i])) return false;
        return true;
    case NK::Block:
        return completes(n->kids[0]);
    case NK::If:
        return completes(n->kids[1]) || completes(n->kids[2]);
    case NK::Return:
        return false;
    case NK::While: {
        const Node* c = n->kids[0];
        bool always = (c->kind == NK::Int && c->ival != 0) || c->kind == NK::List || c->kind == NK::FuncLit;
        return !always;
    }
    case NK::For: {
        if (!completes(n->kids[0])) return false;
        const Node* c = n->kids[1];
        bool always = (c->kind == NK::Int && c->ival != 0) || c->kind == NK::List || c->kind == NK::FuncLit;
        return !always;
    }
    default:
        return true;
    }
}

namespace {

struct Solver {
    const Program& P;
    const Config& C;
    Result& R;
    uint64_t changes = 0;
    uint32_t cur = 0;  // instance being walked
    Instance* CI = nullptr;  // R.insts[cur] (Instances never move: unique_ptr)
    std::vector<uint32_t> count_of;  // DefId -> number of non-generic instances

    // Per-def tables computed once (flat arrays + offsets, DefId-indexed):
    //  wo       ordinals of the expression nodes in the exact order ex() visits
    //           them, so the fixpoint walk never looks a node up;
    //  calls    (ordinal of Call, ordinal of its callee expression), preorder;
    //  lits     ordinals of List literals, preorder;
    //  unclean  LF_UNCLEAN local slots.
    std::vector<uint32_t> wo, wo_off, calls, calls_off, lits, lits_off, unclean, unclean_off;
    std::vector<uint8_t> body_completes;
    const uint32_t* wp = nullptr;  // cursor into wo during a walk
    // Records (DESIGN.md §5.11). Global ordinal of node o of def d = node_base[d] + o.
    // lit_state[global ordinal of a List literal]: 0 ordinary, 1 record candidate,
    // 2 demoted (never a candidate again). Empty in run 0.
    std::vector<uint32_t> node_base;
    std::vector<uint8_t> lit_state;

    Solver(const Program& p, const Config& c, Result& r) : P(p), C(c), R(r) {}

    // ---- lattice -------------------------------------------------------------
    uint32_t find(uint32_t e) {
        while (R.evars[e].parent != e) {
            R.evars[e].parent = R.evars[R.evars[e].parent].parent;
            e = R.evars[e].parent;
        }
        return e;
    }
    bool same(Ty a, Ty b) {
        if (a.k != b.k) return false;
        if (a.k == TyK::List) return find(a.a) == find(b.a);
        if (a.k == TyK::Fn) return a.a == b.a;
        return true;
    }
    void escape(DefId d) {
        if (R.escaped[d]) return;
        R.escaped[d] = 1;
        ++changes;
        uint32_t id = new_instance(d, -1);
        Instance& I = *R.insts[id];
        I.generic = true;
        for (uint32_t s : P.funcs[d].params) I.locals[s] = Ty::any();
        I.ret = Ty::any();
        R.generic_of[d] = (int32_t)id;
    }
    // A value of type t is stored into an Any slot.
    void make_any(Ty t) {
        if (t.k == TyK::List) {
            uint32_t r = find(t.a);
            if (R.evars[r].nf) {  // a record flows into Any: violation (condition 4)
                if (R.evars[r].flags & EV_BAD) return;
                R.evars[r].flags |= EV_BAD;
                // As an ordinary list it would have Any elements: dynamise the fields.
                for (uint32_t j = 0; j < R.evars[r].nf; ++j) make_any(R.evars[find(R.evars[r].fbase + j)].t);
                return;
            }
            R.evars[r].flags |= EV_PLAIN;  // reachable from dynamic code: never a record
            if (R.evars[r].t.k != TyK::Any) {
                Ty old = R.evars[r].t;
                R.evars[r].t = Ty::any();
                ++changes;
                make_any(old);
            }
        } else if (t.k == TyK::Fn) {
            escape(t.a);
        }
    }
    void unify(uint32_t x, uint32_t y) {
        x = find(x);
        y = find(y);
        if (x == y) return;
        R.evars[y].parent = x;
        ++changes;
        auto& ys = R.evars[y].sites;
        auto& xs = R.evars[x].sites;
        xs.insert(xs.end(), ys.begin(), ys.end());
        ys.clear();
        ys.shrink_to_fit();
        if (R.evars[x].nf || R.evars[y].nf) {
            unify_rec(x, y);
            return;
        }
        R.evars[x].flags |= R.evars[y].flags & (EV_PLAIN | EV_BAD);
        Ty ty = R.evars[y].t;
        Ty res = lub(R.evars[x].t, ty);
        uint32_t rx = find(x);
        Ty curt = R.evars[rx].t;
        if (rx != x) res = lub(curt, res);
        if (!same(res, curt)) R.evars[rx].t = res;
    }
    // Records (§5.11): x (new root) and y (now x's child), at least one a record.
    // Two records of the same length unify position-wise (records are mutable
    // references, like the List rule of §2.2); a record meeting an ordinary list
    // or a record of another length is a violation (condition 5). A record
    // meeting an undetermined region (only `[]` sites so far) absorbs it; values
    // stored into that region by literal index flow (conservatively) into every field.
    void unify_rec(uint32_t x, uint32_t y) {
        uint8_t fx = R.evars[x].flags, fy = R.evars[y].flags;
        uint32_t nx = R.evars[x].nf, ny = R.evars[y].nf;
        R.evars[x].flags = fx | (fy & (EV_PLAIN | EV_BAD));
        if (nx && ny) {
            if (nx != ny) {
                R.evars[x].flags |= EV_BAD;
                return;
            }
            uint32_t bx = R.evars[x].fbase, by = R.evars[y].fbase;
            for (uint32_t j = 0; j < nx; ++j) unify(bx + j, by + j);
            return;
        }
        uint8_t of = nx ? fy : fx;
        Ty ot = nx ? R.evars[y].t : R.evars[x].t;
        if (!nx) {
            R.evars[x].fbase = R.evars[y].fbase;
            R.evars[x].nf = (uint8_t)ny;
            R.evars[x].t = Ty::bot();
        }
        if (of & EV_PLAIN) {
            R.evars[x].flags |= EV_BAD;
            return;
        }
        if (ot.k != TyK::Bot) {
            uint32_t b = R.evars[x].fbase, k = R.evars[x].nf;
            for (uint32_t j = 0; j < k; ++j) flow_elem(b + j, ot);
        }
    }
    void bad(uint32_t e) { R.evars[find(e)].flags |= EV_BAD; }
    static bool lit_index(const Node* i, int64_t& j) {
        if (i->kind != NK::Int) return false;
        j = i->ival;
        return true;
    }
    // l[i] where l : List(e). A record needs a literal index (condition 2).
    Ty index_read(Ty l, const Node* idx) {
        uint32_t r = find(l.a);
        int64_t j;
        bool lit = lit_index(idx, j);
        if (R.evars[r].nf) {
            if (!lit) {
                bad(r);
                return Ty::bot();
            }
            if (j < 0 || j >= R.evars[r].nf) return Ty::bot();  // always Index out of bounds
            return R.evars[find(R.evars[r].fbase + (uint32_t)j)].t;
        }
        if (!lit) R.evars[r].flags |= EV_PLAIN;
        return R.evars[r].t;
    }
    // l[i] := v
    void index_store(Ty l, const Node* idx, Ty v) {
        if (l.k == TyK::Any) {
            make_any(v);
            return;
        }
        if (l.k != TyK::List) return;
        uint32_t r = find(l.a);
        int64_t j;
        bool lit = lit_index(idx, j);
        if (R.evars[r].nf) {
            if (!lit) bad(r);
            else if (j >= 0 && j < R.evars[r].nf) flow_elem(R.evars[r].fbase + (uint32_t)j, v);
            return;
        }
        if (!lit) R.evars[r].flags |= EV_PLAIN;
        flow_elem(l.a, v);
    }
    // A list-only use of l : List(e) (append, pop, for-each; condition 3).
    // Returns false (and flags the violation) if l is a record.
    bool list_use(Ty l) {
        uint32_t r = find(l.a);
        if (R.evars[r].nf) {
            R.evars[r].flags |= EV_BAD;
            return false;
        }
        R.evars[r].flags |= EV_PLAIN;
        return true;
    }

    Ty lub(Ty a, Ty b) {
        if (b.k == TyK::Bot) return a;
        if (a.k == TyK::Bot) return b;
        if (a.k == TyK::Any) {
            make_any(b);
            return a;
        }
        if (b.k == TyK::Any) {
            make_any(a);
            return b;
        }
        if (a.k == b.k) {
            if (a.k == TyK::Int) return a;
            if (a.k == TyK::List) {
                unify(a.a, b.a);
                return a;
            }
            if (a.a == b.a) return a;  // Fn, same def
        }
        make_any(a);
        make_any(b);
        return Ty::any();
    }
    void flow(Ty& dst, Ty src) {
        Ty r = lub(dst, src);
        if (!same(r, dst)) {
            dst = r;
            ++changes;
        }
    }
    void flow_elem(uint32_t e, Ty v) {
        uint32_t r = find(e);
        Ty res = lub(R.evars[r].t, v);
        uint32_t r2 = find(e);
        Ty curt = R.evars[r2].t;
        if (r2 != r) res = lub(curt, res);
        if (!same(res, curt)) {
            R.evars[r2].t = res;
            ++changes;
        }
    }

    // ---- instances -----------------------------------------------------------
    uint32_t new_instance(DefId d, int32_t parent) {
        auto I = std::make_unique<Instance>();
        I->id = (uint32_t)R.insts.size();
        I->def = d;
        I->parent = parent;
        I->locals.assign(P.funcs[d].locals.size(), Ty::bot());
        I->nty.assign(R.nnodes[d], Ty::bot());
        I->naux.assign(R.nnodes[d], kNone);
        I->rep = I->id;
        R.insts.push_back(std::move(I));
        ++changes;
        return (uint32_t)R.insts.size() - 1;
    }
    uint32_t shared_instance(DefId d) {
        if (R.shared_of[d] < 0) {
            uint32_t id = new_instance(d, -1);
            R.insts[id]->shared = true;
            R.shared_of[d] = (int32_t)id;
            ++count_of[d];
        }
        return (uint32_t)R.shared_of[d];
    }
    uint32_t callee_instance(uint32_t o, DefId d) {
        uint32_t& slot = R.insts[cur]->naux[o];
        if (slot != kNone && R.insts[slot]->def == d) return slot;
        uint32_t res;
        if (!C.poly) {
            res = shared_instance(d);
        } else {
            res = kNone;
            for (int32_t p = (int32_t)cur; p >= 0; p = R.insts[p]->parent)
                if (R.insts[p]->def == d) {
                    res = (uint32_t)p;
                    break;
                }
            if (res == kNone) {
                if (count_of[d] >= C.max_per_def || R.insts.size() >= C.max_total) {
                    res = shared_instance(d);
                } else {
                    res = new_instance(d, (int32_t)cur);
                    ++count_of[d];
                }
            }
        }
        R.insts[cur]->naux[o] = res;  // (slot may dangle after new_instance? no: naux not resized)
        ++changes;
        return res;
    }

    // ---- walk ----------------------------------------------------------------
    Instance& I() { return *CI; }

    Ty& var(const Node* id) {
        if (id->flags & NF_GLOBAL) return R.globals[(size_t)id->aux];
        return I().locals[(size_t)id->aux];
    }
    void store_into(Ty l, Ty v) {
        if (l.k == TyK::List) flow_elem(l.a, v);
        else if (l.k == TyK::Any) make_any(v);
    }

    Ty ex(const Node* n, bool cond = false) {
        const uint32_t o = *wp++;  // this node's ordinal (ex visiting order == wnum_ex order)
        Ty t = Ty::bot();
        switch (n->kind) {
        case NK::Int:
        case NK::Input:
            t = Ty::i32();
            break;
        case NK::Id:
            t = var(n);
            break;
        case NK::Assign: {
            const Node* tg = n->kids[0];
            if (tg->kind == NK::Id) {
                Ty v = ex(n->kids[1]);
                flow(var(tg), v);
                t = v;
            } else {
                Ty l = ex(tg->kids[0]);
                ex(tg->kids[1]);
                Ty v = ex(n->kids[1]);
                index_store(l, tg->kids[1], v);
                t = v;
            }
            break;
        }
        case NK::Append: {
            Ty l = ex(n->kids[0]);
            Ty v = ex(n->kids[1]);
            if (l.k != TyK::List || list_use(l)) store_into(l, v);
            t = (l.k == TyK::List || l.k == TyK::Any) ? l : Ty::bot();
            break;
        }
        case NK::Add: case NK::Sub: case NK::Mul: case NK::Div:
        case NK::Eq: case NK::Gt: case NK::Lt: case NK::Ge: case NK::Le: case NK::Ne:
            ex(n->kids[0]);
            ex(n->kids[1]);
            t = Ty::i32();
            break;
        case NK::Not:
        case NK::Abs:
            ex(n->kids[0]);
            t = Ty::i32();
            break;
        case NK::Neg: {
            Ty x = ex(n->kids[0]);
            if (x.k == TyK::Int) t = x;
            else if (x.k == TyK::List) t = list_use(x) ? R.evars[find(x.a)].t : Ty::bot();
            else if (x.k == TyK::Any) t = x;
            break;
        }
        case NK::And:
        case NK::Or: {
            Ty a = ex(n->kids[0], cond);
            Ty b = ex(n->kids[1], cond);
            t = cond ? Ty::i32() : lub(a, b);
            break;
        }
        case NK::Index: {
            Ty l = ex(n->kids[0]);
            Ty i = ex(n->kids[1]);
            bool iok = i.k == TyK::Int || i.k == TyK::Any;
            if (l.k == TyK::List && iok) t = index_read(l, n->kids[1]);
            else if (l.k == TyK::List && R.evars[find(l.a)].nf && n->kids[1]->kind != NK::Int) bad(l.a);
            else if (l.k == TyK::Any && iok) t = l;
            break;
        }
        case NK::List: {
            uint32_t e = I().naux[o];
            if (e == kNone) {
                e = (uint32_t)R.evars.size();
                R.evars.push_back(EVar{e, Ty::bot(), {}});
                R.evars[e].sites.push_back(AllocSite{n, cur, n->nkids});
                if (n->nkids) {
                    if (!lit_state.empty() && lit_state[node_base[CI->def] + o] == 1) {
                        uint32_t fb = (uint32_t)R.evars.size();
                        for (uint32_t j = 0; j < n->nkids; ++j)
                            R.evars.push_back(EVar{fb + j, Ty::bot(), {}, kNone, 0, EV_FIELD});
                        R.evars[e].fbase = fb;
                        R.evars[e].nf = (uint8_t)n->nkids;
                    } else {
                        R.evars[e].flags |= EV_PLAIN;
                    }
                }
                I().naux[o] = e;
                ++changes;
            }
            for (uint32_t i = 0; i < n->nkids; ++i) {
                Ty v = ex(n->kids[i]);
                uint32_t r = find(e);
                if (!R.evars[r].nf) flow_elem(e, v);
                else if (i < R.evars[r].nf) flow_elem(R.evars[r].fbase + i, v);
            }
            t = Ty::list(e);
            break;
        }
        case NK::FuncLit:
            t = Ty::fn((uint32_t)n->aux);
            break;
        case NK::Call: {
            const Node* args = n->kids[1];
            Ty at[64];
            std::vector<Ty> big;
            Ty* a = at;
            if (args->nkids > 64) {
                big.resize(args->nkids);
                a = big.data();
            }
            for (uint32_t i = 0; i < args->nkids; ++i) a[i] = ex(args->kids[i]);
            Ty f = ex(n->kids[0]);
            if (f.k == TyK::Fn) {
                const Function& fn = P.funcs[f.a];
                if (fn.params.size() == args->nkids) {
                    uint32_t J = callee_instance(o, f.a);
                    for (uint32_t i = 0; i < args->nkids; ++i) flow(R.insts[J]->locals[fn.params[i]], a[i]);
                    t = R.insts[J]->ret;
                }
            } else if (f.k == TyK::Any) {
                for (uint32_t i = 0; i < args->nkids; ++i) make_any(a[i]);
                t = Ty::any();
            }
            break;
        }
        default:  // Class / Get / SGet are rejected by the resolver
            break;
        }
        I().nty[o] = t;
        return t;
    }

    void st(const Node* n) {
        switch (n->kind) {
        case NK::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) st(n->kids[i]);
            break;
        case NK::ExprStmt:
        case NK::Print:
            ex(n->kids[0]);
            break;
        case NK::Block:
            st(n->kids[0]);
            break;
        case NK::If:
            ex(n->kids[0], true);
            st(n->kids[1]);
            st(n->kids[2]);
            break;
        case NK::While:
            ex(n->kids[0], true);
            st(n->kids[1]);
            break;
        case NK::For:
            st(n->kids[0]);
            ex(n->kids[1], true);
            st(n->kids[3]);
            st(n->kids[2]);
            break;
        case NK::ForEach: {
            const ForEachInfo& fe = P.foreach[(size_t)n->aux];
            Ty L = ex(n->kids[1]);
            flow(I().locals[fe.list_slot], L);
            flow(I().locals[fe.index_slot], Ty::i32());
            const Node* x = n->kids[0];
            if (fe.var_preexisted) flow(var(x), Ty::i32());
            Ty Lt = I().locals[fe.list_slot];
            if (Lt.k == TyK::List) {
                if (list_use(Lt)) flow(var(x), R.evars[find(Lt.a)].t);
            }
            else if (Lt.k == TyK::Any) flow(var(x), Ty::any());
            st(n->kids[2]);
            break;
        }
        case NK::Return: {
            Ty v = ex(n->kids[0]);
            if (I().def != kMainDef) flow(I().ret, v);
            break;
        }
        default:
            ex(n);
            break;
        }
    }

    void walk(uint32_t inst) {
        cur = inst;
        CI = R.insts[inst].get();
        DefId d = CI->def;
        const Function& f = P.funcs[d];
        for (uint32_t i = unclean_off[d]; i < unclean_off[d + 1]; ++i) flow(CI->locals[unclean[i]], Ty::i32());
        wp = wo.data() + wo_off[d];
        st(f.body);
        if (wp != wo.data() + wo_off[d + 1]) __builtin_trap();  // wnum_* out of sync with ex / st
        if (body_completes[d]) flow(CI->ret, Ty::i32());
    }

    // ---- one-time numbering ------------------------------------------------------
    // Preorder ordinals (Result::ord), not descending into nested FuncLits.
    void number(const Node* n, uint32_t& c) {
        R.ord.put(n, c);
        if (n->kind == NK::List) lits.push_back(c);
        if (n->kind == NK::Call) {
            calls.push_back(c);
            calls.push_back(c + 1);  // kids[0] (the callee) is numbered next
        }
        ++c;
        if (n->kind == NK::FuncLit) return;
        for (uint32_t i = 0; i < n->nkids; ++i) number(n->kids[i], c);
    }
    static size_t count_nodes(const Node* n) {
        size_t c = 1;
        if (n->kind == NK::FuncLit) return c;
        for (uint32_t i = 0; i < n->nkids; ++i) c += count_nodes(n->kids[i]);
        return c;
    }
    // Mirrors ex() / st() exactly: records the ordinal of every node ex() visits,
    // in visiting order (the walk checks that it consumed exactly these).
    void wnum_ex(const Node* n) {
        wo.push_back(R.ord.at(n));
        switch (n->kind) {
        case NK::Assign: {
            const Node* tg = n->kids[0];
            if (tg->kind != NK::Id) {
                wnum_ex(tg->kids[0]);
                wnum_ex(tg->kids[1]);
            }
            wnum_ex(n->kids[1]);
            break;
        }
        case NK::Append:
        case NK::Add: case NK::Sub: case NK::Mul: case NK::Div:
        case NK::Eq: case NK::Gt: case NK::Lt: case NK::Ge: case NK::Le: case NK::Ne:
        case NK::And: case NK::Or:
        case NK::Index:
            wnum_ex(n->kids[0]);
            wnum_ex(n->kids[1]);
            break;
        case NK::Not: case NK::Abs: case NK::Neg:
            wnum_ex(n->kids[0]);
            break;
        case NK::List:
            for (uint32_t i = 0; i < n->nkids; ++i) wnum_ex(n->kids[i]);
            break;
        case NK::Call: {
            const Node* args = n->kids[1];
            for (uint32_t i = 0; i < args->nkids; ++i) wnum_ex(args->kids[i]);
            wnum_ex(n->kids[0]);
            break;
        }
        default:
            break;
        }
    }
    void wnum_st(const Node* n) {
        switch (n->kind) {
        case NK::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) wnum_st(n->kids[i]);
            break;
        case NK::ExprStmt: case NK::Print: case NK::Return:
            wnum_ex(n->kids[0]);
            break;
        case NK::Block:
            wnum_st(n->kids[0]);
            break;
        case NK::If:
            wnum_ex(n->kids[0]);
            wnum_st(n->kids[1]);
            wnum_st(n->kids[2]);
            break;
        case NK::While:
            wnum_ex(n->kids[0]);
            wnum_st(n->kids[1]);
            break;
        case NK::For:
            wnum_st(n->kids[0]);
            wnum_ex(n->kids[1]);
            wnum_st(n->kids[3]);
            wnum_st(n->kids[2]);
            break;
        case NK::ForEach:
            wnum_ex(n->kids[1]);
            wnum_st(n->kids[2]);
            break;
        default:
            wnum_ex(n);
            break;
        }
    }

    // ---- post-fixpoint ---------------------------------------------------------
    void mark_reachable() {
        std::vector<uint32_t> work;
        work.reserve(R.insts.size());
        auto push = [&](uint32_t i) {
            if (!R.insts[i]->reachable) {
                R.insts[i]->reachable = true;
                work.push_back(i);
            }
        };
        push(0);
        for (size_t d = 0; d < R.generic_of.size(); ++d)
            if (R.generic_of[d] >= 0) push((uint32_t)R.generic_of[d]);
        while (!work.empty()) {
            uint32_t i = work.back();
            work.pop_back();
            Instance& J = *R.insts[i];
            for (uint32_t k = calls_off[J.def]; k < calls_off[J.def + 1]; k += 2) {
                Ty f = J.nty[calls[k + 1]];
                uint32_t c = J.naux[calls[k]];
                if (f.k == TyK::Fn && c != kNone && R.insts[c]->def == f.a) push(c);
            }
        }
    }

    // Exact-match interning of u32 key strings -> dense class ids in order of
    // first appearance (replaces std::map<std::string, uint32_t>). A candidate key
    // is appended to buf, then commit() either keeps it or drops it as a duplicate.
    struct KeyTab {
        std::vector<uint32_t> buf, off, table;
        std::vector<uint64_t> hash;
        size_t mask = 0;
        void reset(size_t n) {
            buf.clear();
            off.assign(1, 0);
            hash.clear();
            size_t cap = 16;
            while (cap < n * 2) cap <<= 1;
            table.assign(cap, kNone);
            mask = cap - 1;
        }
        size_t size() const { return hash.size(); }
        uint32_t commit() {
            uint32_t b = off.back(), e = (uint32_t)buf.size();
            uint64_t h = 0xcbf29ce484222325ull ^ (e - b);
            for (uint32_t i = b; i < e; ++i) {
                h = (h ^ buf[i]) * 0x100000001b3ull;
                h ^= h >> 29;
            }
            size_t s = (size_t)((h * 0x9E3779B97F4A7C15ull) >> 32) & mask;
            for (;; s = (s + 1) & mask) {
                uint32_t id = table[s];
                if (id == kNone) break;
                if (hash[id] != h) continue;
                uint32_t ob = off[id], oe = off[id + 1];
                if (oe - ob == e - b && std::equal(buf.begin() + ob, buf.begin() + oe, buf.begin() + b)) {
                    buf.resize(b);
                    return id;
                }
            }
            uint32_t id = (uint32_t)hash.size();
            table[s] = id;
            hash.push_back(h);
            off.push_back(e);
            return id;
        }
    };
    void key_ty(std::vector<uint32_t>& k, Ty t) {
        k.push_back((uint32_t)t.k);
        k.push_back(t.k == TyK::List ? find(t.a) : t.k == TyK::Fn ? t.a : 0);
    }
    // Partition refinement over reachable instances.
    void dedup() {
        size_t n = R.insts.size();
        std::vector<uint32_t> cls(n, kNone), ncls(n, kNone);
        KeyTab ids, nids;
        ids.reset(n);
        for (size_t i = 0; i < n; ++i) {
            Instance& J = *R.insts[i];
            if (!J.reachable) continue;
            std::vector<uint32_t>& k = ids.buf;
            k.push_back(J.def);
            k.push_back((J.generic ? 1u : 0u) | (i == 0 ? 2u : 0u));
            for (Ty t : J.locals) key_ty(k, t);
            key_ty(k, J.ret);
            for (Ty t : J.nty) key_ty(k, t);
            // List literal regions (callees are refined below).
            for (uint32_t x = lits_off[J.def]; x < lits_off[J.def + 1]; ++x) {
                uint32_t e = J.naux[lits[x]];
                k.push_back(e == kNone ? kNone : find(e));
            }
            cls[i] = ids.commit();
        }
        for (int round = 0; round < 1000; ++round) {
            nids.reset(n);
            for (size_t i = 0; i < n; ++i) {
                Instance& J = *R.insts[i];
                ncls[i] = kNone;
                if (!J.reachable) continue;
                std::vector<uint32_t>& k = nids.buf;
                k.push_back(cls[i]);
                for (uint32_t x = calls_off[J.def]; x < calls_off[J.def + 1]; x += 2) {
                    uint32_t c = J.naux[calls[x]];
                    k.push_back((c == kNone || !R.insts[c]->reachable) ? kNone : cls[c]);
                }
                ncls[i] = nids.commit();
            }
            bool stable = nids.size() == ids.size();
            cls.swap(ncls);
            std::swap(ids, nids);
            if (stable) break;
        }
        std::vector<int32_t> first(ids.size(), -1);
        for (size_t i = 0; i < n; ++i) {
            if (!R.insts[i]->reachable) continue;
            if (first[cls[i]] < 0) first[cls[i]] = (int32_t)i;
            R.insts[i]->rep = (uint32_t)first[cls[i]];
        }
    }

    void run() {
        R.prog = &P;
        size_t nd = P.funcs.size();
        R.nnodes.assign(nd, 0);
        R.generic_of.assign(nd, -1);
        R.shared_of.assign(nd, -1);
        R.escaped.assign(nd, 0);
        count_of.assign(nd, 0);
        R.globals.assign(P.globals.size(), Ty::bot());
        size_t total = 0;
        for (size_t d = 0; d < nd; ++d) total += count_nodes(P.funcs[d].body);
        R.ord.init(total);
        wo.reserve(total);
        wo_off.resize(nd + 1);
        calls_off.resize(nd + 1);
        lits_off.resize(nd + 1);
        unclean_off.resize(nd + 1);
        body_completes.resize(nd);
        for (size_t d = 0; d < nd; ++d) {
            const Function& f = P.funcs[d];
            calls_off[d] = (uint32_t)calls.size();
            lits_off[d] = (uint32_t)lits.size();
            uint32_t c = 0;
            number(f.body, c);
            R.nnodes[d] = c;
            wo_off[d] = (uint32_t)wo.size();
            wnum_st(f.body);
            unclean_off[d] = (uint32_t)unclean.size();
            for (size_t sl = 0; sl < f.locals.size(); ++sl)
                if (f.locals[sl].flags & LF_UNCLEAN) unclean.push_back((uint32_t)sl);
            body_completes[d] = d != kMainDef && completes(f.body);
        }
        wo_off[nd] = (uint32_t)wo.size();
        calls_off[nd] = (uint32_t)calls.size();
        lits_off[nd] = (uint32_t)lits.size();
        unclean_off[nd] = (uint32_t)unclean.size();
        new_instance(kMainDef, -1);
        fixpoint();
        mark_reachable();
        if (C.dedup) dedup();
        if (C.records) records();
    }

    void fixpoint() {
        for (;;) {
            changes = 0;
            for (uint32_t i = 0; i < R.insts.size(); ++i) walk(i);
            ++R.passes;
            if (changes == 0) break;
        }
    }

    // ---- records: speculate, check, demote, repeat (DESIGN.md §5.11) -------------
    // The variable part of a Result (everything but the numbering).
    struct Snap {
        std::vector<Ty> globals;
        std::vector<EVar> evars;
        std::vector<std::unique_ptr<Instance>> insts;
        std::vector<int32_t> generic_of, shared_of;
        std::vector<uint8_t> escaped;
        uint32_t passes = 0;
    };
    void save(Snap& s) {
        s.globals = std::move(R.globals);
        s.evars = std::move(R.evars);
        s.insts = std::move(R.insts);
        s.generic_of = std::move(R.generic_of);
        s.shared_of = std::move(R.shared_of);
        s.escaped = std::move(R.escaped);
        s.passes = R.passes;
    }
    void load(Snap& s) {
        R.globals = std::move(s.globals);
        R.evars = std::move(s.evars);
        R.insts = std::move(s.insts);
        R.generic_of = std::move(s.generic_of);
        R.shared_of = std::move(s.shared_of);
        R.escaped = std::move(s.escaped);
        R.passes = s.passes;
    }
    // A fresh inference run from scratch with the current lit_state. Returns
    // false if some record region violated a record condition.
    bool solve() {
        size_t nd = P.funcs.size();
        R.globals.assign(P.globals.size(), Ty::bot());
        R.evars.clear();
        R.insts.clear();
        R.generic_of.assign(nd, -1);
        R.shared_of.assign(nd, -1);
        R.escaped.assign(nd, 0);
        count_of.assign(nd, 0);
        R.passes = 0;
        new_instance(kMainDef, -1);
        fixpoint();
        for (auto& v : R.evars)
            if (v.flags & EV_BAD) return false;
        mark_reachable();
        if (C.dedup) dedup();
        return true;
    }
    uint32_t gid(const AllocSite& s) { return node_base[R.insts[s.inst]->def] + R.ord.at(s.node); }
    static bool cand_len(uint32_t len) { return len >= 1 && len <= 16; }
    // Ordinary list regions with Any elements: their literal sites (not yet
    // tried) become candidates. Returns the number of new candidates.
    size_t add_candidates() {
        size_t n = 0;
        for (size_t e = 0; e < R.evars.size(); ++e) {
            const EVar& v = R.evars[e];
            if (v.parent != e || v.nf || (v.flags & EV_FIELD) || v.t.k != TyK::Any) continue;
            for (const AllocSite& s : v.sites) {
                if (!cand_len(s.len)) continue;
                uint8_t& st = lit_state[gid(s)];
                if (st == 0) {
                    st = 1;
                    ++n;
                }
            }
        }
        return n;
    }
    void demote_region(uint32_t r) {
        for (const AllocSite& s : R.evars[r].sites)
            if (s.len) lit_state[gid(s)] = 2;
    }
    size_t active() {
        size_t n = 0;
        for (uint8_t st : lit_state) n += st == 1;
        return n;
    }
    // After a clean run: records whose positions all have one static type and
    // that have no `[]` site are better off as ordinary lists (pair unboxing,
    // int arrays). Returns the number of regions demoted.
    size_t demote_uniform() {
        size_t n = 0;
        for (size_t e = 0; e < R.evars.size(); ++e) {
            const EVar& v = R.evars[e];
            if (v.parent != e || !v.nf) continue;
            bool empty = false;
            for (const AllocSite& s : v.sites) empty |= s.len == 0;
            if (empty) continue;
            Ty f0 = R.evars[find(v.fbase)].t;
            if (f0.k == TyK::Any) continue;
            bool uni = true;
            for (uint32_t j = 1; j < v.nf && uni; ++j) uni = same(f0, R.evars[find(v.fbase + j)].t);
            if (!uni) continue;
            demote_region((uint32_t)e);
            ++n;
        }
        return n;
    }
    void records() {
        size_t total = 0;
        node_base.assign(P.funcs.size() + 1, 0);
        for (size_t d = 0; d < P.funcs.size(); ++d) node_base[d + 1] = node_base[d] + R.nnodes[d];
        total = node_base[P.funcs.size()];
        lit_state.assign(total, 0);
        if (add_candidates() == 0) {  // the common case: nothing is lost, nothing more to do
            lit_state.clear();
            return;
        }
        Snap run0, best;
        save(run0);
        bool have_best = false;  // `best` holds the last clean run with records
        const bool dbg = std::getenv("EPQ_REC_DEBUG") != nullptr;
        enum { CUR, BEST, RUN0 } pick = RUN0;
        uint32_t runs = 0;
        for (;;) {
            if (runs == C.max_record_runs) {  // hard limit: the last clean run, else no records
                pick = have_best ? BEST : RUN0;
                break;
            }
            ++runs;
            if (!solve()) {
                for (size_t e = 0; e < R.evars.size(); ++e)
                    if (R.evars[e].flags & EV_BAD) {
                        if (dbg) {
                            std::fprintf(stderr, "records: run %u: bad region r%zu (root r%u) sites:", runs, e, find((uint32_t)e));
                            for (const AllocSite& s : R.evars[find((uint32_t)e)].sites)
                                std::fprintf(stderr, " %d:%d/len%u", s.node->line, s.node->col, s.len);
                            std::fprintf(stderr, "\n");
                        }
                        demote_region(find((uint32_t)e));
                    }
                if (active() == 0) break;  // nothing left to try: run 0 is the answer
                continue;
            }
            size_t ch = demote_uniform();
            ch += add_candidates();
            if (ch == 0) {
                pick = CUR;
                break;
            }
            if (active() == 0) break;  // (a) demoted every record: run 0 is the answer
            best = Snap{};
            save(best);
            have_best = true;
        }
        if (pick == BEST) load(best);
        else if (pick == RUN0) load(run0);
        R.record_runs = runs;
        R.has_records = false;
        for (size_t e = 0; e < R.evars.size(); ++e) {
            EVar& v = R.evars[e];
            if (v.parent != e || !v.nf) continue;
            R.has_records = true;
            for (const AllocSite& s : v.sites)
                if (s.len == 0) v.flags |= EV_NULL;
        }
        lit_state.clear();
    }
};

}  // namespace

void infer(const Program& prog, const Config& cfg, Result& out) {
    Solver s(prog, cfg, out);
    s.run();
}

namespace {
// Records print as Rec#r(field types) / Rec?#r(...) (nullable); a record
// nested in a field prints as just Rec#r.
std::string ty_str_(const Result& r, Ty t, bool deep) {
    switch (t.k) {
    case TyK::Bot: return "Bot";
    case TyK::Int: return "Int";
    case TyK::Any: return "Any";
    case TyK::Fn: return "Fn(def" + std::to_string(t.a) + ")";
    case TyK::List: {
        uint32_t e = r.find(t.a);
        const EVar& v = r.evars[e];
        if (v.nf) {
            bool null = (v.flags & EV_NULL) != 0;
            for (auto& s : v.sites) null |= s.len == 0;
            std::string o = std::string(null ? "Rec?#" : "Rec#") + std::to_string(e);
            if (!deep) return o;
            o += "(";
            for (uint32_t j = 0; j < v.nf; ++j) o += (j ? ", " : "") + ty_str_(r, r.evars[r.find(v.fbase + j)].t, false);
            return o + ")";
        }
        return "List(r" + std::to_string(e) + ")";
    }
    }
    return "?";
}
}  // namespace

std::string ty_str(const Result& r, Ty t) { return ty_str_(r, t, true); }

void dump_types(std::string& o, const Result& r) {
    const Program& P = *r.prog;
    o += "inference: " + std::to_string(r.passes) + " passes, " + std::to_string(r.insts.size()) + " instances, " +
         std::to_string(r.evars.size()) + " element variables";
    if (r.has_records) o += ", records after " + std::to_string(r.record_runs) + " speculative runs";
    o += "\n";
    o += "globals:\n";
    for (size_t g = 0; g < P.globals.size(); ++g)
        o += "  " + std::string(P.syms->name(P.globals[g])) + " : " + ty_str(r, r.globals[g]) + "\n";
    o += "regions:\n";
    for (size_t e = 0; e < r.evars.size(); ++e) {
        if (r.evars[e].parent != e) continue;
        if (r.evars[e].flags & EV_FIELD) continue;  // record positions print with their record
        if (r.evars[e].nf) o += "  r" + std::to_string(e) + " record " + ty_str(r, Ty::list((uint32_t)e)) + "  sites:";
        else o += "  r" + std::to_string(e) + " elem " + ty_str(r, r.evars[e].t) + "  sites:";
        for (auto& s : r.evars[e].sites)
            o += " " + std::to_string(s.node->line) + ":" + std::to_string(s.node->col) + "/i" +
                 std::to_string(s.inst) + "/len" + std::to_string(s.len);
        o += "\n";
    }
    for (auto& ip : r.insts) {
        const Instance& J = *ip;
        const Function& f = P.funcs[J.def];
        o += "instance i" + std::to_string(J.id) + " def" + std::to_string(J.def);
        if (J.def == kMainDef) o += " <main>";
        else o += " at " + std::to_string(f.node->line) + ":" + std::to_string(f.node->col);
        if (J.generic) o += " generic";
        if (J.shared) o += " shared";
        if (!J.reachable) o += " unreachable";
        if (J.rep != J.id) o += " merged-into i" + std::to_string(J.rep);
        if (J.parent >= 0) o += " parent i" + std::to_string(J.parent);
        o += "\n  (";
        for (size_t i = 0; i < f.params.size(); ++i) {
            if (i) o += ", ";
            o += std::string(P.syms->name(f.locals[f.params[i]].name)) + ": " + ty_str(r, J.locals[f.params[i]]);
        }
        o += ") -> " + ty_str(r, J.ret) + "\n";
        for (size_t s = 0; s < f.locals.size(); ++s) {
            if (f.locals[s].flags & LF_PARAM) continue;
            Sym nm = f.locals[s].name;
            o += "    " + (nm == kNoSym ? std::string("<hidden>") : std::string(P.syms->name(nm))) + "#" +
                 std::to_string(s) + " : " + ty_str(r, J.locals[s]) + "\n";
        }
    }
}

}  // namespace epq::infer
