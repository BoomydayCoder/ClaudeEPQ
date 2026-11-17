// Pass "reclaim" — static reclamation: Free at compile-time-known death points (DESIGN.md §5.7).
//
// Inserts `Free` for lists that provably die when a (non-main) function returns.
// A missed free costs nothing; a wrong free corrupts memory — every rule below
// errs towards "do not free".
//
// Analysis (per function instance, iterated over the whole module to a fixpoint
// because of recursion):
//   * Every list-typed expression has a value set: the list-typed locals whose
//     *current* value it may be (`vals`), plus direct origin bits (`dr`):
//     FRESH (a ListLit, or a call whose summary says "fresh-or-argument"),
//     UNK (anything else: Index, GetGlobal, Pop, calls without that summary...).
//   * Locals are grouped (union-find) whenever a value of one may become the
//     value of another: SetLocal, and call results that may be an argument.
//     Lists of different groups are never the same object.
//   * A group ESCAPES if any value of it reaches a position that may retain it:
//     a list element (ListLit / SetIndex / Append / ElemStore value), a global,
//     Box (dynamic value), an escaping callee parameter, or any position not on
//     the explicit whitelist below. A group is RETURNED if a Return may yield it.
//   * Group roots: FRESH / PARAM (a parameter's incoming value) / UNK.
//   * Summaries: param i escapes = its group escapes; param i may be returned =
//     its group is returned; ret_fresh = every returned list is FRESH-or-param
//     and comes from non-escaping groups without UNK roots.
//     Initial values are optimistic (no escape, not returned, fresh) and the
//     iteration is monotone towards conservative; if it does not converge the
//     pass does nothing.
// Transformation, for every non-main function without a pre-existing Free:
//   * List-valued call arguments that are calls or literals (`merge(sort(l),
//     sort(r))`) are first hoisted into temporaries so they get a name.
//   * A group is freeable iff roots == {FRESH}, it does not escape and it is
//     not returned. At each Return: `tmp := value; Free(x) ...; Return tmp` for
//     every member x of a freeable group that is definitely assigned at that
//     Return. Members of the same group may be the same list: each is freed
//     only if it is not SameList with an earlier-freed member.
//   * Hoisted temporaries that end up freed nowhere are put back.
#include <cstdint>
#include <vector>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

enum : uint8_t { R_FRESH = 1, R_PARAM = 2, R_UNK = 4 };

struct Summary {
    std::vector<bool> esc;     // per parameter position
    std::vector<bool> mayret;  // per parameter position
    bool ret_fresh = true;
    bool operator==(const Summary& o) const { return esc == o.esc && mayret == o.mayret && ret_fresh == o.ret_fresh; }
};

struct V {
    std::vector<uint32_t> locs;
    uint8_t dr = 0;
};

bool is_list(const Node* n) { return n->type.k == TK::List; }

struct FuncAn {
    Module& m;
    Func& f;
    const std::vector<Summary>& sums;
    std::vector<uint32_t> uf;
    std::vector<uint8_t> roots;
    std::vector<bool> esc, ret;
    std::vector<V> returns;  // value sets of list-typed Return values
    bool ret_unknown = false;

    FuncAn(Module& m_, Func& f_, const std::vector<Summary>& s) : m(m_), f(f_), sums(s) {
        size_t n = f.locals.size();
        uf.resize(n);
        for (size_t i = 0; i < n; ++i) uf[i] = (uint32_t)i;
        roots.assign(n, 0);
        esc.assign(n, false);
        ret.assign(n, false);
        for (uint32_t p : f.params) roots[p] |= R_PARAM;
    }
    uint32_t find(uint32_t x) {
        while (uf[x] != x) x = uf[x] = uf[uf[x]];
        return x;
    }
    void unite(uint32_t a, uint32_t b) {
        a = find(a);
        b = find(b);
        if (a != b) uf[a] = b;
    }
    void escape(const V& v) {
        for (uint32_t l : v.locs) esc[l] = true;
    }
    static void add(V& to, const V& from) {
        to.locs.insert(to.locs.end(), from.locs.begin(), from.locs.end());
        to.dr |= from.dr;
    }

    V an(Node* n) {
        V res;
        V prop;  // value contributions of kids to n's own value
        uint32_t nk = n->nkids;
        for (uint32_t i = 0; i < nk; ++i) {
            V k = an(n->kids[i]);
            if (k.locs.empty() && k.dr == 0) continue;
            switch (n->op) {
            // ---- consumed: the list is used, not retained ----
            case Op::Len: case Op::Pop: case Op::ListData: case Op::Print:
            case Op::Eq: case Op::Ne: case Op::Cmp: case Op::SameList: case Op::DynArith:
            case Op::Fail: case Op::Seq:
                break;
            case Op::Index: case Op::SetIndex: case Op::Free:
                if (i != 0) escape(k);
                break;
            case Op::If: case Op::While: case Op::InlineBlock:
                break;  // condition, or a statement whose value is discarded
            case Op::Return:
                for (uint32_t l : k.locs) ret[l] = true;
                break;
            // ---- value propagates to the parent ----
            case Op::SetLocal:
                for (uint32_t l : k.locs) unite(l, (uint32_t)n->a);
                roots[(uint32_t)n->a] |= k.dr;
                add(prop, k);
                break;
            case Op::Append:
                if (i == 0) add(prop, k);
                else escape(k);
                break;
            case Op::Comma:
                if (i + 1 == nk) add(prop, k);
                break;
            case Op::Select:
                if (i != 0) add(prop, k);
                else escape(k);
                break;
            case Op::And: case Op::Or:
                if (!(n->flags & F_COND)) add(prop, k);
                break;
            case Op::CallDirect: {
                const Summary& s = sums[(size_t)n->a];
                if (i + 1 >= nk || i >= s.esc.size() || s.esc[i]) {
                    escape(k);
                    if (i + 1 < nk && i < s.mayret.size() && s.mayret[i]) add(prop, k);
                } else if (s.mayret[i]) {
                    add(prop, k);
                }
                break;
            }
            default:  // ListLit elements, SetIndex/ElemStore/Append values, SetGlobal, Box, unknown
                escape(k);
                break;
            }
        }
        if (n->op == Op::SetLocal && f.locals[(size_t)n->a].type.k != TK::List) {
            // a list value assigned to a non-list local cannot happen (types are exact); be safe
            escape(prop);
        }
        if (!is_list(n)) return res;
        switch (n->op) {
        case Op::Local: res.locs.push_back((uint32_t)n->a); break;
        case Op::ListLit: res.dr = R_FRESH; break;
        case Op::SetLocal:  // the value IS now the local's value
            res = prop;
            res.locs.push_back((uint32_t)n->a);
            break;
        case Op::Append: case Op::Comma: case Op::Select: case Op::And: case Op::Or:
            res = prop;
            break;
        case Op::CallDirect:
            res = prop;
            res.dr |= sums[(size_t)n->a].ret_fresh ? R_FRESH : R_UNK;
            break;
        case Op::Coerce: break;
        default: res.dr = R_UNK; break;
        }
        return res;
    }

    void run() {
        // Return values: recorded through a dedicated walk so we keep their value sets.
        an(f.body);
        returns.clear();
        util::walk(f.body, [&](Node* x, Node*, uint32_t) {
            if (x->op == Op::Return && x->nkids == 1 && is_list(x->kids[0])) returns.push_back(vals_only(x->kids[0]));
            return true;
        });
        // aggregate per group
        size_t n = f.locals.size();
        std::vector<uint8_t> gr(n, 0);
        std::vector<bool> ge(n, false), gt(n, false);
        for (size_t i = 0; i < n; ++i) {
            uint32_t g = find((uint32_t)i);
            gr[g] |= roots[i];
            if (esc[i]) ge[g] = true;
            if (ret[i]) gt[g] = true;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t g = find((uint32_t)i);
            roots[i] = gr[g];
            esc[i] = ge[g];
            ret[i] = gt[g];
        }
    }

    // Value set of an expression WITHOUT side effects on the analysis state
    // (union-find is already complete; used only for Return values).
    V vals_only(Node* n) {
        V res;
        if (!is_list(n)) return res;
        switch (n->op) {
        case Op::Local: res.locs.push_back((uint32_t)n->a); return res;
        case Op::ListLit: res.dr = R_FRESH; return res;
        case Op::SetLocal:
            res = vals_only(n->kids[0]);
            res.locs.push_back((uint32_t)n->a);
            return res;
        case Op::Append: return vals_only(n->kids[0]);
        case Op::Comma: return vals_only(n->kids[n->nkids - 1]);
        case Op::Select:
            res = vals_only(n->kids[1]);
            add(res, vals_only(n->kids[2]));
            return res;
        case Op::And: case Op::Or:
            for (uint32_t i = 0; i < n->nkids; ++i) add(res, vals_only(n->kids[i]));
            return res;
        case Op::CallDirect: {
            const Summary& s = sums[(size_t)n->a];
            for (uint32_t i = 0; i + 1 < n->nkids; ++i)
                if (i < s.mayret.size() && s.mayret[i]) add(res, vals_only(n->kids[i]));
            res.dr |= s.ret_fresh ? R_FRESH : R_UNK;
            return res;
        }
        case Op::Coerce: return res;
        default: res.dr = R_UNK; return res;
        }
    }

    Summary summary() {
        Summary s;
        for (uint32_t p : f.params) {
            s.esc.push_back(esc[p]);
            s.mayret.push_back(ret[p]);
        }
        s.ret_fresh = true;
        for (const V& v : returns) {
            if (v.dr & R_UNK) s.ret_fresh = false;
            for (uint32_t l : v.locs)
                if (esc[l] || (roots[l] & R_UNK)) s.ret_fresh = false;
        }
        if (f.generic) {
            for (size_t i = 0; i < s.esc.size(); ++i) s.esc[i] = s.mayret[i] = true;
            s.ret_fresh = false;
        }
        return s;
    }

    bool freeable(uint32_t l) const {
        return f.locals[l].type.k == TK::List && f.locals[l].kind != LocalKind::Param && roots[l] == R_FRESH && !esc[l] &&
               !ret[l];
    }
};

// ---- definite assignment at each Return -------------------------------------
struct DA {
    size_t n;
    std::vector<std::pair<Node*, std::vector<bool>>> at_return;
    std::vector<bool> dead() const { return std::vector<bool>(n, true); }
    static void meet(std::vector<bool>& a, const std::vector<bool>& b) {
        for (size_t i = 0; i < a.size(); ++i) a[i] = a[i] && b[i];
    }
    void go(Node* x, std::vector<bool>& cur) {
        switch (x->op) {
        case Op::Return:
            for (uint32_t i = 0; i < x->nkids; ++i) go(x->kids[i], cur);
            at_return.push_back({x, cur});
            cur = dead();
            return;
        case Op::Break:
            cur = dead();
            return;
        case Op::InlineBlock: {
            std::vector<bool> before = cur, t = cur;
            for (uint32_t i = 0; i < x->nkids; ++i) go(x->kids[i], t);
            cur = before;  // conservative: whatever was assigned before the block
            return;
        }
        case Op::If: {
            go(x->kids[0], cur);
            std::vector<bool> t = cur, e = cur;
            if (x->nkids > 1) go(x->kids[1], t);
            if (x->nkids > 2) go(x->kids[2], e);
            meet(t, e);
            cur = t;
            return;
        }
        case Op::While: {
            go(x->kids[0], cur);
            std::vector<bool> t = cur;
            for (uint32_t i = 1; i < x->nkids; ++i) go(x->kids[i], t);
            return;
        }
        case Op::And: case Op::Or: {
            go(x->kids[0], cur);
            std::vector<bool> t = cur;
            for (uint32_t i = 1; i < x->nkids; ++i) go(x->kids[i], t);
            return;
        }
        case Op::SetLocal:
            go(x->kids[0], cur);
            if ((size_t)x->a < n) cur[(size_t)x->a] = true;
            return;
        default:
            for (uint32_t i = 0; i < x->nkids; ++i) go(x->kids[i], cur);
            return;
        }
    }
};

struct Hoist {
    Node** slot;
    Node* orig;
    uint32_t tmp;
};

}  // namespace

void pass_reclaim(Module& m) {
    size_t nf = m.funcs.size();
    std::vector<bool> skip(nf, false);
    for (size_t i = 0; i < nf; ++i) {
        Func& f = *m.funcs[i];
        if (f.is_main || f.generic || util::contains_op(f.body, Op::Free)) skip[i] = true;
    }

    // 1. Hoist list-valued call arguments that are calls / literals into temporaries.
    std::vector<std::vector<Hoist>> hoists(nf);
    for (size_t i = 0; i < nf; ++i) {
        if (skip[i]) continue;
        Func& f = *m.funcs[i];
        std::vector<Node*> calls;
        util::walk(f.body, [&](Node* x, Node*, uint32_t) {
            if (x->op == Op::CallDirect) calls.push_back(x);
            return true;
        });
        for (Node* c : calls)
            for (uint32_t k = 0; k + 1 < c->nkids; ++k) {
                Node* a = c->kids[k];
                if (!is_list(a) || (a->op != Op::CallDirect && a->op != Op::ListLit)) continue;
                uint32_t t = m.new_temp(&f, a->type);
                c->kids[k] = util::mk_set(m, f, t, a);
                hoists[i].push_back({&c->kids[k], a, t});
            }
    }

    // 2. Summaries to a fixpoint.
    std::vector<Summary> sums(nf);
    for (size_t i = 0; i < nf; ++i) {
        sums[i].esc.assign(m.funcs[i]->params.size(), false);
        sums[i].mayret.assign(m.funcs[i]->params.size(), false);
        sums[i].ret_fresh = !m.funcs[i]->generic;
        if (m.funcs[i]->generic) {
            sums[i].esc.assign(m.funcs[i]->params.size(), true);
            sums[i].mayret.assign(m.funcs[i]->params.size(), true);
        }
    }
    bool converged = false;
    for (int round = 0; round < 200 && !converged; ++round) {
        converged = true;
        for (size_t i = 0; i < nf; ++i) {
            FuncAn fa(m, *m.funcs[i], sums);
            fa.run();
            Summary s = fa.summary();
            // keep monotone: only ever become more conservative
            for (size_t p = 0; p < s.esc.size(); ++p) {
                s.esc[p] = s.esc[p] || sums[i].esc[p];
                s.mayret[p] = s.mayret[p] || sums[i].mayret[p];
            }
            s.ret_fresh = s.ret_fresh && sums[i].ret_fresh;
            if (!(s == sums[i])) {
                sums[i] = s;
                converged = false;
            }
        }
    }

    for (size_t i = 0; i < nf; ++i) {
        Func& f = *m.funcs[i];
        if (skip[i]) continue;
        std::vector<bool> freed_any(f.locals.size() + 1, false);
        if (converged) {
            FuncAn fa(m, f, sums);
            fa.run();
            DA da{f.locals.size(), {}};
            std::vector<bool> cur(f.locals.size(), false);
            for (uint32_t p : f.params) cur[p] = true;
            da.go(f.body, cur);
            // slots of Returns
            std::vector<std::pair<Node*, uint32_t>> slots;
            util::walk(f.body, [&](Node* x, Node* parent, uint32_t idx) {
                if (x->op == Op::Return && parent) slots.push_back({parent, idx});
                return true;
            });
            const size_t n0 = f.locals.size();
            for (auto& [ret, assigned] : da.at_return) {
                if (ret->nkids != 1 || ret->kids[0]->type.k == TK::Bot) continue;
                // members to free, grouped
                std::vector<std::vector<uint32_t>> groups;
                std::vector<int32_t> gidx(n0, -1);
                for (uint32_t l = 0; l < n0; ++l) {
                    if (!assigned[l] || !fa.freeable(l)) continue;
                    uint32_t g = fa.find(l);
                    if (gidx[g] < 0) {
                        gidx[g] = (int32_t)groups.size();
                        groups.emplace_back();
                    }
                    groups[(size_t)gidx[g]].push_back(l);
                }
                if (groups.empty()) continue;
                Node** slot = nullptr;
                for (auto& [p, k] : slots)
                    if (p->kids[k] == ret) slot = &p->kids[k];
                if (!slot) continue;
                uint32_t line = ret->line;
                std::vector<Node*> seq;
                Node* v = ret->kids[0];
                if (v->op != Op::ConstInt) {
                    uint32_t t = m.new_temp(&f, v->type);
                    Node* s = util::mk_set(m, f, t, v);
                    s->line = line;
                    seq.push_back(s);
                    ret->kids[0] = util::mk_local(m, f, t, line);
                }
                for (auto& g : groups)
                    for (size_t j = 0; j < g.size(); ++j) {
                        Node* fr = m.node(Op::Free, Type::bot(), {util::mk_local(m, f, g[j], line)});
                        fr->line = line;
                        Node* inner = fr;
                        for (size_t q = 0; q < j; ++q) {
                            Node* same = m.node(Op::SameList, Type::i32(),
                                                {util::mk_local(m, f, g[j], line), util::mk_local(m, f, g[q], line)});
                            same->line = line;
                            Node* nop = util::mk_nop(m);
                            Node* iff = m.node(Op::If, Type::bot(), {same, nop, inner});
                            iff->line = line;
                            inner = iff;
                        }
                        seq.push_back(inner);
                        freed_any[g[j]] = true;
                    }
                seq.push_back(ret);
                Node* s = util::mk_seq(m, seq);
                s->line = line;
                *slot = s;
            }
        }
        // 3. Put back hoisted temporaries that are freed nowhere.
        for (auto& h : hoists[i])
            if (!freed_any[h.tmp]) *h.slot = h.orig;
    }
}

}  // namespace epq::tir
