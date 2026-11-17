// Pass "litcmp" — comparisons against list literals without allocating them.
//
// Runs right after the early `simplify` (before `inline`). Every rewrite keeps
// the exact evaluation order of the original (TIR rule: all kids in kid order,
// then the op acts): the only thing removed is the allocation of a list literal
// whose value is observed by nothing but the rewritten operation. Allocation
// cannot fail and is unobservable, so dropping it is invisible.
//
// Rewrites (L = ListLit, optionally under Box; x = the other operand):
//   Eq/Ne  x, []   (either side)
//      x : List      -> EqI32/NeI32(Len x, 0)
//      x : Pair      -> Comma[x, 0 | 1]          (a pair is a 2-element list)
//      x : Int / Fn  -> Comma[x, 0 | 1]          (different kinds never equal; `=` never fails)
//      x : Any       -> DynIsEmptyList(x) / NotI32(DynIsEmptyList(x))   (new op, see tir.hpp)
//   Eq/Ne  x, [c1..ck]  k <= 4, x : List(r) with Int elements, every ci Int and pure
//      (ConstInt only when L is the left operand and x is impure):
//      Comma[SetLocal t x, And(EqI32(Len t, k), And(EqI32(Index(t,0), c1), ...))]
//      (Index is guarded by the length test, so it never fails; t is skipped when
//      x is a Local.)
//   Cmp  x, [] / [], x   with x : List or Pair only (Int/Fn/Any keep the op: it may
//      raise "Invalid comparison"):
//      x < []  -> Comma[x,0]   x <= [] -> Len x == 0   x > [] -> Len x != 0   x >= [] -> Comma[x,1]
//      mirrored when [] is on the left. Pair: constants (length 2).
//   Len(L) / DynAbs(Box L)            -> Comma[e1..ek, k]   (ConstInt k if all ei pure)
//   L / Box L as a condition (If/While kid 0, F_COND And/Or kids)
//                                     -> Comma[e1..ek, 1]   (ConstInt 1 if all pure)
//   L / Box L as a statement (Seq kid, While step, non-final Comma kid)
//                                     -> Comma[e1..ek] / Nop
#include <vector>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

using namespace util;

struct LitCmp {
    Module& m;
    Func& f;
    int fired = 0;

    static Node* lit_of(Node* n) {
        if (n->op == Op::Box && n->nkids == 1) n = n->kids[0];
        return n->op == Op::ListLit ? n : nullptr;
    }
    static bool all_pure(const Node* l) {
        for (uint32_t i = 0; i < l->nkids; ++i)
            if (!is_pure(l->kids[i])) return false;
        return true;
    }
    Node* comma(const std::vector<Node*>& ks) {
        if (ks.size() == 1) return ks[0];
        Node* c = m.node(Op::Comma, ks.back()->type, (uint32_t)ks.size());
        for (size_t i = 0; i < ks.size(); ++i) c->kids[i] = ks[i];
        c->line = ks.back()->line;
        return c;
    }
    Node* i32op(Op o, Node* a, Node* b) {
        Node* n = m.node(o, Type::i32(), {a, b});
        n->line = a->line;
        return n;
    }
    Node* len(Node* x) {
        Node* n = m.node(Op::Len, Type::i32(), {x});
        n->line = x->line;
        return n;
    }
    // Evaluate the literal's elements in order, then yield `v`.
    Node* elems_then(Node* l, int32_t v) {
        Node* k = mk_int(m, v, l->line);
        if (all_pure(l)) return k;
        std::vector<Node*> ks;
        for (uint32_t i = 0; i < l->nkids; ++i) ks.push_back(l->kids[i]);
        ks.push_back(k);
        return comma(ks);
    }
    Node* stmt_of(Node* l) {
        if (all_pure(l)) return mk_nop(m);
        std::vector<Node*> ks;
        for (uint32_t i = 0; i < l->nkids; ++i)
            if (!is_pure(l->kids[i])) ks.push_back(l->kids[i]);
        return comma(ks);
    }
    // x evaluated (for its effects), then constant v.
    Node* then_const(Node* x, int32_t v) {
        Node* k = mk_int(m, v, x->line);
        if (is_pure(x)) return k;
        return comma({x, k});
    }

    // Eq/Ne. Returns replacement or nullptr.
    Node* eq(Node* n) {
        bool ne = n->op == Op::Ne;
        Node* a = n->kids[0];
        Node* b = n->kids[1];
        Node* la = lit_of(a);
        Node* lb = lit_of(b);
        if ((la != nullptr) == (lb != nullptr)) return nullptr;  // none, or both literals
        Node* l = la ? la : lb;
        Node* x = la ? b : a;
        bool lit_left = la != nullptr;
        Type t = x->type;
        if (l->nkids == 0) {
            switch (t.k) {
            case TK::List: return i32op(ne ? Op::NeI32 : Op::EqI32, len(x), mk_int(m, 0, n->line));
            case TK::Pair: case TK::Int: case TK::Fn: return then_const(x, ne ? 1 : 0);
            case TK::Any: {
                Node* e = m.node(Op::DynIsEmptyList, Type::i32(), {x});
                e->line = n->line;
                if (!ne) return e;
                Node* r = m.node(Op::NotI32, Type::i32(), {e});
                r->line = n->line;
                return r;
            }
            default: return nullptr;
            }
        }
        // Non-empty literal: only Int elements against a statically int-element list.
        if (l->nkids > 4 || t.k != TK::List || m.regions[t.a].elem.k != TK::Int) return nullptr;
        for (uint32_t i = 0; i < l->nkids; ++i) {
            Node* e = l->kids[i];
            if (e->type.k != TK::Int || !is_pure(e)) return nullptr;
            if (lit_left && !is_pure(x) && e->op != Op::ConstInt) return nullptr;
        }
        std::vector<Node*> pre;
        uint32_t tid;
        if (x->op == Op::Local) {
            tid = (uint32_t)x->a;
        } else {
            tid = m.new_temp(&f, t);
            pre.push_back(mk_set(m, f, tid, x));
        }
        auto tl = [&] { return mk_local(m, f, tid, n->line); };
        // Build right-nested And chain from the last element backwards.
        Node* chain = nullptr;
        for (uint32_t i = l->nkids; i-- > 0;) {
            Node* idx = m.node(Op::Index, Type::i32(), {tl(), mk_int(m, (int32_t)i, n->line)});
            idx->line = n->line;
            Node* c = i32op(Op::EqI32, idx, l->kids[i]);
            if (!chain) {
                chain = c;
            } else {
                chain = m.node(Op::And, Type::i32(), {c, chain});
                chain->line = n->line;
            }
        }
        Node* lenok = i32op(Op::EqI32, len(tl()), mk_int(m, (int32_t)l->nkids, n->line));
        Node* r = m.node(Op::And, Type::i32(), {lenok, chain});
        r->line = n->line;
        if (ne) {
            r = m.node(Op::NotI32, Type::i32(), {r});
            r->line = n->line;
        }
        pre.push_back(r);
        return comma(pre);
    }

    // Cmp against an empty literal; x statically List or Pair.
    Node* cmp(Node* n) {
        Node* la = lit_of(n->kids[0]);
        Node* lb = lit_of(n->kids[1]);
        if ((la != nullptr) == (lb != nullptr)) return nullptr;
        Node* l = la ? la : lb;
        if (l->nkids != 0) return nullptr;
        Node* x = la ? n->kids[1] : n->kids[0];
        if (x->type.k != TK::List && x->type.k != TK::Pair) return nullptr;
        Op c = (Op)n->c;
        if (la) {  // [] c x  ==  x c' []
            c = c == Op::LtI32 ? Op::GtI32 : c == Op::GtI32 ? Op::LtI32 : c == Op::LeI32 ? Op::GeI32 : Op::LeI32;
        }
        // now: x c []
        switch (c) {
        case Op::LtI32: return then_const(x, 0);
        case Op::GeI32: return then_const(x, 1);
        case Op::LeI32:
            if (x->type.k == TK::Pair) return then_const(x, 0);
            return i32op(Op::EqI32, len(x), mk_int(m, 0, n->line));
        case Op::GtI32:
            if (x->type.k == TK::Pair) return then_const(x, 1);
            return i32op(Op::NeI32, len(x), mk_int(m, 0, n->line));
        default: return nullptr;
        }
    }

    // Expression rewrite after kids are done.
    void expr(Node*& slot) {
        Node* n = slot;
        Node* r = nullptr;
        switch (n->op) {
        case Op::Eq: case Op::Ne: r = eq(n); break;
        case Op::Cmp: r = cmp(n); break;
        case Op::Len:
            if (n->kids[0]->op == Op::ListLit) r = elems_then(n->kids[0], (int32_t)n->kids[0]->nkids);
            break;
        case Op::DynAbs:
            if (n->kids[0]->op == Op::Box && n->kids[0]->kids[0]->op == Op::ListLit) {
                Node* l = n->kids[0]->kids[0];
                r = elems_then(l, (int32_t)l->nkids);
            }
            break;
        default: break;
        }
        if (r) {
            slot = r;
            ++fired;
        }
    }
    void cond(Node*& slot) {
        if (Node* l = lit_of(slot)) {
            slot = elems_then(l, 1);
            ++fired;
        }
    }
    void stmt(Node*& slot) {
        if (Node* l = lit_of(slot)) {
            slot = stmt_of(l);
            ++fired;
        }
    }

    void visit(Node*& slot) {
        Node* n = slot;
        for (uint32_t i = 0; i < n->nkids; ++i) visit(n->kids[i]);
        switch (n->op) {
        case Op::If: cond(n->kids[0]); break;
        case Op::While:
            cond(n->kids[0]);
            if (n->nkids > 2) stmt(n->kids[2]);
            break;
        case Op::And: case Op::Or:
            if (n->flags & F_COND) {
                cond(n->kids[0]);
                cond(n->kids[1]);
            }
            break;
        case Op::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) stmt(n->kids[i]);
            break;
        case Op::Comma:
            for (uint32_t i = 0; i + 1 < n->nkids; ++i) stmt(n->kids[i]);
            break;
        default: break;
        }
        expr(slot);
    }
};

}  // namespace

void pass_litcmp(Module& m) {
    for (auto& fp : m.funcs) {
        if (!fp->body) continue;
        LitCmp p{m, *fp};
        p.visit(fp->body);
    }
}

}  // namespace epq::tir
