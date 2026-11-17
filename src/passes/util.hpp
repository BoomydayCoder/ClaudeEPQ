// Shared helpers for the TIR optimisation passes (src/passes/*.cpp).
//
//   walk(n, fn)                 pre-order: fn(Node* n, Node* parent, uint32_t idx); return false
//                               from fn to skip n's kids
//   rewrite(slot, fn)           post-order rewrite: kids first, then fn(Node*& slot) may replace
//                               *slot (the replacement is not revisited)
//   walk_depth(n, fn)           pre-order with loop depth: fn(Node* n, int depth); a While's
//                               cond/body/step are at depth+1
//   clone(m, n, map, f)         deep copy; map: old local id -> new local id (-1 = keep);
//                               remaps Local/SetLocal/foreach a,b; if f is given every
//                               InlineBlock gets a fresh label of f (Breaks follow)
//   same_expr(a, b)             structural equality of two pure expressions (false if impure)
//   locals_assigned(n, out)     out[id] = true for every local assigned in n (SetLocal,
//                               foreach hidden locals); out is resized to fit
//   assigns_local(n, id)
//   has_call(n)                 CallDirect / CallDyn anywhere in n
//   contains_op(n, op)
//   may_mutate_header(m, n, r)  n may change the header (len/cap/data) of a list of region r:
//                               Append/Pop on r, Free on r, Dyn{Append,Neg}/CallDyn when r's
//                               element type is Any, CallDirect whose callee's summary
//                               (Func::mutates / mutates_any) covers r. Element stores never do.
//                               (Summaries are filled by pass_region_info.)
//   may_store_elems(m, n, r)    SetIndex / ElemStore on r, DynSetIndex if r has Any elements,
//                               or any call (conservative)
//   mk_int / mk_local / mk_set / mk_seq / mk_nop    node constructors
#pragma once

#include <functional>
#include <vector>

#include "../tir.hpp"

namespace epq::tir::util {

template <class Fn>
void walk(Node* n, Fn&& fn, Node* parent = nullptr, uint32_t idx = 0) {
    if (!fn(n, parent, idx)) return;
    for (uint32_t i = 0; i < n->nkids; ++i) walk(n->kids[i], fn, n, i);
}
template <class Fn>
void walk(const Node* n, Fn&& fn) {
    walk(const_cast<Node*>(n), fn);
}

template <class Fn>
void rewrite(Node*& slot, Fn&& fn) {
    Node* n = slot;
    for (uint32_t i = 0; i < n->nkids; ++i) rewrite(n->kids[i], fn);
    fn(slot);
}

template <class Fn>
void walk_depth(Node* n, Fn&& fn, int depth = 0) {
    fn(n, depth);
    int kd = n->op == Op::While ? depth + 1 : depth;
    for (uint32_t i = 0; i < n->nkids; ++i) walk_depth(n->kids[i], fn, kd);
}

inline Node* clone_rec(Module& m, const Node* n, const std::vector<int32_t>& map, Func* f,
                       std::vector<std::pair<int32_t, int32_t>>& labels) {
    Node* c = m.node(n->op, n->type, n->nkids);
    c->flags = n->flags;
    c->a = n->a;
    c->b = n->b;
    c->c = n->c;
    c->line = n->line;
    auto rl = [&](int32_t id) { return (id >= 0 && (size_t)id < map.size() && map[(size_t)id] >= 0) ? map[(size_t)id] : id; };
    if (n->op == Op::Local || n->op == Op::SetLocal) c->a = rl(n->a);
    if (n->op == Op::While && (n->flags & F_FOREACH)) {
        c->a = rl(n->a);
        c->b = rl(n->b);
    }
    bool pushed = false;
    if (n->op == Op::InlineBlock && f) {
        c->a = (int32_t)new_label(*f);
        labels.push_back({n->a, c->a});
        pushed = true;
    }
    if (n->op == Op::Break && f) {
        for (size_t i = labels.size(); i-- > 0;)
            if (labels[i].first == n->a) {
                c->a = labels[i].second;
                break;
            }
    }
    for (uint32_t i = 0; i < n->nkids; ++i) c->kids[i] = clone_rec(m, n->kids[i], map, f, labels);
    if (pushed) labels.pop_back();
    return c;
}
// Deep copy of n. map[old] = new local id (or -1 / out of range: unchanged).
// relabel_in: allocate fresh InlineBlock labels in that function (nullptr: keep labels).
inline Node* clone(Module& m, const Node* n, const std::vector<int32_t>& map = {}, Func* relabel_in = nullptr) {
    std::vector<std::pair<int32_t, int32_t>> labels;
    return clone_rec(m, n, map, relabel_in, labels);
}

inline bool same_expr(const Node* a, const Node* b) {
    if (!is_pure(a) || !is_pure(b)) return false;
    struct R {
        static bool eq(const Node* x, const Node* y) {
            if (x->op != y->op || x->type != y->type || x->nkids != y->nkids || x->a != y->a || x->b != y->b || x->c != y->c ||
                x->flags != y->flags)
                return false;
            for (uint32_t i = 0; i < x->nkids; ++i)
                if (!eq(x->kids[i], y->kids[i])) return false;
            return true;
        }
    };
    return R::eq(a, b);
}

inline void locals_assigned(const Node* n, std::vector<bool>& out) {
    walk(n, [&](Node* x, Node*, uint32_t) {
        auto mark = [&](int32_t id) {
            if (id < 0) return;
            if ((size_t)id >= out.size()) out.resize((size_t)id + 1, false);
            out[(size_t)id] = true;
        };
        if (x->op == Op::SetLocal) mark(x->a);
        if (x->op == Op::While && (x->flags & F_FOREACH)) {
            mark(x->a);
            mark(x->b);
        }
        return true;
    });
}
inline bool assigns_local(const Node* n, uint32_t id) {
    std::vector<bool> v;
    locals_assigned(n, v);
    return id < v.size() && v[id];
}

inline bool contains_op(const Node* n, Op op) {
    bool r = false;
    walk(n, [&](Node* x, Node*, uint32_t) {
        if (x->op == op) r = true;
        return !r;
    });
    return r;
}
inline bool has_call(const Node* n) {
    bool r = false;
    walk(n, [&](Node* x, Node*, uint32_t) {
        if (x->op == Op::CallDirect || x->op == Op::CallDyn) r = true;
        return !r;
    });
    return r;
}

inline bool may_mutate_header(const Module& m, const Node* n, uint32_t r) {
    bool any_elems = m.regions[r].elem.k == TK::Any;
    bool res = false;
    walk(n, [&](Node* x, Node*, uint32_t) {
        switch (x->op) {
        case Op::Append: case Op::Pop: case Op::Free:
            if (x->kids[0]->type.k == TK::List && x->kids[0]->type.a == r) res = true;
            break;
        case Op::DynAppend: case Op::DynNeg: case Op::CallDyn:
            if (any_elems) res = true;
            break;
        case Op::CallDirect: {
            const Func& c = *m.funcs[(size_t)x->a];
            if (c.mutates_any && any_elems) res = true;
            for (uint32_t q : c.mutates)
                if (q == r) res = true;
            break;
        }
        default: break;
        }
        return !res;
    });
    return res;
}
inline bool may_store_elems(const Module& m, const Node* n, uint32_t r) {
    bool any_elems = m.regions[r].elem.k == TK::Any;
    bool res = false;
    walk(n, [&](Node* x, Node*, uint32_t) {
        switch (x->op) {
        case Op::SetIndex: case Op::ElemStore:
            if (x->kids[0]->type.a == r) res = true;
            break;
        case Op::DynSetIndex:
            if (any_elems) res = true;
            break;
        case Op::CallDirect: case Op::CallDyn: res = true; break;
        default: break;
        }
        return !res;
    });
    return res;
}

inline Node* mk_int(Module& m, int32_t v, uint32_t line = 0) {
    Node* n = m.node(Op::ConstInt, Type::i32(), 0);
    n->a = v;
    n->line = line;
    return n;
}
inline Node* mk_local(Module& m, const Func& f, uint32_t id, uint32_t line = 0) {
    Node* n = m.node(Op::Local, f.locals[id].type, 0);
    n->a = (int32_t)id;
    n->line = line;
    return n;
}
inline Node* mk_set(Module& m, const Func& f, uint32_t id, Node* v) {
    Node* n = m.node(Op::SetLocal, f.locals[id].type, {v});
    n->a = (int32_t)id;
    n->line = v->line;
    return n;
}
inline Node* mk_seq(Module& m, const std::vector<Node*>& kids) {
    Node* n = m.node(Op::Seq, Type::bot(), (uint32_t)kids.size());
    for (size_t i = 0; i < kids.size(); ++i) n->kids[i] = kids[i];
    return n;
}
inline Node* mk_nop(Module& m) { return m.node(Op::Nop, Type::bot(), 0); }
// A new Cache local (header-cache locals: Ptr(r) / Int lengths).
inline uint32_t new_cache_local(Func& f, Type t) {
    Local l;
    l.type = t;
    l.kind = LocalKind::Cache;
    f.locals.push_back(l);
    return (uint32_t)f.locals.size() - 1;
}

}  // namespace epq::tir::util
