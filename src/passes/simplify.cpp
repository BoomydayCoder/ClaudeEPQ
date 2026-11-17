// Pass "simplify" — local simplification (DESIGN.md §5.6); also run late as
// "simplify2" (same function). Toggled with --no-pass=simplify / simplify2.
//
// Every rewrite preserves the TIR evaluation rule exactly (src/tir.hpp): all
// kids of an operator are evaluated in kid order before the operator acts, so
// an operator that has no effect of its own and cannot fail may be replaced by
// the in-order evaluation of its kids; an operator that can fail is never
// removed unless its operands are constants for which it provably does not.
// A replacement always has exactly the type of the node it replaces (SetLocal /
// Return / CallDirect need exact types), except in statement and condition
// positions where values are discarded / only tested for truthiness.
//
// Rewrites (value context unless stated):
//   * constant folding of every typed Int op and comparison with 32-bit
//     wrapping semantics (DivI32 only with a non-zero constant divisor;
//     INT_MIN / -1 = INT_MIN; |INT_MIN| = INT_MIN; -INT_MIN = INT_MIN)
//   * identities: x+0, 0+x, x-0, x*1, 1*x, x/1, x*0 / 0*x (-> 0 if x pure, else
//     Comma[x, 0]), x-x (pure), -(-x), !!b (b 0/1-valued), !cmp -> inverse cmp,
//     constants moved to the right of commutative ops / comparisons,
//     (x + c1) + c2 -> x + (c1 + c2)
//   * DynArith / Eq / Ne / Cmp on statically-Int (or Box(Int)) kids -> typed
//     ops; Eq/Ne/Cmp on two Pairs -> Pair compares; DynNot/DynAbs/DynNeg of
//     Box(Int) -> typed; UnboxInt(Box(x)) -> x
//   * PairFirst/Second/PairIndex(const) of PairMake; Select with constant
//     condition or identical arms; constant globals
//   * value-form And/Or with a constant first operand
//   * conditions (If/While kid 0, condition-form And/Or): constants and
//     statically truthy (pure List/Fn/Pair) values; !!x -> x; x != 0 -> x
//   * statements: Seq flattening, Nop / empty-branch removal, code after
//     Return / Break / Fail dropped, pure expression statements removed (and
//     effect-free, non-failing operators reduced to their impure kids),
//     constant If / While, trailing Break of an InlineBlock removed and an
//     InlineBlock without Breaks unwrapped, Comma flattening
//   * dead stores: SetLocal to a local that is never read (value kept)
//   * constant / copy propagation: a local of type Int / Pair assigned exactly
//     once in the function, by a top-level statement of the body whose value is
//     a constant (ConstInt / PairMake of constants) or a parameter that is never
//     assigned, and not read at or before that statement: its uses are replaced
//     and the assignment deleted (also Fn locals assigned a ConstFn).
// Iterated to a fixpoint (bounded).
#include <cstdint>
#include <vector>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

using namespace util;

bool is_bool_valued(const Node* n) {
    switch (n->op) {
    case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
    case Op::NotI32: case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt:
    case Op::PairGe: case Op::SameList:
        return true;
    case Op::ConstInt: return n->a == 0 || n->a == 1;
    default: return false;
    }
}

Op invert_cmp(Op o) {
    switch (o) {
    case Op::EqI32: return Op::NeI32;
    case Op::NeI32: return Op::EqI32;
    case Op::LtI32: return Op::GeI32;
    case Op::GeI32: return Op::LtI32;
    case Op::LeI32: return Op::GtI32;
    case Op::GtI32: return Op::LeI32;
    case Op::PairEq: return Op::PairNe;
    case Op::PairNe: return Op::PairEq;
    case Op::PairLt: return Op::PairGe;
    case Op::PairGe: return Op::PairLt;
    case Op::PairLe: return Op::PairGt;
    case Op::PairGt: return Op::PairLe;
    default: return Op::Nop;
    }
}
// Comparison with swapped operands: a < b  <=>  b > a.
Op swap_cmp(Op o) {
    switch (o) {
    case Op::LtI32: return Op::GtI32;
    case Op::GtI32: return Op::LtI32;
    case Op::LeI32: return Op::GeI32;
    case Op::GeI32: return Op::LeI32;
    default: return o;  // Eq / Ne
    }
}

// Operators with no effect of their own that can never fail: evaluating one
// for its effects is evaluating its kids in order.
bool transparent(Op o) {
    switch (o) {
    case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::NegI32: case Op::AbsI32: case Op::NotI32:
    case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
    case Op::Box: case Op::Len: case Op::Select: case Op::PairMake: case Op::PairFirst: case Op::PairSecond:
    case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
    case Op::SameList: case Op::ListData: case Op::ElemLoad:
        return true;
    default:
        return false;
    }
}

bool is_terminator(const Node* n) {
    return n->op == Op::Return || n->op == Op::Break || n->op == Op::Fail ||
           (n->op == Op::Coerce && n->kids[0]->op == Op::Fail);
}

bool empty_stmt(const Node* n) { return n->op == Op::Nop || (n->op == Op::Seq && n->nkids == 0); }

struct Simp {
    Module& m;
    Func* f = nullptr;
    bool changed = false;
    std::vector<uint32_t> reads;    // per local: number of Local reads (foreach a/b count as reads)
    std::vector<uint32_t> assigns;  // per local: number of assignments (foreach a/b count many)

    explicit Simp(Module& mm) : m(mm) {}

    void count() {
        reads.assign(f->locals.size(), 0);
        assigns.assign(f->locals.size(), 0);
        walk(f->body, [&](Node* x, Node*, uint32_t) {
            if (x->op == Op::Local && (size_t)x->a < reads.size()) reads[(size_t)x->a]++;
            if (x->op == Op::SetLocal && (size_t)x->a < assigns.size()) assigns[(size_t)x->a]++;
            if (x->op == Op::While && (x->flags & F_FOREACH)) {
                for (int32_t id : {x->a, x->b})
                    if (id >= 0 && (size_t)id < reads.size()) {
                        reads[(size_t)id] += 1000;
                        assigns[(size_t)id] += 1000;
                    }
            }
            return true;
        });
    }

    Node* cint(int32_t v, uint32_t line) {
        changed = true;
        return mk_int(m, v, line);
    }
    Node* comma(Node* a, Node* b) {
        Node* c = m.node(Op::Comma, b->type, {a, b});
        c->line = b->line;
        return c;
    }
    // Known truthiness of an evaluated value whose evaluation can be dropped (pure).
    static bool known_truth(const Node* n, bool& t) {
        if (n->op == Op::ConstInt) {
            t = n->a != 0;
            return true;
        }
        if ((n->type.k == TK::List || n->type.k == TK::Fn || n->type.k == TK::Pair) && is_pure(n)) {
            t = true;
            return true;
        }
        return false;
    }
    // Replace n by r only if the types agree exactly.
    Node* repl(Node* n, Node* r) {
        if (r == n) return n;
        if (r->type != n->type) return n;
        changed = true;
        return r;
    }

    // ---- effects of an expression whose value is discarded ----
    void discard(Node* e, std::vector<Node*>& out) {
        if (is_pure(e)) {
            changed = true;
            return;
        }
        if (e->op == Op::Comma) {
            changed = true;
            for (uint32_t i = 0; i < e->nkids; ++i) discard(e->kids[i], out);
            return;
        }
        if (transparent(e->op)) {
            changed = true;
            for (uint32_t i = 0; i < e->nkids; ++i) discard(e->kids[i], out);
            return;
        }
        if (e->op == Op::SetLocal && reads[(size_t)e->a] == 0) {
            changed = true;
            discard(e->kids[0], out);
            return;
        }
        out.push_back(e);
    }

    // ---- statements ----
    void push_stmt(Node* s, std::vector<Node*>& out) {
        if (s->op == Op::Seq) {
            for (uint32_t i = 0; i < s->nkids; ++i) out.push_back(s->kids[i]);
            return;
        }
        if (s->op == Op::Nop) return;
        out.push_back(s);
    }
    Node* seq_of(const std::vector<Node*>& v, uint32_t line) {
        Node* s = mk_seq(m, v);
        s->line = line;
        return s;
    }

    Node* seq(Node* n) {
        std::vector<Node*> out;
        bool flat = true;
        for (uint32_t i = 0; i < n->nkids; ++i) {
            Node* k = n->kids[i];
            if (k->op == Op::Seq || k->op == Op::Nop) flat = false;
            Node* s = stm(k);
            if (s != k) flat = false;
            size_t before = out.size();
            push_stmt(s, out);
            bool term = false;
            for (size_t j = before; j < out.size(); ++j)
                if (is_terminator(out[j])) {
                    out.resize(j + 1);
                    term = true;
                    break;
                }
            if (term) {
                if (i + 1 < n->nkids) flat = false;
                break;
            }
        }
        if (flat && out.size() == n->nkids) return n;
        changed = true;
        return seq_of(out, n->line);
    }

    // Simplify a branch / loop body / step, keeping its shape: a Seq stays a
    // Seq; a single statement stays one unless it expanded.
    Node* branch(Node* k, uint32_t line) {
        bool was_seq = k->op == Op::Seq;
        Node* r = stm(k);
        if (!was_seq && r->op != Op::Seq) return r;
        return as_seq(r, line);
    }
    // Wrap a statement into a Seq.
    Node* as_seq(Node* s, uint32_t line) {
        if (s->op == Op::Seq) return s;
        std::vector<Node*> v;
        push_stmt(s, v);
        return seq_of(v, line);
    }

    // Remove a Break(label) in tail position of s (falling off the end of the
    // block has the same effect). Returns the new statement.
    Node* strip_tail_break(Node* s, int32_t label) {
        switch (s->op) {
        case Op::Break:
            if (s->a == label) {
                changed = true;
                return mk_nop(m);
            }
            return s;
        case Op::Seq:
            if (s->nkids) {
                Node* t = strip_tail_break(s->kids[s->nkids - 1], label);
                if (t != s->kids[s->nkids - 1]) {
                    if (t->op == Op::Nop) {
                        std::vector<Node*> v(s->kids, s->kids + s->nkids - 1);
                        return seq_of(v, s->line);
                    }
                    s->kids[s->nkids - 1] = t;
                }
            }
            return s;
        case Op::If: {
            for (uint32_t i = 1; i <= 2; ++i) {
                Node* t = strip_tail_break(s->kids[i], label);
                if (t != s->kids[i]) s->kids[i] = as_seq(t, s->line);
            }
            return s;
        }
        case Op::InlineBlock: {
            Node* t = strip_tail_break(s->kids[0], label);
            if (t != s->kids[0]) s->kids[0] = t;
            return s;
        }
        default:
            return s;
        }
    }
    static bool has_break(const Node* n, int32_t label) {
        bool r = false;
        walk(n, [&](Node* x, Node*, uint32_t) {
            if (x->op == Op::Break && x->a == label) r = true;
            return !r;
        });
        return r;
    }

    Node* stm(Node* n) {
        switch (n->op) {
        case Op::Seq: return seq(n);
        case Op::Nop: return n;
        case Op::If: {
            n->kids[0] = cnd(n->kids[0]);
            n->kids[1] = branch(n->kids[1], n->line);
            n->kids[2] = branch(n->kids[2], n->line);
            Node* c = n->kids[0];
            bool cond_ok_as_stmt = !(c->op == Op::And || c->op == Op::Or);
            bool t;
            if (known_truth(c, t)) {
                changed = true;
                return n->kids[t ? 1 : 2];
            }
            if (cond_ok_as_stmt && (c->type.k == TK::List || c->type.k == TK::Fn || c->type.k == TK::Pair)) {
                // evaluated, then always true
                std::vector<Node*> v;
                discard(c, v);
                push_stmt(n->kids[1], v);
                changed = true;
                return seq_of(v, n->line);
            }
            if (empty_stmt(n->kids[1]) && empty_stmt(n->kids[2])) {
                if (is_pure(c)) {
                    changed = true;
                    return mk_nop(m);
                }
                if (cond_ok_as_stmt) {
                    std::vector<Node*> v;
                    discard(c, v);
                    changed = true;
                    return seq_of(v, n->line);
                }
            }
            return n;
        }
        case Op::While: {
            n->kids[0] = cnd(n->kids[0]);
            bool t;
            if (known_truth(n->kids[0], t) && !t) {
                changed = true;
                return mk_nop(m);
            }
            n->kids[1] = branch(n->kids[1], n->line);
            n->kids[2] = branch(n->kids[2], n->line);
            return n;
        }
        case Op::Return:
        case Op::Print:
            n->kids[0] = val(n->kids[0]);
            return n;
        case Op::InlineBlock: {
            n->kids[0] = stm(n->kids[0]);
            n->kids[0] = strip_tail_break(n->kids[0], n->a);
            if (empty_stmt(n->kids[0])) {
                changed = true;
                return mk_nop(m);
            }
            if (!has_break(n->kids[0], n->a)) {
                changed = true;
                return n->kids[0];
            }
            return n;
        }
        case Op::Break: return n;
        case Op::Free:
            n->kids[0] = val(n->kids[0]);
            return n;
        default: {
            Node* e = val(n);
            std::vector<Node*> v;
            discard(e, v);
            if (v.size() == 1 && v[0] == e) return e;
            changed = true;
            return seq_of(v, n->line);
        }
        }
    }

    // ---- conditions (value tested for truthiness only) ----
    Node* cnd(Node* n) {
        if ((n->op == Op::And || n->op == Op::Or) && (n->flags & F_COND)) {
            n->kids[0] = cnd(n->kids[0]);
            n->kids[1] = cnd(n->kids[1]);
            Node* a = n->kids[0];
            Node* b = n->kids[1];
            bool ta, tb;
            bool ka = known_truth(a, ta), kb = known_truth(b, tb);
            if (n->op == Op::And) {
                if (ka) {
                    changed = true;
                    return ta ? b : cint(0, n->line);
                }
                if (kb && tb) {
                    changed = true;
                    return a;
                }
                if (kb && !tb && is_pure(a)) return cint(0, n->line);
            } else {
                if (ka) {
                    changed = true;
                    return ta ? cint(1, n->line) : b;
                }
                if (kb && !tb) {
                    changed = true;
                    return a;
                }
                if (kb && tb && is_pure(a)) return cint(1, n->line);
            }
            return n;
        }
        n = val(n);
        // !!x -> x ; x != 0 -> x  (truthiness of an Int)
        if (n->op == Op::NotI32 && n->kids[0]->op == Op::NotI32 && n->kids[0]->kids[0]->type.k == TK::Int) {
            changed = true;
            return n->kids[0]->kids[0];
        }
        if (n->op == Op::NeI32 && n->kids[1]->op == Op::ConstInt && n->kids[1]->a == 0 &&
            n->kids[0]->type.k == TK::Int) {
            changed = true;
            return n->kids[0];
        }
        return n;
    }

    // ---- values ----
    Node* val(Node* n) {
        switch (n->op) {
        case Op::Seq: case Op::If: case Op::While: case Op::Return: case Op::Print:
            return stm(n);  // not expected in value position; handled as a statement
        case Op::InlineBlock:
            n->kids[0] = stm(n->kids[0]);
            n->kids[0] = strip_tail_break(n->kids[0], n->a);
            return n;
        case Op::Comma: return comma_node(n);
        case Op::And: case Op::Or:
            if (n->flags & F_COND) {
                n->kids[0] = cnd(n->kids[0]);
                n->kids[1] = cnd(n->kids[1]);
                return n;
            }
            break;
        default: break;
        }
        for (uint32_t i = 0; i < n->nkids; ++i) n->kids[i] = val(n->kids[i]);
        return fold(n);
    }

    Node* comma_node(Node* n) {
        std::vector<Node*> v;
        bool same = true;
        for (uint32_t i = 0; i < n->nkids; ++i) {
            Node* k = n->kids[i];
            bool last = i + 1 == n->nkids;
            Node* s = val(k);
            if (s != k) same = false;
            if (!last) {
                if (s->op == Op::InlineBlock || s->op == Op::Break || s->op == Op::Free) {
                    v.push_back(s);
                } else if (s->op == Op::Nop) {
                    same = false;
                } else if (s->op == Op::Seq || s->op == Op::If || s->op == Op::While || s->op == Op::Return ||
                           s->op == Op::Print) {
                    v.push_back(s);  // unexpected; keep as is
                } else {
                    size_t before = v.size();
                    discard(s, v);
                    if (!(v.size() == before + 1 && v.back() == s)) same = false;
                }
            } else if (s->op == Op::Comma) {
                for (uint32_t j = 0; j < s->nkids; ++j) v.push_back(s->kids[j]);
                same = false;
            } else {
                v.push_back(s);
            }
        }
        if (v.size() == 1) {
            changed = true;
            return v[0];
        }
        if (same && v.size() == n->nkids) return n;
        changed = true;
        Node* c = m.node(Op::Comma, v.back()->type, (uint32_t)v.size());
        for (size_t i = 0; i < v.size(); ++i) c->kids[i] = v[i];
        c->line = n->line;
        c->flags = n->flags;
        return c;
    }

    static bool ci(const Node* n, int32_t v) { return n->op == Op::ConstInt && n->a == v; }
    static bool isc(const Node* n) { return n->op == Op::ConstInt; }

    Node* typed2(Op op, Node* a, Node* b, uint32_t line) {
        Node* r = m.node(op, Type::i32(), {a, b});
        r->line = line;
        return r;
    }
    // Unwrap Box(Int x) in a "kid of any static type" position.
    Node* unbox_kid(Node* k) {
        if (k->op == Op::Box && k->kids[0]->type.k == TK::Int) {
            changed = true;
            return k->kids[0];
        }
        return k;
    }

    Node* fold(Node* n) {
        auto K = [&](uint32_t i) { return n->kids[i]; };
        switch (n->op) {
        case Op::GetGlobal: {
            const Global& g = m.globals[(size_t)n->a];
            if (!(n->flags & F_CHECKED) && g.constant && g.type.k == TK::Int && n->type.k == TK::Int)
                return cint(g.value, n->line);
            return n;
        }
        case Op::SetLocal:
            if (reads[(size_t)n->a] == 0) return repl(n, K(0));
            return n;
        case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::DivI32: {
            Node *a = K(0), *b = K(1);
            if (isc(a) && isc(b)) {
                uint32_t x = (uint32_t)a->a, y = (uint32_t)b->a;
                switch (n->op) {
                case Op::AddI32: return cint((int32_t)(x + y), n->line);
                case Op::SubI32: return cint((int32_t)(x - y), n->line);
                case Op::MulI32: return cint((int32_t)(x * y), n->line);
                default:
                    if (b->a == 0) return n;  // raises at run time: keep
                    if (a->a == INT32_MIN && b->a == -1) return cint(INT32_MIN, n->line);
                    return cint(a->a / b->a, n->line);
                }
            }
            // constants to the right of commutative ops
            if ((n->op == Op::AddI32 || n->op == Op::MulI32) && isc(a) && !isc(b)) {
                n->kids[0] = b;
                n->kids[1] = a;
                std::swap(a, b);
                changed = true;
            }
            switch (n->op) {
            case Op::AddI32:
                if (ci(b, 0)) return repl(n, a);
                if (isc(b) && a->op == Op::AddI32 && isc(a->kids[1])) {
                    n->kids[0] = a->kids[0];
                    n->kids[1] = mk_int(m, (int32_t)((uint32_t)a->kids[1]->a + (uint32_t)b->a), b->line);
                    changed = true;
                    return fold(n);
                }
                break;
            case Op::SubI32:
                if (ci(b, 0)) return repl(n, a);
                if (a->type.k == TK::Int && same_expr(a, b)) return cint(0, n->line);
                break;
            case Op::MulI32:
                if (ci(b, 1)) return repl(n, a);
                if (ci(b, 0)) {
                    if (is_pure(a)) return cint(0, n->line);
                    changed = true;
                    return comma(a, mk_int(m, 0, n->line));
                }
                break;
            case Op::DivI32:
                if (ci(b, 1)) return repl(n, a);
                break;
            default: break;
            }
            return n;
        }
        case Op::NegI32: {
            Node* a = K(0);
            if (isc(a)) return cint((int32_t)(0u - (uint32_t)a->a), n->line);
            if (a->op == Op::NegI32) return repl(n, a->kids[0]);
            return n;
        }
        case Op::AbsI32: {
            Node* a = K(0);
            if (isc(a)) return cint(a->a == INT32_MIN ? INT32_MIN : (a->a < 0 ? -a->a : a->a), n->line);
            return n;
        }
        case Op::NotI32: {
            Node* a = K(0);
            if (isc(a)) return cint(a->a == 0, n->line);
            if (a->op == Op::NotI32 && is_bool_valued(a->kids[0])) return repl(n, a->kids[0]);
            Op inv = invert_cmp(a->op);
            if (inv != Op::Nop) {
                Node* r = m.node(inv, Type::i32(), {a->kids[0], a->kids[1]});
                r->line = a->line;
                changed = true;
                return r;
            }
            return n;
        }
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32: {
            Node *a = K(0), *b = K(1);
            if (isc(a) && isc(b)) {
                int32_t x = a->a, y = b->a;
                bool r = n->op == Op::EqI32 ? x == y : n->op == Op::NeI32 ? x != y : n->op == Op::LtI32 ? x < y
                       : n->op == Op::LeI32 ? x <= y : n->op == Op::GtI32 ? x > y : x >= y;
                return cint(r, n->line);
            }
            if (isc(a) && !isc(b)) {
                n->kids[0] = b;
                n->kids[1] = a;
                n->op = swap_cmp(n->op);
                changed = true;
            }
            return n;
        }
        case Op::DynArith: {
            n->kids[0] = unbox_kid(K(0));
            n->kids[1] = unbox_kid(K(1));
            if (K(0)->type.is_int_like() && K(1)->type.is_int_like()) {
                changed = true;
                return fold(typed2((Op)n->a, K(0), K(1), n->line));
            }
            return n;
        }
        case Op::DynNot:
            if (K(0)->op == Op::Box && K(0)->kids[0]->type.k == TK::Int) {
                Node* r = m.node(Op::NotI32, Type::i32(), {K(0)->kids[0]});
                r->line = n->line;
                changed = true;
                return fold(r);
            }
            return n;
        case Op::DynAbs:
            if (K(0)->op == Op::Box && K(0)->kids[0]->type.k == TK::Int) {
                Node* r = m.node(Op::AbsI32, Type::i32(), {K(0)->kids[0]});
                r->line = n->line;
                changed = true;
                return fold(r);
            }
            return n;
        case Op::DynNeg:
            if (K(0)->op == Op::Box && K(0)->kids[0]->type.k == TK::Int) {
                Node* r = m.node(Op::NegI32, Type::i32(), {K(0)->kids[0]});
                r->line = n->line;
                Node* b = m.node(Op::Box, Type::any(), {fold(r)});
                b->line = n->line;
                changed = true;
                return b;  // Box(Neg x): Any
            }
            return n;
        case Op::UnboxInt:
            if (K(0)->op == Op::Box && K(0)->kids[0]->type.k == TK::Int) return repl(n, K(0)->kids[0]);
            return n;
        case Op::Eq: case Op::Ne: case Op::Cmp: {
            Node *a = K(0), *b = K(1);
            Node *ua = a, *ub = b;
            if (a->op == Op::Box && a->kids[0]->type.k == TK::Int) ua = a->kids[0];
            if (b->op == Op::Box && b->kids[0]->type.k == TK::Int) ub = b->kids[0];
            if (ua->type.is_int_like() && ub->type.is_int_like()) {
                Op o = n->op == Op::Eq ? Op::EqI32 : n->op == Op::Ne ? Op::NeI32 : (Op)n->c;
                changed = true;
                return fold(typed2(o, ua, ub, n->line));
            }
            if (a->type.k == TK::Pair && b->type.k == TK::Pair) {
                Op o = n->op == Op::Eq ? Op::PairEq : n->op == Op::Ne ? Op::PairNe
                     : (Op)n->c == Op::LtI32 ? Op::PairLt : (Op)n->c == Op::LeI32 ? Op::PairLe
                     : (Op)n->c == Op::GtI32 ? Op::PairGt : Op::PairGe;
                changed = true;
                return typed2(o, a, b, n->line);
            }
            return n;
        }
        case Op::Select: {
            if (isc(K(0))) return repl(n, K(0)->a ? K(1) : K(2));
            if (same_expr(K(1), K(2))) return repl(n, K(1));
            return n;
        }
        case Op::PairFirst:
            if (K(0)->op == Op::PairMake && is_pure(K(0)->kids[1])) return repl(n, K(0)->kids[0]);
            return n;
        case Op::PairSecond:
            if (K(0)->op == Op::PairMake) {
                Node* p = K(0);
                if (is_pure(p->kids[0])) return repl(n, p->kids[1]);
                if (p->kids[1]->type.k == TK::Int) {
                    changed = true;
                    return comma(p->kids[0], p->kids[1]);
                }
            }
            return n;
        case Op::PairIndex:
            if (K(0)->op == Op::PairMake && isc(K(1)) && (K(1)->a == 0 || K(1)->a == 1)) {
                Node* p = K(0);
                if (K(1)->a == 0) {
                    if (is_pure(p->kids[1])) return repl(n, p->kids[0]);
                } else {
                    if (is_pure(p->kids[0])) return repl(n, p->kids[1]);
                    if (p->kids[1]->type.k == TK::Int) {
                        changed = true;
                        return comma(p->kids[0], p->kids[1]);
                    }
                }
            }
            return n;
        case Op::And: case Op::Or: {  // value form
            Node* a = K(0);
            if (!isc(a)) return n;
            bool t = a->a != 0;
            if (n->op == Op::And) return repl(n, t ? K(1) : a);
            return repl(n, t ? a : K(1));
        }
        default:
            return n;
        }
    }

    // ---- constant / copy propagation over the top-level body Seq ----
    bool prop_value_ok(const Node* v, uint32_t x) {
        Type tx = f->locals[x].type;
        if (v->type != tx) return false;
        if (tx.k == TK::Int) {
            if (v->op == Op::ConstInt) return true;
        } else if (tx.k == TK::Pair) {
            if (v->op == Op::PairMake && isc(v->kids[0]) && isc(v->kids[1])) return true;
        } else if (tx.k == TK::Fn) {
            // Fn(d) has no run-time representation: a statically known function value
            if (v->op == Op::ConstFn) return true;
        } else {
            return false;
        }
        if (v->op == Op::Local) {
            uint32_t y = (uint32_t)v->a;
            if (y == x || assigns[y] != 0) return false;
            if (f->locals[y].kind != LocalKind::Param) return false;
            return true;
        }
        return false;
    }
    void propagate() {
        Node* body = f->body;
        if (body->op != Op::Seq) return;
        std::vector<int32_t> first_read(f->locals.size(), -1);
        for (uint32_t i = 0; i < body->nkids; ++i) {
            walk(body->kids[i], [&](Node* x, Node*, uint32_t) {
                if (x->op == Op::Local && first_read[(size_t)x->a] < 0) first_read[(size_t)x->a] = (int32_t)i;
                if (x->op == Op::While && (x->flags & F_FOREACH)) {
                    for (int32_t id : {x->a, x->b})
                        if (first_read[(size_t)id] < 0) first_read[(size_t)id] = (int32_t)i;
                }
                return true;
            });
        }
        for (uint32_t i = 0; i < body->nkids; ++i) {
            Node* s = body->kids[i];
            if (s->op != Op::SetLocal) continue;
            uint32_t x = (uint32_t)s->a;
            if (assigns[x] != 1) continue;
            if (first_read[x] >= 0 && first_read[x] <= (int32_t)i) continue;
            Node* v = s->kids[0];
            if (!prop_value_ok(v, x)) continue;
            // replace uses in later statements
            for (uint32_t j = i + 1; j < body->nkids; ++j) {
                rewrite(body->kids[j], [&](Node*& slot) {
                    if (slot->op == Op::Local && slot->a == (int32_t)x) {
                        Node* c = clone(m, v);
                        c->line = slot->line;
                        slot = c;
                    }
                });
            }
            body->kids[i] = mk_nop(m);
            assigns[x] = 0;
            changed = true;
        }
    }

    void run_func(Func& fn) {
        f = &fn;
        for (int it = 0; it < 16; ++it) {
            changed = false;
            count();
            f->body = stm(f->body);
            if (f->body->op != Op::Seq) f->body = as_seq(f->body, f->body->line);
            count();
            propagate();
            if (!changed) break;
        }
    }
};

}  // namespace

void pass_simplify(Module& m) {
    Simp s(m);
    for (auto& f : m.funcs) s.run_func(*f);
}

}  // namespace epq::tir
