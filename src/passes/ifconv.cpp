// Pass "ifconv" — if-conversion of small conditional bodies to Select (DESIGN.md §5.5).
//
// Shape converted (only inside loops, loop depth >= 1):
//   If(c, body, <empty>)   body = 1..kMaxStmts statements after flattening (Seq, Nop,
//                          a Break-free InlineBlock or one whose only Break is its last
//                          statement, statement-position Comma with a pure last kid), each
//                            SetLocal(x, e)       x of type Int or Pair (not a Cache local)
//                            SetIndex(L, i, e)    L a List local, elem Int/Pair, i pure
//   c: an Int-typed expression (not And/Or).  If(And_cond(c1, c2), S) is first read as
//   If(c1, If(c2, S)) and the inner If converted (c1 keeps guarding c2).
//
// Every right-hand side e must be cheap and unable to fail: constants, locals,
// unchecked globals, Int arithmetic except division, comparisons, Len of a local,
// Pair ops that cannot fail, Select, and Index(M, j) whose bounds are ESTABLISHED.
//
// "Established": an Index(M, j') / SetIndex(M, j', ...) with the same canonical index
// was evaluated unconditionally earlier in the same straight-line sequence or in the
// condition (and completed, hence was in bounds), and since then nothing assigned M or a
// local the canonical index mentions, nothing may have changed M's header
// (util::may_mutate_header), and there was no call at all. Canonical indices expand
// locals through their pure Int definitions (`p := j + 1`), so the inlined swap form
// `p := j + 1; tmp := l[j]; l[j] := l[p]; l[p] := tmp` after `l[j] > l[j + 1]` matches.
//
// Rewrite (sequential semantics statement by statement; "speculative" state = the state
// the body would produce if c were true):
//   t := c
//   SetLocal(x, e)      ->  [nx := e';]  x := Select(t, nx | e', x)
//                           later statements read nx for x (speculative value), so every
//                           index evaluated is the index the body would use — which is
//                           exactly what the established-bounds check is done against.
//   SetIndex(L, i, e)   ->  [ne := e';]  old := L[i'];  L[i'] := Select(t, ne | e', old)
//                           (when t == 0 every store writes back the value just read).
// EPQ is single-threaded, so introducing the stores is legal. Loads and stores executed
// with t == 0 are in bounds by the established facts and have no visible effect.
//
// Default: scalar SetLocal bodies only. Element stores need EPQ_IFCONV_STORES=1 (they
// made bubblesort 4.7x slower). On the C engine at default flags, clang already emits the
// same machine code for the scalar shapes on 8 of the 10 benchmarks where the pass fires
// (heapsort, dijkstra, quicksort, ...), so the gain there is nil but nothing is lost; the
// pass exists mainly for the JIT backend, which does not if-convert by itself.
#include <cstdlib>
#include <cstring>
#include <vector>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

using namespace util;

constexpr size_t kMaxStmts = 4;
constexpr size_t kMaxStores = 3;
constexpr size_t kMaxNodes = 48;

struct Fact {
    uint32_t list;  // local id of the list
    Node* idx;      // canonical index expression
};

struct Facts {
    std::vector<Fact> facts;
    std::vector<Node*> defs;  // local id -> canonical pure Int definition (nullptr: none)
};

bool mentions(const Node* n, uint32_t id) {
    bool r = false;
    walk(n, [&](Node* x, Node*, uint32_t) {
        if (x->op == Op::Local && (uint32_t)x->a == id) r = true;
        return !r;
    });
    return r;
}
bool mentions_any(const Node* n, const std::vector<bool>& A) {
    bool r = false;
    walk(n, [&](Node* x, Node*, uint32_t) {
        if (x->op == Op::Local && (size_t)x->a < A.size() && A[(size_t)x->a]) r = true;
        return !r;
    });
    return r;
}

bool is_int_arith(Op o) {
    switch (o) {
    case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::NegI32: case Op::AbsI32: case Op::NotI32:
    case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
        return true;
    default:
        return false;
    }
}

size_t count_nodes(const Node* n) {
    size_t c = 0;
    walk(n, [&](Node*, Node*, uint32_t) {
        ++c;
        return true;
    });
    return c;
}

struct IfConv {
    Module& m;
    Func* f = nullptr;
    int converted = 0;
    // Element-store conversion (A[i] := Select(t, e, A[i])) is OFF by default: measured
    // as a large slowdown on bubblesort (loop-carried store->load dependency defeats
    // store forwarding, and the branch is predicted well). EPQ_IFCONV_STORES=1 enables it.
    bool stores_enabled = false;

    explicit IfConv(Module& mm) : m(mm) {}

    // ---- canonical index expressions ----------------------------------------------
    // Pure Int expression over constants and Int locals; locals with a recorded
    // definition are expanded. nullptr if not canonicalisable.
    Node* canon(Node* n, const Facts& F) {
        if (n->type.k != TK::Int) return nullptr;
        switch (n->op) {
        case Op::ConstInt: return n;
        case Op::Local: {
            size_t id = (size_t)n->a;
            if (id < F.defs.size() && F.defs[id]) return F.defs[id];
            return n;
        }
        default: break;
        }
        if (!is_int_arith(n->op)) return nullptr;
        Node* c = m.node(n->op, n->type, n->nkids);
        c->a = n->a;
        c->b = n->b;
        c->c = n->c;
        c->flags = n->flags;
        for (uint32_t i = 0; i < n->nkids; ++i) {
            c->kids[i] = canon(n->kids[i], F);
            if (!c->kids[i]) return nullptr;
        }
        return c;
    }

    bool established(const Facts& F, uint32_t list, Node* cidx) {
        for (const Fact& x : F.facts)
            if (x.list == list && same_expr(x.idx, cidx)) return true;
        return false;
    }

    // ---- fact gathering --------------------------------------------------------------
    // Record the accesses of n that are certainly evaluated (and completed) whenever n
    // completes. A: locals assigned anywhere in the enclosing statement — an access whose
    // list or raw index mentions one of them is skipped (its value might differ from
    // the one the definitions describe).
    void add_access(Node* n, Facts& F, const std::vector<bool>& A) {
        Node* l = n->kids[0];
        Node* i = n->kids[1];
        if (l->op != Op::Local || l->type.k != TK::List) return;
        if ((size_t)l->a < A.size() && A[(size_t)l->a]) return;
        if (!is_pure(i) || mentions_any(i, A)) return;
        Node* c = canon(i, F);
        if (!c) return;
        if (!established(F, (uint32_t)l->a, c)) F.facts.push_back({(uint32_t)l->a, c});
    }
    void gather(Node* n, Facts& F, const std::vector<bool>& A) {
        switch (n->op) {
        case Op::And: case Op::Or: case Op::If: case Op::While:
            gather(n->kids[0], F, A);  // only the first kid is unconditional
            return;
        case Op::InlineBlock: case Op::Fail:
            return;
        case Op::Index: case Op::SetIndex:
            for (uint32_t k = 0; k < n->nkids; ++k) gather(n->kids[k], F, A);
            add_access(n, F, A);
            return;
        default:
            for (uint32_t k = 0; k < n->nkids; ++k) gather(n->kids[k], F, A);
            return;
        }
    }
    // Accesses certainly evaluated when condition c is true.
    void gather_true(Node* c, Facts& F, const std::vector<bool>& A) {
        if (c->op == Op::And) {
            gather_true(c->kids[0], F, A);
            gather_true(c->kids[1], F, A);
        } else {
            gather(c, F, A);
        }
    }

    // Drop everything n may invalidate.
    void invalidate(Facts& F, const Node* n) {
        std::vector<bool> A;
        locals_assigned(n, A);
        bool call = has_call(n);
        std::vector<Fact> keep;
        for (const Fact& x : F.facts) {
            if (call) continue;
            if (x.list < A.size() && A[x.list]) continue;
            if (mentions_any(x.idx, A)) continue;
            uint32_t r = f->locals[x.list].type.a;
            if (may_mutate_header(m, n, r)) continue;
            keep.push_back(x);
        }
        F.facts.swap(keep);
        for (size_t id = 0; id < F.defs.size(); ++id) {
            if (!F.defs[id]) continue;
            if ((id < A.size() && A[id]) || mentions_any(F.defs[id], A)) F.defs[id] = nullptr;
        }
    }

    // Straight-line statement s executed: gather, invalidate, record a definition.
    void step(Facts& F, Node* s) {
        std::vector<bool> A;
        locals_assigned(s, A);
        gather(s, F, A);
        invalidate(F, s);
        if (s->op == Op::SetLocal && s->type.k == TK::Int && is_pure(s->kids[0])) {
            Node* c = canon(s->kids[0], F);
            if (c && !mentions(c, (uint32_t)s->a)) {
                if ((size_t)s->a >= F.defs.size()) F.defs.resize((size_t)s->a + 1, nullptr);
                F.defs[(size_t)s->a] = c;
            }
        }
    }

    // ---- traversal ---------------------------------------------------------------
    static bool empty_stmt(const Node* n) { return (n->op == Op::Seq && n->nkids == 0) || n->op == Op::Nop; }

    // Statements nested in expressions (InlineBlocks inside Commas): processed with no facts.
    void expr(Node* n, int depth) {
        for (uint32_t i = 0; i < n->nkids; ++i) {
            Node* k = n->kids[i];
            if (k->op == Op::InlineBlock) {
                Facts G;
                stmt(k->kids[0], G, depth);
            } else {
                expr(k, depth);
            }
        }
    }

    void stmt(Node*& slot, Facts& F, int depth) {
        Node* n = slot;
        switch (n->op) {
        case Op::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) stmt(n->kids[i], F, depth);
            return;
        case Op::If: {
            expr_cond(n->kids[0], depth);
            Facts Fc = F;  // after the condition (unconditional part)
            {
                std::vector<bool> A;
                locals_assigned(n->kids[0], A);
                gather(n->kids[0], Fc, A);
                invalidate(Fc, n->kids[0]);
            }
            Node* orig_then = n->kids[1];
            Node* rep = try_convert(n, F, depth);
            if (rep) {
                slot = rep;
                // Effects of the converted code are those of the original If.
                Node* tmp = m.node(Op::If, Type::bot(), {n->kids[0], orig_then, n->kids[2]});
                invalidate(Fc, tmp);
                F = Fc;
                return;
            }
            Facts Ft = F;
            {
                std::vector<bool> A;
                locals_assigned(n->kids[0], A);
                gather_true(n->kids[0], Ft, A);
                invalidate(Ft, n->kids[0]);
            }
            stmt(n->kids[1], Ft, depth);
            Facts Fe = Fc;
            stmt(n->kids[2], Fe, depth);
            invalidate(Fc, n);
            F = Fc;
            return;
        }
        case Op::While: {
            expr_cond(n->kids[0], depth + 1);
            Facts Fb;
            {
                std::vector<bool> A;
                locals_assigned(n->kids[0], A);
                gather_true(n->kids[0], Fb, A);
                invalidate(Fb, n->kids[0]);
            }
            stmt(n->kids[1], Fb, depth + 1);
            if (n->nkids > 2) {
                Facts Fs;
                stmt(n->kids[2], Fs, depth + 1);
            }
            invalidate(F, n);
            return;
        }
        case Op::InlineBlock: {
            Facts G = F;
            stmt(n->kids[0], G, depth);
            invalidate(F, n);
            return;
        }
        default:
            expr(n, depth);
            step(F, n);
            return;
        }
    }
    void expr_cond(Node* c, int depth) {
        if (c->op == Op::InlineBlock) {
            Facts G;
            stmt(c->kids[0], G, depth);
        } else {
            expr(c, depth);
        }
    }

    // ---- conversion ---------------------------------------------------------------
    // Flatten a body into SetLocal / SetIndex statements; false if anything else.
    bool flatten(Node* n, std::vector<Node*>& out) {
        switch (n->op) {
        case Op::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i)
                if (!flatten(n->kids[i], out)) return false;
            return true;
        case Op::Nop: return true;
        case Op::InlineBlock: {
            std::vector<Node*> inner;
            if (!flatten_ib(n->kids[0], n->a, inner)) return false;
            out.insert(out.end(), inner.begin(), inner.end());
            return true;
        }
        case Op::Comma:
            for (uint32_t i = 0; i + 1 < n->nkids; ++i)
                if (!flatten(n->kids[i], out)) return false;
            return n->nkids == 0 || flatten(n->kids[n->nkids - 1], out);
        case Op::SetLocal: case Op::SetIndex:
            out.push_back(n);
            return true;
        default:
            return is_pure(n);  // a pure expression statement does nothing
        }
    }
    // InlineBlock body: a trailing Break to its own label is dropped; any other Break fails.
    bool flatten_ib(Node* body, int32_t label, std::vector<Node*>& out) {
        Node* last = body;
        Node* last_parent = nullptr;
        while (last->op == Op::Seq && last->nkids > 0) {
            last_parent = last;
            last = last->kids[last->nkids - 1];
        }
        if (last->op == Op::Break && last->a == label) {
            if (!last_parent) return true;  // body is just the Break
            // flatten all but the trailing Break
            std::vector<Node*> path;
            for (Node* s = body; s->op == Op::Seq && s->nkids > 0; s = s->kids[s->nkids - 1]) path.push_back(s);
            for (Node* s : path)
                for (uint32_t i = 0; i + 1 < s->nkids; ++i) {
                    if (contains_op(s->kids[i], Op::Break)) return false;
                    if (!flatten(s->kids[i], out)) return false;
                }
            return true;
        }
        if (contains_op(body, Op::Break)) return false;
        return flatten(body, out);
    }

    // Substitute shadow temps for locals assigned earlier in the body.
    Node* subst(Node* n, const std::vector<int32_t>& sh) {
        return clone(m, n, sh);
    }

    // Cheap, cannot fail, and every Index is established.
    bool cheap(Node* n, const Facts& F) {
        switch (n->op) {
        case Op::ConstInt: case Op::Local: return true;
        case Op::GetGlobal: return !(n->flags & F_CHECKED);
        case Op::PairFirst: case Op::PairSecond: case Op::PairMake:
        case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
        case Op::Select:
            break;
        case Op::Len:
            return n->kids[0]->op == Op::Local;
        case Op::Index: {
            Node* l = n->kids[0];
            Node* i = n->kids[1];
            if (l->op != Op::Local || l->type.k != TK::List || !is_pure(i)) return false;
            Node* c = canon(i, F);
            return c && established(F, (uint32_t)l->a, c);
        }
        default:
            if (!is_int_arith(n->op)) return false;
            break;
        }
        for (uint32_t i = 0; i < n->nkids; ++i)
            if (!cheap(n->kids[i], F)) return false;
        return true;
    }

    static bool slot_type_ok(const Module& m, Type t) {
        return t.k == TK::Int || (t.k == TK::Pair && t.a < m.regions.size() && m.regions[t.a].pair);
    }

    Node* try_convert(Node* ifn, const Facts& F, int depth) {
        if (depth < 1 || !empty_stmt(ifn->kids[2])) return nullptr;
        return convert_cond(ifn->kids[0], ifn->kids[1], ifn->kids[2], F);
    }

    Node* convert_cond(Node* c, Node* body, Node* els, const Facts& F) {
        if (c->op == Op::And && (c->flags & F_COND)) {
            Facts G = F;
            std::vector<bool> A;
            locals_assigned(c->kids[0], A);
            gather_true(c->kids[0], G, A);
            invalidate(G, c->kids[0]);
            Node* inner = convert_cond(c->kids[1], body, els, G);
            if (!inner) return nullptr;
            Node* n = m.node(Op::If, Type::bot(), {c->kids[0], inner, mk_seq(m, {})});
            n->line = c->line;
            return n;
        }
        return convert_core(c, body, F);
    }

    Node* convert_core(Node* c, Node* body, const Facts& F0) {
        if (c->type.k != TK::Int || c->op == Op::And || c->op == Op::Or) return nullptr;
        std::vector<Node*> ss;
        if (!flatten(body, ss) || ss.empty() || ss.size() > kMaxStmts) return nullptr;
        size_t nodes = 0, stores = 0;
        for (Node* s : ss) {
            nodes += count_nodes(s);
            if (s->op == Op::SetIndex) ++stores;
        }
        if (nodes > kMaxNodes || stores > kMaxStores) return nullptr;
        if (stores && !stores_enabled) return nullptr;

        Facts F = F0;
        {
            std::vector<bool> A;
            locals_assigned(c, A);
            gather(c, F, A);
            invalidate(F, c);
        }

        // Validate first (no allocation of locals until we know we convert).
        // Shadow map: local -> shadow temp (or -1). Temps are created lazily below, so
        // validation uses a symbolic pass with the same structure.
        struct Plan {
            Node* s;
            bool shadow;  // SetLocal: needs a shadow temp
        };
        std::vector<Plan> plan;
        for (size_t k = 0; k < ss.size(); ++k) {
            Node* s = ss[k];
            if (s->op == Op::SetLocal) {
                const Local& L = f->locals[(size_t)s->a];
                if (L.kind == LocalKind::Cache || !slot_type_ok(m, L.type) || s->kids[0]->type != L.type) return nullptr;
                bool read_later = false;
                for (size_t j = k + 1; j < ss.size(); ++j)
                    if (mentions(ss[j], (uint32_t)s->a)) read_later = true;
                plan.push_back({s, read_later || !is_pure(s->kids[0])});
            } else {
                Node* l = s->kids[0];
                if (l->op != Op::Local || l->type.k != TK::List) return nullptr;
                Type et = elem_of(m, l->type);
                if (!slot_type_ok(m, et) || s->type != et || s->kids[2]->type != et) return nullptr;
                if (!is_pure(s->kids[1])) return nullptr;
                plan.push_back({s, false});
            }
        }

        // Build (into a scratch list; abandoned temps are harmless but avoid them: we
        // check everything before creating any local).
        std::vector<int32_t> sh(f->locals.size(), -1);
        // Symbolic validation pass with placeholder shadows: shadows are local ids that
        // do not exist yet, so do the real construction directly but verify before
        // committing: collect created temps and roll back on failure.
        size_t nlocals0 = f->locals.size();
        auto rollback = [&]() -> Node* {
            f->locals.resize(nlocals0);
            return nullptr;
        };
        uint32_t t = m.new_temp(f, Type::i32());
        std::vector<Node*> out;
        Node* st = mk_set(m, *f, t, c);
        out.push_back(st);
        auto T = [&]() { return mk_local(m, *f, t, c->line); };
        for (const Plan& p : plan) {
            Node* s = p.s;
            if (s->op == Op::SetLocal) {
                uint32_t x = (uint32_t)s->a;
                Node* e = subst(s->kids[0], sh);
                if (!cheap(e, F)) return rollback();
                Node* val = e;
                if (p.shadow) {
                    uint32_t nx = m.new_temp(f, f->locals[x].type);
                    Node* se = mk_set(m, *f, nx, e);
                    se->line = s->line;
                    out.push_back(se);
                    // definition of the shadow temp (speculative value of x)
                    if (f->locals[x].type.k == TK::Int && is_pure(e)) {
                        Node* ce = canon(e, F);
                        if (ce) {
                            if (F.defs.size() <= nx) F.defs.resize((size_t)nx + 1, nullptr);
                            F.defs[nx] = ce;
                        }
                    }
                    val = mk_local(m, *f, nx, s->line);
                    if (sh.size() <= x) sh.resize((size_t)x + 1, -1);
                    sh[x] = (int32_t)nx;
                }
                Node* sel = m.node(Op::Select, f->locals[x].type, {T(), val, mk_local(m, *f, x, s->line)});
                sel->line = s->line;
                Node* sx = mk_set(m, *f, x, sel);
                sx->line = s->line;
                out.push_back(sx);
            } else {
                Node* l = s->kids[0];
                Node* i = subst(s->kids[1], sh);
                Node* ci = canon(i, F);
                if (!ci || !established(F, (uint32_t)l->a, ci)) return rollback();
                Node* e = subst(s->kids[2], sh);
                if (!cheap(e, F)) return rollback();
                Type et = s->type;
                Node* val = e;
                if (!is_pure(e)) {
                    uint32_t ne = m.new_temp(f, e->type);
                    Node* se = mk_set(m, *f, ne, e);
                    out.push_back(se);
                    val = mk_local(m, *f, ne, s->line);
                }
                uint32_t old = m.new_temp(f, et);
                Node* ld = m.node(Op::Index, et, {clone(m, l), clone(m, i)});
                ld->line = s->line;
                out.push_back(mk_set(m, *f, old, ld));
                Node* sel = m.node(Op::Select, et, {T(), val, mk_local(m, *f, old, s->line)});
                sel->line = s->line;
                Node* stn = m.node(Op::SetIndex, et, {clone(m, l), clone(m, i), sel});
                stn->line = s->line;
                out.push_back(stn);
            }
        }
        ++converted;
        Node* seq = mk_seq(m, out);
        seq->line = c->line;
        return seq;
    }

    void run() {
        for (auto& fp : m.funcs) {
            f = fp.get();
            if (!f->body) continue;
            Facts F;
            stmt(f->body, F, 0);
        }
    }
};

}  // namespace

void pass_ifconv(Module& m) {
    pass_region_info(m);
    IfConv ic(m);
    const char* e = std::getenv("EPQ_IFCONV_STORES");
    ic.stores_enabled = e && *e && std::strcmp(e, "0") != 0;
    ic.run();
}

}  // namespace epq::tir
