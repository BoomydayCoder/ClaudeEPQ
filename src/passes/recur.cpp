// Pass "recur" — recursion optimisations. Runs right after "inline".
//
// 1. Purity (module-wide greatest fixpoint over the call graph): Func::pure
//    (see src/tir.hpp). Int-only instances that write no global, do no I/O,
//    make no dynamic call, read only Fn-typed / constant globals and call only
//    pure instances. May fail / diverge (deterministically in the arguments).
//
// 2. Self tail calls -> loops. In instance F every statement-context
//    Return(CallDirect F (args, callee)) (also through Comma: the prefix kids
//    are kept as statements) becomes
//        [t_i := arg_i ...]   in order (pure args that nothing evaluated later
//                             can disturb are not copied)
//        [callee]             kept if it is not pure
//        p_i := t_i / arg_i   in order
//        Break L
//    and the body becomes
//        While [ConstInt 1, InlineBlock L { resets; body; [Return implicit] }, Seq{}]
//    `resets` re-initialise (Int: 0, Any: boxed 0 — as the backends' frame
//    initialisation) every non-parameter local that may be read before it is
//    written in a fresh activation (a must-assigned scan; unclean definitions).
//    If such a local has another type, or the body may fall off its end and the
//    result type has no plain zero, F is left alone.
//
// 3. Duplicate pure-call elimination (all instances, per top-level statement S
//    of the body): calls of the same pure instance with structurally equal pure
//    arguments whose locals are not assigned anywhere in S compute the same
//    value (or fail / diverge identically at the first one executed). If the
//    first one (in evaluation order) is unconditional in S (not under an If
//    branch, loop, &/| right operand or InlineBlock), the later ones read its
//    temp; otherwise every occurrence becomes
//        Comma[Or(h, Comma[t := call, h := 1]), t]   with h := 0 before S.
//
// 4. One-level self-unrolling of small pure self-recursive instances (fib):
//    one self call site (the one that lets step 3 remove most calls) is
//    replaced by an inlined copy of the original body
//        Comma[InlineBlock L' { binds; resets; body' }, Local res]
//    (pure small arguments of parameters the body never assigns are
//    substituted; Return v -> {res := v; Break L'}), add/sub-by-constant chains
//    are canonicalised (k-1-1 == k-2), step 3 runs, and the result is kept only
//    if it removed at least one call and stays within a node budget.
#include <algorithm>
#include <cstdint>
#include <functional>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

using util::walk;

uint32_t count_nodes(const Node* n) {
    uint32_t c = 1;
    for (uint32_t i = 0; i < n->nkids; ++i) c += count_nodes(n->kids[i]);
    return c;
}

void locals_read(const Node* n, std::vector<bool>& out) {
    walk(n, [&](Node* x, Node*, uint32_t) {
        if (x->op == Op::Local) {
            if ((size_t)x->a >= out.size()) out.resize((size_t)x->a + 1, false);
            out[(size_t)x->a] = true;
        }
        return true;
    });
}
bool intersects(const std::vector<bool>& a, const std::vector<bool>& b) {
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
        if (a[i] && b[i]) return true;
    return false;
}

// Pure (no effect, cannot fail) AND its value can only change through an
// assignment to a local it reads: no list-state reads (Len, ElemLoad,
// ListData) and no reads of non-constant globals (SetGlobal / calls elsewhere
// could change them).
bool stable_pure(const Module& m, const Node* n) {
    if (!is_pure(n)) return false;
    bool ok = true;
    walk(n, [&](Node* x, Node*, uint32_t) {
        switch (x->op) {
        case Op::Len: case Op::ElemLoad: case Op::ListData: ok = false; break;
        case Op::GetGlobal:
            if (x->type.k != TK::Fn && !m.globals[(size_t)x->a].constant) ok = false;
            break;
        default: break;
        }
        return ok;
    });
    return ok;
}

// Visit every statement in statement context (Seq kids, If branches, While
// body/step, InlineBlock body), giving the slot of each non-structural statement.
template <class Fn>
void stmt_walk(Node*& slot, Fn&& fn) {
    Node* n = slot;
    switch (n->op) {
    case Op::Seq:
        for (uint32_t i = 0; i < n->nkids; ++i) stmt_walk(n->kids[i], fn);
        return;
    case Op::If:
        stmt_walk(n->kids[1], fn);
        stmt_walk(n->kids[2], fn);
        return;
    case Op::While:
        stmt_walk(n->kids[1], fn);
        stmt_walk(n->kids[2], fn);
        return;
    case Op::InlineBlock:
        stmt_walk(n->kids[0], fn);
        return;
    default:
        fn(slot);
        return;
    }
}

// Every Return in n is in statement context.
bool returns_in_stmt_context(Node* body) {
    uint32_t total = 0, ctx = 0;
    walk(body, [&](Node* x, Node*, uint32_t) {
        if (x->op == Op::Return) ++total;
        return true;
    });
    Node* b = body;
    stmt_walk(b, [&](Node*& s) {
        if (s->op == Op::Return) {
            ++ctx;
            // a Return nested inside a statement-context Return's value?
            walk(s->kids[0], [&](Node* x, Node*, uint32_t) {
                if (x->op == Op::Return) ++total;  // makes the counts differ
                return true;
            });
        }
    });
    return total == ctx;
}

// May execution fall off the end of n? (conservative: true when unsure)
bool may_complete(const Node* n) {
    switch (n->op) {
    case Op::Return: return false;
    case Op::Seq:
        for (uint32_t i = 0; i < n->nkids; ++i)
            if (!may_complete(n->kids[i])) return false;
        return true;
    case Op::If: return may_complete(n->kids[1]) || may_complete(n->kids[2]);
    default:
        if (!is_stmt_op(n->op) && n->type.k == TK::Bot) return false;  // Fail etc.
        return true;
    }
}

// Must-assigned scan: need[id] = local id may be read before it is written in
// a fresh activation (D starts with the parameters). InlineBlocks: the state
// after the block is the intersection of the states at its Breaks and at the
// end of its body.
struct Scan {
    std::vector<char> need;
    struct Blk {
        int32_t label;
        bool has;
        std::vector<char> acc;
    };
    std::vector<Blk> blocks;
    static void meet(Blk& b, const std::vector<char>& D) {
        if (!b.has) {
            b.acc = D;
            b.has = true;
        } else {
            for (size_t i = 0; i < D.size(); ++i) b.acc[i] = b.acc[i] && D[i];
        }
    }
    void run(const Node* n, std::vector<char>& D) {
        switch (n->op) {
        case Op::Local:
            if (!D[(size_t)n->a]) need[(size_t)n->a] = 1;
            return;
        case Op::SetLocal:
            run(n->kids[0], D);
            D[(size_t)n->a] = 1;
            return;
        case Op::If: {
            run(n->kids[0], D);
            std::vector<char> D1 = D, D2 = D;
            run(n->kids[1], D1);
            run(n->kids[2], D2);
            for (size_t i = 0; i < D.size(); ++i) D[i] = D1[i] && D2[i];
            return;
        }
        case Op::While: {
            run(n->kids[0], D);
            std::vector<char> Dc = D;
            if (n->flags & F_FOREACH) {
                if (!D[(size_t)n->a]) need[(size_t)n->a] = 1;
                if (!D[(size_t)n->b]) need[(size_t)n->b] = 1;
                Dc[(size_t)n->a] = 1;
                Dc[(size_t)n->b] = 1;
            }
            run(n->kids[1], Dc);
            run(n->kids[2], Dc);
            return;
        }
        case Op::And: case Op::Or: {
            run(n->kids[0], D);
            std::vector<char> Dc = D;
            run(n->kids[1], Dc);
            return;
        }
        case Op::InlineBlock: {
            blocks.push_back({n->a, false, {}});
            std::vector<char> Dc = D;
            run(n->kids[0], Dc);
            Blk b = std::move(blocks.back());
            blocks.pop_back();
            meet(b, Dc);
            D = b.acc;
            return;
        }
        case Op::Break:
            for (size_t i = blocks.size(); i-- > 0;)
                if (blocks[i].label == n->a) {
                    meet(blocks[i], D);
                    break;
                }
            return;
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) run(n->kids[i], D);
            return;
        }
    }
};

std::vector<char> needs_reset(const Func& f, const Node* body) {
    Scan sc;
    std::vector<char> D(f.locals.size(), 0);
    sc.need.assign(f.locals.size(), 0);
    for (uint32_t p : f.params) D[p] = 1;
    sc.run(body, D);
    for (uint32_t p : f.params) sc.need[p] = 0;
    return sc.need;
}

// Initial value of a fresh frame's local (backends: Int 0, Any boxed 0).
Node* zero_of(Module& m, Type t) {
    if (t.k == TK::Int) return util::mk_int(m, 0);
    if (t.k == TK::Any) return m.node(Op::Box, Type::any(), {util::mk_int(m, 0)});
    return nullptr;
}

// ---------------------------------------------------------------------------
// 1. Purity
// ---------------------------------------------------------------------------
bool locally_pure(const Module& m, const Func& f) {
    if (f.is_main || f.generic || f.ret.k != TK::Int) return false;
    for (auto& l : f.locals)
        if (l.type.k != TK::Int) return false;
    bool ok = true;
    walk(f.body, [&](Node* x, Node*, uint32_t) {
        TK k = x->type.k;
        if (k != TK::Int && k != TK::Bot && k != TK::Fn) ok = false;
        switch (x->op) {
        case Op::SetGlobal: case Op::Print: case Op::Input: case Op::CallDyn:
        case Op::Free: case Op::ListLit:
            ok = false;
            break;
        case Op::GetGlobal:
            if (x->type.k != TK::Fn && !m.globals[(size_t)x->a].constant) ok = false;
            break;
        default: break;
        }
        return ok;
    });
    return ok;
}

void compute_purity(Module& m) {
    for (auto& f : m.funcs) f->pure = locally_pure(m, *f);
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& f : m.funcs) {
            if (!f->pure) continue;
            bool ok = true;
            walk(f->body, [&](Node* x, Node*, uint32_t) {
                if (x->op == Op::CallDirect && !m.funcs[(size_t)x->a]->pure) ok = false;
                return ok;
            });
            if (!ok) {
                f->pure = false;
                changed = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Self tail calls
// ---------------------------------------------------------------------------
bool is_self_tail(const Func& f, Node* ret, std::vector<Node*>* pre, Node** call) {
    if (ret->op != Op::Return) return false;
    Node* v = ret->kids[0];
    std::vector<Node*> p;
    while (v->op == Op::Comma) {
        for (uint32_t i = 0; i + 1 < v->nkids; ++i) p.push_back(v->kids[i]);
        v = v->kids[v->nkids - 1];
    }
    if (v->op != Op::CallDirect || (uint32_t)v->a != f.id || v->nkids != f.params.size() + 1) return false;
    if (v->type != f.ret) return false;
    if (pre) *pre = std::move(p);
    if (call) *call = v;
    return true;
}

Node* tail_replacement(Module& m, Func& f, std::vector<Node*> pre, Node* call, uint32_t label) {
    size_t n = f.params.size();
    std::vector<Node*> out = std::move(pre);
    std::vector<int> mode(n, 0);  // 0 skip (arg is the param itself), 1 temp, 2 deferred
    std::vector<int32_t> temp(n, -1);
    std::vector<bool> assigned_params(f.locals.size(), false);
    for (size_t i = 0; i < n; ++i) {
        Node* a = call->kids[i];
        if (a->op == Op::Local && (uint32_t)a->a == f.params[i]) continue;
        mode[i] = 1;
        if (stable_pure(m, a)) {
            std::vector<bool> rd;
            locals_read(a, rd);
            bool ok = !intersects(rd, assigned_params);
            for (size_t j = i + 1; j <= n && ok; ++j) {
                std::vector<bool> as;
                util::locals_assigned(call->kids[j], as);
                if (intersects(rd, as)) ok = false;
            }
            if (ok) mode[i] = 2;
        }
        assigned_params[f.params[i]] = true;
    }
    for (size_t i = 0; i < n; ++i)
        if (mode[i] == 1) {
            temp[i] = (int32_t)m.new_temp(&f, f.locals[f.params[i]].type);
            out.push_back(util::mk_set(m, f, (uint32_t)temp[i], call->kids[i]));
        }
    Node* callee = call->kids[n];
    if (!is_pure(callee)) out.push_back(callee);
    for (size_t i = 0; i < n; ++i) {
        if (mode[i] == 1) out.push_back(util::mk_set(m, f, f.params[i], util::mk_local(m, f, (uint32_t)temp[i], call->line)));
        else if (mode[i] == 2) out.push_back(util::mk_set(m, f, f.params[i], call->kids[i]));
    }
    Node* br = m.node(Op::Break, Type::bot(), 0);
    br->a = (int32_t)label;
    br->line = call->line;
    out.push_back(br);
    return util::mk_seq(m, out);
}

// Returns the inner Seq (the loop body's statement list) if F was transformed.
Node* tail_calls(Module& m, Func& f) {
    if (f.is_main) return nullptr;
    uint32_t sites = 0;
    stmt_walk(f.body, [&](Node*& s) {
        if (is_self_tail(f, s, nullptr, nullptr)) ++sites;
    });
    if (!sites) return nullptr;
    std::vector<char> need = needs_reset(f, f.body);
    std::vector<Node*> resets;
    for (size_t i = 0; i < need.size(); ++i) {
        if (!need[i]) continue;
        Node* z = zero_of(m, f.locals[i].type);
        if (!z) return nullptr;
        resets.push_back(util::mk_set(m, f, (uint32_t)i, z));
    }
    Node* tail_ret = nullptr;
    if (may_complete(f.body)) {
        Node* z = zero_of(m, f.ret);
        if (!z) return nullptr;
        tail_ret = m.node(Op::Return, Type::bot(), {z});
    }
    uint32_t label = new_label(f);
    stmt_walk(f.body, [&](Node*& s) {
        std::vector<Node*> pre;
        Node* call = nullptr;
        if (is_self_tail(f, s, &pre, &call)) s = tail_replacement(m, f, std::move(pre), call, label);
    });
    std::vector<Node*> inner = resets;
    for (uint32_t i = 0; i < f.body->nkids; ++i) inner.push_back(f.body->kids[i]);
    if (tail_ret) inner.push_back(tail_ret);
    Node* seq = util::mk_seq(m, inner);
    Node* ib = m.node(Op::InlineBlock, Type::bot(), {seq});
    ib->a = (int32_t)label;
    Node* loop = m.node(Op::While, Type::bot(), {util::mk_int(m, 1), ib, util::mk_seq(m, {})});
    f.body = util::mk_seq(m, {loop});
    return seq;
}

// ---------------------------------------------------------------------------
// 3. Duplicate pure-call elimination
// ---------------------------------------------------------------------------
struct CallSite {
    Node** slot;
    bool uncond;
};

bool callee_ok(const Node* c) { return is_pure(c) || (c->op == Op::GetGlobal && c->type.k == TK::Fn); }

bool cse_candidate(const Module& m, const Node* n) {
    if (n->op != Op::CallDirect || !m.funcs[(size_t)n->a]->pure) return false;
    for (uint32_t i = 0; i + 1 < n->nkids; ++i)
        if (!stable_pure(m, n->kids[i])) return false;
    return callee_ok(n->kids[n->nkids - 1]);
}

bool same_call(const Node* a, const Node* b) {
    if (a->a != b->a || a->nkids != b->nkids) return false;
    for (uint32_t i = 0; i + 1 < a->nkids; ++i)
        if (!util::same_expr(a->kids[i], b->kids[i])) return false;
    const Node* x = a->kids[a->nkids - 1];
    const Node* y = b->kids[b->nkids - 1];
    if (x->op == Op::GetGlobal && y->op == Op::GetGlobal) return x->a == y->a;
    return util::same_expr(x, y);
}

void collect_calls(const Module& m, Node*& slot, bool uncond, std::vector<CallSite>& out) {
    Node* n = slot;
    switch (n->op) {
    case Op::If:
        collect_calls(m, n->kids[0], uncond, out);
        collect_calls(m, n->kids[1], false, out);
        collect_calls(m, n->kids[2], false, out);
        return;
    case Op::While:
        for (uint32_t i = 0; i < 3; ++i) collect_calls(m, n->kids[i], false, out);
        return;
    case Op::And: case Op::Or:
        collect_calls(m, n->kids[0], uncond, out);
        collect_calls(m, n->kids[1], false, out);
        return;
    case Op::InlineBlock:
        collect_calls(m, n->kids[0], false, out);
        return;
    default:
        for (uint32_t i = 0; i < n->nkids; ++i) collect_calls(m, n->kids[i], uncond, out);
        if (cse_candidate(m, n)) out.push_back({&slot, uncond});  // post-order = evaluation order
        return;
    }
}

// CSE inside each top-level statement of the Seq `seq`. Returns calls removed.
uint32_t cse_seq(Module& m, Func& f, Node* seq) {
    if (seq->op != Op::Seq) return 0;
    uint32_t removed = 0;
    std::vector<Node*> out;
    bool changed = false;
    for (uint32_t si = 0; si < seq->nkids; ++si) {
        Node*& S = seq->kids[si];
        std::vector<CallSite> calls;
        collect_calls(m, S, true, calls);
        if (calls.size() < 2) {
            out.push_back(S);
            continue;
        }
        std::vector<bool> assigned;
        util::locals_assigned(S, assigned);
        std::vector<bool> used(calls.size(), false);
        for (size_t i = 0; i < calls.size(); ++i) {
            if (used[i]) continue;
            Node* c1 = *calls[i].slot;
            std::vector<bool> rd;
            for (uint32_t k = 0; k + 1 < c1->nkids; ++k) locals_read(c1->kids[k], rd);
            if (intersects(rd, assigned)) continue;
            std::vector<size_t> group{i};
            for (size_t j = i + 1; j < calls.size(); ++j)
                if (!used[j] && same_call(c1, *calls[j].slot)) group.push_back(j);
            if (group.size() < 2) continue;
            for (size_t g : group) used[g] = true;
            Type rt = c1->type;
            uint32_t t = m.new_temp(&f, rt);
            uint32_t line = c1->line;
            if (calls[i].uncond) {
                *calls[i].slot = m.node(Op::Comma, rt, {util::mk_set(m, f, t, c1), util::mk_local(m, f, t, line)});
                for (size_t g = 1; g < group.size(); ++g) *calls[group[g]].slot = util::mk_local(m, f, t, line);
            } else {
                uint32_t h = m.new_temp(&f, Type::i32());
                out.push_back(util::mk_set(m, f, h, util::mk_int(m, 0, line)));
                for (size_t g : group) {
                    Node* c = *calls[g].slot;
                    Node* comp = m.node(Op::Comma, Type::i32(),
                                        {util::mk_set(m, f, t, c), util::mk_set(m, f, h, util::mk_int(m, 1, line))});
                    Node* orn = m.node(Op::Or, Type::i32(), {util::mk_local(m, f, h, line), comp});
                    orn->line = line;
                    *calls[g].slot = m.node(Op::Comma, rt, {orn, util::mk_local(m, f, t, line)});
                }
            }
            removed += (uint32_t)group.size() - 1;
            changed = true;
        }
        out.push_back(S);
    }
    if (changed) {
        Node* ns = util::mk_seq(m, out);
        seq->nkids = ns->nkids;
        seq->kids = ns->kids;
    }
    return removed;
}

// ---------------------------------------------------------------------------
// 4. Self-unrolling
// ---------------------------------------------------------------------------
// Canonical linear form of add/sub-by-constant chains.
Node* lin_base(Node* n, uint32_t& off) {
    if (n->op == Op::ConstInt) {
        off += (uint32_t)n->a;
        return nullptr;
    }
    if (n->op == Op::AddI32 && n->kids[1]->op == Op::ConstInt) {
        off += (uint32_t)n->kids[1]->a;
        return lin_base(n->kids[0], off);
    }
    if (n->op == Op::AddI32 && n->kids[0]->op == Op::ConstInt) {
        off += (uint32_t)n->kids[0]->a;
        return lin_base(n->kids[1], off);
    }
    if (n->op == Op::SubI32 && n->kids[1]->op == Op::ConstInt) {
        off -= (uint32_t)n->kids[1]->a;
        return lin_base(n->kids[0], off);
    }
    return n;
}
void canonicalise(Module& m, Node*& root) {
    util::rewrite(root, [&](Node*& s) {
        Node* n = s;
        if (n->op != Op::AddI32 && n->op != Op::SubI32) return;
        if (n->kids[0]->op != Op::ConstInt && n->kids[1]->op != Op::ConstInt) return;
        uint32_t off = 0;
        Node* b = lin_base(n, off);
        int32_t o = (int32_t)off;
        Node* r;
        if (!b) r = util::mk_int(m, o, n->line);
        else if (o == 0) r = b;
        else if (o < 0 && o != INT32_MIN) r = m.node(Op::SubI32, Type::i32(), {b, util::mk_int(m, -o, n->line)});
        else r = m.node(Op::AddI32, Type::i32(), {b, util::mk_int(m, o, n->line)});
        r->line = n->line;
        s = r;
    });
}

// Inline expansion of `call` (a self call of f) using the original body `orig`.
Node* expand(Module& m, Func& f, const Node* orig, Node* call) {
    size_t np = f.params.size();
    size_t nloc = f.locals.size();
    std::vector<int32_t> map(nloc, -1);
    std::vector<char> need = needs_reset(f, orig);
    for (size_t i = 0; i < nloc; ++i) {
        Local l = f.locals[i];
        l.kind = LocalKind::Temp;
        l.unique = false;
        f.locals.push_back(l);
        map[i] = (int32_t)(f.locals.size() - 1);
    }
    std::vector<Node*> stmts;
    std::vector<Node*> subst(f.locals.size(), nullptr);  // param local -> substituted arg
    for (size_t i = 0; i < np; ++i) {
        Node* a = call->kids[i];
        uint32_t p = f.params[i];
        bool sub = false;
        if (stable_pure(m, a) && count_nodes(a) <= 6 && !util::assigns_local(orig, p)) {
            std::vector<bool> rd;
            locals_read(a, rd);
            sub = true;
            for (size_t j = 0; j <= np && sub; ++j) {
                if (j == i) continue;
                std::vector<bool> as;
                util::locals_assigned(call->kids[j], as);
                if (intersects(rd, as)) sub = false;
            }
        }
        if (sub) subst[(size_t)map[p]] = a;
        else stmts.push_back(util::mk_set(m, f, (uint32_t)map[p], a));
    }
    if (!is_pure(call->kids[np])) stmts.push_back(call->kids[np]);
    for (size_t i = 0; i < nloc; ++i)
        if (need[i]) stmts.push_back(util::mk_set(m, f, (uint32_t)map[i], util::mk_int(m, 0)));
    Node* body = util::clone(m, orig, map, &f);
    util::rewrite(body, [&](Node*& s) {
        if (s->op == Op::Local && (size_t)s->a < subst.size() && subst[(size_t)s->a]) s = util::clone(m, subst[(size_t)s->a]);
    });
    uint32_t res = m.new_temp(&f, f.ret);
    uint32_t label = new_label(f);
    stmt_walk(body, [&](Node*& s) {
        if (s->op != Op::Return) return;
        Node* v = s->kids[0];
        if (v->type.k == TK::Bot) v = m.node(Op::Coerce, f.ret, {v});
        Node* br = m.node(Op::Break, Type::bot(), 0);
        br->a = (int32_t)label;
        s = util::mk_seq(m, {util::mk_set(m, f, res, v), br});
    });
    stmts.push_back(body);
    Node* ib = m.node(Op::InlineBlock, Type::bot(), {util::mk_seq(m, stmts)});
    ib->a = (int32_t)label;
    ib->line = call->line;
    Node* r = m.node(Op::Comma, f.ret, {ib, util::mk_local(m, f, res, call->line)});
    r->line = call->line;
    return r;
}

std::vector<Node**> self_call_slots(const Func& f, Node*& root) {
    std::vector<Node**> v;
    std::function<void(Node*&)> rec = [&](Node*& s) {
        for (uint32_t i = 0; i < s->nkids; ++i) rec(s->kids[i]);
        if (s->op == Op::CallDirect && (uint32_t)s->a == f.id) v.push_back(&s);
    };
    rec(root);
    return v;
}

constexpr uint32_t kUnrollMaxBody = 60;
constexpr uint32_t kUnrollMaxResult = 260;

// Try unrolling self call site k; returns calls removed (0 = failed) and the new body.
uint32_t try_unroll(Module& m, Func& f, size_t k, Node** out_body) {
    Node* body = util::clone(m, f.body);
    auto slots = self_call_slots(f, body);
    if (k >= slots.size()) return 0;
    *slots[k] = expand(m, f, f.body, *slots[k]);
    canonicalise(m, body);
    uint32_t removed = cse_seq(m, f, body);
    if (!removed || count_nodes(body) > kUnrollMaxResult) return 0;
    *out_body = body;
    return removed;
}

bool unroll(Module& m, Func& f) {
    if (!f.pure || f.is_main) return false;
    if (count_nodes(f.body) > kUnrollMaxBody || may_complete(f.body) || !returns_in_stmt_context(f.body)) return false;
    // >= 2 self calls in one top-level statement
    bool two = false;
    for (uint32_t i = 0; i < f.body->nkids; ++i) {
        uint32_t c = 0;
        walk(f.body->kids[i], [&](Node* x, Node*, uint32_t) {
            if (x->op == Op::CallDirect && (uint32_t)x->a == f.id) ++c;
            return true;
        });
        if (c >= 2) two = true;
    }
    if (!two) return false;
    size_t nsites = self_call_slots(f, f.body).size();
    size_t saved_locals = f.locals.size();
    uint32_t saved_label = f.next_label;
    uint32_t best = 0;
    size_t best_k = 0;
    for (size_t k = 0; k < nsites; ++k) {
        Node* b = nullptr;
        uint32_t r = try_unroll(m, f, k, &b);
        f.locals.resize(saved_locals);
        f.next_label = saved_label;
        if (r > best) {
            best = r;
            best_k = k;
        }
    }
    if (!best) return false;
    Node* b = nullptr;
    try_unroll(m, f, best_k, &b);
    f.body = b;
    return true;
}

}  // namespace

void pass_recur(Module& m) {
    compute_purity(m);
    bool changed = false;
    std::vector<Node*> loop_seq(m.funcs.size(), nullptr);
    for (auto& f : m.funcs) {
        loop_seq[f->id] = tail_calls(m, *f);
        if (loop_seq[f->id]) changed = true;
    }
    for (auto& f : m.funcs) {
        if (!loop_seq[f->id] && unroll(m, *f)) {
            changed = true;
            continue;
        }
        Node* seq = loop_seq[f->id] ? loop_seq[f->id] : f->body;
        if (cse_seq(m, *f, seq)) changed = true;
    }
    if (changed) pass_region_info(m);
}

}  // namespace epq::tir
