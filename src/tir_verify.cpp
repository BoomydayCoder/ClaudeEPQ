// tir::verify — structural and type checker for TIR (src/tir.hpp).
//
// Checks every node of every function against the op table documented in
// tir.hpp: kid count, kid types, result type, ids (locals, globals, funcs,
// regions, descriptors), SetLocal/SetGlobal value types, Return type == the
// function's return type, CallDirect arity / argument types == callee params,
// Select kids pure, Break inside an InlineBlock with the same label, Pair
// types only of pair regions, Ptr types only in Cache locals or as values of
// ListData / ElemLoad bases. "Int" positions also accept Bot (a Bot value never
// exists). On the first failure prints function, op, path and expectation to
// stderr and aborts.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tir.hpp"

namespace epq::tir {

namespace {

struct Verifier {
    Module& M;
    const char* after;
    const Func* F = nullptr;
    std::vector<int32_t> labels;          // enclosing InlineBlock labels
    std::vector<const Node*> path;

    [[noreturn]] void fail(const Node* n, const std::string& what) {
        std::string p;
        for (const Node* q : path) {
            if (!p.empty()) p += " > ";
            p += op_name(q->op);
        }
        std::fprintf(stderr, "epq: TIR verifier (after %s): in %s, %s (line %u): %s\n  path: %s\n", after,
                     F ? F->name.c_str() : "?", op_name(n->op), n->line, what.c_str(), p.c_str());
        if (std::getenv("EPQ_VERIFY_DUMP")) {  // EPQ_VERIFY_DUMP=1: also dump the module
            std::string d;
            dump(d, M);
            std::fputs(d.c_str(), stderr);
        }
        std::abort();
    }
    std::string ts(Type t) { return type_str(M, t); }

    void type_ok(const Node* n, Type t, const char* what) {
        switch (t.k) {
        case TK::List:
        case TK::Ptr:
            if (t.a >= M.regions.size()) fail(n, std::string(what) + ": bad region id " + std::to_string(t.a));
            break;
        case TK::Pair:
            if (t.a >= M.regions.size()) fail(n, std::string(what) + ": bad region id " + std::to_string(t.a));
            if (!M.regions[t.a].pair) fail(n, std::string(what) + ": Pair of region " + std::to_string(t.a) + " which is not a pair region");
            break;
        case TK::Fn:
            if (t.a >= M.defs.size()) fail(n, std::string(what) + ": bad DefId " + std::to_string(t.a));
            break;
        default:
            break;
        }
    }
    void nk(const Node* n, uint32_t k) {
        if (n->nkids != k) fail(n, "expected " + std::to_string(k) + " kids, has " + std::to_string(n->nkids));
    }
    void nk_min(const Node* n, uint32_t k) {
        if (n->nkids < k) fail(n, "expected at least " + std::to_string(k) + " kids, has " + std::to_string(n->nkids));
    }
    void kid_is(const Node* n, uint32_t i, Type t) {
        Type k = n->kids[i]->type;
        if (t.k == TK::Int && k.is_int_like()) return;
        if (k != t) fail(n, "kid " + std::to_string(i) + " has type " + ts(k) + ", expected " + ts(t));
    }
    void kid_int(const Node* n, uint32_t i) { kid_is(n, i, Type::i32()); }
    void kid_kind(const Node* n, uint32_t i, TK k, const char* name) {
        if (n->kids[i]->type.k != k) fail(n, "kid " + std::to_string(i) + " has type " + ts(n->kids[i]->type) + ", expected " + name);
    }
    void kid_valued(const Node* n, uint32_t i) {  // "any static type" position
        TK k = n->kids[i]->type.k;
        if (k == TK::Ptr) fail(n, "kid " + std::to_string(i) + " is a Ptr where a value of any static type is expected");
    }
    void res(const Node* n, Type t) {
        if (n->type != t) fail(n, "result type " + ts(n->type) + ", expected " + ts(t));
    }
    void res_int(const Node* n) {
        if (n->type.k != TK::Int) fail(n, "result type " + ts(n->type) + ", expected Int");
    }
    void local_ok(const Node* n, int32_t id) {
        if (id < 0 || (size_t)id >= F->locals.size()) fail(n, "bad local id " + std::to_string(id));
    }
    void desc_ok(const Node* n, int32_t d, Type t, const char* which) {
        if (d < 0 || (size_t)d >= M.descs.size()) fail(n, std::string(which) + ": bad descriptor " + std::to_string(d));
        uint32_t want;
        switch (t.k) {
        case TK::Bot: case TK::Int: want = 0; break;
        case TK::Any: want = 1; break;
        case TK::Fn: want = 2; break;
        case TK::List: want = 3; break;
        case TK::Pair: want = 4; break;
        default: fail(n, std::string(which) + ": operand of type " + ts(t) + " has no descriptor");
        }
        if (M.descs[(size_t)d].kind != want)
            fail(n, std::string(which) + ": descriptor " + std::to_string(d) + " has kind " + std::to_string(M.descs[(size_t)d].kind) +
                        " but operand has type " + ts(t));
    }
    bool is_cmp_op(int32_t c) {
        Op o = (Op)c;
        return o == Op::LtI32 || o == Op::LeI32 || o == Op::GtI32 || o == Op::GeI32;
    }

    void node(const Node* n) {
        path.push_back(n);
        type_ok(n, n->type, "node type");
        for (uint32_t i = 0; i < n->nkids; ++i) {
            if (!n->kids[i]) fail(n, "kid " + std::to_string(i) + " is null");
        }
        // Kids first, except InlineBlock which must register its label.
        if (n->op == Op::InlineBlock) {
            nk(n, 1);
            labels.push_back(n->a);
            node(n->kids[0]);
            labels.pop_back();
        } else {
            for (uint32_t i = 0; i < n->nkids; ++i) node(n->kids[i]);
        }
        check(n);
        path.pop_back();
    }

    void check(const Node* n) {
        switch (n->op) {
        case Op::Seq: break;
        case Op::If: nk(n, 3); kid_valued(n, 0); break;
        case Op::While:
            nk(n, 3);
            kid_valued(n, 0);
            if (n->flags & F_FOREACH) {
                local_ok(n, n->a);
                local_ok(n, n->b);
            }
            break;
        case Op::Return: {
            nk(n, 1);
            if (!F->is_main) {
                Type k = n->kids[0]->type;
                if (k != F->ret && !(k.k == TK::Bot))
                    fail(n, "returns " + ts(k) + ", function returns " + ts(F->ret));
            }
            break;
        }
        case Op::Print: nk(n, 1); kid_valued(n, 0); desc_ok(n, n->a, n->kids[0]->type, "print"); break;
        case Op::ConstInt: nk(n, 0); res_int(n); break;
        case Op::ConstFn: nk(n, 0); res(n, Type::fn((uint32_t)n->a)); break;
        case Op::Local:
            nk(n, 0);
            local_ok(n, n->a);
            res(n, F->locals[(size_t)n->a].type);
            break;
        case Op::GetGlobal:
            nk(n, 0);
            if (n->a < 0 || (size_t)n->a >= M.globals.size()) fail(n, "bad global id " + std::to_string(n->a));
            res(n, M.globals[(size_t)n->a].type);
            break;
        case Op::Input: nk(n, 0); res_int(n); break;
        case Op::SetLocal: {
            nk(n, 1);
            local_ok(n, n->a);
            Type t = F->locals[(size_t)n->a].type;
            res(n, t);
            if (n->kids[0]->type != t) fail(n, "value type " + ts(n->kids[0]->type) + " != local type " + ts(t));
            break;
        }
        case Op::SetGlobal: {
            nk(n, 1);
            if (n->a < 0 || (size_t)n->a >= M.globals.size()) fail(n, "bad global id " + std::to_string(n->a));
            Type t = M.globals[(size_t)n->a].type;
            res(n, t);
            if (n->kids[0]->type != t) fail(n, "value type " + ts(n->kids[0]->type) + " != global type " + ts(t));
            break;
        }
        case Op::Box: {
            nk(n, 1);
            TK k = n->kids[0]->type.k;
            if (k != TK::Int && k != TK::List && k != TK::Fn) fail(n, "Box of " + ts(n->kids[0]->type) + " (expected Int|List|Fn)");
            if (k == TK::List && M.regions[n->kids[0]->type.a].elem.k != TK::Any)
                fail(n, "Box of a list whose region element type is not Any");
            res(n, Type::any());
            break;
        }
        case Op::UnboxInt: nk(n, 1); kid_kind(n, 0, TK::Any, "Any"); res_int(n); break;
        case Op::Coerce: nk(n, 1); kid_kind(n, 0, TK::Bot, "Bot"); break;
        case Op::Comma: nk_min(n, 1); res(n, n->kids[n->nkids - 1]->type); break;
        case Op::Fail: res(n, Type::bot()); break;
        case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::DivI32:
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
            nk(n, 2); kid_int(n, 0); kid_int(n, 1); res_int(n); break;
        case Op::NegI32: case Op::AbsI32: case Op::NotI32: nk(n, 1); kid_int(n, 0); res_int(n); break;
        case Op::DynArith: {
            nk(n, 2);
            kid_valued(n, 0);
            kid_valued(n, 1);
            Op o = (Op)n->a;
            if (o != Op::AddI32 && o != Op::SubI32 && o != Op::MulI32 && o != Op::DivI32) fail(n, "bad arithmetic op in a");
            res_int(n);
            break;
        }
        case Op::DynNeg: nk(n, 1); kid_kind(n, 0, TK::Any, "Any"); res(n, Type::any()); break;
        case Op::DynAbs: case Op::DynNot: nk(n, 1); kid_kind(n, 0, TK::Any, "Any"); res_int(n); break;
        case Op::Eq: case Op::Ne: case Op::Cmp:
            nk(n, 2);
            kid_valued(n, 0);
            kid_valued(n, 1);
            desc_ok(n, n->a, n->kids[0]->type, "desc a");
            desc_ok(n, n->b, n->kids[1]->type, "desc b");
            if (n->op == Op::Cmp && !is_cmp_op(n->c)) fail(n, "bad comparison op in c");
            res_int(n);
            break;
        case Op::ListLit: {
            if (n->type.k != TK::List) fail(n, "ListLit of non-list type " + ts(n->type));
            Type e = elem_of(M, n->type);
            for (uint32_t i = 0; i < n->nkids; ++i) kid_is(n, i, e);
            break;
        }
        case Op::Len: nk(n, 1); kid_kind(n, 0, TK::List, "List"); res_int(n); break;
        case Op::Index:
            nk(n, 2); kid_kind(n, 0, TK::List, "List"); kid_int(n, 1); res(n, elem_of(M, n->kids[0]->type)); break;
        case Op::SetIndex:
            nk(n, 3); kid_kind(n, 0, TK::List, "List"); kid_int(n, 1);
            kid_is(n, 2, elem_of(M, n->kids[0]->type)); res(n, elem_of(M, n->kids[0]->type)); break;
        case Op::Append:
            nk(n, 2); kid_kind(n, 0, TK::List, "List"); kid_is(n, 1, elem_of(M, n->kids[0]->type));
            res(n, n->kids[0]->type); break;
        case Op::Pop: nk(n, 1); kid_kind(n, 0, TK::List, "List"); res(n, elem_of(M, n->kids[0]->type)); break;
        case Op::DynIndex: {
            nk(n, 2); kid_kind(n, 0, TK::Any, "Any");
            TK k = n->kids[1]->type.k;
            if (k != TK::Int && k != TK::Any && k != TK::Bot) fail(n, "index kid must be Int|Any");
            res(n, Type::any());
            break;
        }
        case Op::DynSetIndex: {
            nk(n, 3); kid_kind(n, 0, TK::Any, "Any");
            TK k = n->kids[1]->type.k;
            if (k != TK::Int && k != TK::Any && k != TK::Bot) fail(n, "index kid must be Int|Any");
            kid_kind(n, 2, TK::Any, "Any");
            res(n, Type::any());
            break;
        }
        case Op::DynAppend: nk(n, 2); kid_kind(n, 0, TK::Any, "Any"); kid_kind(n, 1, TK::Any, "Any"); res(n, Type::any()); break;
        case Op::And: case Op::Or:
            nk(n, 2);
            if (!(n->flags & F_COND)) {
                kid_is(n, 0, n->type);
                kid_is(n, 1, n->type);
            }
            break;
        case Op::CallDirect: {
            if (n->a < 0 || (size_t)n->a >= M.funcs.size()) fail(n, "bad callee func id " + std::to_string(n->a));
            const Func& c = *M.funcs[(size_t)n->a];
            if (n->nkids != c.params.size() + 1)
                fail(n, "call of " + c.name + " with " + std::to_string(n->nkids - 1) + " args, expected " + std::to_string(c.params.size()));
            for (size_t i = 0; i < c.params.size(); ++i)
                if (n->kids[i]->type != c.locals[c.params[i]].type)
                    fail(n, "arg " + std::to_string(i) + " of " + c.name + " has type " + ts(n->kids[i]->type) + ", param type " +
                                ts(c.locals[c.params[i]].type));
            kid_kind(n, n->nkids - 1, TK::Fn, "Fn");
            res(n, c.ret);
            break;
        }
        case Op::CallDyn:
            nk_min(n, 1);
            for (uint32_t i = 0; i < n->nkids; ++i) kid_kind(n, i, TK::Any, "Any");
            res(n, Type::any());
            break;
        // ---- extension ----
        case Op::Select:
            nk(n, 3);
            kid_int(n, 0);
            kid_is(n, 1, n->type);
            kid_is(n, 2, n->type);
            for (uint32_t i = 0; i < 3; ++i)
                if (!is_pure(n->kids[i])) fail(n, "kid " + std::to_string(i) + " (" + op_name(n->kids[i]->op) + ") is not pure");
            break;
        case Op::InlineBlock: break;  // kid count checked in node()
        case Op::Break: {
            nk(n, 0);
            bool found = false;
            for (int32_t l : labels) found |= l == n->a;
            if (!found) fail(n, "Break label" + std::to_string(n->a) + " not inside an InlineBlock with that label");
            break;
        }
        case Op::PairMake: nk(n, 2); kid_int(n, 0); kid_int(n, 1); if (n->type.k != TK::Pair) fail(n, "result must be Pair"); break;
        case Op::PairFirst: case Op::PairSecond: nk(n, 1); kid_kind(n, 0, TK::Pair, "Pair"); res_int(n); break;
        case Op::PairIndex: nk(n, 2); kid_kind(n, 0, TK::Pair, "Pair"); kid_int(n, 1); res_int(n); break;
        case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
            nk(n, 2); kid_kind(n, 0, TK::Pair, "Pair"); kid_kind(n, 1, TK::Pair, "Pair"); res_int(n); break;
        case Op::ListData:
            nk(n, 1); kid_kind(n, 0, TK::List, "List"); res(n, Type::ptr(n->kids[0]->type.a)); break;
        case Op::ElemLoad:
            nk(n, 2); kid_kind(n, 0, TK::Ptr, "Ptr"); kid_int(n, 1); res(n, elem_of(M, n->kids[0]->type)); break;
        case Op::ElemStore:
            nk(n, 3); kid_kind(n, 0, TK::Ptr, "Ptr"); kid_int(n, 1);
            kid_is(n, 2, elem_of(M, n->kids[0]->type)); res(n, elem_of(M, n->kids[0]->type)); break;
        case Op::CheckIndex: nk(n, 2); kid_int(n, 0); kid_int(n, 1); res_int(n); break;
        case Op::Free: nk(n, 1); kid_kind(n, 0, TK::List, "List"); break;
        case Op::SameList: nk(n, 2); kid_kind(n, 0, TK::List, "List"); kid_kind(n, 1, TK::List, "List"); res_int(n); break;
        case Op::Nop: nk(n, 0); break;
        // ---- appendopt ----
        case Op::Reserve: nk(n, 2); kid_kind(n, 0, TK::List, "List"); kid_int(n, 1); break;
        case Op::AppendRange:
            nk(n, 4); kid_kind(n, 0, TK::List, "List"); kid_kind(n, 1, TK::List, "List"); kid_int(n, 2); kid_int(n, 3); res_int(n);
            if (elem_of(M, n->kids[0]->type) != elem_of(M, n->kids[1]->type) ||
                M.elem_is_i32(n->kids[0]->type.a) != M.elem_is_i32(n->kids[1]->type.a))
                fail(n, "lists have different element types");
            break;
        // ---- end appendopt ----
        }
        // Ptr values only come from ListData / Local reads of Cache locals / assignments to them.
        if (n->type.k == TK::Ptr && n->op != Op::ListData && n->op != Op::Local && n->op != Op::SetLocal && n->op != Op::Comma &&
            n->op != Op::Select)
            fail(n, "produces a Ptr (only ListData, cache locals, SetLocal, Comma, Select may)");
    }

    void func(const Func& f) {
        F = &f;
        for (size_t i = 0; i < f.locals.size(); ++i) {
            Type t = f.locals[i].type;
            if (t.k == TK::Ptr && f.locals[i].kind != LocalKind::Cache)
                std::fprintf(stderr, "epq: TIR verifier (after %s): in %s: local %zu has type Ptr but kind is not Cache\n", after,
                             f.name.c_str(), i),
                    std::abort();
            if ((t.k == TK::List || t.k == TK::Ptr || t.k == TK::Pair) && t.a >= M.regions.size())
                std::fprintf(stderr, "epq: TIR verifier (after %s): in %s: local %zu has a bad region\n", after, f.name.c_str(), i),
                    std::abort();
        }
        for (uint32_t p : f.params)
            if (p >= f.locals.size())
                std::fprintf(stderr, "epq: TIR verifier (after %s): in %s: bad param local %u\n", after, f.name.c_str(), p), std::abort();
        if (!f.body) std::fprintf(stderr, "epq: TIR verifier (after %s): %s has no body\n", after, f.name.c_str()), std::abort();
        labels.clear();
        path.clear();
        node(f.body);
    }
};

}  // namespace

void verify(Module& m, const char* after_pass) {
    Verifier V{m, after_pass ? after_pass : "?"};
    for (size_t r = 0; r < m.regions.size(); ++r) {
        const Region& R = m.regions[r];
        if (R.elem.k == TK::Ptr) std::fprintf(stderr, "epq: TIR verifier (after %s): region %zu has Ptr elements\n", V.after, r), std::abort();
        if (R.pair && !R.elem.is_int_like())
            std::fprintf(stderr, "epq: TIR verifier (after %s): pair region %zu has element type %s\n", V.after, r, type_str(m, R.elem).c_str()),
                std::abort();
    }
    for (size_t i = 0; i < m.funcs.size(); ++i) {
        if (m.funcs[i]->id != i)
            std::fprintf(stderr, "epq: TIR verifier (after %s): funcs[%zu] has id %u\n", V.after, i, m.funcs[i]->id), std::abort();
        V.func(*m.funcs[i]);
    }
}

}  // namespace epq::tir
