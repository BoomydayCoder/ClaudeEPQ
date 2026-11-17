#include "tirbuild.hpp"

#include <cstdio>
#include <cstdlib>

namespace epq {

using namespace tir;
using TNode = tir::Node;
using infer::Ty;
using infer::TyK;

namespace {

[[noreturn]] void internal(const char* what, uint32_t line) {
    std::fprintf(stderr, "epq: internal error in TIR builder: %s (line %u)\n", what, line);
    std::abort();
}

struct Builder {
    const Program& P;
    infer::Result& R;
    const BuildOptions& O;
    Module& M;

    std::vector<int32_t> region_of_root;
    std::vector<int32_t> func_of_inst;
    std::vector<uint32_t> work;
    // Scratch stack for kid lists under construction. Every builder function
    // leaves it at the size it found it; across recursion only indices into
    // it are kept (pushes may reallocate).
    std::vector<TNode*> stk;
    Func* F = nullptr;
    uint32_t inst = 0;
    uint32_t line = 0;

    Builder(const Program& p, infer::Result& r, const BuildOptions& o, Module& m) : P(p), R(r), O(o), M(m) {}

    // ---- types -----------------------------------------------------------------
    uint32_t region(uint32_t e) {
        uint32_t root = R.find(e);
        if (region_of_root[root] >= 0) return (uint32_t)region_of_root[root];
        uint32_t id = (uint32_t)M.regions.size();
        M.regions.emplace_back();
        region_of_root[root] = (int32_t)id;
        Type el = conv(R.evars[root].t);
        M.regions[id].elem = el;
        M.regions[id].flows_to_any = el.k == TK::Any;
        return id;
    }
    Type conv(Ty t) {
        switch (t.k) {
        case TyK::Bot: return Type::bot();
        case TyK::Int: return Type::i32();
        case TyK::Any: return Type::any();
        case TyK::Fn: return Type::fn(t.a);
        case TyK::List: return Type::list(region(t.a));
        }
        return Type::bot();
    }
    Type elem(Type l) { return M.regions[l.a].elem; }

    uint32_t func_for(uint32_t i) {
        i = R.insts[i]->rep;
        if (func_of_inst[i] >= 0) return (uint32_t)func_of_inst[i];
        const infer::Instance& J = *R.insts[i];
        const Function& fn = P.funcs[J.def];
        auto f = std::make_unique<Func>();
        f->id = (uint32_t)M.funcs.size();
        f->def = J.def;
        f->inst = i;
        f->generic = J.generic;
        f->is_main = J.def == kMainDef;
        f->params = fn.params;
        f->locals.resize(fn.locals.size());
        for (size_t s = 0; s < fn.locals.size(); ++s) {
            Local& l = f->locals[s];
            l.type = conv(J.locals[s]);
            l.name = fn.locals[s].name;
            l.kind = (fn.locals[s].flags & LF_PARAM) ? LocalKind::Param
                     : (fn.locals[s].flags & LF_HIDDEN) ? LocalKind::Hidden : LocalKind::User;
        }
        f->ret = f->is_main ? Type::bot() : conv(J.ret);
        std::string nm = f->is_main ? "main" : M.defs[J.def].name != kNoSym ? std::string(P.syms->name(M.defs[J.def].name)) : "anon";
        f->name = "f" + std::to_string(f->id) + "_" + nm + (J.generic ? "_generic" : "");
        func_of_inst[i] = (int32_t)f->id;
        M.funcs.push_back(std::move(f));
        work.push_back((uint32_t)M.funcs.size() - 1);
        return (uint32_t)M.funcs.size() - 1;
    }

    // ---- node helpers ------------------------------------------------------------
    TNode* mk(Op op, Type t, std::initializer_list<TNode*> k) {
        TNode* n = M.node(op, t, k);
        n->line = line;
        return n;
    }
    TNode* mkv(Op op, Type t, TNode* const* k, size_t nk) {
        TNode* n = M.node(op, t, (uint32_t)nk);
        for (size_t i = 0; i < nk; ++i) n->kids[i] = k[i];
        n->line = line;
        return n;
    }
    // Node whose kids are stk[base..]; pops them.
    TNode* mkv_top(Op op, Type t, size_t base) {
        TNode* n = mkv(op, t, stk.data() + base, stk.size() - base);
        stk.resize(base);
        return n;
    }
    TNode* cint(int32_t v) {
        TNode* n = mk(Op::ConstInt, Type::i32(), {});
        n->a = v;
        return n;
    }
    TNode* local(uint32_t id) {
        TNode* n = mk(Op::Local, F->locals[id].type, {});
        n->a = (int32_t)id;
        return n;
    }
    TNode* set_local(uint32_t id, TNode* v) {
        TNode* n = mk(Op::SetLocal, F->locals[id].type, {coerce(v, F->locals[id].type)});
        n->a = (int32_t)id;
        return n;
    }
    TNode* coerce(TNode* n, Type t) {
        if (n->type == t) return n;
        if (n->type.k == TK::Bot) {
            if (t.k == TK::Bot) return n;
            return mk(Op::Coerce, t, {n});
        }
        if (t.k == TK::Any) return mk(Op::Box, Type::any(), {n});
        internal("impossible coercion", n->line);
    }
    TNode* fail(Err e, TNode* const* k, size_t nk) {
        TNode* n = mkv(Op::Fail, Type::bot(), k, nk);
        n->a = (int32_t)e;
        return n;
    }
    TNode* fail(Err e, std::initializer_list<TNode*> k) { return fail(e, k.begin(), k.size()); }
    // If some kid has type Bot, the operator is never reached: evaluate the kids
    // up to that one. Returns nullptr when no kid is Bot.
    TNode* bot_short(TNode* const* k, size_t nk) {
        for (size_t i = 0; i < nk; ++i) {
            if (k[i]->type.k == TK::Bot) {
                if (i == 0) return k[0];
                return mkv(Op::Comma, Type::bot(), k, i + 1);
            }
        }
        return nullptr;
    }
    // Replace every non-constant kid by a temporary assigned in order; pushes
    // the prefix of SetLocal nodes onto stk.
    void spill(TNode** k, size_t nk) {
        for (size_t i = 0; i < nk; ++i) {
            TNode*& x = k[i];
            if (x->op == Op::ConstInt || x->op == Op::ConstFn) continue;
            uint32_t t = M.new_temp(F, x->type);
            stk.push_back(set_local(t, x));
            x = local(t);
        }
    }
    // Comma(stk[base..], n), or n itself when nothing was pushed.
    TNode* with_prefix(size_t base, TNode* n) {
        if (stk.size() == base) return n;
        stk.push_back(n);
        return mkv_top(Op::Comma, n->type, base);
    }
    static bool is_list(Type t) { return t.k == TK::List; }
    static bool is_nonint_static(Type t) { return t.k == TK::List || t.k == TK::Fn; }

    // ---- variables -------------------------------------------------------------
    TNode* read_var(const epq::Node* id) {
        if (id->flags & NF_GLOBAL) {
            Global& g = M.globals[(size_t)id->aux];
            TNode* n = mk(Op::GetGlobal, g.type, {});
            n->a = id->aux;
            n->flags |= F_CHECKED;
            g.ever_checked = true;
            return n;
        }
        return local((uint32_t)id->aux);
    }
    Type var_type(const epq::Node* id) {
        if (id->flags & NF_GLOBAL) return M.globals[(size_t)id->aux].type;
        return F->locals[(size_t)id->aux].type;
    }
    TNode* write_var(const epq::Node* id, TNode* v) {
        if (id->flags & NF_GLOBAL) {
            Type t = M.globals[(size_t)id->aux].type;
            TNode* n = mk(Op::SetGlobal, t, {coerce(v, t)});
            n->a = id->aux;
            return n;
        }
        return set_local((uint32_t)id->aux, v);
    }

    // ---- expressions -------------------------------------------------------------
    TNode* expr(const epq::Node* n, bool discard = false) {
        uint32_t saved = line;
        line = n->line;
        TNode* r = expr_(n, discard);
        line = saved;
        return r;
    }

    TNode* expr_(const epq::Node* n, bool discard) {
        switch (n->kind) {
        case NK::Int: return cint(n->ival);
        case NK::Input: return mk(Op::Input, Type::i32(), {});
        case NK::Id: return read_var(n);
        case NK::FuncLit: {
            TNode* c = mk(Op::ConstFn, Type::fn((uint32_t)n->aux), {});
            c->a = n->aux;
            return c;
        }
        case NK::Assign: return assign(n, discard);
        case NK::Append: {
            TNode* k[2] = {expr(n->kids[0]), expr(n->kids[1])};
            if (TNode* b = bot_short(k, 2)) return b;
            Type l = k[0]->type;
            if (l.k == TK::List) return mk(Op::Append, l, {k[0], coerce(k[1], elem(l))});
            if (l.k == TK::Any) return mk(Op::DynAppend, l, {k[0], coerce(k[1], Type::any())});
            return fail(Err::OperandList, k, 2);
        }
        case NK::Add: case NK::Sub: case NK::Mul: case NK::Div: return arith(n);
        case NK::Eq: case NK::Ne: return equality(n);
        case NK::Lt: case NK::Gt: case NK::Le: case NK::Ge: return compare(n);
        case NK::Not: {
            TNode* x = expr(n->kids[0]);
            switch (x->type.k) {
            case TK::Bot: return x;
            case TK::Int: return mk(Op::NotI32, Type::i32(), {x});
            case TK::Any: return mk(Op::DynNot, Type::i32(), {x});
            default: return fail(Err::OperandInt, {x});
            }
        }
        case NK::Abs: {
            TNode* x = expr(n->kids[0]);
            switch (x->type.k) {
            case TK::Bot: return x;
            case TK::Int: return mk(Op::AbsI32, Type::i32(), {x});
            case TK::List: return mk(Op::Len, Type::i32(), {x});
            case TK::Any: return mk(Op::DynAbs, Type::i32(), {x});
            default: return fail(Err::IntOrList, {x});
            }
        }
        case NK::Neg: {
            TNode* x = expr(n->kids[0]);
            switch (x->type.k) {
            case TK::Bot: return x;
            case TK::Int: return mk(Op::NegI32, Type::i32(), {x});
            case TK::List: return mk(Op::Pop, elem(x->type), {x});
            case TK::Any: return mk(Op::DynNeg, Type::any(), {x});
            default: return fail(Err::IntOrList, {x});
            }
        }
        case NK::And:
        case NK::Or: {
            Type T = conv(R.type_of(inst, n));
            TNode* a = expr(n->kids[0]);
            TNode* b = expr(n->kids[1]);
            if (a->type.k == TK::Bot) return a;
            if (T.k == TK::Bot) internal("and/or type", n->line);
            return mk(n->kind == NK::And ? Op::And : Op::Or, T, {coerce(a, T), coerce(b, T)});
        }
        case NK::Index: {
            TNode* k[2] = {expr(n->kids[0]), expr(n->kids[1])};
            if (TNode* b = bot_short(k, 2)) return b;
            Type l = k[0]->type, i = k[1]->type;
            if (is_nonint_static(i) || l.k == TK::Int || l.k == TK::Fn) return fail(Err::InvalidOp, k, 2);
            if (l.k == TK::List) {
                TNode* idx = i.k == TK::Int ? k[1] : unbox(k[1], Err::InvalidOp);
                return mk(Op::Index, elem(l), {k[0], idx});
            }
            return mk(Op::DynIndex, Type::any(), {k[0], k[1]});  // l Any, i Int|Any
        }
        case NK::List: {
            uint32_t e = R.aux_of(inst, n);
            Type t = Type::list(region(e));
            Type el = elem(t);
            size_t base = stk.size();
            for (uint32_t i = 0; i < n->nkids; ++i) {
                TNode* x = expr(n->kids[i]);
                stk.push_back(x);
            }
            if (TNode* b = bot_short(stk.data() + base, n->nkids)) {
                stk.resize(base);
                return b;
            }
            for (size_t i = base; i < stk.size(); ++i) stk[i] = coerce(stk[i], el);
            return mkv_top(Op::ListLit, t, base);
        }
        case NK::Call: return call(n);
        default:
            internal("unexpected AST node", n->line);
        }
    }

    TNode* unbox(TNode* x, Err e) {
        TNode* u = mk(Op::UnboxInt, Type::i32(), {x});
        u->a = (int32_t)e;
        return u;
    }

    TNode* assign(const epq::Node* n, bool discard) {
        const epq::Node* tg = n->kids[0];
        if (tg->kind == NK::Id) {
            TNode* v = expr(n->kids[1]);
            if (v->type.k == TK::Bot) return v;
            Type vt = v->type, xt = var_type(tg);
            if (discard || vt == xt) return write_var(tg, v);
            // The expression's value keeps v's (narrower) type.
            uint32_t t = M.new_temp(F, vt);
            return mk(Op::Comma, vt, {set_local(t, v), write_var(tg, local(t)), local(t)});
        }
        // l[i] := v
        TNode* k[3] = {expr(tg->kids[0]), expr(tg->kids[1]), expr(n->kids[1])};
        if (TNode* b = bot_short(k, 3)) return b;
        Type l = k[0]->type, i = k[1]->type, vt = k[2]->type;
        if (is_nonint_static(i) || l.k == TK::Int || l.k == TK::Fn) return fail(Err::InvalidOp, k, 3);
        Type et = l.k == TK::List ? elem(l) : Type::any();
        size_t base = stk.size();
        bool keep_v = !discard && vt != et;
        if (keep_v || (l.k == TK::List && i.k == TK::Any && !is_pure(k[2]))) spill(k, 3);
        TNode* st;
        if (l.k == TK::List) {
            TNode* idx = i.k == TK::Int ? k[1] : unbox(k[1], Err::InvalidOp);
            st = mk(Op::SetIndex, et, {k[0], idx, coerce(k[2], et)});
        } else {
            st = mk(Op::DynSetIndex, Type::any(), {k[0], k[1], coerce(k[2], Type::any())});
        }
        if (keep_v) {
            stk.push_back(st);
            stk.push_back(k[2]);  // the temp holding v
            return mkv_top(Op::Comma, vt, base);
        }
        return with_prefix(base, st);
    }

    TNode* arith(const epq::Node* n) {
        TNode* k[2] = {expr(n->kids[0]), expr(n->kids[1])};
        if (TNode* b = bot_short(k, 2)) return b;
        Op op = n->kind == NK::Add ? Op::AddI32 : n->kind == NK::Sub ? Op::SubI32 : n->kind == NK::Mul ? Op::MulI32 : Op::DivI32;
        Type a = k[0]->type, b = k[1]->type;
        if (a.k == TK::Int && b.k == TK::Int) return mk(op, Type::i32(), {k[0], k[1]});
        if (op != Op::DivI32 && (is_nonint_static(a) || is_nonint_static(b))) return fail(Err::OperandsInt, k, 2);
        if (op == Op::DivI32 && is_nonint_static(b)) return fail(Err::OperandsInt, k, 2);
        TNode* d = mk(Op::DynArith, Type::i32(), {k[0], k[1]});
        d->a = (int32_t)op;
        return d;
    }

    TNode* equality(const epq::Node* n) {
        TNode* k[2] = {expr(n->kids[0]), expr(n->kids[1])};
        if (TNode* b = bot_short(k, 2)) return b;
        bool ne = n->kind == NK::Ne;
        Type a = k[0]->type, b = k[1]->type;
        if (a.k == TK::Int && b.k == TK::Int) return mk(ne ? Op::NeI32 : Op::EqI32, Type::i32(), {k[0], k[1]});
        if (a.k != TK::Any && b.k != TK::Any && !(a.k == TK::List && b.k == TK::List)) {
            bool eq = a.k == b.k && (a.k != TK::Fn || a.a == b.a);
            return mk(Op::Comma, Type::i32(), {k[0], k[1], cint(eq != ne ? 1 : 0)});
        }
        TNode* r = mk(ne ? Op::Ne : Op::Eq, Type::i32(), {k[0], k[1]});
        r->a = (int32_t)M.desc_of(a);
        r->b = (int32_t)M.desc_of(b);
        return r;
    }

    TNode* compare(const epq::Node* n) {
        TNode* k[2] = {expr(n->kids[0]), expr(n->kids[1])};
        if (TNode* b = bot_short(k, 2)) return b;
        Op op = n->kind == NK::Lt ? Op::LtI32 : n->kind == NK::Gt ? Op::GtI32 : n->kind == NK::Le ? Op::LeI32 : Op::GeI32;
        Type a = k[0]->type, b = k[1]->type;
        if (a.k == TK::Int && b.k == TK::Int) return mk(op, Type::i32(), {k[0], k[1]});
        if (a.k == TK::Fn || b.k == TK::Fn || (a.k == TK::Int && b.k == TK::List) || (a.k == TK::List && b.k == TK::Int))
            return fail(Err::InvalidCmp, k, 2);
        TNode* r = mk(Op::Cmp, Type::i32(), {k[0], k[1]});
        r->a = (int32_t)M.desc_of(a);
        r->b = (int32_t)M.desc_of(b);
        r->c = (int32_t)op;
        return r;
    }

    TNode* call(const epq::Node* n) {
        const epq::Node* args = n->kids[1];
        size_t base = stk.size();
        for (uint32_t i = 0; i < args->nkids; ++i) {
            TNode* a = expr(args->kids[i]);
            stk.push_back(a);
        }
        TNode* callee_e = expr(n->kids[0]);
        stk.push_back(callee_e);
        // No pushes below: k stays valid.
        TNode** k = stk.data() + base;
        size_t nk = stk.size() - base;
        TNode* r;
        Type f = callee_e->type;
        uint32_t na = args->nkids;
        if ((r = bot_short(k, nk))) {
        } else if (f.k == TK::Fn) {
            const Function& fn = P.funcs[f.a];
            if (fn.params.size() != na) {
                r = fail(Err::Arity, k, nk);
            } else {
                uint32_t ci = R.aux_of(inst, n);
                if (ci == infer::kNone || R.insts[ci]->def != f.a) internal("call without instance", n->line);
                uint32_t fid = func_for(ci);
                Func* callee = M.funcs[fid].get();
                for (uint32_t i = 0; i < na; ++i) k[i] = coerce(k[i], callee->locals[callee->params[i]].type);
                r = mkv(Op::CallDirect, callee->ret, k, nk);
                r->a = (int32_t)fid;
            }
        } else if (f.k == TK::Any) {
            for (uint32_t i = 0; i < na; ++i) k[i] = coerce(k[i], Type::any());
            r = mkv(Op::CallDyn, Type::any(), k, nk);
        } else {
            r = fail(Err::OperandFn, k, nk);
        }
        stk.resize(base);
        return r;
    }

    // ---- conditions & statements ---------------------------------------------------
    TNode* cond(const epq::Node* n) {
        if (n->kind == NK::And || n->kind == NK::Or) {
            uint32_t saved = line;
            line = n->line;
            TNode* a = cond(n->kids[0]);
            TNode* b = cond(n->kids[1]);
            TNode* r;
            if (a->type.k == TK::Bot) r = a;
            else {
                r = mk(n->kind == NK::And ? Op::And : Op::Or, Type::i32(), {a, b});
                r->flags |= F_COND;
            }
            line = saved;
            return r;
        }
        return expr(n);
    }

    TNode* seq(std::initializer_list<TNode*> k) { return mkv(Op::Seq, Type::bot(), k.begin(), k.size()); }
    TNode* seq_top(size_t base) { return mkv_top(Op::Seq, Type::bot(), base); }

    TNode* stmt(const epq::Node* n) {
        uint32_t saved = line;
        line = n->line;
        TNode* r = stmt_(n);
        line = saved;
        return r;
    }

    TNode* stmt_(const epq::Node* n) {
        switch (n->kind) {
        case NK::Seq: {
            size_t base = stk.size();
            for (uint32_t i = 0; i < n->nkids; ++i) {
                TNode* s = stmt(n->kids[i]);
                stk.push_back(s);
            }
            return seq_top(base);
        }
        case NK::ExprStmt: return expr(n->kids[0], true);
        case NK::Print: {
            TNode* v = expr(n->kids[0]);
            if (v->type.k == TK::Bot) return v;
            TNode* p = mk(Op::Print, Type::bot(), {v});
            p->a = (int32_t)M.desc_of(v->type);
            return p;
        }
        case NK::Block: return stmt(n->kids[0]);
        case NK::If: return mk(Op::If, Type::bot(), {cond(n->kids[0]), stmt(n->kids[1]), stmt(n->kids[2])});
        case NK::While: return mk(Op::While, Type::bot(), {cond(n->kids[0]), stmt(n->kids[1]), seq({})});
        case NK::For: {
            TNode* init = stmt(n->kids[0]);
            TNode* c = cond(n->kids[1]);
            TNode* body = stmt(n->kids[3]);
            TNode* step = stmt(n->kids[2]);
            TNode* w = mk(Op::While, Type::bot(), {c, body, step});
            w->flags |= F_FOR;
            return seq({init, w});
        }
        case NK::ForEach: return foreach(n);
        case NK::Return: {
            TNode* v = expr(n->kids[0]);
            if (!F->is_main) v = coerce(v, F->ret);
            return mk(Op::Return, Type::bot(), {v});
        }
        default:
            return expr(n, true);
        }
    }

    TNode* foreach(const epq::Node* n) {
        const ForEachInfo& fe = P.foreach[(size_t)n->aux];
        const epq::Node* x = n->kids[0];
        uint32_t idx = fe.index_slot, L = fe.list_slot;
        size_t base = stk.size();
        stk.push_back(set_local(idx, cint(0)));
        TNode* lv = expr(n->kids[1]);
        if (lv->type.k == TK::Bot) {
            stk.push_back(lv);
            return seq_top(base);
        }
        stk.push_back(set_local(L, lv));
        if (fe.var_preexisted) stk.push_back(write_var(x, cint(0)));
        Type lt = F->locals[L].type;
        TNode* c;
        TNode* get;
        switch (lt.k) {
        case TK::List:
            c = mk(Op::LtI32, Type::i32(), {local(idx), mk(Op::Len, Type::i32(), {local(L)})});
            get = mk(Op::Index, elem(lt), {local(L), local(idx)});
            break;
        case TK::Any:
            c = mk(Op::LtI32, Type::i32(), {local(idx), mk(Op::DynAbs, Type::i32(), {local(L)})});
            get = mk(Op::DynIndex, Type::any(), {local(L), local(idx)});
            break;
        case TK::Int:
            c = mk(Op::LtI32, Type::i32(), {local(idx), mk(Op::AbsI32, Type::i32(), {local(L)})});
            get = fail(Err::InvalidOp, {local(L), local(idx)});
            break;
        default:  // Fn
            c = fail(Err::IntOrList, {local(L)});
            get = fail(Err::InvalidOp, {});
            break;
        }
        TNode* body = seq({write_var(x, get), stmt(n->kids[2])});
        TNode* step = set_local(idx, mk(Op::AddI32, Type::i32(), {local(idx), cint(1)}));
        TNode* w = mk(Op::While, Type::bot(), {c, body, step});
        w->flags |= F_FOREACH;
        w->a = (int32_t)idx;
        w->b = (int32_t)L;
        stk.push_back(w);
        return seq_top(base);
    }

    void build_func(uint32_t fid) {
        F = M.funcs[fid].get();
        inst = F->inst;
        const Function& fn = P.funcs[F->def];
        line = fn.node ? fn.node->line : 1;
        TNode* body = stmt(fn.body);
        if (body->op != Op::Seq) body = seq({body});
        if (!F->is_main && infer::completes(fn.body)) {
            size_t base = stk.size();
            stk.insert(stk.end(), body->kids, body->kids + body->nkids);
            TNode* r = mk(Op::Return, Type::bot(), {coerce(cint(0), F->ret)});
            stk.push_back(r);
            body = seq_top(base);
        }
        F->body = body;
    }

    void name_defs(const epq::Node* n) {
        if (n->kind == NK::Assign && n->kids[0]->kind == NK::Id && n->kids[1]->kind == NK::FuncLit) {
            DefInfo& d = M.defs[(size_t)n->kids[1]->aux];
            if (d.name == kNoSym) d.name = n->kids[0]->sym;
        }
        for (uint32_t i = 0; i < n->nkids; ++i) name_defs(n->kids[i]);
    }

    void run() {
        M.prog = &P;
        stk.reserve(256);
        M.funcs.reserve(R.insts.size());
        region_of_root.assign(R.evars.size(), -1);
        func_of_inst.assign(R.insts.size(), -1);
        M.defs.resize(P.funcs.size());
        for (size_t d = 0; d < P.funcs.size(); ++d) M.defs[d].arity = (uint32_t)P.funcs[d].params.size();
        name_defs(P.root);
        M.globals.resize(P.globals.size());
        for (size_t g = 0; g < P.globals.size(); ++g) {
            M.globals[g].name = P.globals[g];
            M.globals[g].type = conv(R.globals[g]);
        }
        M.desc_of(Type::any());  // creates the fixed descriptors
        func_for(0);
        for (size_t d = 0; d < P.funcs.size(); ++d)
            if (R.generic_of[d] >= 0) M.defs[d].generic_func = (int32_t)func_for((uint32_t)R.generic_of[d]);
        while (!work.empty()) {
            uint32_t f = work.back();
            work.pop_back();
            build_func(f);
        }
        // Allocation sites per region.
        for (size_t e = 0; e < R.evars.size(); ++e) {
            if (region_of_root[e] < 0) continue;
            for (auto& s : R.evars[e].sites) {
                int32_t fi = func_of_inst[R.insts[s.inst]->rep];
                if (fi < 0 || R.insts[s.inst]->rep != s.inst) continue;
                M.regions[(size_t)region_of_root[e]].sites.push_back(AllocSiteInfo{(uint32_t)fi, s.len, s.node->line});
            }
        }
    }
};

}  // namespace

void build_tir(const Program& prog, infer::Result& inf, const BuildOptions& opt, Module& out) {
    Builder b(prog, inf, opt, out);
    b.run();
}

}  // namespace epq
