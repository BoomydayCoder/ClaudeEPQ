#include "resolver.hpp"

#include <string>

namespace epq {

const char* scope_kind_name(ScopeKind k) {
    switch (k) {
    case ScopeKind::Function: return "function";
    case ScopeKind::Block: return "block";
    case ScopeKind::ListLit: return "list-literal";
    case ScopeKind::IfStmt: return "if";
    case ScopeKind::IfThen: return "if-then";
    case ScopeKind::IfElse: return "if-else";
    case ScopeKind::WhileStmt: return "while";
    case ScopeKind::WhileBody: return "while-body";
    case ScopeKind::ForStmt: return "for";
    case ScopeKind::ForBody: return "for-body";
    case ScopeKind::ForEachStmt: return "foreach";
    case ScopeKind::ForEachBody: return "foreach-body";
    }
    return "?";
}

namespace {

struct Abort {};

class Resolver {
public:
    Resolver(const Interner& syms, Program& p, ResolveError& err) : syms_(syms), p_(p), err_(err) {
        p_.syms = &syms;
        p_.global_of.assign(syms.size(), -1);
        bind_.assign(syms.size(), -1);
    }

    void run(Node* root) {
        p_.root = root;
        discover(root);
        // main
        p_.funcs.emplace_back();
        Function& m = p_.funcs.back();
        m.id = kMainDef;
        m.parent = -1;
        m.node = nullptr;
        m.body = root;
        m.scopes.push_back(Scope{-1, ScopeKind::Function, nullptr});
        fn_ = kMainDef;
        scope_ = 0;
        stmt(root, true);
    }

private:
    const Interner& syms_;
    Program& p_;
    ResolveError& err_;

    // Current function state.
    DefId fn_ = 0;
    int32_t scope_ = 0;
    std::vector<int32_t> bind_;        // Sym -> local slot in the current function, -1
    std::vector<Sym> active_;          // names bound in open scopes (all functions)
    std::vector<size_t> scope_marks_;  // active_.size() at each open scope of fn_
    size_t fn_active_begin_ = 0;

    [[noreturn]] void fail(const std::string& msg, const Node* at) {
        err_.msg = msg;
        err_.line = at ? at->line : 0;
        err_.col = at ? at->col : 0;
        throw Abort{};
    }

    // ---- global discovery: compiler.cc resolve_globals ----------------------
    void discover(Node* n) {
        switch (n->kind) {
        case NK::Block:
        case NK::For:
        case NK::FuncLit:
            return;
        case NK::If:
            discover(n->kids[0]);
            return;
        case NK::Assign:
            if (n->kids[0]->kind == NK::Id) {
                Sym s = n->kids[0]->sym;
                if (p_.global_of[s] < 0) {
                    p_.global_of[s] = (int32_t)p_.globals.size();
                    p_.globals.push_back(s);
                }
            }
            break;
        default:
            break;
        }
        for (uint32_t i = 0; i < n->nkids; ++i) discover(n->kids[i]);
    }

    // ---- scopes --------------------------------------------------------------
    Function& fn() { return p_.funcs[fn_]; }

    void begin(ScopeKind k, const Node* opener) {
        Function& f = fn();
        f.scopes.push_back(Scope{scope_, k, opener});
        scope_ = (int32_t)f.scopes.size() - 1;
        scope_marks_.push_back(active_.size());
    }
    void end() {
        size_t mark = scope_marks_.back();
        scope_marks_.pop_back();
        for (size_t i = mark; i < active_.size(); ++i) bind_[active_[i]] = -1;
        active_.resize(mark);
        scope_ = fn().scopes[(size_t)scope_].parent;
    }

    uint32_t new_local(Sym name, uint8_t flags, const Node* site) {
        Function& f = fn();
        uint32_t slot = (uint32_t)f.locals.size();
        f.locals.push_back(LocalVar{name, (uint32_t)scope_, flags, site});
        if (name != kNoSym) {
            bind_[name] = (int32_t)slot;
            active_.push_back(name);
        }
        return slot;
    }

    // compiler.cc set_var: global -> global; existing local -> it; else define.
    // Returns true if the name already existed (global or local).
    bool set_var(Node* id, bool clean, uint8_t extra_flags = 0) {
        Sym s = id->sym;
        if (int32_t g = p_.global_of[s]; g >= 0) {
            id->flags |= NF_GLOBAL;
            id->aux = g;
            return true;
        }
        if (int32_t l = bind_[s]; l >= 0) {
            id->flags |= NF_LOCAL;
            id->aux = l;
            return true;
        }
        uint8_t lf = extra_flags | (clean ? 0 : LF_UNCLEAN);
        uint32_t slot = new_local(s, lf, id);
        id->flags |= NF_LOCAL | NF_DEFINE | (clean ? 0 : NF_UNCLEAN);
        id->aux = (int32_t)slot;
        return false;
    }

    // compiler.cc get_var.
    void get_var(Node* id) {
        Sym s = id->sym;
        if (int32_t g = p_.global_of[s]; g >= 0) {
            id->flags |= NF_GLOBAL;
            id->aux = g;
            return;
        }
        if (int32_t l = bind_[s]; l >= 0) {
            id->flags |= NF_LOCAL;
            id->aux = l;
            return;
        }
        fail("Undefined variable " + std::string(syms_.name(s)), id);
    }

    // ---- the compile walk (compiler.cc compile), same visiting order --------
    // `clean`: an expression statement directly here is a clean definition point.
    void stmt(Node* n, bool clean) {
        switch (n->kind) {
        case NK::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) stmt(n->kids[i], true);
            return;
        case NK::ExprStmt:
            top_expr(n->kids[0], clean);
            return;
        case NK::Print:
        case NK::Return:
            expr(n->kids[0]);
            return;
        case NK::Block:
            begin(ScopeKind::Block, n);
            stmt(n->kids[0], true);
            end();
            return;
        case NK::If:
            begin(ScopeKind::IfStmt, n);
            top_expr(n->kids[0], true);  // the if condition runs once per scope entry
            begin(ScopeKind::IfThen, n);
            stmt(n->kids[1], true);
            end();
            begin(ScopeKind::IfElse, n);
            stmt(n->kids[2], true);
            end();
            end();
            return;
        case NK::While:
            begin(ScopeKind::WhileStmt, n);
            expr(n->kids[0]);
            begin(ScopeKind::WhileBody, n);
            stmt(n->kids[1], true);
            end();
            end();
            return;
        case NK::For:
            // Original order: init, condition, body, step.
            begin(ScopeKind::ForStmt, n);
            stmt(n->kids[0], true);
            expr(n->kids[1]);
            begin(ScopeKind::ForBody, n);
            stmt(n->kids[3], true);
            end();
            stmt(n->kids[2], false);
            end();
            return;
        case NK::ForEach: {
            // Original: hidden i := null; L := list; x := null (set_var rule);
            // loop { body scope: x := L[i]; body }.
            begin(ScopeKind::ForEachStmt, n);
            uint32_t islot = new_local(kNoSym, LF_HIDDEN, n);
            expr(n->kids[1]);
            uint32_t lslot = new_local(kNoSym, LF_HIDDEN, n);
            bool pre = set_var(n->kids[0], true, LF_FOREACH_VAR);
            n->aux = (int32_t)p_.foreach.size();
            p_.foreach.push_back(ForEachInfo{fn_, islot, lslot, pre});
            begin(ScopeKind::ForEachBody, n);
            stmt(n->kids[2], true);
            end();
            end();
            return;
        }
        default:
            // Not reachable: the parser only puts statements here.
            expr(n);
            return;
        }
    }

    void top_expr(Node* e, bool clean) {
        if (e->kind == NK::Assign) assign(e, clean);
        else expr(e);
    }

    void assign(Node* n, bool clean) {
        Node* t = n->kids[0];
        switch (t->kind) {
        case NK::Id:
            expr(n->kids[1]);
            set_var(t, clean);
            return;
        case NK::Index:
            expr(t->kids[0]);
            expr(t->kids[1]);
            expr(n->kids[1]);
            return;
        default:  // Get / SGet
            fail("Classes are not implemented", n);
        }
    }

    void expr(Node* n) {
        switch (n->kind) {
        case NK::Assign:
            assign(n, false);
            return;
        case NK::Id:
            get_var(n);
            return;
        case NK::Int:
        case NK::Input:
            return;
        case NK::List:  // list literal: its own scope
            begin(ScopeKind::ListLit, n);
            for (uint32_t i = 0; i < n->nkids; ++i) expr(n->kids[i]);
            end();
            return;
        case NK::Call: {
            Node* args = n->kids[1];
            for (uint32_t i = 0; i < args->nkids; ++i) expr(args->kids[i]);
            expr(n->kids[0]);
            return;
        }
        case NK::FuncLit:
            funclit(n);
            return;
        case NK::Class:
        case NK::Get:
        case NK::SGet:
            fail("Classes are not implemented", n);
        default:
            // Binary / unary operators and Index: children left to right.
            for (uint32_t i = 0; i < n->nkids; ++i) expr(n->kids[i]);
            return;
        }
    }

    void funclit(Node* n) {
        // Enter a fresh function context; the enclosing function's locals are
        // invisible inside (no closures), so unbind them and restore after.
        DefId saved_fn = fn_;
        int32_t saved_scope = scope_;
        size_t saved_begin = fn_active_begin_;
        std::vector<size_t> saved_marks;
        saved_marks.swap(scope_marks_);
        std::vector<std::pair<Sym, int32_t>> saved_bind;
        saved_bind.reserve(active_.size() - fn_active_begin_);
        for (size_t i = fn_active_begin_; i < active_.size(); ++i) {
            saved_bind.emplace_back(active_[i], bind_[active_[i]]);
            bind_[active_[i]] = -1;
        }
        fn_active_begin_ = active_.size();

        DefId id = (DefId)p_.funcs.size();
        n->aux = (int32_t)id;
        p_.funcs.emplace_back();
        {
            Function& f = p_.funcs.back();
            f.id = id;
            f.parent = (int32_t)saved_fn;
            f.node = n;
            f.body = n->kids[1];
            f.scopes.push_back(Scope{-1, ScopeKind::Function, n});
        }
        fn_ = id;
        scope_ = 0;

        Node* params = n->kids[0];
        for (uint32_t i = 0; i < params->nkids; ++i) {
            Node* prm = params->kids[i];
            if (prm->kind != NK::Id) fail("Invalid argument list", n);
            if (p_.global_of[prm->sym] >= 0) fail("Argument shadows global variable", prm);
            if (bind_[prm->sym] >= 0) fail("Duplicate argument name", prm);
            uint32_t slot = new_local(prm->sym, LF_PARAM, prm);
            prm->flags |= NF_LOCAL | NF_DEFINE;
            prm->aux = (int32_t)slot;
            p_.funcs[id].params.push_back(slot);
        }
        stmt(n->kids[1], true);

        // Leave: drop this function's bindings (incl. its never-closed outer
        // scope), restore the enclosing function's.
        for (size_t i = fn_active_begin_; i < active_.size(); ++i) bind_[active_[i]] = -1;
        active_.resize(fn_active_begin_);
        for (auto& [s, slot] : saved_bind) bind_[s] = slot;
        fn_active_begin_ = saved_begin;
        scope_marks_.swap(saved_marks);
        fn_ = saved_fn;
        scope_ = saved_scope;
    }
};

}  // namespace

bool resolve(Node* root, const Interner& syms, Program& out, ResolveError& err) {
    Resolver r(syms, out, err);
    try {
        r.run(root);
        return true;
    } catch (const Abort&) {
        return false;
    }
}

// ---- --dump-resolved --------------------------------------------------------

namespace {

struct ResolvedDumper {
    std::ostream& os;
    const Program& p;

    std::string name(Sym s) const { return s == kNoSym ? std::string("<hidden>") : std::string(p.syms->name(s)); }

    void bindings(const Node* n, DefId fn) {
        if (n->kind == NK::FuncLit) fn = (DefId)n->aux;
        if (n->kind == NK::Get) {  // member name is not a variable
            bindings(n->kids[0], fn);
            return;
        }
        if (n->kind == NK::SGet) return;
        if (n->kind == NK::Id) {
            os << "  " << n->line << ':' << n->col << ' ' << p.syms->name(n->sym) << " -> ";
            if (n->flags & NF_GLOBAL) os << "global " << n->aux;
            else if (n->flags & NF_LOCAL) os << "local def" << fn << " slot " << n->aux;
            else os << "unresolved";
            if (n->flags & NF_DEFINE) os << " [define]";
            if (n->flags & NF_UNCLEAN) os << " [unclean]";
            os << '\n';
        }
        for (uint32_t i = 0; i < n->nkids; ++i) bindings(n->kids[i], fn);
    }

    void run() {
        os << "globals (" << p.globals.size() << "):\n";
        for (size_t g = 0; g < p.globals.size(); ++g) os << "  g" << g << ' ' << name(p.globals[g]) << '\n';
        for (const Function& f : p.funcs) {
            os << "def " << f.id;
            if (f.id == kMainDef) os << " <main>";
            else os << " at " << f.node->line << ':' << f.node->col << " parent def" << f.parent;
            os << " params(";
            for (size_t i = 0; i < f.params.size(); ++i) os << (i ? " " : "") << f.params[i];
            os << ")\n";
            for (size_t s = 0; s < f.scopes.size(); ++s) {
                const Scope& sc = f.scopes[s];
                os << "  scope " << s << ' ' << scope_kind_name(sc.kind) << " parent " << sc.parent;
                if (sc.opener) os << " at " << sc.opener->line << ':' << sc.opener->col;
                os << '\n';
            }
            for (size_t l = 0; l < f.locals.size(); ++l) {
                const LocalVar& v = f.locals[l];
                os << "  local " << l << ' ' << name(v.name) << " scope " << v.scope;
                if (v.flags & LF_PARAM) os << " [param]";
                if (v.flags & LF_HIDDEN) os << " [hidden]";
                if (v.flags & LF_FOREACH_VAR) os << " [foreach-var]";
                if (v.flags & LF_UNCLEAN) os << " [unclean]";
                if (v.site) os << " at " << v.site->line << ':' << v.site->col;
                os << '\n';
            }
        }
        os << "foreach loops (" << p.foreach.size() << "):\n";
        for (size_t i = 0; i < p.foreach.size(); ++i) {
            const ForEachInfo& fe = p.foreach[i];
            os << "  #" << i << " def" << fe.fn << " index slot " << fe.index_slot << " list slot " << fe.list_slot
               << (fe.var_preexisted ? " var pre-existed" : " var new") << '\n';
        }
        os << "bindings:\n";
        bindings(p.root, kMainDef);
    }
};

}  // namespace

void dump_resolved(std::ostream& os, const Program& p) {
    ResolvedDumper d{os, p};
    d.run();
}

}  // namespace epq
