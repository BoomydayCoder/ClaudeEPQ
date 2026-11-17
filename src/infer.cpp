#include "infer.hpp"

#include <algorithm>
#include <map>
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
    std::vector<uint32_t> count_of;  // DefId -> number of non-generic instances

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
        Ty ty = R.evars[y].t;
        Ty res = lub(R.evars[x].t, ty);
        uint32_t rx = find(x);
        Ty curt = R.evars[rx].t;
        if (rx != x) res = lub(curt, res);
        if (!same(res, curt)) R.evars[rx].t = res;
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
    Instance& I() { return *R.insts[cur]; }

    Ty& var(const Node* id) {
        if (id->flags & NF_GLOBAL) return R.globals[(size_t)id->aux];
        return I().locals[(size_t)id->aux];
    }
    void store_into(Ty l, Ty v) {
        if (l.k == TyK::List) flow_elem(l.a, v);
        else if (l.k == TyK::Any) make_any(v);
    }

    Ty ex(const Node* n, bool cond = false) {
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
                store_into(l, v);
                t = v;
            }
            break;
        }
        case NK::Append: {
            Ty l = ex(n->kids[0]);
            Ty v = ex(n->kids[1]);
            store_into(l, v);
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
            else if (x.k == TyK::List) t = R.evars[find(x.a)].t;
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
            if (l.k == TyK::List && iok) t = R.evars[find(l.a)].t;
            else if (l.k == TyK::Any && iok) t = l;
            break;
        }
        case NK::List: {
            uint32_t o = R.ordinal(n);
            uint32_t e = I().naux[o];
            if (e == kNone) {
                e = (uint32_t)R.evars.size();
                R.evars.push_back(EVar{e, Ty::bot(), {}});
                R.evars[e].sites.push_back(AllocSite{n, cur, n->nkids});
                I().naux[o] = e;
                ++changes;
            }
            for (uint32_t i = 0; i < n->nkids; ++i) flow_elem(e, ex(n->kids[i]));
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
                    uint32_t J = callee_instance(R.ordinal(n), f.a);
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
        I().nty[R.ordinal(n)] = t;
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
            if (Lt.k == TyK::List) flow(var(x), R.evars[find(Lt.a)].t);
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
        DefId d = R.insts[inst]->def;
        const Function& f = P.funcs[d];
        for (size_t s = 0; s < f.locals.size(); ++s)
            if (f.locals[s].flags & LF_UNCLEAN) flow(I().locals[s], Ty::i32());
        st(f.body);
        if (d != kMainDef && completes(f.body)) flow(I().ret, Ty::i32());
    }

    void number(const Node* n, uint32_t& c) {
        R.ord[n] = c++;
        if (n->kind == NK::FuncLit) return;
        for (uint32_t i = 0; i < n->nkids; ++i) number(n->kids[i], c);
    }

    // ---- post-fixpoint ---------------------------------------------------------
    void mark_reachable() {
        std::vector<uint32_t> work;
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
            reach_calls(J, P.funcs[J.def].body, push);
        }
    }
    template <class F>
    void reach_calls(Instance& J, const Node* n, F& push) {
        if (n->kind == NK::FuncLit) return;
        if (n->kind == NK::Call) {
            uint32_t o = R.ordinal(n);
            Ty f = J.nty[R.ordinal(n->kids[0])];
            uint32_t c = J.naux[o];
            if (f.k == TyK::Fn && c != kNone && R.insts[c]->def == f.a) push(c);
        }
        for (uint32_t i = 0; i < n->nkids; ++i) reach_calls(J, n->kids[i], push);
    }

    void key_ty(std::string& k, Ty t) {
        k += (char)('0' + (int)t.k);
        if (t.k == TyK::List) k += std::to_string(find(t.a));
        else if (t.k == TyK::Fn) k += std::to_string(t.a);
        k += ',';
    }
    // Partition refinement over reachable instances.
    void dedup() {
        size_t n = R.insts.size();
        std::vector<uint32_t> cls(n, kNone);
        std::vector<std::string> base(n);
        std::map<std::string, uint32_t> ids;
        for (size_t i = 0; i < n; ++i) {
            Instance& J = *R.insts[i];
            if (!J.reachable) continue;
            std::string& k = base[i];
            k = std::to_string(J.def) + (J.generic ? "g" : "n") + ":";
            if (i == 0) k += "main";
            for (Ty t : J.locals) key_ty(k, t);
            k += '|';
            key_ty(k, J.ret);
            for (size_t o = 0; o < J.nty.size(); ++o) key_ty(k, J.nty[o]);
            k += '|';
            // List literal regions (callees are refined below).
            lits_key(J, P.funcs[J.def].body, k);
            auto it = ids.emplace(k, (uint32_t)ids.size()).first;
            cls[i] = it->second;
        }
        for (int round = 0; round < 1000; ++round) {
            std::map<std::string, uint32_t> nids;
            std::vector<uint32_t> ncls(n, kNone);
            for (size_t i = 0; i < n; ++i) {
                Instance& J = *R.insts[i];
                if (!J.reachable) continue;
                std::string k = std::to_string(cls[i]) + ":";
                calls_key(J, P.funcs[J.def].body, cls, k);
                auto it = nids.emplace(k, (uint32_t)nids.size()).first;
                ncls[i] = it->second;
            }
            bool stable = nids.size() == ids.size();
            cls.swap(ncls);
            ids.swap(nids);
            if (stable) break;
        }
        std::vector<int32_t> first(ids.size(), -1);
        for (size_t i = 0; i < n; ++i) {
            if (!R.insts[i]->reachable) continue;
            if (first[cls[i]] < 0) first[cls[i]] = (int32_t)i;
            R.insts[i]->rep = (uint32_t)first[cls[i]];
        }
    }
    void lits_key(Instance& J, const Node* n, std::string& k) {
        if (n->kind == NK::FuncLit) return;
        if (n->kind == NK::List) {
            uint32_t e = J.naux[R.ordinal(n)];
            k += (e == kNone ? std::string("-") : std::to_string(find(e))) + ",";
        }
        for (uint32_t i = 0; i < n->nkids; ++i) lits_key(J, n->kids[i], k);
    }
    void calls_key(Instance& J, const Node* n, const std::vector<uint32_t>& cls, std::string& k) {
        if (n->kind == NK::FuncLit) return;
        if (n->kind == NK::Call) {
            uint32_t c = J.naux[R.ordinal(n)];
            k += (c == kNone || !R.insts[c]->reachable) ? std::string("-") : std::to_string(cls[c]);
            k += ',';
        }
        for (uint32_t i = 0; i < n->nkids; ++i) calls_key(J, n->kids[i], cls, k);
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
        for (size_t d = 0; d < nd; ++d) {
            uint32_t c = 0;
            number(P.funcs[d].body, c);
            R.nnodes[d] = c;
        }
        new_instance(kMainDef, -1);
        for (;;) {
            changes = 0;
            for (uint32_t i = 0; i < R.insts.size(); ++i) walk(i);
            ++R.passes;
            if (changes == 0) break;
        }
        mark_reachable();
        if (C.dedup) dedup();
    }
};

}  // namespace

void infer(const Program& prog, const Config& cfg, Result& out) {
    Solver s(prog, cfg, out);
    s.run();
}

std::string ty_str(const Result& r, Ty t) {
    switch (t.k) {
    case TyK::Bot: return "Bot";
    case TyK::Int: return "Int";
    case TyK::Any: return "Any";
    case TyK::Fn: return "Fn(def" + std::to_string(t.a) + ")";
    case TyK::List: {
        uint32_t e = r.find(t.a);
        return "List(r" + std::to_string(e) + ")";
    }
    }
    return "?";
}

void dump_types(std::string& o, const Result& r) {
    const Program& P = *r.prog;
    o += "inference: " + std::to_string(r.passes) + " passes, " + std::to_string(r.insts.size()) + " instances, " +
         std::to_string(r.evars.size()) + " element variables\n";
    o += "globals:\n";
    for (size_t g = 0; g < P.globals.size(); ++g)
        o += "  " + std::string(P.syms->name(P.globals[g])) + " : " + ty_str(r, r.globals[g]) + "\n";
    o += "regions:\n";
    for (size_t e = 0; e < r.evars.size(); ++e) {
        if (r.evars[e].parent != e) continue;
        o += "  r" + std::to_string(e) + " elem " + ty_str(r, r.evars[e].t) + "  sites:";
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
