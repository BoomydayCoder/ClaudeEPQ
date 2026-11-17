// Pass "inline" — inlining of small non-recursive instances as InlineBlocks (DESIGN.md §5.1).
//
// A CallDirect of a suitable callee c
//     CallDirect c [a1..an, fexpr]
// becomes (expression form)
//     Comma[ SetLocal(p1', a1) .. SetLocal(pn', an),   (copied parameters only)
//            fexpr,                                    (only if it is impure)
//            InlineBlock L { Seq[ resets..., body' ] },
//            Local(res) ]
// or, when the call's value is discarded (statement position), the same as a
// Seq without `res`. body' is a clone of c's body in which every callee local
// is a fresh caller local, and every `Return v` is `Seq[SetLocal(res, v),
// Break L]` (`Seq[v, Break L]` when v is Bot or the value is unused).
//
// Semantics preserved:
//  * evaluation order: arguments in kid order, then the callee expression
//    (which may fail, e.g. a checked global read), then the body.
//  * parameter substitution: a parameter that c never assigns is replaced by
//    its argument when that is a constant (ConstInt / ConstFn) or a caller
//    Local that no later argument and not the callee expression assigns. EPQ has
//    no closures, so the inlined body cannot touch caller locals; reading the
//    local later therefore yields the value it had when the argument was
//    evaluated.
//  * fresh frames: the C backend zero-initialises every local once per call
//    (Int 0, Any boxed 0, Pair bits 0). A callee local that may be read before
//    it is written on some path (only possible outside the well-defined subset)
//    is re-initialised at the start of the block, so an inlined call inside a
//    loop sees the same state as a real call on every entry.
//  * bodies never fall off the end (tirbuild appends the implicit Return; the C
//    backend relies on it with __builtin_unreachable), so `res` is always set
//    when the block completes normally.
//
// Candidates: not main, not generic, not in a call-graph cycle (Tarjan SCCs
// over CallDirect edges), non-Bot return type, and small (<= kSmall nodes) or
// called from exactly one site and of moderate size. Functions are processed
// bottom-up (callees before callers), so leaves are inlined into their callers
// before those are considered. Growth of each caller is bounded. Afterwards
// instances no longer reachable from main or a generic entry are removed (ids
// renumbered) and region info is recomputed.
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <functional>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

using namespace util;

constexpr size_t kSmall = 40;        // always-inline size (TIR nodes)
constexpr size_t kSingleSite = 400;  // size limit for a callee with one call site
constexpr size_t kCallerMax = 6000;  // stop inlining into a caller beyond this size

size_t tree_size(const Node* n) {
    size_t s = 1;
    for (uint32_t i = 0; i < n->nkids; ++i) s += tree_size(n->kids[i]);
    return s;
}

void call_targets(const Node* n, std::vector<uint32_t>& out) {
    walk(n, [&](Node* x, Node*, uint32_t) {
        if (x->op == Op::CallDirect) out.push_back((uint32_t)x->a);
        return true;
    });
}

// Locals of f that may be read before being written on some path through the
// body (parameters count as written). Conservative.
struct ReadBeforeWrite {
    std::vector<char> need;
    std::vector<std::pair<int32_t, std::vector<char>>> blocks;  // open InlineBlocks: label, state at Breaks
    static void meet(std::vector<char>& into, const std::vector<char>& x) {
        for (size_t i = 0; i < into.size(); ++i) into[i] = into[i] && x[i];
    }
    void run(const Node* n, std::vector<char>& da) {
        switch (n->op) {
        case Op::Local:
            if ((size_t)n->a < da.size() && !da[(size_t)n->a]) need[(size_t)n->a] = 1;
            return;
        case Op::SetLocal:
            run(n->kids[0], da);
            da[(size_t)n->a] = 1;
            return;
        case Op::If: {
            run(n->kids[0], da);
            std::vector<char> b = da;
            run(n->kids[1], da);
            run(n->kids[2], b);
            meet(da, b);
            return;
        }
        case Op::While: {
            if (n->flags & F_FOREACH) {
                da[(size_t)n->a] = 1;
                da[(size_t)n->b] = 1;
            }
            run(n->kids[0], da);
            std::vector<char> c = da;
            run(n->kids[1], c);
            run(n->kids[2], c);
            return;
        }
        case Op::And:
        case Op::Or: {
            run(n->kids[0], da);
            std::vector<char> c = da;
            run(n->kids[1], c);
            return;
        }
        case Op::InlineBlock: {
            blocks.push_back({n->a, std::vector<char>(da.size(), 1)});
            run(n->kids[0], da);
            meet(da, blocks.back().second);
            blocks.pop_back();
            return;
        }
        case Op::Break:
            for (size_t i = blocks.size(); i-- > 0;)
                if (blocks[i].first == n->a) {
                    meet(blocks[i].second, da);
                    break;
                }
            std::fill(da.begin(), da.end(), 1);  // unreachable from here
            return;
        case Op::Return:
        case Op::Fail:
            for (uint32_t i = 0; i < n->nkids; ++i) run(n->kids[i], da);
            std::fill(da.begin(), da.end(), 1);
            return;
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) run(n->kids[i], da);
            return;
        }
    }
    explicit ReadBeforeWrite(const Func& f) {
        need.assign(f.locals.size(), 0);
        std::vector<char> da(f.locals.size(), 0);
        for (uint32_t p : f.params) da[p] = 1;
        run(f.body, da);
    }
};

struct Inliner {
    Module& m;
    std::vector<char> recursive;
    std::vector<uint32_t> sites;     // number of CallDirect sites per callee
    std::vector<size_t> size;        // current body size per func
    bool verbose = std::getenv("EPQ_INLINE_VERBOSE") != nullptr;
    size_t inlined = 0;

    explicit Inliner(Module& mm) : m(mm) {}

    // Tarjan SCCs; returns funcs in an order where callees come before callers.
    std::vector<uint32_t> bottom_up(const std::vector<std::vector<uint32_t>>& g) {
        size_t n = g.size();
        std::vector<int> idx(n, -1), low(n, 0);
        std::vector<char> on(n, 0);
        std::vector<uint32_t> st, order;
        int counter = 0;
        recursive.assign(n, 0);
        std::function<void(uint32_t)> dfs = [&](uint32_t v) {
            idx[v] = low[v] = counter++;
            st.push_back(v);
            on[v] = 1;
            for (uint32_t w : g[v]) {
                if (w == v) recursive[v] = 1;
                if (idx[w] < 0) {
                    dfs(w);
                    low[v] = std::min(low[v], low[w]);
                } else if (on[w]) {
                    low[v] = std::min(low[v], idx[w]);
                }
            }
            if (low[v] == idx[v]) {
                std::vector<uint32_t> comp;
                for (;;) {
                    uint32_t w = st.back();
                    st.pop_back();
                    on[w] = 0;
                    comp.push_back(w);
                    if (w == v) break;
                }
                if (comp.size() > 1)
                    for (uint32_t w : comp) recursive[w] = 1;
                for (uint32_t w : comp) order.push_back(w);
            }
        };
        for (uint32_t v = 0; v < n; ++v)
            if (idx[v] < 0) dfs(v);
        return order;
    }

    bool eligible(const Func& caller, const Node* call) {
        const Func& c = *m.funcs[(size_t)call->a];
        if (&c == &caller || c.is_main || c.generic || recursive[c.id]) return false;
        if (c.ret.k == TK::Bot || call->type != c.ret) return false;
        if (call->nkids != c.params.size() + 1) return false;
        for (size_t i = 0; i < c.params.size(); ++i)
            if (call->kids[i]->type != c.locals[c.params[i]].type) return false;
        size_t sz = size[c.id];
        if (sz > kSmall && !(sites[c.id] == 1 && sz <= kSingleSite)) return false;
        if (size[caller.id] + sz > kCallerMax) return false;
        return true;
    }

    Node* reset_value(const Local& l, uint32_t line) {
        Node* v = nullptr;
        switch (l.type.k) {
        case TK::Int: v = mk_int(m, 0, line); break;
        case TK::Any: v = m.node(Op::Box, Type::any(), {mk_int(m, 0, line)}); break;
        case TK::Pair:
            // bit pattern 0 = (0, INT_MIN): what the C backend's `EPair l = 0` holds
            v = m.node(Op::PairMake, l.type, {mk_int(m, 0, line), mk_int(m, INT_MIN, line)});
            break;
        default: return nullptr;  // List / Fn / Ptr: no expressible zero (and never read uninitialised)
        }
        v->line = line;
        return v;
    }

    // Replace *slot (a CallDirect) by its inlined form.
    void inline_call(Func& F, Node*& slot, bool stmt) {
        Node* call = slot;
        const Func& c = *m.funcs[(size_t)call->a];
        uint32_t line = call->line;
        uint32_t na = call->nkids - 1;
        Node* fexpr = call->kids[na];

        std::vector<bool> c_assigned;
        locals_assigned(c.body, c_assigned);
        c_assigned.resize(c.locals.size(), false);

        std::vector<int32_t> map(c.locals.size(), -1);
        std::vector<Node*> subst_const(c.locals.size(), nullptr);  // param -> constant
        std::vector<Node*> kids;

        for (uint32_t i = 0; i < na; ++i) {
            uint32_t p = c.params[i];
            Node* arg = call->kids[i];
            bool sub = false;
            if (!c_assigned[p]) {
                if (arg->op == Op::ConstInt || arg->op == Op::ConstFn) {
                    subst_const[p] = arg;
                    sub = true;
                } else if (arg->op == Op::Local) {
                    bool later = assigns_local(fexpr, (uint32_t)arg->a);
                    for (uint32_t j = i + 1; j < na && !later; ++j) later = assigns_local(call->kids[j], (uint32_t)arg->a);
                    if (!later) {
                        map[p] = arg->a;
                        sub = true;
                    }
                }
            }
            if (sub) continue;
            Local l = c.locals[p];
            l.kind = LocalKind::Temp;
            l.unique = false;
            l.global = -1;
            F.locals.push_back(l);
            map[p] = (int32_t)F.locals.size() - 1;
            kids.push_back(mk_set(m, F, (uint32_t)map[p], arg));
        }
        if (!is_pure(fexpr)) kids.push_back(fexpr);

        std::vector<bool> is_param(c.locals.size(), false);
        for (uint32_t p : c.params) is_param[p] = true;
        for (size_t i = 0; i < c.locals.size(); ++i) {
            if (is_param[i]) continue;
            Local l = c.locals[i];
            if (l.kind == LocalKind::Param || l.kind == LocalKind::Demoted) l.kind = LocalKind::Temp;
            l.unique = false;
            l.global = -1;
            F.locals.push_back(l);
            map[i] = (int32_t)F.locals.size() - 1;
        }
        uint32_t label = new_label(F);
        int32_t res = -1;
        if (!stmt) res = (int32_t)m.new_temp(&F, c.ret);

        // Constant-substituted parameters (never assigned, so only Local reads
        // exist) are first renamed to a sentinel caller local, whose reads are
        // then replaced by the constant (the slot itself stays unused).
        std::vector<int32_t> sentinel(c.locals.size(), -1);
        for (uint32_t p : c.params)
            if (subst_const[p]) {
                Local l = c.locals[p];
                l.kind = LocalKind::Temp;
                l.unique = false;
                l.global = -1;
                F.locals.push_back(l);
                sentinel[p] = (int32_t)F.locals.size() - 1;
                map[p] = sentinel[p];
            }
        Node* body = clone(m, c.body, map, &F);

        std::vector<Node*> by_sentinel(F.locals.size(), nullptr);
        bool any_sentinel = false;
        for (uint32_t p : c.params)
            if (sentinel[p] >= 0) {
                by_sentinel[(size_t)sentinel[p]] = subst_const[p];
                any_sentinel = true;
            }

        rewrite(body, [&](Node*& s) {
            Node* n = s;
            if (any_sentinel && n->op == Op::Local && (size_t)n->a < by_sentinel.size() && by_sentinel[(size_t)n->a]) {
                Node* k = clone(m, by_sentinel[(size_t)n->a]);
                k->line = n->line;
                s = k;
                return;
            }
            if (n->op == Op::Return) {
                Node* v = n->kids[0];
                Node* br = m.node(Op::Break, Type::bot(), 0);
                br->a = (int32_t)label;
                br->line = n->line;
                Node* first;
                if (res >= 0 && v->type.k != TK::Bot) first = mk_set(m, F, (uint32_t)res, v);
                else first = v;
                Node* sq = mk_seq(m, {first, br});
                sq->line = n->line;
                s = sq;
            }
        });

        // Re-initialise locals that may be read before written (fresh frame).
        ReadBeforeWrite rbw(c);
        std::vector<Node*> blk;
        for (size_t i = 0; i < c.locals.size(); ++i) {
            if (!rbw.need[i] || is_param[i]) continue;
            Node* v = reset_value(c.locals[i], line);
            if (v) blk.push_back(mk_set(m, F, (uint32_t)map[i], v));
        }
        blk.push_back(body);
        Node* ib = m.node(Op::InlineBlock, Type::bot(), {mk_seq(m, blk)});
        ib->a = (int32_t)label;
        ib->line = line;
        kids.push_back(ib);

        Node* out;
        if (stmt) {
            out = mk_seq(m, kids);
        } else {
            kids.push_back(mk_local(m, F, (uint32_t)res, line));
            out = m.node(Op::Comma, c.ret, (uint32_t)kids.size());
            for (size_t i = 0; i < kids.size(); ++i) out->kids[i] = kids[i];
        }
        out->line = line;
        slot = out;
        size[F.id] += size[c.id] + kids.size();
        ++inlined;
        if (verbose) std::fprintf(stderr, "epq: inline: %s into %s (line %u)\n", c.name.c_str(), F.name.c_str(), line);
    }

    // Statement context: kids of Seq, If then/else, While body/step, InlineBlock body.
    void visit(Func& F, Node*& slot, bool stmt) {
        Node* n = slot;
        for (uint32_t i = 0; i < n->nkids; ++i) {
            bool ks = n->op == Op::Seq || n->op == Op::InlineBlock || ((n->op == Op::If || n->op == Op::While) && i > 0);
            visit(F, n->kids[i], ks);
        }
        if (n->op == Op::CallDirect && eligible(F, n)) inline_call(F, slot, stmt);
    }

    void remove_dead() {
        size_t n = m.funcs.size();
        std::vector<char> live(n, 0);
        std::vector<uint32_t> work;
        auto mark = [&](uint32_t f) {
            if (!live[f]) {
                live[f] = 1;
                work.push_back(f);
            }
        };
        mark(0);
        for (auto& f : m.funcs)
            if (f->is_main || f->generic) mark(f->id);
        for (auto& d : m.defs)
            if (d.generic_func >= 0) mark((uint32_t)d.generic_func);
        while (!work.empty()) {
            uint32_t f = work.back();
            work.pop_back();
            std::vector<uint32_t> t;
            call_targets(m.funcs[f]->body, t);
            for (uint32_t c : t) mark(c);
        }
        std::vector<int32_t> nid(n, -1);
        uint32_t k = 0;
        for (size_t i = 0; i < n; ++i)
            if (live[i]) nid[i] = (int32_t)k++;
        if (k == n) return;
        std::vector<std::unique_ptr<Func>> nf;
        for (size_t i = 0; i < n; ++i)
            if (live[i]) {
                nf.push_back(std::move(m.funcs[i]));
                nf.back()->id = (uint32_t)nid[i];
            }
        m.funcs = std::move(nf);
        for (auto& f : m.funcs)
            walk(f->body, [&](Node* x, Node*, uint32_t) {
                if (x->op == Op::CallDirect) x->a = nid[(size_t)x->a];
                return true;
            });
        for (auto& d : m.defs)
            if (d.generic_func >= 0) d.generic_func = nid[(size_t)d.generic_func];
        // Allocation sites of removed instances now live in their callers;
        // the func field is informational only (dumps).
        for (auto& r : m.regions)
            for (auto& s : r.sites) s.func = nid[s.func] >= 0 ? (uint32_t)nid[s.func] : 0;
    }

    void run() {
        size_t n = m.funcs.size();
        std::vector<std::vector<uint32_t>> g(n);
        sites.assign(n, 0);
        size.assign(n, 0);
        for (auto& f : m.funcs) {
            call_targets(f->body, g[f->id]);
            for (uint32_t c : g[f->id]) ++sites[c];
            size[f->id] = tree_size(f->body);
        }
        std::vector<uint32_t> order = bottom_up(g);
        for (uint32_t fid : order) {
            Func& F = *m.funcs[fid];
            visit(F, F.body, true);
        }
        if (!inlined) return;
        remove_dead();
        pass_region_info(m);
    }
};

}  // namespace

void pass_inline(Module& m) {
    Inliner in(m);
    in.run();
}

}  // namespace epq::tir
