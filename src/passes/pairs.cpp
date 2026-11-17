// Pass "pairs" — pair unboxing (DESIGN.md §5.4).
//
// A region r becomes a pair region (Region::pair; every List(r) becomes Pair(r),
// one int64, see src/tir.hpp) iff, over the WHOLE module (every function
// instance, every node):
//   * r's element type is exactly Int;
//   * r has at least one ListLit and every ListLit of type List(r) has exactly
//     two kids;
//   * every use of a List(r)-typed value is in a position that does not observe
//     list identity and does not mutate it (whitelist below). In particular a
//     List(r) value is never the base of Append / Pop / SetIndex, never Boxed
//     (never flows into Any), never the operand of ListData / Free / SameList
//     (Low-TIR ops of later passes, so the pass is safe after them).
// Anything not on the whitelist disqualifies the region ("if in doubt, not a
// pair"). Identity of an immutable list is unobservable in EPQ (= is deep), so
// such lists are values and the 64-bit packed form is exact.
//
// Rewrites (kid order / failure points preserved):
//   ListLit [a, b]            -> PairMake [a, b]
//   Index [p, 0] / [p, 1]     -> PairFirst [p] / PairSecond [p]
//   Index [p, i]              -> PairIndex [p, i]          (IndexOOB unless i in {0,1})
//   Len [p]                   -> 2 (p pure) / Comma [p, 2]
//   Eq/Ne/Cmp [p, q] both Pair-> PairEq .. PairGe
//   Eq/Ne/Cmp/Print one Pair  -> descriptor updated (runtime handles PAIR vs LIST)
//   condition on a Pair       -> 1 (p pure) / Comma [p, 1]
// Types are rewritten everywhere: node types, locals (params, hidden for-each
// locals), function return types, globals, element types of regions whose
// elements are List(r) (these lists then hold packed pairs in 8-byte slots).
// The region's own LIST descriptor (if one was created) is turned into a PAIR
// descriptor in place, so descriptors of outer lists ("list of List(r)") now
// describe "list of PAIR".
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

struct PairsPass {
    Module& M;
    std::vector<uint8_t> cand;   // per region: still a candidate
    std::vector<uint32_t> nlit;  // per region: ListLit count
    bool verbose = std::getenv("EPQ_PAIRS_VERBOSE") != nullptr;

    bool is_pr(Type t) const { return t.k == TK::List && t.a < cand.size() && cand[t.a]; }
    void kill(Type t) {
        if (t.k == TK::List && t.a < cand.size()) cand[t.a] = 0;
    }

    // Is a List(r)-typed kid at position i of n allowed (no identity / mutation)?
    static bool use_ok(const Node* n, uint32_t i) {
        switch (n->op) {
        case Op::Seq: case Op::If: case Op::While:   // statements / conditions
        case Op::Return: case Op::Print:
        case Op::SetLocal: case Op::SetGlobal:
        case Op::Comma: case Op::Fail:
        case Op::Eq: case Op::Ne: case Op::Cmp:
        case Op::DynArith:                           // always OperandsInt for a list or pair
        case Op::ListLit:                            // element of an outer list
        case Op::And: case Op::Or:
        case Op::InlineBlock:
            return true;
        case Op::Len: case Op::Index: return i == 0;
        case Op::SetIndex: return i == 2;            // stored into an outer list
        case Op::Append: return i == 1;              // appended to an outer list
        case Op::ElemStore: return i == 2;           // stored into an outer list (Low TIR)
        case Op::Select: return i == 1 || i == 2;
        case Op::CallDirect: return i + 1 < n->nkids;
        default: return false;                       // Box, Pop/Append/SetIndex base, ListData, Free, SameList, ...
        }
    }

    void analyse_node(Node* n) {
        if (n->op == Op::ListLit && n->type.k == TK::List && n->type.a < cand.size()) {
            nlit[n->type.a]++;
            if (n->nkids != 2) cand[n->type.a] = 0;
        }
        for (uint32_t i = 0; i < n->nkids; ++i) {
            Node* k = n->kids[i];
            if (k->type.k == TK::List && !use_ok(n, i)) kill(k->type);
            analyse_node(k);
        }
    }

    Type rt(Type t) const { return is_pr(t) ? Type::pair(t.a) : t; }

    Node* comma(Node* a, Node* b) {
        Node* c = M.node(Op::Comma, b->type, {a, b});
        c->line = a->line;
        return c;
    }
    Node* const_or_comma(Node* p, int32_t v) {
        Node* c = util::mk_int(M, v, p->line);
        return is_pure(p) ? c : comma(p, c);
    }

    // Condition positions: kid 0 of If / While, kids of F_COND And / Or.
    void fix_cond(Node*& slot) {
        Node* c = slot;
        if (c->type.k == TK::Pair) slot = const_or_comma(c, 1);
    }

    // Post-order: kids already retyped and rewritten.
    void rewrite_node(Node*& slot) {
        Node* n = slot;
        bool was_pair_lit = n->op == Op::ListLit && is_pr(n->type);
        n->type = rt(n->type);
        switch (n->op) {
        case Op::ListLit:
            if (was_pair_lit) n->op = Op::PairMake;  // same two kids, same order
            break;
        case Op::Index:
            if (n->kids[0]->type.k == TK::Pair) {
                Node* i = n->kids[1];
                if (i->op == Op::ConstInt && (i->a == 0 || i->a == 1)) {
                    Node* f = M.node(i->a == 0 ? Op::PairFirst : Op::PairSecond, Type::i32(), {n->kids[0]});
                    f->line = n->line;
                    slot = f;
                } else {
                    n->op = Op::PairIndex;
                }
            }
            break;
        case Op::Len:
            if (n->kids[0]->type.k == TK::Pair) slot = const_or_comma(n->kids[0], 2);
            break;
        case Op::Eq: case Op::Ne: case Op::Cmp: {
            Type ta = n->kids[0]->type, tb = n->kids[1]->type;
            if (ta.k == TK::Pair && tb.k == TK::Pair) {
                Op o;
                if (n->op == Op::Eq) o = Op::PairEq;
                else if (n->op == Op::Ne) o = Op::PairNe;
                else {
                    switch ((Op)n->c) {
                    case Op::LtI32: o = Op::PairLt; break;
                    case Op::LeI32: o = Op::PairLe; break;
                    case Op::GtI32: o = Op::PairGt; break;
                    default: o = Op::PairGe; break;
                    }
                }
                n->op = o;
                n->a = n->b = n->c = 0;
            } else {
                if (ta.k == TK::Pair) n->a = (int32_t)M.desc_of(ta);
                if (tb.k == TK::Pair) n->b = (int32_t)M.desc_of(tb);
            }
            break;
        }
        case Op::Print:
            if (n->kids[0]->type.k == TK::Pair) n->a = (int32_t)M.desc_of(n->kids[0]->type);
            break;
        case Op::If:
        case Op::While:
            fix_cond(n->kids[0]);
            break;
        case Op::And:
        case Op::Or:
            if (n->flags & F_COND) {
                fix_cond(n->kids[0]);
                fix_cond(n->kids[1]);
            }
            break;
        default:
            break;
        }
    }

    void run() {
        size_t R = M.regions.size();
        cand.assign(R, 0);
        nlit.assign(R, 0);
        for (size_t r = 0; r < R; ++r) {
            const Region& g = M.regions[r];
            cand[r] = !g.pair && g.elem.k == TK::Int;
        }
        for (auto& f : M.funcs) analyse_node(f->body);
        // Box can never take such a list (elem is not Any), but be explicit:
        // regions flagged as flowing into Any are never pairs.
        size_t npairs = 0;
        for (size_t r = 0; r < R; ++r) {
            if (cand[r] && (nlit[r] == 0 || M.regions[r].flows_to_any)) cand[r] = 0;
            npairs += cand[r];
        }
        if (verbose) {
            for (size_t r = 0; r < R; ++r)
                if (cand[r]) std::fprintf(stderr, "pairs: region %zu -> pair (%u literal sites)\n", r, nlit[r]);
        }
        if (!npairs) return;

        // Region flags and element types.
        for (size_t r = 0; r < R; ++r) {
            Region& g = M.regions[r];
            if (cand[r]) g.pair = true;
            g.elem = rt(g.elem);
        }
        // The pair region's own LIST descriptor (if any) becomes a PAIR descriptor:
        // outer list descriptors referring to it now describe lists of pairs.
        for (size_t r = 0; r < R && r < M.region_desc_memo.size(); ++r) {
            int32_t d = M.region_desc_memo[r];
            if (cand[r] && d >= 4) M.descs[(size_t)d] = Desc{4, 0};
        }
        for (auto& g : M.globals) g.type = rt(g.type);
        for (auto& fp : M.funcs) {
            Func& f = *fp;
            for (auto& l : f.locals) l.type = rt(l.type);
            f.ret = rt(f.ret);
        }
        for (auto& fp : M.funcs) util::rewrite(fp->body, [&](Node*& s) { rewrite_node(s); });
        pass_region_info(M);
    }
};

}  // namespace

void pass_pairs(Module& m) {
    PairsPass p{m, {}, {}};
    p.run();
    // Test hook: EPQ_PAIRS_TWICE=1 runs the pass a second time on its own
    // output (idempotency check; the second run must find nothing to do).
    if (std::getenv("EPQ_PAIRS_TWICE")) {
        verify(m, "pairs (first run)");
        PairsPass q{m, {}, {}};
        q.run();
    }
}

}  // namespace epq::tir
