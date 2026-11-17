// Region analyses for the optimisation passes (DESIGN.md §5.2 / §5.4 inputs).
//
//  * Func::mutates / Func::mutates_any — regions whose list HEADERS (len/cap/data)
//    the function may change, transitively through calls: Append / Pop on a
//    List(r) base mutate r; DynAppend / DynNeg mutate "the Any region" (every
//    region whose element type is Any); a dynamic call may run any generic entry
//    and mutates the Any region. Element stores never count.
//  * Local::unique — the local is assigned exactly once, from a list literal,
//    and every read of it is the base of Index / SetIndex / Append / Pop / Len
//    or the value of a Return: no other name can ever refer to that list.
//  * Region::header_mutated / Region::elem_stored — some function appends/pops
//    (header) or stores elements (SetIndex) into lists of the region.
#include <algorithm>

#include "tir.hpp"

namespace epq::tir {

namespace {

struct Walk {
    Module& M;
    Func* F = nullptr;
    std::vector<uint32_t> sets, bad_reads;
    std::vector<uint8_t> lit_set;
    std::vector<uint32_t> calls;  // CallDirect targets of all functions, preorder (flat)

    void add_mut(Func* f, uint32_t r) {
        if (std::find(f->mutates.begin(), f->mutates.end(), r) == f->mutates.end()) f->mutates.push_back(r);
    }

    void visit(Node* n, Node* parent, uint32_t idx) {
        switch (n->op) {
        case Op::Append:
        case Op::Pop:
        case Op::Reserve:      // appendopt: mutate kid 0's header like Append
        case Op::AppendRange:  // appendopt
            add_mut(F, n->kids[0]->type.a);
            M.regions[n->kids[0]->type.a].header_mutated = true;
            break;
        case Op::SetIndex:
            M.regions[n->kids[0]->type.a].elem_stored = true;
            break;
        case Op::DynAppend:
        case Op::DynNeg:
        case Op::CallDyn:
            F->mutates_any = true;
            break;
        case Op::CallDirect:
            calls.push_back((uint32_t)n->a);
            break;
        case Op::SetLocal:
            sets[(size_t)n->a]++;
            if (n->kids[0]->op == Op::ListLit) lit_set[(size_t)n->a] = 1;
            break;
        case Op::Local: {
            bool base = parent && idx == 0 &&
                        (parent->op == Op::Index || parent->op == Op::SetIndex || parent->op == Op::Append ||
                         parent->op == Op::Pop || parent->op == Op::Len ||
                         parent->op == Op::Reserve || parent->op == Op::AppendRange);  // appendopt
            if (parent && idx == 1 && parent->op == Op::AppendRange) base = true;  // appendopt: copy source, like an Index base
            bool ret = parent && parent->op == Op::Return;
            if (!(base || ret)) bad_reads[(size_t)n->a]++;
            break;
        }
        default:
            break;
        }
        for (uint32_t i = 0; i < n->nkids; ++i) visit(n->kids[i], n, i);
    }
};

}  // namespace

void pass_region_info(Module& m) {
    Walk w{m, nullptr, {}, {}, {}, {}};
    for (auto& r : m.regions) r.header_mutated = r.elem_stored = false;
    size_t nf = m.funcs.size();
    // CallDirect targets of function id i: w.calls[cb[i] .. ce[i])
    std::vector<uint32_t> cb(nf, 0), ce(nf, 0);
    for (auto& fp : m.funcs) {
        Func* f = fp.get();
        f->mutates.clear();
        f->mutates_any = false;
        w.F = f;
        size_t nl = f->locals.size();
        w.sets.assign(nl, 0);
        w.bad_reads.assign(nl, 0);
        w.lit_set.assign(nl, 0);
        cb[f->id] = (uint32_t)w.calls.size();
        w.visit(f->body, nullptr, 0);
        ce[f->id] = (uint32_t)w.calls.size();
        for (size_t l = 0; l < nl; ++l) {
            Local& L = f->locals[l];
            L.unique = L.kind != LocalKind::Param && L.type.k == TK::List && w.sets[l] == 1 && w.lit_set[l] &&
                       w.bad_reads[l] == 0;
        }
    }
    // Generic entries are reachable from any dynamic call.
    std::vector<uint32_t> generics;
    for (auto& f : m.funcs)
        if (f->generic) generics.push_back(f->id);
    auto absorb = [&](Func* f, uint32_t c, bool& changed) {
        Func* g = m.funcs[c].get();
        if (g->mutates_any && !f->mutates_any) {
            f->mutates_any = true;
            changed = true;
        }
        if (g == f) return;
        for (uint32_t r : g->mutates) {
            if (std::find(f->mutates.begin(), f->mutates.end(), r) == f->mutates.end()) {
                f->mutates.push_back(r);
                changed = true;
            }
        }
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& fp : m.funcs) {
            Func* f = fp.get();
            bool any = f->mutates_any;  // generics join the callees as of the start of f's visit
            for (uint32_t i = cb[f->id]; i < ce[f->id]; ++i) absorb(f, w.calls[i], changed);
            if (any)
                for (uint32_t c : generics) absorb(f, c, changed);
        }
    }
    for (auto& f : m.funcs) std::sort(f->mutates.begin(), f->mutates.end());
}

}  // namespace epq::tir
