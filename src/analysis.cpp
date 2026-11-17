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
    std::vector<uint32_t> sets, ok_reads, bad_reads;
    std::vector<uint8_t> lit_set;

    void add_mut(Func* f, uint32_t r) {
        if (std::find(f->mutates.begin(), f->mutates.end(), r) == f->mutates.end()) f->mutates.push_back(r);
    }

    void visit(Node* n, Node* parent, uint32_t idx) {
        switch (n->op) {
        case Op::Append:
        case Op::Pop:
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
        case Op::SetLocal:
            sets[(size_t)n->a]++;
            if (n->kids[0]->op == Op::ListLit) lit_set[(size_t)n->a] = 1;
            break;
        case Op::Local: {
            bool base = parent && idx == 0 &&
                        (parent->op == Op::Index || parent->op == Op::SetIndex || parent->op == Op::Append ||
                         parent->op == Op::Pop || parent->op == Op::Len);
            bool ret = parent && parent->op == Op::Return;
            if (base || ret) ok_reads[(size_t)n->a]++;
            else bad_reads[(size_t)n->a]++;
            break;
        }
        default:
            break;
        }
        for (uint32_t i = 0; i < n->nkids; ++i) visit(n->kids[i], n, i);
    }

    void callees(Node* n, std::vector<uint32_t>& out) {
        if (n->op == Op::CallDirect) out.push_back((uint32_t)n->a);
        for (uint32_t i = 0; i < n->nkids; ++i) callees(n->kids[i], out);
    }
};

}  // namespace

void pass_region_info(Module& m) {
    Walk w{m, nullptr, {}, {}, {}, {}};
    for (auto& r : m.regions) r.header_mutated = r.elem_stored = false;
    std::vector<std::vector<uint32_t>> calls(m.funcs.size());
    for (auto& fp : m.funcs) {
        Func* f = fp.get();
        f->mutates.clear();
        f->mutates_any = false;
        w.F = f;
        size_t nl = f->locals.size();
        w.sets.assign(nl, 0);
        w.ok_reads.assign(nl, 0);
        w.bad_reads.assign(nl, 0);
        w.lit_set.assign(nl, 0);
        w.visit(f->body, nullptr, 0);
        for (size_t l = 0; l < nl; ++l) {
            Local& L = f->locals[l];
            L.unique = L.kind != LocalKind::Param && L.type.k == TK::List && w.sets[l] == 1 && w.lit_set[l] &&
                       w.bad_reads[l] == 0;
        }
        w.callees(f->body, calls[f->id]);
    }
    // Generic entries are reachable from any dynamic call.
    std::vector<uint32_t> generics;
    for (auto& f : m.funcs)
        if (f->generic) generics.push_back(f->id);
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& fp : m.funcs) {
            Func* f = fp.get();
            std::vector<uint32_t> cs = calls[f->id];
            if (f->mutates_any) cs.insert(cs.end(), generics.begin(), generics.end());
            for (uint32_t c : cs) {
                Func* g = m.funcs[c].get();
                if (g->mutates_any && !f->mutates_any) {
                    f->mutates_any = true;
                    changed = true;
                }
                for (uint32_t r : g->mutates) {
                    if (std::find(f->mutates.begin(), f->mutates.end(), r) == f->mutates.end()) {
                        f->mutates.push_back(r);
                        changed = true;
                    }
                }
            }
        }
    }
    for (auto& f : m.funcs) std::sort(f->mutates.begin(), f->mutates.end());
}

}  // namespace epq::tir
