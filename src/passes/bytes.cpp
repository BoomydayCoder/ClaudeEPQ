// Pass "bytes": byte lists (DESIGN.md §5.12).
//
// A region r becomes a byte region (Region::bytes; Module::elem_width(r) == 1)
// iff its element type is Int, it is neither a pair nor a record region, and
// every value that can ever be stored into an element of a list of r is
// provably in [0, 255]. Its lists then hold one byte per element.
//
// Why this is sound: inference unifies the element variables of every alias of
// a list, so all lists that can ever be the same list share one region, and
// every store into any of them is one of the element-storing ops below, in
// some function instance of the module (every instance is in Module::funcs,
// generic entries included). Lists of Int regions never flow into Any (Box of
// a List(r) only happens when elem(r) is Any), so dynamic code (Dyn*, CallDyn,
// the runtime's generic helpers) never stores into them. Element-storing ops:
//   ListLit (every kid), SetIndex (kid 2), ElemStore (kid 2), Append (kid 1),
//   AppendRange (copies bitwise from region m: r and m must have the same width,
//   so they are coupled — either both are byte regions or neither is).
// A Box of a List(r) disqualifies r (defensive: the invariant says it never
// happens for an Int region).
//
// "Provably in [0, 255]" (in_range), a greatest fixpoint over regions, Int
// locals, Int globals and Int function results, starting from "everything
// qualifies" and removing violators until nothing changes:
//   ConstInt 0..255; a comparison / NotI32 / DynNot / Eq / Ne / Cmp / PairEq.. /
//   DynIsEmptyList / SameList (0 or 1); any Bot-typed node (never produces a
//   value); Select / value-form And / Or: both value kids; Comma: its last kid;
//   SetLocal / SetGlobal / SetIndex / ElemStore used as a value: the stored value;
//   Index / ElemLoad / Pop of a region still in the set; a Local whose every
//   assignment (SetLocal, in its function) is in range — parameters are unknown;
//   a GetGlobal of a global whose every SetGlobal is in range; a CallDirect of a
//   (non-main, non-generic) instance whose every Return value is in range.
// Anything else (arithmetic, Input, UnboxInt, PairFirst, RecGet, ...) is not.
//
// Position: last in the pipeline (after reclaim). Nothing in TIR bakes in an
// element width: Ptr(r), ElemLoad/ElemStore (listopt), Reserve/AppendRange
// (appendopt) and Free (reclaim) all derive the width from the region when the
// back ends emit code, so setting Region::bytes after every other pass is safe,
// and only then is the set of element-storing ops final. appendopt only creates
// AppendRange between regions of equal width (all Int regions are 4 bytes when
// it runs); the coupling rule above keeps them equal. The verifier checks the
// result (src/tir_verify.cpp: byte regions are Int, not pair/record, and the two
// lists of an AppendRange have equal widths).
//
// Descriptors: a byte region's LIST descriptor gets the element descriptor
// INT8 ({INT, 1}, Module::desc_int8), so the runtime's deep equality, ordering
// and printing read its elements as bytes — also against ordinary int lists.
#include <vector>

#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

namespace {

struct Bytes {
    Module& M;
    std::vector<char> reg;                     // region still qualifies
    std::vector<std::vector<char>> loc;        // [func][local] Int local all of whose assignments qualify
    std::vector<char> glob;                    // Int global all of whose assignments qualify
    std::vector<char> ret;                     // Int function result: every Return qualifies

    struct Store { uint32_t r, f; const Node* v; };  // f: the function it occurs in
    struct LocSet { uint32_t f, l; const Node* v; };
    struct Coupled { uint32_t a, b; };
    std::vector<Store> stores;
    std::vector<LocSet> locsets;
    std::vector<Store> globsets;                          // r: the global
    std::vector<std::pair<uint32_t, const Node*>> rets;  // (function, returned value)
    std::vector<Coupled> coupled;

    explicit Bytes(Module& m) : M(m) {}

    bool cur(uint32_t f, uint32_t l) const { return loc[f][l]; }

    bool in_range(uint32_t f, const Node* n) const {
        if (n->type.k == TK::Bot) return true;  // never produces a value
        if (n->type.k != TK::Int) return false;
        switch (n->op) {
        case Op::ConstInt: return n->a >= 0 && n->a <= 255;
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
        case Op::NotI32: case Op::DynNot: case Op::Eq: case Op::Ne: case Op::Cmp:
        case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
        case Op::DynIsEmptyList: case Op::SameList:
            return true;
        case Op::Select: return in_range(f, n->kids[1]) && in_range(f, n->kids[2]);
        case Op::And: case Op::Or:
            if (n->flags & F_COND) return false;
            return in_range(f, n->kids[0]) && in_range(f, n->kids[1]);
        case Op::Comma: return n->nkids > 0 && in_range(f, n->kids[n->nkids - 1]);
        case Op::SetLocal: case Op::SetGlobal: return in_range(f, n->kids[0]);
        case Op::SetIndex: case Op::ElemStore: return in_range(f, n->kids[2]);
        case Op::Index: case Op::Pop: case Op::ElemLoad: {
            Type t = n->kids[0]->type;
            return (t.k == TK::List || t.k == TK::Ptr) && reg[t.a];
        }
        case Op::Local: return cur(f, (uint32_t)n->a);
        case Op::GetGlobal: return glob[(size_t)n->a];
        case Op::CallDirect: return ret[(size_t)n->a];
        default: return false;
        }
    }

    void collect(uint32_t f, Node* body) {
        util::walk(body, [&](Node* n, Node*, uint32_t) {
            switch (n->op) {
            case Op::ListLit:
                for (uint32_t i = 0; i < n->nkids; ++i) stores.push_back({n->type.a, f, n->kids[i]});
                break;
            case Op::SetIndex:
                if (n->kids[0]->type.k == TK::List) stores.push_back({n->kids[0]->type.a, f, n->kids[2]});
                break;
            case Op::ElemStore:
                if (n->kids[0]->type.k == TK::Ptr) stores.push_back({n->kids[0]->type.a, f, n->kids[2]});
                break;
            case Op::Append:
                if (n->kids[0]->type.k == TK::List) stores.push_back({n->kids[0]->type.a, f, n->kids[1]});
                break;
            case Op::AppendRange:
                coupled.push_back({n->kids[0]->type.a, n->kids[1]->type.a});
                break;
            case Op::Box:
                if (n->kids[0]->type.k == TK::List) reg[n->kids[0]->type.a] = 0;
                break;
            case Op::SetLocal: locsets.push_back({f, (uint32_t)n->a, n->kids[0]}); break;
            case Op::SetGlobal: globsets.push_back({(uint32_t)n->a, f, n->kids[0]}); break;
            case Op::Return:
                if (n->nkids) rets.push_back({f, n->kids[0]});
                break;
            default: break;
            }
            return true;
        });
    }

    void run() {
        const size_t R = M.regions.size();
        reg.assign(R, 0);
        for (size_t r = 0; r < R; ++r) {
            const Region& g = M.regions[r];
            reg[r] = !g.pair && !g.record && g.elem.k == TK::Int;
        }
        loc.resize(M.funcs.size());
        ret.assign(M.funcs.size(), 0);
        for (size_t f = 0; f < M.funcs.size(); ++f) {
            const Func& F = *M.funcs[f];
            loc[f].assign(F.locals.size(), 0);
            for (size_t l = 0; l < F.locals.size(); ++l)
                loc[f][l] = F.locals[l].type.k == TK::Int && F.locals[l].kind != LocalKind::Param;
            ret[f] = !F.is_main && !F.generic && F.ret.k == TK::Int;
        }
        glob.assign(M.globals.size(), 0);
        for (size_t g = 0; g < M.globals.size(); ++g) glob[g] = M.globals[g].type.k == TK::Int;
        for (size_t f = 0; f < M.funcs.size(); ++f) collect((uint32_t)f, M.funcs[f]->body);

        for (bool changed = true; changed;) {
            changed = false;
            auto drop = [&](char& c) {
                if (c) {
                    c = 0;
                    changed = true;
                }
            };
            for (auto& s : stores)
                if (reg[s.r] && !in_range(s.f, s.v)) drop(reg[s.r]);
            for (auto& c : coupled)
                if (reg[c.a] != reg[c.b]) {
                    drop(reg[c.a]);
                    drop(reg[c.b]);
                }
            for (auto& s : locsets)
                if (loc[s.f][s.l] && !in_range(s.f, s.v)) drop(loc[s.f][s.l]);
            for (auto& s : globsets)
                if (glob[s.r] && !in_range(s.f, s.v)) drop(glob[s.r]);
            for (auto& s : rets)
                if (ret[s.first] && !in_range(s.first, s.second)) drop(ret[s.first]);
        }
        for (size_t r = 0; r < R; ++r) M.regions[r].bytes = reg[r] != 0;
        // Existing LIST descriptors of byte regions describe byte elements.
        for (size_t r = 0; r < R && r < M.region_desc_memo.size(); ++r) {
            int32_t d = M.region_desc_memo[r];
            if (reg[r] && d >= 4 && M.descs[(size_t)d].kind == 3) M.descs[(size_t)d].elem = M.desc_int8();
        }
    }
};

}  // namespace

void pass_bytes(Module& m) {
    Bytes b(m);
    b.run();
}

}  // namespace epq::tir
