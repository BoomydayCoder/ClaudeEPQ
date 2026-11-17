// Whole-program type inference and function specialisation (DESIGN.md §2).
//
// Lattice:  T ::= Bot | Int | List(e) | Fn(d) | Any
//   e is an *element variable* (EVar): a union-find node that itself has a T.
//   Every list object has exactly one element representation, fixed by the
//   element variable of its allocation site; every variable that can hold it
//   has the same (unified) element variable or is Any — in which case the
//   element variable is Any too (dynamisation). An EVar root is therefore a
//   Steensgaard points-to class ("region"): lists with different roots never
//   alias.
//
// Solving: abstract interpretation to a fixpoint. Every pass walks the body of
// every instance and re-applies all flow constraints (flow(src -> dst) =
// dst := join(dst, src), with unification of element variables and
// dynamisation); passes repeat until nothing changes. All rules are monotone
// on a finite lattice (instances and element variables are capped).
//
// Specialisation (§2.4): an *instance* is one copy of a function body with its
// own tvars (locals, return value, per-node types). In polyvariant mode every
// call site of every instance gets its own callee instance, except that a
// callee already on the caller's instantiation chain is reused (recursion);
// beyond the caps, sites share one merged instance per function. Escaped
// functions (an Fn(d) that flows into Any) get one *generic* instance whose
// parameters and result are Any (§2.5). After the fixpoint, instances with
// identical resolved types (including region identity) and equivalent callees
// are merged by partition refinement (Instance::rep).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "resolver.hpp"

namespace epq::infer {

enum class TyK : uint8_t { Bot, Int, List, Fn, Any };

struct Ty {
    TyK k = TyK::Bot;
    uint32_t a = 0;  // List: an element variable (maybe not a root: use find); Fn: DefId
    static Ty bot() { return {}; }
    static Ty i32() { return {TyK::Int, 0}; }
    static Ty any() { return {TyK::Any, 0}; }
    static Ty list(uint32_t e) { return {TyK::List, e}; }
    static Ty fn(uint32_t d) { return {TyK::Fn, d}; }
};

// An allocation site recorded in an element variable (for pair unboxing, §5.4).
struct AllocSite {
    const Node* node;  // the List literal
    uint32_t inst;     // instance whose body contains it
    uint32_t len;      // number of elements of the literal
};

constexpr uint32_t kNone = 0xffffffffu;

// EVar flags (records, DESIGN.md §5.11).
enum : uint8_t {
    EV_PLAIN = 1 << 0,  // region is definitely an ordinary list (non-record site or a list-only use)
    EV_BAD = 1 << 1,    // record region violated a record condition (this run is discarded)
    EV_FIELD = 1 << 2,  // a position variable of a record, not a region
    EV_NULL = 1 << 3,   // (final result, roots) record region with an empty-literal site
};

struct EVar {
    uint32_t parent;               // union-find parent (== own index for roots)
    Ty t;                          // element type (meaningful on roots; unused for records)
    std::vector<AllocSite> sites;  // merged into the root on union
    // Records (§5.11): a root with nf > 0 is a *record region*; position j has
    // its own type variable, the field EVar find(fbase + j) (its t is the type).
    // A root with nf == 0 is an ordinary list region (or, with neither
    // EV_PLAIN nor any non-empty site, still undetermined: only `[]` sites).
    uint32_t fbase = kNone;
    uint8_t nf = 0;
    uint8_t flags = 0;
};

// Open-addressing map AST node -> ordinal (built once by infer(), read-only
// afterwards). Replaces std::unordered_map: one allocation, no per-entry nodes.
class NodeMap {
public:
    void init(size_t n) {
        size_t cap = 16;
        while (cap < n * 2) cap <<= 1;
        keys_.assign(cap, nullptr);
        vals_.assign(cap, 0);
        shift_ = 64 - (unsigned)__builtin_ctzll(cap);
    }
    void put(const Node* n, uint32_t v) {
        size_t i = slot(n);
        while (keys_[i] && keys_[i] != n) i = (i + 1) & (keys_.size() - 1);
        keys_[i] = n;
        vals_[i] = v;
    }
    uint32_t at(const Node* n) const {
        size_t i = slot(n);
        while (keys_[i] != n) {
            if (!keys_[i]) __builtin_trap();  // not an expression node of any function
            i = (i + 1) & (keys_.size() - 1);
        }
        return vals_[i];
    }

private:
    size_t slot(const Node* n) const {
        return (size_t)(((uint64_t)(uintptr_t)n * 0x9E3779B97F4A7C15ull) >> shift_);
    }
    std::vector<const Node*> keys_;
    std::vector<uint32_t> vals_;
    unsigned shift_ = 60;
};

struct Instance {
    uint32_t id = 0;
    DefId def = 0;
    int32_t parent = -1;   // instance that created it (instantiation chain), -1 for roots
    bool generic = false;  // the generic (all-Any) instance of an escaped function
    bool shared = false;   // the merged instance used beyond the caps / in mono mode
    std::vector<Ty> locals;   // resolver slot -> type
    Ty ret;                   // return value
    std::vector<Ty> nty;      // node ordinal -> type of the expression node
    std::vector<uint32_t> naux;  // node ordinal -> List literal: EVar; Call: callee instance
    uint32_t rep = 0;         // after dedup: representative instance (== id if itself)
    bool reachable = false;   // after the fixpoint: reachable from main / generic instances
};

struct Config {
    bool poly = true;            // per-call-site instances (false: one instance per function)
    uint32_t max_per_def = 8;    // §2.4 caps
    uint32_t max_total = 256;
    bool dedup = true;
    bool records = true;         // §5.11 record typing (--no-records / EPQ_NO_RECORDS=1: off)
    uint32_t max_record_runs = 10;  // speculative runs after run 0 (hard limit)
};

struct Result {
    const Program* prog = nullptr;
    std::vector<Ty> globals;
    std::vector<EVar> evars;
    std::vector<std::unique_ptr<Instance>> insts;  // insts[0] is main
    std::vector<int32_t> generic_of;  // DefId -> generic instance, -1 if not escaped
    std::vector<int32_t> shared_of;   // DefId -> shared instance, -1
    std::vector<uint8_t> escaped;     // DefId -> escaped
    NodeMap ord;                      // node -> ordinal within its function (preorder)
    std::vector<uint32_t> nnodes;     // DefId -> number of ordinals
    uint32_t passes = 0;
    uint32_t record_runs = 0;   // inference runs after run 0 (0: no record candidates)
    bool has_records = false;   // some region of the final result is a record region

    uint32_t find(uint32_t e) const {
        while (evars[e].parent != e) e = evars[e].parent;
        return e;
    }
    // Element type of the region of e (the root's type).
    Ty elem(uint32_t e) const { return evars[find(e)].t; }
    // Records (§5.11). e: any EVar of a list region.
    bool is_rec(uint32_t e) const { return evars[find(e)].nf > 0; }
    uint32_t rec_len(uint32_t e) const { return evars[find(e)].nf; }
    bool rec_nullable(uint32_t e) const { return (evars[find(e)].flags & EV_NULL) != 0; }
    Ty field(uint32_t e, uint32_t j) const { return evars[find(evars[find(e)].fbase + j)].t; }
    uint32_t ordinal(const Node* n) const { return ord.at(n); }
    const Ty& type_of(uint32_t inst, const Node* n) const { return insts[inst]->nty[ordinal(n)]; }
    uint32_t aux_of(uint32_t inst, const Node* n) const { return insts[inst]->naux[ordinal(n)]; }
};

void infer(const Program& prog, const Config& cfg, Result& out);

// Statement can complete normally (falls through); used for "falling off the end".
bool completes(const Node* stmt);

// For --dump-types.
std::string ty_str(const Result& r, Ty t);
void dump_types(std::string& out, const Result& r);

}  // namespace epq::infer
