// TIR — typed intermediate representation (DESIGN.md §4, "High TIR").
//
// The single input of both backends (C, ARM64 JIT) and of the optimisation
// passes. One tree per function *instance* (a specialised copy of a function
// literal, see src/infer.hpp).
//
// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
//   Bot      never produces a value (the node diverges or raises); represented
//            like Int (int32 0) wherever a slot of that type must exist
//   Int      int32_t, wrapping arithmetic
//   List(r)  pointer to a list header; r is a *region* (Module::regions): all
//            lists of a region share one element type, and two list values of
//            different regions can never be the same list (Steensgaard
//            classes from inference). Region element type Int/Bot => int32
//            element array, otherwise 8-byte EAny words (runtime/epq_rt.h).
//   Fn(d)    a function value whose DefId d is statically known; no run-time
//            representation (C uses a dummy int32 0)
//   Any      tagged 64-bit dynamic value (runtime/epq_rt.h). Soundness
//            invariant: a List(r) is only ever Boxed when r's element type is Any.
//
// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------
// One node shape (Node) for statements and expressions; `op` decides the
// meaning of kids and of the payload fields a/b/c. Every node carries its
// resolved static type (statements: Bot). All nodes are arena-allocated and
// may be freely rewritten in place (kids is a mutable array; replace kids[i]).
//
// EVALUATION RULE (the important invariant): every operator evaluates ALL its
// kids, in kid order, and only then performs its checks and its operation.
// The kid order IS the source evaluation order (e.g. CallDirect / CallDyn have
// the arguments first and the callee last). Exceptions, by design:
//   And / Or (short circuit), If, While, and UnboxInt (checks its own kid
//   immediately — the builder only places it where nothing evaluated after it
//   in the parent could observe the difference).
//
// Checks and failure points are explicit:
//   * Fail(err, kids...)         evaluates kids, then raises err. Type Bot.
//                                Used for statically certain type errors.
//   * GetGlobal with F_CHECKED   raises "Undefined global" if not yet assigned.
//   * UnboxInt(err)              Any -> Int, raises err if not an integer.
//   * Typed ops document their own failure points (DivI32: divisor; Index /
//     SetIndex: bounds; Pop: empty; Dyn*: tag checks after evaluating kids).
// Conversions are explicit: Box (Int/List/Fn -> Any), UnboxInt, Coerce (Bot ->
// T; never executes past its kid). A typed op's kids have exactly the types
// listed below; the builder inserts conversions.
//
// Conditions (If/While kid 0) are in *branch form*: any expression is allowed
// and is tested for truthiness by type (Int: != 0; Any: != boxed 0; List/Fn:
// always true after evaluation; Bot: never returns). And/Or with F_COND are
// condition-only (kids are conditions, no value is materialised).
//
// Statements: Seq, If, While, Return, Print, and any expression (value
// discarded). For loops and for-each are desugared into While with a step kid:
//   While [cond, body, step]   loop: while (cond) { body; step; }
//   flags F_FOR / F_FOREACH; for-each: a = hidden index local, b = hidden
//   list local; the loop is `idx := 0; L := e; [x := 0;] while (idx < |L|)
//   { x := L[idx]; body; idx := idx + 1 }`.
//
// ---------------------------------------------------------------------------
// Extension for the optimisation passes (DESIGN.md §5; Low TIR)
// ---------------------------------------------------------------------------
// Types:
//   Pair(r)  an immutable two-integer list unboxed into one int64:
//              (int64)first << 32 | ((uint32)second ^ 0x80000000)
//            so signed 64-bit comparison is lexicographic comparison of the
//            two signed ints and equality is bit equality. r is the region the
//            pair belongs to; a Pair-typed value ALWAYS belongs to a region
//            with Region::pair == true (whose elem is Int). A region whose
//            element type is Pair stores packed pairs in 8-byte slots.
//            Wherever an op accepts kids "of any static type" (DynArith,
//            Eq/Ne/Cmp with descriptors, Print, conditions) a Pair behaves
//            exactly like the two-element integer list it stands for
//            (descriptor kind PAIR; DynArith: always OperandsInt; truthy).
//            Pairs never flow into Any (no Box of a Pair).
//   Ptr(r)   raw pointer to the element buffer of a list of region r: int32
//            elements if elem(r) is Int/Bot, else 8-byte slots. Only ever
//            held in compiler-created locals (LocalKind::Cache). It becomes
//            stale when the list's header changes (Append may move the buffer
//            and frees the old one; Free).
// Ops (all evaluate all kids in kid order, then act, unless stated):
//   Select [c: Int, x: T, y: T] -> T      all kids pure and cannot fail (is_pure);
//                                         x if c != 0 else y
//   InlineBlock [body], a = label        statement (type Bot); body is any statement
//   Break, a = label                     statement; jumps to the end of the innermost
//                                         enclosing InlineBlock with label a.
//     An inlined call is Comma[InlineBlock{...; SetLocal(res, v); Break; ...}, Local(res)].
//     Labels are allocated per function with new_label(Func&) (Func::next_label).
//   PairMake [a: Int, b: Int] -> Pair(r)  (r = node type's region)
//   PairFirst [p: Pair] -> Int, PairSecond [p: Pair] -> Int
//   PairIndex [p: Pair, i: Int] -> Int    IndexOOB unless i is 0 or 1
//   PairEq/PairNe/PairLt/PairLe/PairGt/PairGe [p: Pair, q: Pair] -> Int (0/1)
//   ListData [l: List(r)] -> Ptr(r)       loads the header's data pointer
//   ElemLoad [d: Ptr(r), i: Int] -> elem(r)            unchecked
//   ElemStore [d: Ptr(r), i: Int, v: elem(r)] -> elem(r)  unchecked; value v
//   CheckIndex [i: Int, len: Int] -> Int  value i; IndexOOB unless 0 <= i < len
//   Free [l: List(r)]                     statement; returns header + buffer to the
//                                         allocator's free lists (only the list itself)
//   SameList [a: List, b: List] -> Int    pointer identity (any two list regions)
//   Nop                                   statement, does nothing
// is_pure() is true for Select, PairMake/First/Second, PairEq..PairGe,
// SameList, ListData, ElemLoad, Nop (when their kids are pure). It is NOT
// true for PairIndex / CheckIndex (may fail) or ElemStore / Free.
// tir::verify() checks all of the above (src/tir_verify.cpp).
//
// ---------------------------------------------------------------------------
// Records (DESIGN.md §5.11; set by inference, built by tirbuild)
// ---------------------------------------------------------------------------
// Types:
//   Rec(r)   a fixed-shape heterogeneous list of a *record region* r
//            (Region::record): every list of r has exactly rec_len elements
//            (or, if Region::nullable, is empty), and position j always holds a
//            value of static type Region::fields[j]. Representation: pointer to
//            rec_len 8-byte slots, NO header; a slot holds the field's word
//            (Int/Bot: (uint64)(uint32)v; List/Rec: pointer; Pair: the int64;
//            Fn: the tagged function word; Any: the tagged word). The empty list
//            of a nullable region is the null pointer. Records are heap objects
//            with identity (stores are seen through every alias), but: a region
//            with record == true never appears as List(r); records never flow
//            into Any (no Box of a Rec); nothing ever appends to / pops / iterates
//            a record or indexes it with a non-constant. Wherever an op accepts
//            kids "of any static type" (Eq/Ne/Cmp/Print with descriptors,
//            conditions, DynArith) a Rec behaves exactly like the list it stands
//            for (descriptor kind RECORD; truthy; DynArith: OperandsInt).
// Ops:
//   RecLit [k kids: fields[i]] -> Rec(r)   allocates k slots, stores the kids. k == rec_len
//   RecNull -> Rec(r)                      the empty list of a nullable r (null); pure
//   RecGet a = position [rec] -> fields[a] one load; b = 1 iff r is nullable: then a null
//                                          rec raises IndexOOB (after evaluating rec)
//   RecSet a = position [rec, v: fields[a]] -> fields[a]   stores v into slot a (value v);
//                                          b as RecGet (null check after evaluating both kids).
//                                          Never touches any list header.
//   RecLen [rec] -> Int                    rec_len, or 0 if null (nullable r only; for a
//                                          non-nullable r the builder emits Comma(rec, k))
// A literal index outside [0, k) is built as Fail(IndexOOB, rec, i[, v]).
// `t = []` / `t != []` with t : Rec(r) is built as EqI32/NeI32(RecLen(t), 0)
// (or a constant after evaluating t when r is not nullable): no allocation, no call.
// is_pure: RecNull, RecLen, and RecGet with b == 0 (when their kids are pure).
// Escape: a list or record stored into a RecLit kid or a RecSet value escapes
// exactly like one stored into a list element.
//
// Locals: each Func has a typed local table: parameters first-class (Func::
// params lists their local ids in order), user locals (resolver slots keep
// their slot numbers), hidden for-each locals, compiler temporaries (kind
// Temp, appended). Globals live in Module::globals.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "resolver.hpp"

namespace epq::tir {

enum class TK : uint8_t { Bot, Int, List, Fn, Any, Pair, Ptr, Rec };

struct Type {
    TK k = TK::Bot;
    uint32_t a = 0;  // List: region id; Fn: DefId
    static Type bot() { return {}; }
    static Type i32() { return {TK::Int, 0}; }
    static Type any() { return {TK::Any, 0}; }
    static Type list(uint32_t r) { return {TK::List, r}; }
    static Type fn(uint32_t d) { return {TK::Fn, d}; }
    static Type pair(uint32_t r) { return {TK::Pair, r}; }  // a: the pair's region (Region::pair)
    static Type ptr(uint32_t r) { return {TK::Ptr, r}; }    // a: region of the list
    static Type rec(uint32_t r) { return {TK::Rec, r}; }    // a: the record region (Region::record)
    bool has_arg() const { return k == TK::List || k == TK::Fn || k == TK::Pair || k == TK::Ptr || k == TK::Rec; }
    bool operator==(const Type& o) const { return k == o.k && (!has_arg() ? true : a == o.a); }
    bool operator!=(const Type& o) const { return !(*this == o); }
    bool is_int_like() const { return k == TK::Int || k == TK::Bot; }
};

// Run-time error kinds; values match runtime/epq_rt.h EPQ_E_*.
enum class Err : uint8_t {
    None = 0,
    OperandsInt,   // Operands must be integers
    DivZero,       // Division by zero
    ListEmpty,     // List is empty
    IntOrList,     // Operand must be an integer or a list
    OperandList,   // Operand must be a list
    OperandInt,    // Operand must be an integer
    UndefGlobal,   // Undefined global
    InvalidOp,     // Invalid operation
    IndexOOB,      // Index out of bounds
    OperandFn,     // Operand must be a function
    Arity,         // Invalid arity
    InvalidCmp,    // Invalid comparison
};
const char* err_name(Err e);

enum class Op : uint8_t {
    // ---- statements ----------------------------------------------------------
    Seq,        // kids: statements (any node; expression values discarded)
    If,         // [cond, then, else]
    While,      // [cond, body, step]          see header comment
    Return,     // [value]  value has the function's return type; in main: ends the program (exit 0)
    Print,      // [value]  a = descriptor of value's type   (">> value\n")
    // ---- leaves ----------------------------------------------------------------
    ConstInt,   // a = value                                  Int
    ConstFn,    // a = DefId                                  Fn(d)
    Local,      // a = local id                               local's type
    GetGlobal,  // a = global id; F_CHECKED: may be undefined  global's type
    Input,      // reads an integer (prints "<< ")            Int
    // ---- assignment --------------------------------------------------------------
    SetLocal,   // a = local id, [v: local's type]   value = v
    SetGlobal,  // a = global id, [v: global's type] value = v
    // ---- conversions ------------------------------------------------------------
    Box,        // [v: Int|List|Fn] -> Any
    UnboxInt,   // [v: Any] -> Int; a = Err raised if not an integer
    Coerce,     // [v: Bot] -> node type (never completes)
    Comma,      // [k1..kn]  evaluate all; value of kn (type of kn)
    Fail,       // a = Err, [kids...]  evaluate kids, raise        Bot
    // ---- typed integer ops (kids Int, result Int) --------------------------------
    AddI32, SubI32, MulI32,
    DivI32,     // raises DivZero; INT_MIN / -1 = INT_MIN
    NegI32, AbsI32,
    NotI32,     // 1 if 0 else 0
    EqI32, NeI32, LtI32, LeI32, GtI32, GeI32,
    // ---- dynamic integer ops: kids of ANY static type (Int, Any, or a statically
    //      known non-integer List/Fn which always fails); result Int ----------------
    DynArith,   // a = Op (AddI32/SubI32/MulI32/DivI32), [x, y]; after both: (Div: divisor
                // is integer 0 -> DivZero), any non-integer -> OperandsInt
    DynNeg,     // [x: Any] -> Any: int negate / list pop (ListEmpty) / fn -> IntOrList
    DynAbs,     // [x: Any] -> Int: int abs / list length / fn -> IntOrList
    DynNot,     // [x: Any] -> Int; non-integer -> OperandInt
    // ---- deep comparisons (descriptor driven; kids of any static type) -------------
    Eq,         // [x, y], a = desc(x), b = desc(y)      Int 0/1; never fails
    Ne,         // as Eq, negated
    Cmp,        // [x, y], a = desc(x), b = desc(y), c = Op (LtI32/LeI32/GtI32/GeI32)
                //   ints by value, lists lexicographically; else InvalidCmp
    // ---- lists ------------------------------------------------------------------
    ListLit,    // kids: elements (each of the region's element type)    List(r)
    Len,        // [l: List]                       Int
    Index,      // [l: List(r), i: Int]            elem(r); IndexOOB
    SetIndex,   // [l: List(r), i: Int, v: elem(r)]  value = v (type elem(r)); IndexOOB
    Append,     // [l: List(r), v: elem(r)]        value = l
    Pop,        // [l: List(r)]                    elem(r); ListEmpty
    DynIndex,   // [l: Any, i: Int|Any] -> Any; not list / not int -> InvalidOp; IndexOOB
    DynSetIndex,// [l: Any, i: Int|Any, v: Any] -> Any; same errors
    DynAppend,  // [l: Any, v: Any] -> Any (l); not a list -> OperandList
    // ---- control-flow expressions --------------------------------------------------
    And,        // [x, y] short circuit; value form: kids and result of the node type
    Or,         //   (F_COND: condition form, kids are conditions)
    // ---- calls ------------------------------------------------------------------
    CallDirect, // a = callee Func id; kids = [args (param types)..., callee expr]   ret type
                // the callee expression (type Fn) is evaluated after the arguments
    CallDyn,    // kids = [args: Any..., callee: Any] -> Any; not fn -> OperandFn; arity -> Arity
    // ---- extension (see "Extension for the optimisation passes" above) ----------
    Select,     // [c: Int, x: T, y: T] -> T; kids pure
    InlineBlock,// [body], a = label                       statement
    Break,      // a = label                               statement
    PairMake,   // [a: Int, b: Int] -> Pair(r)
    PairFirst,  // [p: Pair] -> Int
    PairSecond, // [p: Pair] -> Int
    PairIndex,  // [p: Pair, i: Int] -> Int; IndexOOB unless i in {0, 1}
    PairEq, PairNe, PairLt, PairLe, PairGt, PairGe,  // [p: Pair, q: Pair] -> Int
    ListData,   // [l: List(r)] -> Ptr(r)
    ElemLoad,   // [d: Ptr(r), i: Int] -> elem(r)          unchecked
    ElemStore,  // [d: Ptr(r), i: Int, v: elem(r)] -> elem(r)  unchecked; value v
    CheckIndex, // [i: Int, len: Int] -> Int (i); IndexOOB unless 0 <= i < len
    Free,       // [l: List(r)]                            statement
    SameList,   // [a: List, b: List] -> Int               pointer identity
    Nop,        // statement
    // ---- litcmp extension (src/passes/litcmp.cpp) ---------------------------------
    // DynIsEmptyList [x: Any] -> Int: evaluates x, then 1 if x is a list (tag
    //   (x & 3) == 0) whose length is 0, else 0. Never fails; pure if x is pure.
    //   It is `x = []` for a dynamic x without allocating the literal.
    //   C: epq_is_empty_list(x) (runtime/epq_rt.h). JIT: tst x,#3; b.ne -> 0;
    //   ldr w, [x, #8] (EList::len); cmp w, #0; cset.
    DynIsEmptyList,
    // ---- end litcmp extension ------------------------------------------------------
    // ---- appendopt (src/passes/appendopt.cpp) -----------------------------------
    // Both ops MUTATE THE HEADER OF l's REGION LIKE Append (other passes: treat them
    // exactly as an Append to kid 0 — any cached data pointer / length of a list of
    // that region is stale afterwards). Neither ever touches m's header or any
    // element of any list other than l's new slots.
    Reserve,    // [l: List(r), n: Int]  statement. Capacity hint: afterwards cap(l) >=
                // len(l) + min(max(n, 0), 2^26). Never changes len or contents, never
                // fails; does nothing if l is NULL (never-assigned local). May MOVE l's
                // data buffer (frees the old out-of-line buffer like Append's growth).
    AppendRange,// [l: List(r), m: List(r2), lo: Int, hi: Int] -> Int; elem(r) == elem(r2).
                // After evaluating the kids: if hi <= lo: does nothing, value 1. Else if
                // l and m are non-NULL, l != m, 0 <= lo and hi <= len(m): appends
                // m[lo..hi) to l (bitwise element copy; may move l's data buffer; len(l)
                // += hi - lo), value 1. Otherwise does nothing, value 0. Never fails.
                // The pass uses it as If(AppendRange(...), <set loop var>, <original loop>).
    // ---- records (see "Records" above) ----------------------------------------
    RecLit,     // [k kids: fields] -> Rec(r)          allocates
    RecNull,    // -> Rec(r)                            null (nullable r)
    RecGet,     // a = position, b = null check, [rec] -> fields[a]
    RecSet,     // a = position, b = null check, [rec, v] -> fields[a]
    RecLen,     // [rec] -> Int (k or 0)
};
const char* op_name(Op o);

enum : uint8_t {
    F_CHECKED = 1 << 0,  // GetGlobal: needs the "Undefined global" check
    F_COND = 1 << 1,     // And/Or: condition form
    F_FOR = 1 << 2,      // While: desugared for loop
    F_FOREACH = 1 << 3,  // While: desugared for-each loop
};

struct Node {
    Op op = Op::Seq;
    uint8_t flags = 0;
    uint16_t pad_ = 0;
    uint32_t nkids = 0;
    Type type;
    int32_t a = 0, b = 0, c = 0;
    uint32_t line = 0;  // source line (diagnostics / dumps)
    Node** kids = nullptr;
    Node* kid(uint32_t i) const { return kids[i]; }
};

enum class LocalKind : uint8_t { Param, User, Hidden, Temp, Demoted, Cache };  // Cache: header-cache locals (passes)

struct Local {
    Type type;
    LocalKind kind = LocalKind::User;
    Sym name = kNoSym;     // source name (kNoSym for hidden / temps)
    int32_t global = -1;   // Demoted: the global it replaces
    bool unique = false;   // region-info: sole name of a literal-allocated list (§5.2)
};

struct Func {
    uint32_t id = 0;
    DefId def = 0;
    uint32_t inst = 0;          // inference instance
    bool generic = false;       // generic entry of an escaped function (all Any)
    bool is_main = false;
    std::vector<uint32_t> params;  // local ids, in order
    std::vector<Local> locals;
    Type ret;
    Node* body = nullptr;       // a Seq
    std::string name;           // e.g. "f3_swap"
    // region-info analysis (§5.2): regions whose list headers this function may
    // mutate, transitively; mutates_any: also every region with Any elements.
    std::vector<uint32_t> mutates;
    bool mutates_any = false;
    uint32_t next_label = 0;    // InlineBlock / Break labels: new_label(f)
    // ---- recur pass (src/passes/recur.cpp) ----
    // pure: set by pass "recur" (false until it runs / if disabled). The result
    // depends only on the arguments: Int-only locals/params/result, no SetGlobal,
    // Print, Input, CallDyn, no list values at all, reads only Fn-typed or
    // Global::constant globals, calls only pure instances. It MAY fail or not
    // terminate (deterministically, given the arguments). Two calls with equal
    // arguments are interchangeable (call-CSE); a call whose result is unused may
    // NOT be dropped (it may fail / diverge).
    bool pure = false;
    // ---- end recur pass ----
};
inline uint32_t new_label(Func& f) { return f.next_label++; }

struct Global {
    Type type;
    Sym name = kNoSym;
    bool demoted = false;  // replaced by a local of main (no storage)
    bool ever_checked = false;  // some GetGlobal of it had F_CHECKED when built
    bool constant = false;      // constants pass: always holds `value` once assigned
    int32_t value = 0;
};

struct AllocSiteInfo {
    uint32_t func;  // Func id containing the literal
    uint32_t len;   // literal length
    uint32_t line;
};

struct Region {
    Type elem;
    std::vector<AllocSiteInfo> sites;  // literal allocation sites (from inference)
    // Filled by analyses (§5.2/§5.4 inputs):
    bool flows_to_any = false;   // elem == Any
    bool header_mutated = false; // region-info: some Append / Pop on the region
    bool elem_stored = false;    // region-info: some SetIndex on the region
    // Pair region (§5.4, set by the pairs pass): values of this region are
    // Pair(r), elem must be Int. Every Pair-typed value belongs to such a region.
    bool pair = false;
    // Record region (§5.11, from inference): values are Rec(r), never List(r);
    // elem is unused (Bot). fields[j] is the static type of position j.
    bool record = false;
    bool nullable = false;   // record region with `[]` sites (the null record)
    uint32_t rec_len = 0;    // k, 1..16
    std::vector<Type> fields;
};

// Static type descriptors (runtime EDesc). Index 0 INT, 1 ANY, 2 LIST(ANY), 3 FN.
// RECORD (kind 5): elem = rec_len | (nullable ? 0x10000 : 0); it is followed by
// rec_len FIELD entries (kind 6) whose elem is the descriptor of that position.
struct Desc {
    uint32_t kind;  // 0 INT, 1 ANY, 2 FN, 3 LIST, 4 PAIR, 5 RECORD, 6 FIELD (runtime EPQ_K_*)
    uint32_t elem;  // LIST: element descriptor
};

struct DefInfo {
    uint32_t arity = 0;
    int32_t generic_func = -1;  // Func id of the generic entry, if escaped
    Sym name = kNoSym;          // name of the global it was first assigned to (cosmetic)
};

struct Module {
    Arena arena;
    const Program* prog = nullptr;
    std::vector<std::unique_ptr<Func>> funcs;   // funcs[0] is main
    std::vector<Global> globals;
    std::vector<Region> regions;
    std::vector<Desc> descs;
    std::vector<DefInfo> defs;  // by DefId

    // ---- construction helpers ----
    Node* node(Op op, Type t, uint32_t nkids);
    Node* node(Op op, Type t, std::initializer_list<Node*> kids);
    uint32_t desc_of(Type t);
    uint32_t new_temp(Func* f, Type t);
    bool elem_is_i32(uint32_t region) const { return regions[region].elem.is_int_like(); }
    std::vector<int32_t> region_desc_memo;
};

// ---- utilities ---------------------------------------------------------------
// No side effects and cannot fail (constants, locals, unchecked globals, pure
// integer arithmetic of such, Box/Len of such).
bool is_pure(const Node* n);
bool is_stmt_op(Op o);  // Seq If While Return Print InlineBlock Break Free Nop Reserve
// Element type of a List(r) / Ptr(r) type.
inline Type elem_of(const Module& m, Type t) { return m.regions[t.a].elem; }
std::string type_str(const Module& m, Type t);
void dump(std::string& out, const Module& m);

// ---- pass manager ----------------------------------------------------------------
struct Pass {
    const char* name;
    const char* what;
    void (*run)(Module&);
};
// Global-variable passes (src/globals.cpp), DESIGN.md §2.6.
void pass_definite_assignment(Module& m);
void pass_constants(Module& m);
void pass_demote_globals(Module& m);
// Region analyses (src/analysis.cpp): Func::mutates, Local::unique, Region flags.
void pass_region_info(Module& m);
// Constant folding (src/tir.cpp).
void pass_fold(Module& m);
// Structural / type checker (src/tir_verify.cpp). On failure prints a precise
// message (function, op, expectation) to stderr and aborts.
void verify(Module& m, const char* after_pass);
// Pass manager (src/passes/registry.cpp). The registry lists every pass in
// pipeline order; `optimisation` passes are those --only-pass selects among.
struct PassOptions {
    std::vector<std::string> disabled;  // --no-pass / EPQ_NO_PASS
    std::vector<std::string> only;      // --only-pass: if non-empty, only these optimisation passes run
    std::vector<std::string> enabled;   // --enable-pass / EPQ_ENABLE_PASS: default-off passes to run
    bool verbose = false;
    bool verify = false;                // --verify-tir / EPQ_VERIFY_TIR=1: verify after every pass
    std::string dump_after;             // --dump-tir-after=NAME
};
// Returns true if --dump-tir-after fired (the dump went to stdout; caller stops).
bool run_passes(Module& m, const PassOptions& o);
// Compatibility form: skipping those named in `disabled`.
void run_passes(Module& m, const std::vector<std::string>& disabled, bool verbose);
const std::vector<Pass>& passes();
bool pass_is_optimisation(const char* name);
// Passes that are registered but OFF by default (registry.cpp kDefaultOff, with
// the reason); --enable-pass=NAME / EPQ_ENABLE_PASS=NAME (or --only-pass naming
// them) runs them.
bool pass_default_off(const char* name);

}  // namespace epq::tir
