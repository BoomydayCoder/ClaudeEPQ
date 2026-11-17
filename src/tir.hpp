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

enum class TK : uint8_t { Bot, Int, List, Fn, Any };

struct Type {
    TK k = TK::Bot;
    uint32_t a = 0;  // List: region id; Fn: DefId
    static Type bot() { return {}; }
    static Type i32() { return {TK::Int, 0}; }
    static Type any() { return {TK::Any, 0}; }
    static Type list(uint32_t r) { return {TK::List, r}; }
    static Type fn(uint32_t d) { return {TK::Fn, d}; }
    bool operator==(const Type& o) const { return k == o.k && (k != TK::List && k != TK::Fn ? true : a == o.a); }
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

enum class LocalKind : uint8_t { Param, User, Hidden, Temp, Demoted };

struct Local {
    Type type;
    LocalKind kind = LocalKind::User;
    Sym name = kNoSym;     // source name (kNoSym for hidden / temps)
    int32_t global = -1;   // Demoted: the global it replaces
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
};

struct Global {
    Type type;
    Sym name = kNoSym;
    bool demoted = false;  // replaced by a local of main (no storage)
    bool ever_checked = false;  // some GetGlobal of it has F_CHECKED
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
};

// Static type descriptors (runtime EDesc). Index 0 INT, 1 ANY, 2 LIST(ANY), 3 FN.
struct Desc {
    uint32_t kind;  // 0 INT, 1 ANY, 2 FN, 3 LIST (runtime EPQ_K_*)
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
std::string type_str(const Module& m, Type t);
void dump(std::string& out, const Module& m);

// ---- pass manager ----------------------------------------------------------------
struct Pass {
    const char* name;
    const char* what;
    void (*run)(Module&);
};
// Runs the registered passes in order, skipping those named in `disabled`.
void run_passes(Module& m, const std::vector<std::string>& disabled, bool verbose);
const std::vector<Pass>& passes();

}  // namespace epq::tir
