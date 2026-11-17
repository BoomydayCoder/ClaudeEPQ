// Resolver: LANGUAGE.md section 4 (globals, locals, scopes) and the original's
// compile-time errors, reproduced in the order the original compiler
// (reference/EPQ/compiler.cc) would hit them.
//
// Output, all attached to / indexed from the AST:
//   * every Id use and assignment target: Node::flags has NF_GLOBAL or NF_LOCAL
//     and Node::aux is the global index or the local slot of the function the
//     Id appears in (lexically: the innermost enclosing FuncLit, or main);
//   * NF_DEFINE on the Id occurrence that creates a local, NF_UNCLEAN when that
//     definition is not at a clean position (LANGUAGE.md 9.1);
//   * every FuncLit: Node::aux = its DefId (1..N in compile pre-order; DefId 0
//     is the top-level program, "main");
//   * every ForEach: Node::aux = index into Program::foreach (hidden slots).
#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "ast.hpp"

namespace epq {

using DefId = uint32_t;
constexpr DefId kMainDef = 0;

enum class ScopeKind : uint8_t {
    Function,     // outermost scope of a function (parameters; main's top level)
    Block,        // { ... }
    ListLit,      // [ ... ]
    IfStmt,       // whole if statement (covers the condition)
    IfThen,
    IfElse,
    WhileStmt,    // covers the condition
    WhileBody,
    ForStmt,      // covers init, condition, step
    ForBody,
    ForEachStmt,  // covers the hidden variables and a new loop variable
    ForEachBody,
};
const char* scope_kind_name(ScopeKind k);

enum : uint8_t {
    LF_PARAM = 1 << 0,    // function parameter
    LF_HIDDEN = 1 << 1,   // for-each hidden index / list variable (no name)
    LF_UNCLEAN = 1 << 2,  // defined at an unclean position: zero-initialise at scope entry
    LF_FOREACH_VAR = 1 << 3,  // created by a for-each as its loop variable
};

struct LocalVar {
    Sym name;          // kNoSym for hidden variables
    uint32_t scope;    // index into Function::scopes of the defining scope
    uint8_t flags;     // LF_*
    const Node* site;  // defining Id node (param Id, assignment target, for-each var);
                       // the ForEach node for hidden variables
};

struct Scope {
    int32_t parent;     // -1 for the function scope
    ScopeKind kind;
    const Node* opener; // node that opened it (FuncLit/nullptr for Function scope)
};

struct Function {
    DefId id;
    int32_t parent;            // lexically enclosing DefId (-1 for main)
    const Node* node;          // the FuncLit (nullptr for main)
    const Node* body;          // FuncLit's Block, or the program's Seq for main
    std::vector<uint32_t> params;   // local slots of the parameters, in order
    std::vector<LocalVar> locals;   // slot -> variable (slots are never reused)
    std::vector<Scope> scopes;      // scope 0 is the Function scope
};

struct ForEachInfo {
    DefId fn;               // function containing the loop
    uint32_t index_slot;    // hidden `i`
    uint32_t list_slot;     // hidden `L`
    bool var_preexisted;    // loop variable was an existing global/local
                            // (then `x := 0` overwrites it before the loop)
};

struct Program {
    Node* root = nullptr;             // the top-level Seq
    const Interner* syms = nullptr;
    std::vector<Sym> globals;         // global index -> name (discovery order)
    std::vector<int32_t> global_of;   // Sym -> global index or -1
    std::vector<Function> funcs;      // DefId -> function; funcs[0] is main
    std::vector<ForEachInfo> foreach;
};

struct ResolveError {
    std::string msg;  // exact message of the original, e.g. "Undefined variable x"
    uint32_t line = 0, col = 0;
};

// Resolves `root` in place. Returns false and fills `err` on the first
// compile-time error.
bool resolve(Node* root, const Interner& syms, Program& out, ResolveError& err);

// Human-readable dump for --dump-resolved.
void dump_resolved(std::ostream& os, const Program& p);

}  // namespace epq
