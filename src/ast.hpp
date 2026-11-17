// AST: compact arena-allocated nodes, one shape for every kind.
//
// Node layout (32 bytes): kind, flags, source position, one 32-bit payload
// (integer value or interned identifier), one 32-bit resolver slot (`aux`),
// and an arena array of children. Child conventions per kind are listed below;
// they mirror the original's Ast (exptree.hh) closely, which keeps the `-e`
// dump trivially byte-identical.
#pragma once

#include <cstdint>
#include <ostream>

#include "common.hpp"

namespace epq {

enum class NK : uint8_t {
    // ---- statements -------------------------------------------------------
    Seq,       // kids: statements. Also the empty statement ';' and an absent else.
    ExprStmt,  // [expr]                       `e;`
    Print,     // [expr]                       `>> e;`
    Block,     // [Seq]                        `{ ... }`
    If,        // [cond, then, else]           else is an empty Seq when absent
    While,     // [cond, body]
    For,       // [init, cond, step, body]     init/step are statements
    ForEach,   // [var(Id), list, body]        `(x : e) @ body`
    Return,    // [expr]                       `=> e;`
    // ---- expressions ------------------------------------------------------
    Assign,    // [target, value]  target is Id | Index | Get | SGet
    Append,    // [list, value]    `:+`
    Add, Sub, Mul, Div,
    Eq, Gt, Lt, Ge, Le, Ne,
    Or, And,   // short-circuit `|`, `&`
    Not,       // [x]  `!`
    Neg,       // [x]  unary `-`
    Abs,       // [x]  `|x|`
    Int,       // ival
    List,      // kids: elements. List literal `[...]`; also the parameter list of
               // FuncLit and the argument list of Call (dumped the same way).
    Id,        // sym
    Index,     // [object, index]
    Input,     // `<<`
    FuncLit,   // [params(List of Id), body(Block)]   aux = DefId (after resolve)
    Call,      // [callee, args(List)]
    Class,     // [Block]          `# { ... }`      (rejected by the resolver)
    Get,       // [object, Id]     `e.x`            (rejected by the resolver)
    SGet,      // [Id]             `.x`             (rejected by the resolver)
};

const char* nk_name(NK k);

// Node::flags bits (set by the resolver).
enum : uint8_t {
    NF_GLOBAL = 1 << 0,   // Id: bound to global `aux`
    NF_LOCAL = 1 << 1,    // Id: bound to local slot `aux` of the enclosing function
    NF_DEFINE = 1 << 2,   // Id (assignment target / for-each var): this occurrence
                          // creates the local `aux` (its first definition point)
    NF_UNCLEAN = 1 << 3,  // Id with NF_DEFINE: definition is not at a "clean"
                          // position (LANGUAGE.md 9.1); zero-initialise the slot
                          // at scope entry
};

struct Node {
    NK kind;
    uint8_t flags = 0;
    uint16_t pad_ = 0;
    uint32_t line = 0;   // source position of the node's first token
    uint32_t col = 0;
    uint32_t nkids = 0;
    union {
        int32_t ival;  // Int
        Sym sym;       // Id
    };
    int32_t aux = -1;  // resolver: Id -> global index / local slot; FuncLit -> DefId;
                       // ForEach -> index into Program::foreach
    Node** kids = nullptr;

    Node() : kind(NK::Seq), ival(0) {}
    Node* kid(unsigned i) const { return kids[i]; }
};
static_assert(sizeof(Node) == 32, "keep Node compact");

inline bool is_statement(NK k) { return k <= NK::Return; }

// Dumps `n` in the exact format of the original's Ast::print_self (`-e`).
void dump_ast(std::ostream& os, const Node* n, const Interner& syms);

}  // namespace epq
