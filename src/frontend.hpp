// Front-end driver: source loading, lexing, parsing, diagnostics.
#pragma once

#include <string>
#include <string_view>

#include "ast.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "resolver.hpp"

namespace epq {

// Owns everything the front end produces. Nodes live in `arena`, names in `syms`.
struct CompileUnit {
    std::string file;    // name used in messages
    std::string source;
    Interner syms;
    Arena arena;
    LexResult lex;
    Node* root = nullptr;
    Program prog;
    // C stack available to the thread running the front end; bounds the
    // parser's recursion depth (see parse_program). main() sizes it from the
    // source length and runs the pipeline on a big-stack thread if needed.
    size_t stack_bytes = 8u << 20;
};

// Reads `path` ("-" = stdin) into cu.source. On failure prints
// "cannot open PATH" to stderr and returns false.
bool load_source(CompileUnit& cu, const std::string& path);

// Lexes and parses cu.source. Prints the lexer's "invalid character"
// diagnostics (only those the original would print before stopping) and, on
// failure, "FILE:LOC: syntax error, ..." to stderr. Returns false on a syntax error.
bool parse_unit(CompileUnit& cu);

// Resolves cu.root into cu.prog. On failure prints the original's message
// (e.g. "Undefined variable x") to stderr and returns false.
bool resolve_unit(CompileUnit& cu);

}  // namespace epq
