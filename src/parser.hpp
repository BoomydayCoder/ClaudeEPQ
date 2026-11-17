// Parser: hand-written recursive descent (statements) + precedence climbing
// (expressions) accepting exactly the language of the original bison grammar
// (reference/EPQ/parser.yy) with bison's conflict resolutions. See parser.cpp
// for the mapping from LR conflict states to parser decisions.
#pragma once

#include <string>

#include "ast.hpp"
#include "lexer.hpp"

namespace epq {

struct SyntaxError {
    uint32_t tok_index = 0;  // index of the offending token in LexResult::toks
    Loc loc;                 // its location
    std::string msg;         // "syntax error, unexpected X[, expecting Y]"
};

// Parses the token stream into a Seq node (the program). Returns nullptr and
// fills `err` on a syntax error. `max_depth` bounds parser recursion (each
// unit costs ~200 bytes of C stack, ~4x that under ASan; a source of N bytes
// needs at most 2N units); exceeding it is reported as
// "syntax error, nesting too deep".
Node* parse_program(const LexResult& lx, Arena& arena, SyntaxError& err, unsigned max_depth);

}  // namespace epq
