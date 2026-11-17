// Lexer: a hand-written equivalent of the original scanner.ll (flex).
//
// The whole source is tokenised up front into a flat array (the parser needs
// two tokens of lookahead in a couple of places). Diagnostics for invalid
// characters are recorded, not printed: the original prints them lazily as the
// parser pulls tokens, so a syntax error stops them. Tok::nerr_before says how
// many diagnostics precede a token, letting the driver print exactly the ones
// the original would have printed.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"

namespace epq {

// Token kinds. Single-character tokens use their character code (as bison does);
// multi-character tokens and classes use values >= 256.
enum Tk : uint16_t {
    T_END = 0,
    T_ASSIGN = 256,  // :=
    T_PRINT,         // >>
    T_DEF,           // =>
    T_NUMBER,
    T_ID,
    T_INPUT,   // <<
    T_GE,      // >=
    T_LE,      // <=
    T_NE,      // !=
    T_APPEND,  // :+
    T_GT,      // >
    T_LT,      // <
};

struct Tok {
    uint16_t kind;
    uint16_t pad_ = 0;
    int32_t val;           // NUMBER: value (strtol-then-truncate); ID: Sym
    Loc loc;               // bison location (used in messages)
    uint32_t nerr_before;  // number of lexer diagnostics emitted before this token
};

struct LexDiag {
    Loc loc;  // message is always "invalid character"
};

struct LexResult {
    std::vector<Tok> toks;  // always terminated by a T_END token
    std::vector<LexDiag> diags;
};

// Tokenises [src, src+len). Identifiers are interned into `syms`.
void lex(const char* src, size_t len, Interner& syms, LexResult& out);

// Bison-style token name for messages: "ID", "';'", "END", ...
std::string token_name(uint16_t kind);

}  // namespace epq
