#include "lexer.hpp"

#include <climits>

namespace epq {

std::string format_loc(std::string_view file, const Loc& loc) {
    std::string s(file);
    s += ':';
    s += std::to_string(loc.begin.line);
    s += '.';
    s += std::to_string(loc.begin.col);
    // bison: last = end - 1 column, with columns clamped at 1.
    uint32_t last_line = loc.end.line;
    uint32_t last_col = loc.end.col > 1 ? loc.end.col - 1 : 1;
    if (loc.begin.line != last_line) {
        s += '-';
        s += std::to_string(last_line);
        s += '.';
        s += std::to_string(last_col);
    } else if (loc.begin.col != last_col) {
        s += '-';
        s += std::to_string(last_col);
    }
    return s;
}

namespace {

// Character classes.
enum : uint8_t { C_OTHER, C_BLANK, C_NL, C_DIGIT, C_ALPHA, C_PUNCT };

struct CharTable {
    uint8_t cls[256];
    constexpr CharTable() : cls() {
        for (int i = 0; i < 256; ++i) cls[i] = C_OTHER;
        cls[(int)' '] = cls[(int)'\t'] = C_BLANK;
        cls[(int)'\n'] = C_NL;
        for (int c = '0'; c <= '9'; ++c) cls[c] = C_DIGIT;
        for (int c = 'a'; c <= 'z'; ++c) cls[c] = C_ALPHA;
        for (int c = 'A'; c <= 'Z'; ++c) cls[c] = C_ALPHA;
        const char* p = "-+*/();{}=!?:&|@,.#[]<>";
        for (; *p; ++p) cls[(unsigned char)*p] = C_PUNCT;
    }
};
constexpr CharTable kChars;

inline bool is_ident_char(unsigned char c) {
    uint8_t k = kChars.cls[c];
    return k == C_ALPHA || k == C_DIGIT || c == '_';
}

}  // namespace

void lex(const char* src, size_t len, Interner& syms, LexResult& out) {
    out.toks.clear();
    out.diags.clear();
    out.toks.reserve(len / 3 + 16);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(src);
    const unsigned char* end = p + len;

    // Emulation of the scanner's yylloc: `step()` makes begin = end; every
    // matched rule first advances end.col by its length (YY_USER_ACTION).
    Pos b, e;  // begin / end
    auto step = [&] { b = e; };

    for (;;) {
        step();  // done at each entry to yylex()
        for (;;) {
            if (p >= end) {
                out.toks.push_back(Tok{T_END, 0, 0, Loc{b, e}, (uint32_t)out.diags.size()});
                return;
            }
            unsigned char c = *p;
            switch (kChars.cls[c]) {
            case C_BLANK: {
                const unsigned char* q = p;
                while (q < end && (*q == ' ' || *q == '\t')) ++q;
                e.col += (uint32_t)(q - p);
                p = q;
                step();
                continue;
            }
            case C_NL: {
                const unsigned char* q = p;
                while (q < end && *q == '\n') ++q;
                e.line += (uint32_t)(q - p);
                e.col = 1;
                p = q;
                step();
                continue;
            }
            case C_DIGIT: {
                // strtol(yytext) then truncation to int: values above LONG_MAX
                // saturate to LONG_MAX (whose low 32 bits are -1).
                const unsigned char* q = p;
                unsigned long long v = 0;
                bool sat = false;
                while (q < end && kChars.cls[*q] == C_DIGIT) {
                    if (!sat) {
                        unsigned d = *q - '0';
                        if (v > ((unsigned long long)LONG_MAX - d) / 10) {
                            sat = true;
                            v = (unsigned long long)LONG_MAX;
                        } else {
                            v = v * 10 + d;
                        }
                    }
                    ++q;
                }
                e.col += (uint32_t)(q - p);
                int32_t iv = (int32_t)(uint32_t)(v & 0xffffffffu);
                out.toks.push_back(Tok{T_NUMBER, 0, iv, Loc{b, e}, (uint32_t)out.diags.size()});
                p = q;
                break;
            }
            case C_ALPHA: {
                const unsigned char* q = p + 1;
                while (q < end && is_ident_char(*q)) ++q;
                e.col += (uint32_t)(q - p);
                Sym s = syms.intern(std::string_view(reinterpret_cast<const char*>(p), (size_t)(q - p)));
                out.toks.push_back(Tok{T_ID, 0, (int32_t)s, Loc{b, e}, (uint32_t)out.diags.size()});
                p = q;
                break;
            }
            case C_PUNCT: {
                unsigned char n = (p + 1 < end) ? p[1] : 0;
                uint16_t k = c;
                int l = 1;
                switch (c) {
                case '/':
                    if (n == '/') {  // comment: "//".* (up to, not including, '\n')
                        const unsigned char* q = p + 2;
                        while (q < end && *q != '\n') ++q;
                        e.col += (uint32_t)(q - p);
                        p = q;
                        step();
                        continue;  // continues the scanning loop
                    }
                    break;
                case ':':
                    if (n == '=') k = T_ASSIGN, l = 2;
                    else if (n == '+') k = T_APPEND, l = 2;
                    break;
                case '>':
                    if (n == '>') k = T_PRINT, l = 2;
                    else if (n == '=') k = T_GE, l = 2;
                    else k = T_GT;
                    break;
                case '<':
                    if (n == '<') k = T_INPUT, l = 2;
                    else if (n == '=') k = T_LE, l = 2;
                    else k = T_LT;
                    break;
                case '!':
                    if (n == '=') k = T_NE, l = 2;
                    break;
                case '=':
                    if (n == '>') k = T_DEF, l = 2;
                    break;
                default:
                    break;
                }
                e.col += (uint32_t)l;
                out.toks.push_back(Tok{k, 0, 0, Loc{b, e}, (uint32_t)out.diags.size()});
                p += l;
                break;
            }
            default:
                // Invalid character: reported and skipped. Note the scanner does
                // not step() here, so the location keeps growing over a run of
                // invalid characters and into the next token. '\r' is skipped
                // silently (the original reports it like any other).
                e.col += 1;
                if (c != '\r') out.diags.push_back(LexDiag{Loc{b, e}});
                ++p;
                continue;
            }
            break;  // a token was produced
        }
    }
}

std::string token_name(uint16_t k) {
    switch (k) {
    case T_END: return "END";
    case T_ASSIGN: return "ASSIGN";
    case T_PRINT: return "PRINT";
    case T_DEF: return "DEF";
    case T_NUMBER: return "NUMBER";
    case T_ID: return "ID";
    case T_INPUT: return "INPUT";
    case T_GE: return "GE";
    case T_LE: return "LE";
    case T_NE: return "NE";
    case T_APPEND: return "APPEND";
    case T_GT: return "GT";
    case T_LT: return "LT";
    default: {
        std::string s = "'";
        s += (char)k;
        s += "'";
        return s;
    }
    }
}

}  // namespace epq
