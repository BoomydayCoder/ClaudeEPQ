// Parser for EPQ. The original grammar is LALR(1) with precedence declarations
// (lowest first):
//
//   nonassoc ID; right ASSIGN; nonassoc EXPLIST; left APPEND; nonassoc '@';
//   right '?' ':'; left '|'; left '&'; right '!'; left = > < >= <= !=;
//   left + -; left * /; right UMINUS; left '.'; left '['; left '('; left ABS
//
// bison 2.3/2.4 report *no* unresolved conflicts for it (every conflict is
// settled by precedence), and its tables are identical to the ones compiled
// into the baseline. The resolved conflicts and how this parser reproduces
// each of them (state numbers from `bison -v` on the grammar):
//
//  * Binary operators (states 69-81): ordinary left-associative precedence
//    climbing over the levels APPEND < '|' < '&' < comparisons < +- < */.
//  * '!' exp (27): reduces before APPEND '|' '&', shifts comparisons and
//    arithmetic, so the operand of '!' is an expression at comparison level:
//    `!a = b` is `!(a = b)`, `!a & b` is `(!a) & b` -- regardless of what
//    surrounds the '!'.
//  * '-' exp (28): reduces before every binary operator, shifts '.' '[' '(':
//    the operand is a unary/postfix expression.
//  * idtype ASSIGN exp (68): shifts everything, i.e. the right-hand side is a
//    full expression. ASSIGN is not in FOLLOW(exp), so an idtype followed by
//    ':=' always becomes an assignment, however tightly bound the left
//    context is: `a + x := 1` is `a + (x := 1)`, `-x := 1` is `-(x := 1)`.
//    A parenthesised expression is never an idtype.
//  * '|' exp '|' (60), the abs/or ambiguity: inside `| ... |`, when every
//    pending operator has reduced (we are back in the abs's own operator
//    loop) and the lookahead is '|', bison shifts it into a state that has
//    both `'|' exp '|' .` and `exp '|' . exp`. There, NUMBER ID INPUT '!' '#'
//    shift (the '|' was a logical or; these tokens are not in the reduce
//    lookahead set, so no conflict even exists) and every other token closes
//    the abs ('|' '-' '.' '[' '(' by precedence, the rest by default
//    reduction). A '|' seen while a pending APPEND or ASSIGN is the innermost
//    operator is always shifted as a logical or (69, 68), so `|x := 1|` and
//    `|a :+ b|` can never close.
//  * '(' exp . ')' (36, 57): shift; `(e) => {}` uses the one-parameter rule.
//    Both produce the same tree.
//  * '(' exp ')' '?' stmt . ':' (100): shift -- `else` binds to the nearest if.
//
// Statement-level '(' needs no backtracking either: see parse_paren_stmt.

#include "parser.hpp"

#include <vector>

namespace epq {

namespace {

struct Fail {};  // thrown on syntax error (err already filled)

enum : int { P_NONE = 0, P_APPEND = 1, P_OR = 2, P_AND = 3, P_CMP = 4, P_ADD = 5, P_MUL = 6 };

inline int binprec(uint16_t k) {
    switch (k) {
    case T_APPEND: return P_APPEND;
    case '|': return P_OR;
    case '&': return P_AND;
    case '=': case T_GT: case T_LT: case T_GE: case T_LE: case T_NE: return P_CMP;
    case '+': case '-': return P_ADD;
    case '*': case '/': return P_MUL;
    default: return P_NONE;
    }
}

inline NK binkind(uint16_t k) {
    switch (k) {
    case T_APPEND: return NK::Append;
    case '|': return NK::Or;
    case '&': return NK::And;
    case '=': return NK::Eq;
    case T_GT: return NK::Gt;
    case T_LT: return NK::Lt;
    case T_GE: return NK::Ge;
    case T_LE: return NK::Le;
    case T_NE: return NK::Ne;
    case '+': return NK::Add;
    case '-': return NK::Sub;
    case '*': return NK::Mul;
    default: return NK::Div;
    }
}

// After `'|' exp '|'`, these lookaheads make the second bar a logical or.
inline bool bar_is_or_before(uint16_t k) {
    return k == T_NUMBER || k == T_ID || k == T_INPUT || k == '!' || k == '#';
}

// An expression plus whether it is an `idtype` (can be assigned to).
struct Ex {
    Node* n;
    bool lvalue;
};

// Result of a statement-level '(' construct: a complete statement, or an
// expression primary (parenthesised expression / function literal) that the
// caller continues.
struct ParenResult {
    Node* n;
    bool is_stmt;
};

class Parser {
public:
    Parser(const LexResult& lx, Arena& a, SyntaxError& err, unsigned max_depth)
        : toks_(lx.toks.data()), ntoks_(lx.toks.size()), a_(a), err_(err), max_depth_(max_depth) {
        tmp_.reserve(256);
    }

    Node* program() {
        Node* seq = sequence(T_END);
        // tok is T_END here
        return seq;
    }

private:
    const Tok* toks_;
    size_t ntoks_;
    size_t i_ = 0;
    Arena& a_;
    SyntaxError& err_;
    std::vector<Node*> tmp_;  // child-collection stack shared by all lists
    unsigned max_depth_;
    unsigned depth_ = 0;

    // ---- token access ----------------------------------------------------
    uint16_t tok() const { return toks_[i_].kind; }
    uint16_t peek(size_t k) const {
        size_t j = i_ + k;
        return j < ntoks_ ? toks_[j].kind : (uint16_t)T_END;
    }
    const Tok& cur() const { return toks_[i_]; }
    void advance() {
        if (i_ + 1 < ntoks_) ++i_;
    }

    [[noreturn]] void error(const char* expecting = nullptr) {
        err_.tok_index = (uint32_t)i_;
        err_.loc = cur().loc;
        err_.msg = "syntax error, unexpected " + token_name(tok());
        if (expecting) {
            err_.msg += ", expecting ";
            err_.msg += expecting;
        }
        throw Fail{};
    }
    void expect(uint16_t k) {
        if (tok() != k) {
            std::string name = token_name(k);
            error(name.c_str());
        }
        advance();
    }

    // Recursion guard: pathological nesting would otherwise overflow the C
    // stack. Each source byte opens at most two guarded levels; main() sizes
    // the thread stack from the source length so that, below the 32 GB stack
    // cap, the limit (stack/1KB) is never reached by any input.
    struct DepthGuard {
        Parser& p;
        explicit DepthGuard(Parser& pp) : p(pp) {
            if (++p.depth_ > p.max_depth_) {
                p.err_.tok_index = (uint32_t)p.i_;
                p.err_.loc = p.cur().loc;
                p.err_.msg = "syntax error, nesting too deep";
                throw Fail{};
            }
        }
        ~DepthGuard() { --p.depth_; }
    };

    // ---- node construction ------------------------------------------------
    Node* mk(NK k, const Tok& at) {
        Node* n = a_.make<Node>();
        n->kind = k;
        n->line = at.loc.begin.line;
        n->col = at.loc.begin.col;
        return n;
    }
    Node* mk(NK k, const Node* at) {
        Node* n = a_.make<Node>();
        n->kind = k;
        n->line = at->line;
        n->col = at->col;
        return n;
    }
    void set_kids(Node* n, std::initializer_list<Node*> ks) {
        n->nkids = (uint32_t)ks.size();
        n->kids = a_.array<Node*>(ks.size());
        size_t j = 0;
        for (Node* k : ks) n->kids[j++] = k;
    }
    // Moves tmp_[mark..] into n's children.
    void take_kids(Node* n, size_t mark) {
        size_t cnt = tmp_.size() - mark;
        n->nkids = (uint32_t)cnt;
        n->kids = a_.array<Node*>(cnt);
        for (size_t j = 0; j < cnt; ++j) n->kids[j] = tmp_[mark + j];
        tmp_.resize(mark);
    }
    Node* empty_seq(const Tok& at) { return mk(NK::Seq, at); }

    // ---- statements ------------------------------------------------------
    // sequence: stmt* up to (not including) `term`.
    Node* sequence(uint16_t term) {
        Node* seq = mk(NK::Seq, cur());
        size_t mark = tmp_.size();
        while (tok() != term) tmp_.push_back(stmt());
        take_kids(seq, mark);
        return seq;
    }

    Node* block() {
        const Tok& at = cur();
        expect('{');
        Node* seq = sequence('}');
        advance();  // '}'
        Node* b = mk(NK::Block, at);
        set_kids(b, {seq});
        return b;
    }

    Node* stmt() {
        DepthGuard g(*this);
        const Tok& at = cur();
        switch (tok()) {
        case ';':
            advance();
            return empty_seq(at);
        case T_PRINT: {
            advance();
            Node* e = expr();
            expect(';');
            Node* n = mk(NK::Print, at);
            set_kids(n, {e});
            return n;
        }
        case '{':
            return block();
        case T_DEF: {
            advance();
            Node* e = expr();
            expect(';');
            Node* n = mk(NK::Return, at);
            set_kids(n, {e});
            return n;
        }
        case '(': {
            ParenResult r = paren_stmt();
            if (r.is_stmt) return r.n;
            Node* e = binary(P_APPEND, false, Ex{r.n, false}).n;
            return expr_stmt_end(e, at);
        }
        default: {
            Node* e = expr();
            return expr_stmt_end(e, at);
        }
        }
    }

    Node* expr_stmt_end(Node* e, const Tok& at) {
        expect(';');
        Node* n = mk(NK::ExprStmt, at);
        set_kids(n, {e});
        return n;
    }

    // A '(' where a statement may start. Possible readings:
    //   (e) ? s [: s]      (e) @ s      (x : e) @ s      (init cond; step) @ s
    //   or an expression statement starting with `(e)`, `(a,b) => {}`, `() => {}`.
    // The decision is deferred exactly as far as the LR parser defers it:
    //   - `( ID :`                        -> for-each
    //   - `( ;`, `( >>`, `( {`, `( =>`     -> for (init can only be a statement)
    //   - `( )`                            -> `() => block` expression
    //   - `( (`                            -> recurse: the inner construct is
    //        either a complete control statement (then it is the for-init) or
    //        an expression primary, which we continue into a full expression
    //   - otherwise parse an expression e, then the next token decides:
    //        ';' -> for with init `e;`;  ',' -> function literal;
    //        ')' -> then '?' if, '@' while, '=>' function literal, else `(e)`.
    ParenResult paren_stmt() {
        DepthGuard g(*this);
        const Tok& open = cur();
        advance();  // '('
        uint16_t t = tok();
        if (t == T_ID && peek(1) == ':') {
            Node* var = mk(NK::Id, cur());
            var->sym = (Sym)cur().val;
            advance();
            advance();  // ':'
            Node* list = expr();
            expect(')');
            expect('@');
            Node* body = stmt();
            Node* n = mk(NK::ForEach, open);
            set_kids(n, {var, list, body});
            return {n, true};
        }
        if (t == ';' || t == T_PRINT || t == '{' || t == T_DEF) {
            Node* init = stmt();
            return {for_rest(init, open), true};
        }
        if (t == ')') {
            advance();
            return {funclit_tail(open, tmp_.size()), false};
        }
        Node* e;
        if (t == '(') {
            ParenResult r = paren_stmt();
            if (r.is_stmt) return {for_rest(r.n, open), true};
            e = binary(P_APPEND, false, Ex{r.n, false}).n;
        } else {
            e = expr();
        }
        switch (tok()) {
        case ';': {
            advance();
            Node* init = mk(NK::ExprStmt, e);
            set_kids(init, {e});
            return {for_rest(init, open), true};
        }
        case ',': {
            size_t mark = tmp_.size();
            tmp_.push_back(e);
            while (tok() == ',') {
                advance();
                tmp_.push_back(expr());
            }
            expect(')');
            return {funclit_tail(open, mark), false};
        }
        case ')': {
            advance();
            uint16_t k = tok();
            if (k == '?') {
                advance();
                Node* s1 = stmt();
                Node* s2;
                if (tok() == ':') {  // dangling else: shift (state 100)
                    advance();
                    s2 = stmt();
                } else {
                    s2 = empty_seq(cur());
                }
                Node* n = mk(NK::If, open);
                set_kids(n, {e, s1, s2});
                return {n, true};
            }
            if (k == '@') {
                advance();
                Node* body = stmt();
                Node* n = mk(NK::While, open);
                set_kids(n, {e, body});
                return {n, true};
            }
            if (k == T_DEF) {
                size_t mark = tmp_.size();
                tmp_.push_back(e);
                return {funclit_def(open, mark), false};
            }
            return {e, false};
        }
        default:
            error();
        }
    }

    // After the for-init statement: `cond ; step ) @ body`.
    Node* for_rest(Node* init, const Tok& open) {
        Node* cond = expr();
        expect(';');
        Node* step = stmt();
        expect(')');
        expect('@');
        Node* body = stmt();
        Node* n = mk(NK::For, open);
        set_kids(n, {init, cond, step, body});
        return n;
    }

    // `=> block` after a closed parameter list held in tmp_[mark..].
    Node* funclit_tail(const Tok& open, size_t mark) {
        if (tok() != T_DEF) {
            tmp_.resize(mark);
            error("DEF");
        }
        return funclit_def(open, mark);
    }
    Node* funclit_def(const Tok& open, size_t mark) {
        advance();  // '=>'
        Node* params = mk(NK::List, open);
        take_kids(params, mark);
        Node* body = block();
        Node* n = mk(NK::FuncLit, open);
        set_kids(n, {params, body});
        return n;
    }

    // ---- expressions -----------------------------------------------------
    Node* expr() { return binary(P_APPEND, false, Ex{nullptr, false}).n; }

    // Precedence climbing over the binary levels >= min_prec. `in_abs` is true
    // only for the operator loop directly inside `| ... |` (see header comment).
    // If `pre.n` is set it is an already-parsed primary that starts the
    // expression (used for statement-level parentheses).
    Ex binary(int min_prec, bool in_abs, Ex pre) {
        DepthGuard g(*this);
        Ex lhs = unary(pre);
        for (;;) {
            uint16_t op = tok();
            int p = binprec(op);
            if (p == P_NONE || p < min_prec) break;
            if (op == '|' && in_abs && !bar_is_or_before(peek(1))) break;  // closes the abs
            advance();
            // Every right operand is parsed with in_abs = false: for operators
            // above '|' it cannot matter (a '|' ends the operand anyway); for
            // APPEND, a '|' inside its right operand is always a logical or.
            Ex rhs = binary(p + 1, false, Ex{nullptr, false});
            Node* n = mk(binkind(op), lhs.n);
            set_kids(n, {lhs.n, rhs.n});
            lhs = Ex{n, false};
        }
        return lhs;
    }

    Ex unary(Ex pre) {
        if (pre.n) return postfix(pre);
        const Tok& at = cur();
        switch (tok()) {
        case '!': {
            advance();
            Node* x = binary(P_CMP, false, Ex{nullptr, false}).n;
            Node* n = mk(NK::Not, at);
            set_kids(n, {x});
            return {n, false};
        }
        case '-': {
            advance();
            DepthGuard g(*this);
            Node* x = unary(Ex{nullptr, false}).n;
            Node* n = mk(NK::Neg, at);
            set_kids(n, {x});
            return {n, false};
        }
        default:
            return postfix(primary());
        }
    }

    // Postfix chain: calls, indexing, member access; then `:=` if the result
    // is an idtype.
    Ex postfix(Ex e) {
        for (;;) {
            uint16_t k = tok();
            if (k == '(') {
                const Tok& at = cur();
                advance();
                Node* args = explist(')', at);
                expect(')');
                Node* n = mk(NK::Call, e.n);
                set_kids(n, {e.n, args});
                e = {n, false};
            } else if (k == '[') {
                advance();
                Node* idx = expr();
                expect(']');
                Node* n = mk(NK::Index, e.n);
                set_kids(n, {e.n, idx});
                e = {n, true};
            } else if (k == '.') {
                advance();
                if (tok() != T_ID) error("ID");
                Node* id = mk(NK::Id, cur());
                id->sym = (Sym)cur().val;
                advance();
                Node* n = mk(NK::Get, e.n);
                set_kids(n, {e.n, id});
                e = {n, true};
            } else {
                break;
            }
        }
        if (e.lvalue && tok() == T_ASSIGN) {
            advance();
            Node* rhs = expr();
            Node* n = mk(NK::Assign, e.n);
            set_kids(n, {e.n, rhs});
            return {n, false};
        }
        return e;
    }

    // explist up to (not including) `close`.
    Node* explist(uint16_t close, const Tok& at) {
        Node* l = mk(NK::List, at);
        size_t mark = tmp_.size();
        if (tok() != close) {
            tmp_.push_back(expr());
            while (tok() == ',') {
                advance();
                tmp_.push_back(expr());
            }
        }
        take_kids(l, mark);
        return l;
    }

    Ex primary() {
        DepthGuard g(*this);
        const Tok& at = cur();
        switch (tok()) {
        case T_NUMBER: {
            Node* n = mk(NK::Int, at);
            n->ival = at.val;
            advance();
            return {n, false};
        }
        case T_ID: {
            Node* n = mk(NK::Id, at);
            n->sym = (Sym)at.val;
            advance();
            return {n, true};
        }
        case T_INPUT: {
            advance();
            return {mk(NK::Input, at), false};
        }
        case '[': {
            advance();
            Node* l = explist(']', at);
            expect(']');
            return {l, false};
        }
        case '|': {
            advance();
            Node* x = binary(P_APPEND, true, Ex{nullptr, false}).n;
            expect('|');
            Node* n = mk(NK::Abs, at);
            set_kids(n, {x});
            return {n, false};
        }
        case '.': {
            advance();
            if (tok() != T_ID) error("ID");
            Node* id = mk(NK::Id, cur());
            id->sym = (Sym)cur().val;
            advance();
            Node* n = mk(NK::SGet, at);
            set_kids(n, {id});
            return {n, true};
        }
        case '#': {
            advance();
            if (tok() != '{') error("'{'");
            Node* b = block();
            Node* n = mk(NK::Class, at);
            set_kids(n, {b});
            return {n, false};
        }
        case '(':
            return {paren_expr(), false};
        default:
            error();
        }
    }

    // '(' in expression context: `(e)`, `(e) => {}`, `(a, b, ...) => {}`, `() => {}`.
    Node* paren_expr() {
        const Tok& open = cur();
        advance();
        size_t mark = tmp_.size();
        if (tok() == ')') {
            advance();
            return funclit_tail(open, mark);
        }
        Node* e = expr();
        if (tok() == ',') {
            tmp_.push_back(e);
            while (tok() == ',') {
                advance();
                tmp_.push_back(expr());
            }
            expect(')');
            return funclit_tail(open, mark);
        }
        expect(')');
        if (tok() == T_DEF) {
            tmp_.push_back(e);
            return funclit_def(open, mark);
        }
        return e;  // parenthesised: not an idtype
    }
};

}  // namespace

Node* parse_program(const LexResult& lx, Arena& arena, SyntaxError& err, unsigned max_depth) {
    Parser p(lx, arena, err, max_depth);
    try {
        return p.program();
    } catch (const Fail&) {
        return nullptr;
    }
}

}  // namespace epq
