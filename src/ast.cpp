#include "ast.hpp"

#include <string>

namespace epq {

const char* nk_name(NK k) {
    switch (k) {
    case NK::Seq: return "Seq";
    case NK::ExprStmt: return "ExprStmt";
    case NK::Print: return "Print";
    case NK::Block: return "Block";
    case NK::If: return "If";
    case NK::While: return "While";
    case NK::For: return "For";
    case NK::ForEach: return "ForEach";
    case NK::Return: return "Return";
    case NK::Assign: return "Assign";
    case NK::Append: return "Append";
    case NK::Add: return "Add";
    case NK::Sub: return "Sub";
    case NK::Mul: return "Mul";
    case NK::Div: return "Div";
    case NK::Eq: return "Eq";
    case NK::Gt: return "Gt";
    case NK::Lt: return "Lt";
    case NK::Ge: return "Ge";
    case NK::Le: return "Le";
    case NK::Ne: return "Ne";
    case NK::Or: return "Or";
    case NK::And: return "And";
    case NK::Not: return "Not";
    case NK::Neg: return "Neg";
    case NK::Abs: return "Abs";
    case NK::Int: return "Int";
    case NK::List: return "List";
    case NK::Id: return "Id";
    case NK::Index: return "Index";
    case NK::Input: return "Input";
    case NK::FuncLit: return "FuncLit";
    case NK::Call: return "Call";
    case NK::Class: return "Class";
    case NK::Get: return "Get";
    case NK::SGet: return "SGet";
    }
    return "?";
}

namespace {

// Mirrors exptree.cc Ast::print_self case by case.
struct Dumper {
    std::string& o;
    const Interner& syms;

    void list_items(const Node* l) {
        for (uint32_t i = 0; i < l->nkids; ++i) {
            dump(l->kids[i]);
            if (i != l->nkids - 1) o += ", ";
        }
    }

    void binop(const Node* n, const char* op) {
        o += '(';
        dump(n->kids[0]);
        o += ' ';
        o += op;
        o += ' ';
        dump(n->kids[1]);
        o += ')';
    }

    void dump(const Node* n) {
        switch (n->kind) {
        case NK::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) {
                dump(n->kids[i]);
                o += '\n';
            }
            break;
        case NK::ExprStmt: dump(n->kids[0]); break;
        case NK::Add: binop(n, "+"); break;
        case NK::Sub: binop(n, "-"); break;
        case NK::Mul: binop(n, "*"); break;
        case NK::Div: binop(n, "/"); break;
        case NK::Assign: binop(n, ":="); break;
        case NK::Eq: binop(n, "=="); break;
        case NK::And: binop(n, "&&"); break;
        case NK::Or: binop(n, "||"); break;
        case NK::Gt: binop(n, ">"); break;
        case NK::Lt: binop(n, "<"); break;
        case NK::Ge: binop(n, ">="); break;
        case NK::Le: binop(n, "<="); break;
        case NK::Ne: binop(n, "!="); break;
        case NK::Append: binop(n, ":+"); break;
        case NK::Neg: o += '-'; dump(n->kids[0]); break;
        case NK::Int: o += std::to_string(n->ival); break;
        case NK::List: o += '['; list_items(n); o += ']'; break;
        case NK::Id: o += syms.name(n->sym); break;
        case NK::Index:
            dump(n->kids[0]);
            o += '[';
            dump(n->kids[1]);
            o += ']';
            break;
        case NK::Print: o += "print "; dump(n->kids[0]); break;
        case NK::Block: o += "{\n"; dump(n->kids[0]); o += '}'; break;
        case NK::Input: o += "input"; break;
        case NK::Not: o += '!'; dump(n->kids[0]); break;
        case NK::If:
            o += "if ";
            dump(n->kids[0]);
            o += " then ";
            dump(n->kids[1]);
            o += " else ";
            dump(n->kids[2]);
            break;
        case NK::While:
            o += "while ";
            dump(n->kids[0]);
            o += " do ";
            dump(n->kids[1]);
            break;
        case NK::For:
            o += "for [";
            dump(n->kids[0]);
            o += ", ";
            dump(n->kids[1]);
            o += ", ";
            dump(n->kids[2]);
            o += "] do ";
            dump(n->kids[3]);
            break;
        case NK::ForEach:
            o += "for [";
            dump(n->kids[0]);
            o += " in ";
            dump(n->kids[1]);
            o += "] do ";
            dump(n->kids[2]);
            break;
        case NK::Call:
            dump(n->kids[0]);
            o += '(';
            list_items(n->kids[1]);
            o += ')';
            break;
        case NK::FuncLit:
            o += '(';
            list_items(n->kids[0]);
            o += ") => ";
            dump(n->kids[1]);
            break;
        case NK::Return: o += "return "; dump(n->kids[0]); break;
        case NK::Abs: o += '|'; dump(n->kids[0]); o += '|'; break;
        case NK::Class: o += "class "; dump(n->kids[0]); break;
        case NK::Get:
            dump(n->kids[0]);
            o += '.';
            dump(n->kids[1]);
            break;
        case NK::SGet: o += '.'; dump(n->kids[0]); break;
        }
    }
};

}  // namespace

void dump_ast(std::ostream& os, const Node* n, const Interner& syms) {
    std::string out;
    out.reserve(4096);
    Dumper d{out, syms};
    d.dump(n);
    os.write(out.data(), (std::streamsize)out.size());
}

}  // namespace epq
