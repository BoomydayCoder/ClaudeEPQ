#include "cgen.hpp"

#include <cstdio>
#include <cstdlib>

namespace epq {

using namespace tir;

namespace {

using Node = tir::Node;
using Type = tir::Type;

#include "epq_rt_src.inc"

struct CGen {
    const Module& M;
    std::string out;
    const Func* F = nullptr;
    int tmpn = 0;
    int labeln = 0;                                   // C labels (InlineBlock), per function
    std::vector<std::pair<int32_t, int>> blocks;     // enclosing InlineBlocks: (TIR label, C label)

    explicit CGen(const Module& m) : M(m) {}

    std::string ctype(Type t) const {
        switch (t.k) {
        case TK::List: return "EList*";
        case TK::Any: return "EAny";
        case TK::Pair: return "EPair";
        case TK::Ptr: return std::string(elem_struct(t.a)) + "*";
        case TK::Rec: return "ERec*";  // records (cgen_rec.inc)
        default: return "int32_t";
        }
    }
    // C element struct of lists of region r (runtime/epq_rt.h): EU8 / EI32 / EW64.
    const char* elem_struct(uint32_t r) const {
        uint32_t w = M.elem_width(r);
        return w == 1 ? "EU8" : w == 4 ? "EI32" : "EW64";
    }
    std::string tmp() { return "_t" + std::to_string(tmpn++); }
    static std::string num(int64_t v) {
        if (v == INT32_MIN) return "(-2147483647-1)";
        return std::to_string(v);
    }
    static std::string se(const std::string& pre, const std::string& e) {
        if (pre.empty()) return e;
        return "({ " + pre + e + "; })";
    }
    static bool is_const(const Node* n) { return n->op == Op::ConstInt || n->op == Op::ConstFn; }

    // Evaluate kids [from, to) honouring the TIR evaluation order. All but the
    // last non-constant kid go to temporaries when there is any impure kid.
    struct Kids {
        std::string pre;
        std::vector<std::string> v;
    };
    Kids kids(const Node* n, uint32_t from, uint32_t to, bool all_temps = false, bool force = false) {
        Kids k;
        bool any_impure = force;
        int last_nc = -1;
        for (uint32_t i = from; i < to; ++i) {
            if (!is_pure(n->kids[i])) any_impure = true;
            if (!is_const(n->kids[i])) last_nc = (int)i;
        }
        for (uint32_t i = from; i < to; ++i) {
            const Node* c = n->kids[i];
            std::string e = E(c);
            bool spill = (any_impure && !is_const(c) && ((int)i != last_nc || all_temps));
            if (spill) {
                std::string t = tmp();
                k.pre += ctype(c->type) + " " + t + " = " + e + "; ";
                k.v.push_back(t);
            } else {
                k.v.push_back(e);
            }
        }
        return k;
    }
    // Value of kid (of static type t, as C expression e) as an EAny word for
    // descriptor-driven helpers / word-layout element slots.
    std::string word(Type t, const std::string& e, bool pure) {
        switch (t.k) {
        case TK::Int:
        case TK::Bot: return "((EAny)(uint32_t)(" + e + "))";
        case TK::List: return "((EAny)(uintptr_t)(" + e + "))";
        case TK::Rec: return "((EAny)(uintptr_t)(" + e + "))";  // records
        case TK::Fn:
            if (pure) return "epq_box_fn(" + std::to_string(t.a) + ")";
            return "((void)(" + e + "), epq_box_fn(" + std::to_string(t.a) + "))";
        case TK::Any: return e;
        case TK::Pair: return "((EAny)(" + e + "))";
        case TK::Ptr: break;
        }
        return e;
    }
    // Boxed (Any) form of an Int|Any operand.
    std::string any_of(Type t, const std::string& e) {
        if (t.k == TK::Any) return e;
        return "epq_box_int(" + e + ")";
    }
    // A word loaded from an 8-byte slot, as a value of static type t.
    std::string from_word(Type t, const std::string& w) {
        switch (t.k) {
        case TK::List: return "((EList*)(uintptr_t)(" + w + "))";
        case TK::Rec: return "((ERec*)(uintptr_t)(" + w + "))";  // records
        case TK::Any: return w;
        case TK::Fn: return "((void)(" + w + "), 0)";
        case TK::Pair: return "((EPair)(" + w + "))";
        default: return "((int32_t)(" + w + "))";
        }
    }
    std::string truthy(Type t, const std::string& e) {
        switch (t.k) {
        case TK::Int: return "(" + e + " != 0)";
        case TK::Any: return "(" + e + " != 1)";
        case TK::Bot: return "((void)(" + e + "), 0)";
        default: return "((void)(" + e + "), 1)";
        }
    }
    std::string err(int32_t e) { return std::to_string(e); }

    std::string C(const Node* n) {
        switch (n->op) {
        case Op::And:
            if (n->flags & F_COND) return "(" + C(n->kids[0]) + " && " + C(n->kids[1]) + ")";
            break;
        case Op::Or:
            if (n->flags & F_COND) return "(" + C(n->kids[0]) + " || " + C(n->kids[1]) + ")";
            break;
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
        case Op::NotI32:
            return E(n);
        case Op::ConstInt:
            return n->a ? "1" : "0";
        default:
            break;
        }
        if (is_pure(n) && n->type.k != TK::Int && n->type.k != TK::Any) return n->type.k == TK::Bot ? "0" : "1";
        return truthy(n->type, E(n));
    }

    std::string E(const Node* n) {
        switch (n->op) {
        case Op::ConstInt: return num(n->a);
        case Op::ConstFn: return "0";
        case Op::Local: return "l" + std::to_string(n->a);
        case Op::GetGlobal: {
            std::string g = "g" + std::to_string(n->a);
            if (!(n->flags & F_CHECKED)) return g;
            return "({ if (EPQ_UNLIKELY(!gd" + std::to_string(n->a) + ")) epq_fail(EPQ_E_UNDEF_GLOBAL); " + g + "; })";
        }
        case Op::Input: return "epq_read_int()";
        case Op::SetLocal: return "(l" + std::to_string(n->a) + " = " + E(n->kids[0]) + ")";
        case Op::SetGlobal: {
            std::string g = std::to_string(n->a);
            return "(g" + g + " = " + E(n->kids[0]) + ", gd" + g + " = 1, g" + g + ")";
        }
        case Op::Box: {
            const Node* k = n->kids[0];
            Type t = k->type;
            if (t.k == TK::Int) return "epq_box_int(" + E(k) + ")";
            return word(t, E(k), is_pure(k));
        }
        case Op::UnboxInt: return "epq_unbox_int(" + E(n->kids[0]) + ", " + err(n->a) + ")";
        case Op::Coerce: return "({ (void)(" + E(n->kids[0]) + "); (" + ctype(n->type) + ")0; })";
        case Op::Comma: {
            std::string s = "({ ";
            for (uint32_t i = 0; i + 1 < n->nkids; ++i) s += "(void)(" + E(n->kids[i]) + "); ";
            s += E(n->kids[n->nkids - 1]) + "; })";
            return s;
        }
        case Op::Fail: {
            std::string s = "({ ";
            for (uint32_t i = 0; i < n->nkids; ++i) s += "(void)(" + E(n->kids[i]) + "); ";
            return s + "epq_fail(" + err(n->a) + "); (int32_t)0; })";
        }
        case Op::AddI32: case Op::SubI32: case Op::MulI32:
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32: {
            static const char* sym[] = {"+", "-", "*", "==", "!=", "<", "<=", ">", ">="};
            int idx = n->op == Op::AddI32 ? 0 : n->op == Op::SubI32 ? 1 : n->op == Op::MulI32 ? 2
                    : n->op == Op::EqI32 ? 3 : n->op == Op::NeI32 ? 4 : n->op == Op::LtI32 ? 5
                    : n->op == Op::LeI32 ? 6 : n->op == Op::GtI32 ? 7 : 8;
            Kids k = kids(n, 0, 2);
            return se(k.pre, "(" + k.v[0] + " " + sym[idx] + " " + k.v[1] + ")");
        }
        case Op::DivI32: {
            Kids k = kids(n, 0, 2);
            return se(k.pre, "epq_div_i32(" + k.v[0] + ", " + k.v[1] + ")");
        }
        case Op::NegI32: return "epq_neg_i32(" + E(n->kids[0]) + ")";
        case Op::AbsI32: return "epq_abs_i32(" + E(n->kids[0]) + ")";
        case Op::NotI32: return "(" + E(n->kids[0]) + " == 0)";
        case Op::DynArith: return dyn_arith(n);
        case Op::DynNeg: return "epq_dyn_neg(" + E(n->kids[0]) + ")";
        case Op::DynAbs: return "epq_dyn_abs(" + E(n->kids[0]) + ")";
        case Op::DynNot: return "epq_dyn_not(" + E(n->kids[0]) + ")";
        case Op::Eq:
        case Op::Ne: {
            Kids k = kids(n, 0, 2, true);
            if (std::string f = fast_cmp(n, k); !f.empty()) return se(k.pre, f);
            std::string a = word(n->kids[0]->type, k.v[0], true), b = word(n->kids[1]->type, k.v[1], true);
            std::string call = "epq_eq(" + std::to_string(n->a) + ", " + a + ", " + std::to_string(n->b) + ", " + b + ")";
            return se(k.pre, n->op == Op::Ne ? "(!" + call + ")" : call);
        }
        case Op::Cmp: {
            Kids k = kids(n, 0, 2, true);
            if (std::string f = fast_cmp(n, k); !f.empty()) return se(k.pre, f);
            std::string a = word(n->kids[0]->type, k.v[0], true), b = word(n->kids[1]->type, k.v[1], true);
            std::string da = std::to_string(n->a), db = std::to_string(n->b);
            std::string lt_ab = "epq_lt(" + da + ", " + a + ", " + db + ", " + b + ")";
            std::string lt_ba = "epq_lt(" + db + ", " + b + ", " + da + ", " + a + ")";
            Op c = (Op)n->c;
            std::string r = c == Op::LtI32 ? lt_ab : c == Op::GtI32 ? lt_ba : c == Op::LeI32 ? "(!" + lt_ba + ")" : "(!" + lt_ab + ")";
            return se(k.pre, r);
        }
        case Op::ListLit: return list_lit(n);
        case Op::Len: return "(" + E(n->kids[0]) + ")->len";
        case Op::Index: {
            Kids k = kids(n, 0, 2);
            uint32_t r = n->kids[0]->type.a;
            if (M.elem_width(r) == 1) return se(k.pre, "epq_ld_u8(" + k.v[0] + ", " + k.v[1] + ")");
            if (M.elem_width(r) == 4) return se(k.pre, "epq_ld_i32(" + k.v[0] + ", " + k.v[1] + ")");
            return se(k.pre, from_word(n->type, "epq_ld_w(" + k.v[0] + ", " + k.v[1] + ")"));
        }
        case Op::SetIndex: {
            Kids k = kids(n, 0, 3);
            uint32_t r = n->kids[0]->type.a;
            if (M.elem_width(r) == 1) return se(k.pre, "epq_st_u8(" + k.v[0] + ", " + k.v[1] + ", " + k.v[2] + ")");
            if (M.elem_width(r) == 4) return se(k.pre, "epq_st_i32(" + k.v[0] + ", " + k.v[1] + ", " + k.v[2] + ")");
            std::string w = word(n->kids[2]->type, k.v[2], is_pure(n->kids[2]));
            return se(k.pre, from_word(n->type, "epq_st_w(" + k.v[0] + ", " + k.v[1] + ", " + w + ")"));
        }
        case Op::Append: {
            Kids k = kids(n, 0, 2);
            uint32_t r = n->kids[0]->type.a;
            if (M.elem_width(r) == 1) return se(k.pre, "epq_push_u8(" + k.v[0] + ", " + k.v[1] + ")");
            if (M.elem_width(r) == 4) return se(k.pre, "epq_push_i32(" + k.v[0] + ", " + k.v[1] + ")");
            std::string w = word(n->kids[1]->type, k.v[1], is_pure(n->kids[1]));
            return se(k.pre, "epq_push_w(" + k.v[0] + ", " + w + ")");
        }
        case Op::Pop: {
            uint32_t r = n->kids[0]->type.a;
            if (M.elem_width(r) == 1) return "epq_pop_u8(" + E(n->kids[0]) + ")";
            if (M.elem_width(r) == 4) return "epq_pop_i32(" + E(n->kids[0]) + ")";
            return from_word(n->type, "epq_pop_w(" + E(n->kids[0]) + ")");
        }
        case Op::DynIndex: {
            Kids k = kids(n, 0, 2);
            return se(k.pre, "epq_dyn_index(" + k.v[0] + ", " + any_of(n->kids[1]->type, k.v[1]) + ")");
        }
        case Op::DynSetIndex: {
            Kids k = kids(n, 0, 3, true);
            return se(k.pre, "epq_dyn_set_index(" + k.v[0] + ", " + any_of(n->kids[1]->type, k.v[1]) + ", " + k.v[2] + ")");
        }
        case Op::DynAppend: {
            Kids k = kids(n, 0, 2);
            return se(k.pre, "epq_dyn_append(" + k.v[0] + ", " + k.v[1] + ")");
        }
        case Op::And:
        case Op::Or: {
            if (n->flags & F_COND) return "(int32_t)" + C(n);
            std::string t = tmp();
            std::string T = ctype(n->type);
            std::string a = T + " " + t + " = " + E(n->kids[0]) + "; ";
            std::string tr = truthy(n->type, t);
            if (n->op == Op::And) return "({ " + a + tr + " ? " + E(n->kids[1]) + " : " + t + "; })";
            return "({ " + a + tr + " ? " + t + " : " + E(n->kids[1]) + "; })";
        }
        case Op::CallDirect: {
            const Func* callee = M.funcs[(size_t)n->a].get();
            uint32_t na = n->nkids - 1;
            const Node* fk = n->kids[na];
            bool fpure = is_pure(fk);
            Kids k = kids(n, 0, na, !fpure, !fpure);
            std::string pre = k.pre;
            if (!fpure) pre += "(void)(" + E(fk) + "); ";
            std::string call = callee->name + "(";
            for (uint32_t i = 0; i < na; ++i) call += (i ? ", " : "") + k.v[i];
            call += ")";
            return se(pre, call);
        }
        case Op::CallDyn: {
            uint32_t na = n->nkids - 1;
            Kids k = kids(n, 0, n->nkids, true);
            std::string pre = k.pre;
            std::string arr;
            if (na == 0) {
                arr = "0";
            } else {
                arr = tmp();
                pre += "EAny " + arr + "[" + std::to_string(na) + "] = {";
                for (uint32_t i = 0; i < na; ++i) pre += (i ? ", " : "") + k.v[i];
                pre += "}; ";
            }
            return se(pre, "epq_call_dyn(" + k.v[na] + ", " + std::to_string(na) + ", " + arr + ")");
        }
        // ---- litcmp extension ----
        case Op::DynIsEmptyList: return "epq_is_empty_list(" + E(n->kids[0]) + ")";
        // ---- end litcmp extension ----
        default:
            if (std::string r; low_E(n, r)) return r;
            if (std::string r; rec_E(n, r)) return r;  // records (cgen_rec.inc)
            std::fprintf(stderr, "epq: cgen: unexpected op %s in expression\n", op_name(n->op));
            std::abort();
        }
    }

    // Inline fast paths for Eq / Ne / Cmp. Kids are temporaries or pure.
    std::string fast_cmp(const Node* n, Kids& k) {
        Type ta = n->kids[0]->type, tb = n->kids[1]->type;
        Op c = n->op == Op::Cmp ? (Op)n->c : n->op == Op::Eq ? Op::EqI32 : Op::NeI32;
        auto intish = [](Type t) { return t.k == TK::Int || t.k == TK::Any; };
        if (intish(ta) && intish(tb)) {
            // Tagged integers compare like the integers (signed 64-bit, same tag).
            std::string a = any_of(ta, k.v[0]), b = any_of(tb, k.v[1]);
            std::string wa = tmp(), wb = tmp();
            k.pre += "EAny " + wa + " = " + a + "; EAny " + wb + " = " + b + "; ";
            std::string sa = "(int64_t)" + wa, sb = "(int64_t)" + wb;
            std::string fast, slow;
            switch (c) {
            case Op::EqI32: fast = wa + " == " + wb; slow = "epq_eq(1, " + wa + ", 1, " + wb + ")"; break;
            case Op::NeI32: fast = wa + " != " + wb; slow = "!epq_eq(1, " + wa + ", 1, " + wb + ")"; break;
            case Op::LtI32: fast = sa + " < " + sb; slow = "epq_lt(1, " + wa + ", 1, " + wb + ")"; break;
            case Op::GtI32: fast = sa + " > " + sb; slow = "epq_lt(1, " + wb + ", 1, " + wa + ")"; break;
            case Op::LeI32: fast = sa + " <= " + sb; slow = "!epq_lt(1, " + wb + ", 1, " + wa + ")"; break;
            default: fast = sa + " >= " + sb; slow = "!epq_lt(1, " + wa + ", 1, " + wb + ")"; break;
            }
            return "(EPQ_LIKELY(" + wa + " & " + wb + " & 1) ? (" + fast + ") : " + slow + ")";
        }
        if (ta.k == TK::Pair && tb.k == TK::Pair) return low_pair_cmp(c, k.v[0], k.v[1]);
        if (ta.k == TK::Rec || tb.k == TK::Rec) return rec_cmp(n, c, k);  // records (cgen_rec.inc)
        if (ta.k == TK::List && tb.k == TK::List && M.elem_width(ta.a) == 4 && M.elem_width(tb.a) == 4) {
            std::string a = k.v[0], b = k.v[1];
            switch (c) {
            case Op::EqI32: return "epq_eq_li(" + a + ", " + b + ")";
            case Op::NeI32: return "(!epq_eq_li(" + a + ", " + b + "))";
            case Op::LtI32: return "epq_lt_li(" + a + ", " + b + ")";
            case Op::GtI32: return "epq_lt_li(" + b + ", " + a + ")";
            case Op::LeI32: return "(!epq_lt_li(" + b + ", " + a + "))";
            default: return "(!epq_lt_li(" + a + ", " + b + "))";
            }
        }
        // ---- litcmp extension: both lists with Any elements -> inline loop ----
        if (ta.k == TK::List && tb.k == TK::List && M.regions[ta.a].elem.k == TK::Any && M.regions[tb.a].elem.k == TK::Any) {
            std::string a = k.v[0], b = k.v[1];
            switch (c) {
            case Op::EqI32: return "epq_eq_la(" + a + ", " + b + ")";
            case Op::NeI32: return "(!epq_eq_la(" + a + ", " + b + "))";
            case Op::LtI32: return "(epq_cmp3_la(" + a + ", " + b + ") < 0)";
            case Op::GtI32: return "(epq_cmp3_la(" + a + ", " + b + ") > 0)";
            case Op::LeI32: return "(epq_cmp3_la(" + a + ", " + b + ") <= 0)";
            default: return "(epq_cmp3_la(" + a + ", " + b + ") >= 0)";
            }
        }
        // ---- end litcmp extension ----
        return "";
    }

    std::string dyn_arith(const Node* n) {
        const Node* x = n->kids[0];
        const Node* y = n->kids[1];
        std::string a = tmp(), b = tmp();
        std::string pre = ctype(x->type) + " " + a + " = " + E(x) + "; " + ctype(y->type) + " " + b + " = " + E(y) + "; ";
        Op op = (Op)n->a;
        if (op == Op::DivI32) {
            if (y->type.k == TK::Int) pre += "if (EPQ_UNLIKELY(" + b + " == 0)) epq_fail(EPQ_E_DIV_ZERO); ";
            else if (y->type.k == TK::Any) pre += "if (EPQ_UNLIKELY(" + b + " == 1)) epq_fail(EPQ_E_DIV_ZERO); ";
        }
        bool xs = x->type.k == TK::List || x->type.k == TK::Fn || x->type.k == TK::Pair || x->type.k == TK::Rec,
             ys = y->type.k == TK::List || y->type.k == TK::Fn || y->type.k == TK::Pair || y->type.k == TK::Rec;
        if (xs || ys) return "({ " + pre + "epq_fail(EPQ_E_OPERANDS_INT); (int32_t)0; })";
        bool xa = x->type.k == TK::Any, ya = y->type.k == TK::Any;
        if (xa && ya) pre += "if (EPQ_UNLIKELY(!(" + a + " & " + b + " & 1))) epq_fail(EPQ_E_OPERANDS_INT); ";
        else if (xa) pre += "if (EPQ_UNLIKELY(!(" + a + " & 1))) epq_fail(EPQ_E_OPERANDS_INT); ";
        else if (ya) pre += "if (EPQ_UNLIKELY(!(" + b + " & 1))) epq_fail(EPQ_E_OPERANDS_INT); ";
        std::string ia = xa ? "epq_int_of(" + a + ")" : a, ib = ya ? "epq_int_of(" + b + ")" : b;
        std::string r;
        switch (op) {
        case Op::AddI32: r = "(int32_t)(" + ia + " + " + ib + ")"; break;
        case Op::SubI32: r = "(int32_t)(" + ia + " - " + ib + ")"; break;
        case Op::MulI32: r = "(int32_t)(" + ia + " * " + ib + ")"; break;
        default: r = "epq_div_nz(" + ia + ", " + ib + ")"; break;
        }
        return "({ " + pre + r + "; })";
    }

    std::string list_lit(const Node* n) {
        uint32_t r = n->type.a;
        uint32_t w = M.elem_width(r);
        bool i32 = w <= 4;  // integer elements (EI32 or EU8): stored as values, not words
        std::string esz = std::to_string(w), es = elem_struct(r);
        if (n->nkids == 0) return "epq_list_new(0, " + esz + ")";
        Kids k = kids(n, 0, n->nkids, true);
        std::string t = tmp();
        std::string pre = k.pre + "EList* " + t + " = epq_list_new(" + std::to_string(n->nkids) + ", " + esz + "); ";
        std::string d = tmp();
        pre += es + "* " + d + " = (" + es + "*)" + t + "->data; ";
        for (uint32_t i = 0; i < n->nkids; ++i) {
            std::string v = i32 ? k.v[i] : word(n->kids[i]->type, k.v[i], true);
            pre += d + "[" + std::to_string(i) + "].v = " + v + "; ";
        }
        return "({ " + pre + t + "; })";
    }

    void ind(int d) { out.append((size_t)d * 4, ' '); }

    // Low-TIR / extension ops (Select, InlineBlock/Break, Pair*, ListData,
    // ElemLoad/ElemStore, CheckIndex, Free, SameList, Nop): src/cgen_low.inc.
#include "cgen_low.inc"
    // Records (RecLit/RecNull/RecGet/RecSet/RecLen, typed record comparisons).
#include "cgen_rec.inc"

    void S(const Node* n, int d) {
        switch (n->op) {
        case Op::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) S(n->kids[i], d);
            return;
        case Op::If: {
            ind(d);
            out += "if (" + C(n->kids[0]) + ") {\n";
            S(n->kids[1], d + 1);
            const Node* e = n->kids[2];
            if (!(e->op == Op::Seq && e->nkids == 0)) {
                ind(d);
                out += "} else {\n";
                S(e, d + 1);
            }
            ind(d);
            out += "}\n";
            return;
        }
        case Op::While: {
            ind(d);
            out += "while (" + C(n->kids[0]) + ") {\n";
            S(n->kids[1], d + 1);
            S(n->kids[2], d + 1);
            ind(d);
            out += "}\n";
            return;
        }
        case Op::Return: {
            const Node* v = n->kids[0];
            ind(d);
            if (F->is_main) {
                if (!is_pure(v)) out += "(void)(" + E(v) + "); ";
                out += "return;\n";
            } else {
                out += "return " + E(v) + ";\n";
            }
            return;
        }
        case Op::Print: {
            const Node* v = n->kids[0];
            ind(d);
            if (v->type.k == TK::Int) out += "epq_print_int(" + E(v) + ");\n";
            else out += "epq_print(" + std::to_string(n->a) + ", " + word(v->type, E(v), is_pure(v)) + ");\n";
            return;
        }
        default:
            if (low_S(n, d)) return;
            if (is_pure(n)) return;
            ind(d);
            out += "(void)(" + E(n) + ");\n";
            return;
        }
    }

    std::string proto(const Func& f) {
        std::string s = "static ";
        s += f.is_main ? "void" : ctype(f.ret);
        s += " " + f.name + "(";
        if (f.params.empty()) s += "void";
        for (size_t i = 0; i < f.params.size(); ++i) {
            if (i) s += ", ";
            s += ctype(f.locals[f.params[i]].type) + " l" + std::to_string(f.params[i]);
        }
        return s + ")";
    }

    void func(const Func& f) {
        F = &f;
        tmpn = 0;
        labeln = 0;
        blocks.clear();
        out += proto(f) + " {\n";
        std::vector<bool> is_param(f.locals.size(), false);
        for (uint32_t p : f.params) is_param[p] = true;
        for (size_t i = 0; i < f.locals.size(); ++i) {
            if (is_param[i]) continue;
            Type t = f.locals[i].type;
            out += "    " + ctype(t) + " l" + std::to_string(i) + " = " + (t.k == TK::Any ? "1" : "0") + ";\n";
        }
        S(f.body, 1);
        if (!f.is_main) out += "    __builtin_unreachable();\n";
        out += "}\n\n";
    }

    void run() {
        out.reserve(1 << 16);
        out.append((const char*)epq_rt_h, epq_rt_h_len);
        out += "\n/* ---- program ---- */\n";
        out += "static const EDesc epq_descs[] = {";
        for (size_t i = 0; i < M.descs.size(); ++i)
            out += (i ? ", " : "") + std::string("{") + std::to_string(M.descs[i].kind) + ", " + std::to_string(M.descs[i].elem) + "}";
        out += "};\n";
        for (size_t g = 0; g < M.globals.size(); ++g) {
            if (M.globals[g].demoted) continue;
            Type t = M.globals[g].type;
            out += "static " + ctype(t) + " g" + std::to_string(g) + (t.k == TK::Any ? " = 1" : "") + "; static uint8_t gd" + std::to_string(g) + ";\n";
        }
        for (auto& f : M.funcs) out += proto(*f) + ";\n";
        out += "\n";
        size_t rec_at = out.size();  // records: generated comparison helpers go here
        for (auto& f : M.funcs) func(*f);
        out.insert(rec_at, rec_protos + rec_bodies);
        // Generic entry points for dynamic calls.
        for (size_t d = 0; d < M.defs.size(); ++d) {
            int32_t gf = M.defs[d].generic_func;
            if (gf < 0) continue;
            const Func& f = *M.funcs[(size_t)gf];
            out += "static EAny epq_gw" + std::to_string(d) + "(const EAny* a) { return " + f.name + "(";
            for (size_t i = 0; i < f.params.size(); ++i) out += (i ? ", a[" : "a[") + std::to_string(i) + "]";
            out += "); }\n";
        }
        out += "static const EFnInfo epq_fns[] = {";
        for (size_t d = 0; d < M.defs.size(); ++d) {
            out += (d ? ", {" : "{") + std::to_string(M.defs[d].arity) + ", ";
            out += M.defs[d].generic_func >= 0 ? "epq_gw" + std::to_string(d) : std::string("0");
            out += "}";
        }
        out += "};\n\n";
        out += "__attribute__((visibility(\"default\"))) void epq_entry_flush(void) { epq_flush(); }\n";
        out += "__attribute__((visibility(\"default\"))) int epq_entry(void) {\n";
        out += "    epq_rt_desc = epq_descs;\n    epq_rt_fns = epq_fns;\n";
        out += "    " + M.funcs[0]->name + "();\n    epq_flush();\n    return 0;\n}\n";
    }
};

}  // namespace

std::string emit_c(const Module& m) {
    CGen g(m);
    g.run();
    return std::move(g.out);
}

}  // namespace epq
