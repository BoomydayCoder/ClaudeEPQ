// C backend: records (src/tir.hpp "Records"; DESIGN.md §5.11). Textually
// included inside struct CGen (src/cgen.cpp), like cgen_low.inc.
//
//   Rec(r)     ERec* (k 8-byte slots, no header; NULL = the empty list of a
//              nullable region). ERec is its own struct type, so type-based
//              alias analysis keeps record slots apart from list headers/elements.
//   RecLit     epq_rec_new(k) + k slot stores
//   RecGet     one load at a constant offset (+ null test when b)
//   RecSet     one store (+ null test when b, after both kids)
//   RecLen     (r ? k : 0)
//   Eq/Ne/Cmp with a Rec operand: typed comparison functions generated per pair
//   of static operand types (epq_rq<N> equality, epq_rc<N> three-way), emitted
//   in front of the program's functions (rec_protos / rec_bodies). They follow
//   LANGUAGE.md §6 exactly: equality never fails; three-way comparison reaches
//   the same element pairs as `x < y` then `y < x` and raises Invalid
//   comparison at the first mixed-kind pair reached. Anything involving Any
//   (or a Pair against a list) goes through the descriptor-driven runtime.

std::string rec_protos, rec_bodies;
std::vector<std::pair<std::string, std::string>> rec_memo;  // key -> function name
bool rec_gen_failed = false;

static std::string rec_key(Type t) { return std::to_string((int)t.k) + ":" + std::to_string(t.has_arg() ? t.a : 0); }

// Descriptor index of static type t (it exists: Cmp/Eq/Print operand descriptors
// are created recursively), or -1.
int64_t rec_desc_id(Type t) const {
    switch (t.k) {
    case TK::Bot: case TK::Int: return 0;
    case TK::Any: return 1;
    case TK::Fn: return 3;
    case TK::Pair:
        for (size_t i = 4; i < M.descs.size(); ++i)
            if (M.descs[i].kind == 4) return (int64_t)i;
        return -1;
    case TK::List: case TK::Rec:
        if (t.a < M.region_desc_memo.size() && M.region_desc_memo[t.a] >= 0) return M.region_desc_memo[t.a];
        return -1;
    default: return -1;
    }
}
std::string rec_dyn(const char* fn, Type a, Type b, const std::string& x, const std::string& y) {
    int64_t da = rec_desc_id(a), db = rec_desc_id(b);
    if (da < 0 || db < 0) {
        rec_gen_failed = true;
        return "0";
    }
    return std::string(fn) + "(" + std::to_string(da) + ", " + word(a, x, true) + ", " + std::to_string(db) + ", " + word(b, y, true) + ")";
}
// Element j of a value x of list-like type t (j: C expression), as its static type.
std::string rec_elem(Type t, const std::string& x, const std::string& j, uint32_t pos) {
    if (t.k == TK::Rec) return from_word(M.regions[t.a].fields[pos], x + "[" + j + "].v");
    if (M.elem_is_i32(t.a)) return "((EI32*)" + x + "->data)[" + j + "].v";
    return from_word(M.regions[t.a].elem, "((EW64*)" + x + "->data)[" + j + "].v");
}
Type rec_elem_type(Type t, uint32_t pos) { return t.k == TK::Rec ? M.regions[t.a].fields[pos] : M.regions[t.a].elem; }
std::string rec_len_expr(Type t, const std::string& x) {
    if (t.k == TK::List) return x + "->len";
    const Region& R = M.regions[t.a];
    if (!R.nullable) return std::to_string(R.rec_len);
    return "(" + x + " ? " + std::to_string(R.rec_len) + " : 0)";
}

// Three-way comparison expression (-1/0/1) of x : a with y : b.
std::string rec_c3(Type a, Type b, const std::string& x, const std::string& y) {
    if (a.is_int_like() && b.is_int_like()) return "epq_c3i(" + x + ", " + y + ")";
    if (a.k == TK::Any || b.k == TK::Any) return rec_dyn("epq_c3_dyn", a, b, x, y);
    if (a.k == TK::Fn || b.k == TK::Fn) return "epq_c3_invalid()";
    if (a.k == TK::Pair && b.k == TK::Pair) return "epq_c3p(" + x + ", " + y + ")";
    if (a.is_int_like() || b.is_int_like()) return "epq_c3_invalid()";  // int against a list
    if (a.k == TK::Pair || b.k == TK::Pair) return rec_dyn("epq_c3_dyn", a, b, x, y);
    if (a.k == TK::List && b.k == TK::List && M.elem_is_i32(a.a) && M.elem_is_i32(b.a)) return "epq_c3_li(" + x + ", " + y + ")";
    return rec_fn(false, a, b) + "(" + x + ", " + y + ")";
}
// Equality expression (0/1) of x : a with y : b. Never fails.
std::string rec_q(Type a, Type b, const std::string& x, const std::string& y) {
    if (a.is_int_like() && b.is_int_like()) return "(" + x + " == " + y + ")";
    if (a.k == TK::Any || b.k == TK::Any) return rec_dyn("epq_eq", a, b, x, y);
    if (a.k == TK::Fn || b.k == TK::Fn) return (a.k == TK::Fn && b.k == TK::Fn && a.a == b.a) ? "1" : "0";
    if (a.k == TK::Pair && b.k == TK::Pair) return "(" + x + " == " + y + ")";
    if (a.is_int_like() || b.is_int_like()) return "0";
    if (a.k == TK::Pair || b.k == TK::Pair) return rec_dyn("epq_eq", a, b, x, y);
    if (a.k == TK::List && b.k == TK::List && M.elem_is_i32(a.a) && M.elem_is_i32(b.a)) return "epq_eq_li(" + x + ", " + y + ")";
    return rec_fn(true, a, b) + "(" + x + ", " + y + ")";
}
// Name of the generated function comparing two list-like static types.
std::string rec_fn(bool eq, Type a, Type b) {
    std::string key = std::string(eq ? "q" : "c") + rec_key(a) + "/" + rec_key(b);
    for (auto& m : rec_memo)
        if (m.first == key) return m.second;
    std::string name = std::string(eq ? "epq_rq" : "epq_rc") + std::to_string(rec_memo.size());
    rec_memo.push_back({key, name});
    std::string sig = "static int " + name + "(" + ctype(a) + " a, " + ctype(b) + " b)";
    rec_protos += sig + ";\n";
    std::string o = sig + " {\n    int32_t na = " + rec_len_expr(a, "a") + ", nb = " + rec_len_expr(b, "b") + ";\n";
    bool ra = a.k == TK::Rec, rb = b.k == TK::Rec;
    if (!ra && !rb) {  // two lists: run-time loop
        Type ea = rec_elem_type(a, 0), eb = rec_elem_type(b, 0);
        if (eq) {
            o += "    if (na != nb) return 0;\n    for (int32_t j = 0; j < na; j++)\n        if (!" +
                 rec_q(ea, eb, rec_elem(a, "a", "j", 0), rec_elem(b, "b", "j", 0)) + ") return 0;\n    return 1;\n";
        } else {
            o += "    int32_t n = na < nb ? na : nb;\n    for (int32_t j = 0; j < n; j++) {\n        int c = " +
                 rec_c3(ea, eb, rec_elem(a, "a", "j", 0), rec_elem(b, "b", "j", 0)) +
                 ";\n        if (c) return c;\n    }\n    return (na > nb) - (na < nb);\n";
        }
    } else {  // unrolled over the record's positions
        uint32_t ka = ra ? M.regions[a.a].rec_len : 0, kb = rb ? M.regions[b.a].rec_len : 0;
        uint32_t k = ra && rb ? (ka < kb ? ka : kb) : ra ? ka : kb;
        bool dyn = !ra || !rb || M.regions[a.a].nullable || M.regions[b.a].nullable;
        if (eq) {
            o += "    if (na != nb) return 0;\n    if (na == 0) return 1;\n";
            if (ra && rb && ka != kb) {
                o += "    return 0;\n";
            } else {
                for (uint32_t j = 0; j < k; ++j) {
                    std::string js = std::to_string(j);
                    o += "    if (!" + rec_q(rec_elem_type(a, j), rec_elem_type(b, j), rec_elem(a, "a", js, j), rec_elem(b, "b", js, j)) +
                         ") return 0;\n";
                }
                o += "    return 1;\n";
            }
        } else {
            o += "    int c;\n";
            for (uint32_t j = 0; j < k; ++j) {
                std::string js = std::to_string(j);
                if (dyn && (j == 0 || !ra || !rb)) o += "    if (" + js + " >= na || " + js + " >= nb) return (na > nb) - (na < nb);\n";
                o += "    c = " + rec_c3(rec_elem_type(a, j), rec_elem_type(b, j), rec_elem(a, "a", js, j), rec_elem(b, "b", js, j)) +
                     ";\n    if (c) return c;\n";
            }
            o += "    return (na > nb) - (na < nb);\n";
        }
    }
    o += "}\n";
    rec_bodies += o;
    return name;
}

// Eq/Ne/Cmp fast path when an operand is a Rec (kids are temporaries or pure).
std::string rec_cmp(const Node* n, Op c, Kids& k) {
    Type ta = n->kids[0]->type, tb = n->kids[1]->type;
    size_t p = rec_protos.size(), b = rec_bodies.size(), m = rec_memo.size();
    rec_gen_failed = false;
    std::string r;
    if (c == Op::EqI32 || c == Op::NeI32) {
        std::string e = rec_q(ta, tb, k.v[0], k.v[1]);
        r = c == Op::NeI32 ? "((int32_t)!" + e + ")" : "((int32_t)" + e + ")";
    } else {
        std::string e = rec_c3(ta, tb, k.v[0], k.v[1]);
        const char* s = c == Op::LtI32 ? " < 0" : c == Op::GtI32 ? " > 0" : c == Op::LeI32 ? " <= 0" : " >= 0";
        r = "((int32_t)(" + e + s + "))";
    }
    if (rec_gen_failed) {  // a descriptor was missing: use the generic runtime call
        rec_protos.resize(p);
        rec_bodies.resize(b);
        rec_memo.resize(m);
        return "";
    }
    return r;
}

bool rec_E(const Node* n, std::string& r) {
    switch (n->op) {
    case Op::RecLit: {
        Kids k = kids(n, 0, n->nkids, true);
        std::string t = tmp();
        std::string pre = k.pre + "ERec* " + t + " = epq_rec_new(" + std::to_string(n->nkids) + "); ";
        for (uint32_t i = 0; i < n->nkids; ++i)
            pre += t + "[" + std::to_string(i) + "].v = " + word(n->kids[i]->type, k.v[i], true) + "; ";
        r = "({ " + pre + t + "; })";
        return true;
    }
    case Op::RecNull:
        r = "((ERec*)0)";
        return true;
    case Op::RecGet: {
        std::string x = E(n->kids[0]);
        std::string j = std::to_string(n->a);
        if (n->b) {
            std::string t = tmp();
            r = "({ ERec* " + t + " = " + x + "; if (EPQ_UNLIKELY(!" + t + ")) epq_fail_index(); " + from_word(n->type, t + "[" + j + "].v") + "; })";
        } else {
            r = from_word(n->type, "(" + x + ")[" + j + "].v");
        }
        return true;
    }
    case Op::RecSet: {
        Kids k = kids(n, 0, 2, true);
        std::string pre = k.pre;
        if (n->b) pre += "if (EPQ_UNLIKELY(!" + k.v[0] + ")) epq_fail_index(); ";
        pre += "(" + k.v[0] + ")[" + std::to_string(n->a) + "].v = " + word(n->type, k.v[1], true) + "; ";
        r = "({ " + pre + k.v[1] + "; })";
        return true;
    }
    case Op::RecLen:
        r = "((" + E(n->kids[0]) + ") ? " + std::to_string(M.regions[n->kids[0]->type.a].rec_len) + " : 0)";
        return true;
    default:
        return false;
    }
}
