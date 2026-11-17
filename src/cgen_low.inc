// C backend: extension / Low-TIR ops (src/tir.hpp "Extension for the
// optimisation passes"). Textually included inside struct CGen (src/cgen.cpp),
// so these are member functions with access to E / S / kids / word / ctype.
//
//   Pair        EPair (int64_t), runtime helpers epq_pair_*
//   Ptr(r)      EI32* / EW64* (the one-member element structs keep type-based
//               alias analysis effective: element stores never alias headers)
//   Select      C conditional expression
//   InlineBlock { body } _B<n>: ;   (expression form: GNU statement expression)
//   Break       goto _B<n>;  (n = C label of the innermost enclosing InlineBlock
//               with the same TIR label)
//   CheckIndex  epq_check_index: one unsigned compare + cold noreturn failure

std::string capture_stmts(const Node* body, int d) {
    std::string saved;
    saved.swap(out);
    S(body, d);
    std::string r;
    r.swap(out);
    out.swap(saved);
    return r;
}

int break_target(const Node* n) {
    for (size_t i = blocks.size(); i-- > 0;)
        if (blocks[i].first == n->a) return blocks[i].second;
    std::fprintf(stderr, "epq: cgen: Break label%d outside an InlineBlock with that label (in %s)\n", n->a,
                 F ? F->name.c_str() : "?");
    std::abort();
}

std::string low_pair_cmp(Op c, const std::string& a, const std::string& b) {
    const char* sym = c == Op::EqI32 ? "==" : c == Op::NeI32 ? "!=" : c == Op::LtI32 ? "<" : c == Op::LeI32 ? "<="
                    : c == Op::GtI32 ? ">" : ">=";
    return "((int32_t)(" + a + " " + sym + " " + b + "))";
}

std::string elem_size(uint32_t region) { return M.elem_is_i32(region) ? "4" : "8"; }

bool low_E(const Node* n, std::string& r) {
    switch (n->op) {
    case Op::Select:
        r = "(" + E(n->kids[0]) + " ? " + E(n->kids[1]) + " : " + E(n->kids[2]) + ")";
        return true;
    case Op::InlineBlock: {
        int L = labeln++;
        blocks.push_back({n->a, L});
        std::string body = capture_stmts(n->kids[0], 0);
        blocks.pop_back();
        r = "({ " + body + " _B" + std::to_string(L) + ": ; })";
        return true;
    }
    case Op::Break:
        r = "({ goto _B" + std::to_string(break_target(n)) + "; (int32_t)0; })";
        return true;
    case Op::Nop:
        r = "0";
        return true;
    case Op::PairMake: {
        Kids k = kids(n, 0, 2);
        r = se(k.pre, "epq_pair_make(" + k.v[0] + ", " + k.v[1] + ")");
        return true;
    }
    case Op::PairFirst: r = "epq_pair_first(" + E(n->kids[0]) + ")"; return true;
    case Op::PairSecond: r = "epq_pair_second(" + E(n->kids[0]) + ")"; return true;
    case Op::PairIndex: {
        Kids k = kids(n, 0, 2);
        r = se(k.pre, "epq_pair_index(" + k.v[0] + ", " + k.v[1] + ")");
        return true;
    }
    case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe: {
        static const Op map[] = {Op::EqI32, Op::NeI32, Op::LtI32, Op::LeI32, Op::GtI32, Op::GeI32};
        Kids k = kids(n, 0, 2);
        r = se(k.pre, low_pair_cmp(map[(int)n->op - (int)Op::PairEq], k.v[0], k.v[1]));
        return true;
    }
    case Op::ListData: {
        uint32_t reg = n->kids[0]->type.a;
        r = "((" + ctype(Type::ptr(reg)) + ")(" + E(n->kids[0]) + ")->data)";
        return true;
    }
    case Op::ElemLoad: {
        Kids k = kids(n, 0, 2);
        uint32_t reg = n->kids[0]->type.a;
        std::string ld = k.v[0] + "[" + k.v[1] + "].v";
        r = se(k.pre, M.elem_is_i32(reg) ? ld : from_word(n->type, ld));
        return true;
    }
    case Op::ElemStore: {
        Kids k = kids(n, 0, 3);
        uint32_t reg = n->kids[0]->type.a;
        if (M.elem_is_i32(reg)) {
            r = se(k.pre, "(" + k.v[0] + "[" + k.v[1] + "].v = " + k.v[2] + ")");
        } else {
            std::string w = word(n->kids[2]->type, k.v[2], is_pure(n->kids[2]));
            r = se(k.pre, from_word(n->type, "(" + k.v[0] + "[" + k.v[1] + "].v = " + w + ")"));
        }
        return true;
    }
    case Op::CheckIndex: {
        Kids k = kids(n, 0, 2);
        r = se(k.pre, "epq_check_index(" + k.v[0] + ", " + k.v[1] + ")");
        return true;
    }
    case Op::Free:
        r = "(epq_list_free(" + E(n->kids[0]) + ", " + elem_size(n->kids[0]->type.a) + "), 0)";
        return true;
    case Op::SameList: {
        Kids k = kids(n, 0, 2);
        r = se(k.pre, "((int32_t)((void*)" + k.v[0] + " == (void*)" + k.v[1] + "))");
        return true;
    }
    default:
        return false;
    }
}

bool low_S(const Node* n, int d) {
    switch (n->op) {
    case Op::Nop:
        return true;
    case Op::InlineBlock: {
        int L = labeln++;
        blocks.push_back({n->a, L});
        ind(d);
        out += "{\n";
        S(n->kids[0], d + 1);
        ind(d);
        out += "}\n";
        blocks.pop_back();
        ind(d);
        out += "_B" + std::to_string(L) + ": ;\n";
        return true;
    }
    case Op::Break:
        ind(d);
        out += "goto _B" + std::to_string(break_target(n)) + ";\n";
        return true;
    case Op::Free:
        ind(d);
        out += "epq_list_free(" + E(n->kids[0]) + ", " + elem_size(n->kids[0]->type.a) + ");\n";
        return true;
    default:
        return false;
    }
}
