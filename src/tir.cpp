#include "tir.hpp"

#include <cstdio>
#include <cstring>
#include <iterator>

namespace epq::tir {

const char* err_name(Err e) {
    switch (e) {
    case Err::None: return "None";
    case Err::OperandsInt: return "OperandsInt";
    case Err::DivZero: return "DivZero";
    case Err::ListEmpty: return "ListEmpty";
    case Err::IntOrList: return "IntOrList";
    case Err::OperandList: return "OperandList";
    case Err::OperandInt: return "OperandInt";
    case Err::UndefGlobal: return "UndefGlobal";
    case Err::InvalidOp: return "InvalidOp";
    case Err::IndexOOB: return "IndexOOB";
    case Err::OperandFn: return "OperandFn";
    case Err::Arity: return "Arity";
    case Err::InvalidCmp: return "InvalidCmp";
    }
    return "?";
}

const char* op_name(Op o) {
    switch (o) {
#define C(x) case Op::x: return #x;
        C(Seq) C(If) C(While) C(Return) C(Print) C(ConstInt) C(ConstFn) C(Local) C(GetGlobal) C(Input)
        C(SetLocal) C(SetGlobal) C(Box) C(UnboxInt) C(Coerce) C(Comma) C(Fail) C(AddI32) C(SubI32) C(MulI32)
        C(DivI32) C(NegI32) C(AbsI32) C(NotI32) C(EqI32) C(NeI32) C(LtI32) C(LeI32) C(GtI32) C(GeI32)
        C(DynArith) C(DynNeg) C(DynAbs) C(DynNot) C(Eq) C(Ne) C(Cmp) C(ListLit) C(Len) C(Index) C(SetIndex)
        C(Append) C(Pop) C(DynIndex) C(DynSetIndex) C(DynAppend) C(And) C(Or) C(CallDirect) C(CallDyn)
        C(Select) C(InlineBlock) C(Break) C(PairMake) C(PairFirst) C(PairSecond) C(PairIndex) C(PairEq) C(PairNe)
        C(PairLt) C(PairLe) C(PairGt) C(PairGe) C(ListData) C(ElemLoad) C(ElemStore) C(CheckIndex) C(Free)
        C(SameList) C(Nop)
        C(DynIsEmptyList)  // litcmp extension
        C(Reserve) C(AppendRange)  // appendopt
        C(RecLit) C(RecNull) C(RecGet) C(RecSet) C(RecLen)  // records (§5.11)
#undef C
    }
    return "?";
}

Node* Module::node(Op op, Type t, uint32_t nkids) {
    Node* n = arena.make<Node>();
    n->op = op;
    n->type = t;
    n->nkids = nkids;
    n->kids = arena.array<Node*>(nkids);
    return n;
}

Node* Module::node(Op op, Type t, std::initializer_list<Node*> kids) {
    Node* n = node(op, t, (uint32_t)kids.size());
    uint32_t i = 0;
    for (Node* k : kids) n->kids[i++] = k;
    return n;
}

uint32_t Module::desc_of(Type t) {
    if (descs.empty()) {
        descs.push_back({0, 0});  // INT
        descs.push_back({1, 0});  // ANY
        descs.push_back({3, 1});  // LIST(ANY)
        descs.push_back({2, 0});  // FN
    }
    switch (t.k) {
    case TK::Bot:
    case TK::Int: return 0;
    case TK::Any: return 1;
    case TK::Fn: return 3;
    case TK::Pair:
    case TK::Ptr: {  // one shared PAIR descriptor (Ptr never needs one; mapped for totality)
        for (size_t i = 4; i < descs.size(); ++i)
            if (descs[i].kind == 4) return (uint32_t)i;
        descs.push_back({4, 0});
        return (uint32_t)descs.size() - 1;
    }
    case TK::List: {
        if (region_desc_memo.size() < regions.size()) region_desc_memo.resize(regions.size(), -1);
        if (region_desc_memo[t.a] >= 0) return (uint32_t)region_desc_memo[t.a];
        if (regions[t.a].elem.k == TK::Any) return region_desc_memo[t.a] = 2;
        uint32_t id = (uint32_t)descs.size();
        descs.push_back({3, 0});
        region_desc_memo[t.a] = (int32_t)id;
        uint32_t e = desc_of(regions[t.a].elem);
        descs[id].elem = e;
        return id;
    }
    case TK::Rec: {  // records (§5.11): RECORD {5, k | nullable << 16}, then k FIELD entries
        if (region_desc_memo.size() < regions.size()) region_desc_memo.resize(regions.size(), -1);
        if (region_desc_memo[t.a] >= 0) return (uint32_t)region_desc_memo[t.a];
        const uint32_t k = regions[t.a].rec_len;
        uint32_t id = (uint32_t)descs.size();
        descs.push_back({5, k | (regions[t.a].nullable ? 0x10000u : 0u)});
        for (uint32_t j = 0; j < k; ++j) descs.push_back({6, 0});
        region_desc_memo[t.a] = (int32_t)id;
        for (uint32_t j = 0; j < k; ++j) {
            uint32_t e = desc_of(regions[t.a].fields[j]);
            descs[id + 1 + j].elem = e;
        }
        return id;
    }
    }
    return 1;
}

uint32_t Module::new_temp(Func* f, Type t) {
    Local l;
    l.type = t;
    l.kind = LocalKind::Temp;
    f->locals.push_back(l);
    return (uint32_t)f->locals.size() - 1;
}

bool is_pure(const Node* n) {
    switch (n->op) {
    case Op::ConstInt:
    case Op::ConstFn:
    case Op::Local:
        return true;
    case Op::GetGlobal:
        return !(n->flags & F_CHECKED);
    case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::NegI32: case Op::AbsI32: case Op::NotI32:
    case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
    case Op::Box: case Op::Len:
    case Op::Select: case Op::PairMake: case Op::PairFirst: case Op::PairSecond:
    case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
    case Op::SameList: case Op::ListData: case Op::ElemLoad: case Op::Nop:
    case Op::DynIsEmptyList:  // litcmp extension
    case Op::RecNull: case Op::RecLen:
        for (uint32_t i = 0; i < n->nkids; ++i)
            if (!is_pure(n->kids[i])) return false;
        return true;
    case Op::RecGet:  // records: pure unless it carries a null check
        if (n->b) return false;
        for (uint32_t i = 0; i < n->nkids; ++i)
            if (!is_pure(n->kids[i])) return false;
        return true;
    default:
        return false;
    }
}

bool is_stmt_op(Op o) {
    switch (o) {
    case Op::Seq: case Op::If: case Op::While: case Op::Return: case Op::Print:
    case Op::InlineBlock: case Op::Break: case Op::Free: case Op::Nop:
    case Op::Reserve:  // appendopt
        return true;
    default:
        return false;
    }
}

std::string type_str(const Module& m, Type t) {
    switch (t.k) {
    case TK::Bot: return "Bot";
    case TK::Int: return "Int";
    case TK::Any: return "Any";
    case TK::Fn: return "Fn" + std::to_string(t.a);
    case TK::List: return "L" + std::to_string(t.a);
    case TK::Pair: return "Pair" + std::to_string(t.a);
    case TK::Ptr: return "Ptr" + std::to_string(t.a);
    case TK::Rec: return "R" + std::to_string(t.a);
    }
    return "?";
}

namespace {

struct Dumper {
    const Module& m;
    std::string& o;
    const Func* f = nullptr;

    void ind(int d) { o.append((size_t)d * 2, ' '); }
    std::string lname(uint32_t id) {
        const Local& l = f->locals[id];
        std::string s;
        if (l.name != kNoSym) s = std::string(m.prog->syms->name(l.name));
        else if (l.kind == LocalKind::Temp) s = "tmp";
        else if (l.kind == LocalKind::Cache) s = "cache";
        else s = "hid";
        return s + "%" + std::to_string(id);
    }
    std::string gname(int32_t g) { return "@" + std::string(m.prog->syms->name(m.globals[(size_t)g].name)); }

    void node(const Node* n, int d) {
        ind(d);
        o += op_name(n->op);
        switch (n->op) {
        case Op::ConstInt: o += " " + std::to_string(n->a); break;
        case Op::ConstFn: o += " def" + std::to_string(n->a); break;
        case Op::Local: case Op::SetLocal: o += " " + lname((uint32_t)n->a); break;
        case Op::GetGlobal: case Op::SetGlobal: o += " " + gname(n->a); break;
        case Op::UnboxInt: case Op::Fail: o += std::string(" ") + err_name((Err)n->a); break;
        case Op::Print: o += " desc" + std::to_string(n->a); break;
        case Op::DynArith: o += std::string(" ") + op_name((Op)n->a); break;
        case Op::Eq: case Op::Ne: o += " desc" + std::to_string(n->a) + ",desc" + std::to_string(n->b); break;
        case Op::Cmp:
            o += std::string(" ") + op_name((Op)n->c) + " desc" + std::to_string(n->a) + ",desc" + std::to_string(n->b);
            break;
        case Op::CallDirect: o += " " + m.funcs[(size_t)n->a]->name; break;
        case Op::InlineBlock: case Op::Break: o += " label" + std::to_string(n->a); break;
        case Op::RecGet: case Op::RecSet: o += " #" + std::to_string(n->a) + (n->b ? " nullchk" : ""); break;
        case Op::While:
            if (n->flags & F_FOREACH) o += " foreach(idx=" + lname((uint32_t)n->a) + ", list=" + lname((uint32_t)n->b) + ")";
            if (n->flags & F_FOR) o += " for";
            break;
        default: break;
        }
        if (n->flags & F_CHECKED) o += " checked";
        if (n->flags & F_COND) o += " cond";
        if (!is_stmt_op(n->op)) o += " : " + type_str(m, n->type);
        bool leaf = n->nkids == 0;
        if (leaf) {
            o += "\n";
            return;
        }
        // Compact single-line form for small pure subtrees.
        o += "\n";
        for (uint32_t i = 0; i < n->nkids; ++i) node(n->kids[i], d + 1);
    }
};

}  // namespace

void dump(std::string& o, const Module& m) {
    Dumper D{m, o};
    o += "globals:\n";
    for (size_t g = 0; g < m.globals.size(); ++g) {
        const Global& G = m.globals[g];
        o += "  @" + std::string(m.prog->syms->name(G.name)) + " : " + type_str(m, G.type);
        if (G.demoted) o += " demoted";
        o += "\n";
    }
    o += "regions:\n";
    for (size_t r = 0; r < m.regions.size(); ++r) {
        o += "  L" + std::to_string(r) + " elem " + type_str(m, m.regions[r].elem);
        if (m.regions[r].header_mutated) o += " header-mutated";
        if (m.regions[r].elem_stored) o += " elem-stored";
        if (m.regions[r].pair) o += " pair";
        if (m.regions[r].record) {
            o += m.regions[r].nullable ? " record? (" : " record (";
            for (size_t j = 0; j < m.regions[r].fields.size(); ++j) o += (j ? ", " : "") + type_str(m, m.regions[r].fields[j]);
            o += ")";
        }
        o += " sites:";
        for (auto& s : m.regions[r].sites)
            o += " f" + std::to_string(s.func) + "/line" + std::to_string(s.line) + "/len" + std::to_string(s.len);
        o += "\n";
    }
    o += "descs:";
    for (size_t i = 0; i < m.descs.size(); ++i) {
        static const char* kn[] = {"INT", "ANY", "FN", "LIST", "PAIR", "RECORD", "FIELD"};
        o += " " + std::to_string(i) + "=" + kn[m.descs[i].kind];
        if (m.descs[i].kind == 3 || m.descs[i].kind == 6) o += "(" + std::to_string(m.descs[i].elem) + ")";
        if (m.descs[i].kind == 5) o += "(" + std::to_string(m.descs[i].elem & 0xffff) + (m.descs[i].elem >> 16 ? "?)" : ")");
    }
    o += "\n";
    for (auto& fp : m.funcs) {
        const Func& f = *fp;
        D.f = &f;
        o += "\nfunc " + f.name + " (instance i" + std::to_string(f.inst) + ")";
        if (f.generic) o += " generic";
        o += " (";
        for (size_t i = 0; i < f.params.size(); ++i) {
            if (i) o += ", ";
            o += D.lname(f.params[i]) + ": " + type_str(m, f.locals[f.params[i]].type);
        }
        o += ") -> " + type_str(m, f.ret) + "\n  mutates:";
        for (uint32_t r : f.mutates) o += " L" + std::to_string(r);
        if (f.mutates_any) o += " any";
        o += "\n  locals:";
        for (size_t i = 0; i < f.locals.size(); ++i) {
            static const char* kn[] = {"param", "user", "hidden", "temp", "demoted", "cache"};
            o += " " + D.lname((uint32_t)i) + ":" + type_str(m, f.locals[i].type) + "(" + kn[(int)f.locals[i].kind] +
                 (f.locals[i].unique ? ",unique" : "") + ")";
        }
        o += "\n";
        D.node(f.body, 1);
    }
}

// ---- constant folding (the pass manager is in src/passes/registry.cpp) --------------

namespace {
// Constant folding of pure integer subtrees (tiny; mostly a placeholder for the
// optimisation passes of DESIGN.md §5, which slot in here).
Node* fold(Module& m, Node* n) {
    for (uint32_t i = 0; i < n->nkids; ++i) n->kids[i] = fold(m, n->kids[i]);
    auto k = [&](uint32_t i) { return n->kids[i]->op == Op::ConstInt; };
    auto v = [&](uint32_t i) { return (uint32_t)n->kids[i]->a; };
    int64_t r;
    switch (n->op) {
    case Op::AddI32: if (!(k(0) && k(1))) return n; r = (int32_t)(v(0) + v(1)); break;
    case Op::SubI32: if (!(k(0) && k(1))) return n; r = (int32_t)(v(0) - v(1)); break;
    case Op::MulI32: if (!(k(0) && k(1))) return n; r = (int32_t)(v(0) * v(1)); break;
    case Op::NegI32: if (!k(0)) return n; r = (int32_t)(0u - v(0)); break;
    default: return n;
    }
    Node* c = m.node(Op::ConstInt, Type::i32(), 0);
    c->a = (int32_t)r;
    c->line = n->line;
    return c;
}
}  // namespace

void pass_fold(Module& m) {
    for (auto& f : m.funcs) f->body = fold(m, f->body);
}

}  // namespace epq::tir
