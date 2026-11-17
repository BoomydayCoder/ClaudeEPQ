// Global-variable passes (DESIGN.md §2.6), run on TIR:
//
//  definite-assignment  Forward must-analysis of "global g has been assigned".
//                       Sets only grow along a path (globals are never
//                       unassigned), so a loop body's entry state is the state
//                       after the first evaluation of the condition and no
//                       iteration is needed inside a function. Across calls:
//                       S(f) = intersection of the states at all call sites of
//                       f (dynamic call sites count for every generic entry),
//                       iterated to a fixpoint over the call graph; main
//                       starts with nothing. Reads proven assigned lose
//                       F_CHECKED.
//  constants            A global of type Int all of whose assignments store the
//                       same integer literal: every unchecked read is that literal.
//  demote-globals       A global referenced by no function body other than main,
//                       and with no remaining checked read, becomes a local of main.
#include <cstdint>
#include <cstring>
#include <vector>

#include "tir.hpp"

namespace epq::tir {

namespace {

// A set of globals, or `top` (unreachable / unconstrained: contains every
// global; the words are then meaningless). All Bits of one analysis have the
// same word count; up to kInline words live inline (no heap traffic for the
// copies at every If / While / And / Or).
struct Bits {
    static constexpr uint32_t kInline = 4;
    bool top = true;
    uint32_t n;
    uint64_t* w;
    uint64_t inl[kInline];

    explicit Bits(uint32_t nw) : n(nw), w(nw <= kInline ? inl : new uint64_t[nw]) {
        std::memset(w, 0, sizeof(uint64_t) * n);
    }
    Bits(const Bits& o) : top(o.top), n(o.n), w(o.n <= kInline ? inl : new uint64_t[o.n]) {
        std::memcpy(w, o.w, sizeof(uint64_t) * n);
    }
    Bits& operator=(const Bits& o) {
        if (this != &o) {
            top = o.top;
            std::memcpy(w, o.w, sizeof(uint64_t) * n);  // same n throughout an analysis
        }
        return *this;
    }
    ~Bits() {
        if (w != inl) delete[] w;
    }
    bool has(uint32_t g) const { return top || (w[g >> 6] >> (g & 63) & 1); }
    void add(uint32_t g) {
        if (!top) w[g >> 6] |= uint64_t(1) << (g & 63);
    }
    void meet(const Bits& o) {
        if (o.top) return;
        if (top) {
            *this = o;
            return;
        }
        for (uint32_t i = 0; i < n; ++i) w[i] &= o.w[i];
    }
    bool operator==(const Bits& o) const {
        return top == o.top && (top || std::memcmp(w, o.w, sizeof(uint64_t) * n) == 0);
    }
    void set_empty() {
        top = false;
        std::memset(w, 0, sizeof(uint64_t) * n);
    }
};

struct DA {
    Module& M;
    uint32_t nw;
    std::vector<Bits> S, Snew, next;
    Bits Sdyn;
    bool clear = false;

    explicit DA(Module& m) : M(m), nw((uint32_t)((m.globals.size() + 63) / 64)), Sdyn(nw) {}

    void walk(Node* n, Bits& s) {
        switch (n->op) {
        case Op::If: {
            walk(n->kids[0], s);
            Bits a = s, b = s;
            walk(n->kids[1], a);
            walk(n->kids[2], b);
            a.meet(b);
            s = a;
            return;
        }
        case Op::While: {
            walk(n->kids[0], s);
            Bits b = s;
            walk(n->kids[1], b);
            walk(n->kids[2], b);
            return;
        }
        case Op::And:
        case Op::Or: {
            walk(n->kids[0], s);
            Bits t = s;
            walk(n->kids[1], t);
            break;
        }
        case Op::GetGlobal:
            if ((n->flags & F_CHECKED) && clear && s.has((uint32_t)n->a)) n->flags &= (uint8_t)~F_CHECKED;
            s.add((uint32_t)n->a);
            break;
        case Op::SetGlobal:
            walk(n->kids[0], s);
            s.add((uint32_t)n->a);
            break;
        case Op::CallDirect:
            for (uint32_t i = 0; i < n->nkids; ++i) walk(n->kids[i], s);
            Snew[(size_t)n->a].meet(s);
            break;
        case Op::CallDyn:
            for (uint32_t i = 0; i < n->nkids; ++i) walk(n->kids[i], s);
            Sdyn.meet(s);
            break;
        case Op::Return:
        case Op::Fail:
            for (uint32_t i = 0; i < n->nkids; ++i) walk(n->kids[i], s);
            s.top = true;
            return;
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) walk(n->kids[i], s);
            break;
        }
        if (n->op > Op::Print && n->type.k == TK::Bot && n->op != Op::Seq) s.top = true;
    }

    void round() {
        for (auto& b : Snew) b.top = true;
        Sdyn.top = true;
        for (auto& f : M.funcs) {
            Bits s = S[f->id];
            walk(f->body, s);
        }
    }

    void run() {
        size_t nf = M.funcs.size();
        S.assign(nf, Bits(nw));
        Snew.assign(nf, Bits(nw));
        S[0].set_empty();
        for (int it = 0; it < 1000; ++it) {
            round();
            next = Snew;
            next[0].set_empty();
            for (auto& f : M.funcs)
                if (f->generic) next[f->id].meet(Sdyn);
            // Monotone decrease: intersect with the previous estimate.
            bool same = true;
            for (size_t i = 0; i < nf; ++i) {
                Bits v = S[i];
                v.meet(next[i]);
                if (!(v == S[i])) same = false;
                S[i] = v;
            }
            if (same) break;
        }
        clear = true;
        round();
    }
};

void collect(Node* n, std::vector<Node*>& sets, std::vector<Node*>& gets) {
    if (n->op == Op::SetGlobal) sets.push_back(n);
    if (n->op == Op::GetGlobal) gets.push_back(n);
    for (uint32_t i = 0; i < n->nkids; ++i) collect(n->kids[i], sets, gets);
}

// Nodes grouped by global (n->a), keeping their order: by[off[g] .. off[g+1]).
struct ByGlobal {
    std::vector<uint32_t> off;
    std::vector<Node*> by;
    void build(const std::vector<Node*>& v, size_t ng) {
        off.assign(ng + 1, 0);
        for (Node* x : v) off[(size_t)x->a + 1]++;
        for (size_t g = 0; g < ng; ++g) off[g + 1] += off[g];
        by.resize(v.size());
        std::vector<uint32_t> pos(off.begin(), off.end() - 1);
        for (Node* x : v) by[pos[(size_t)x->a]++] = x;
    }
};

}  // namespace

void pass_definite_assignment(Module& m) {
    if (m.globals.empty()) return;
    DA d(m);
    d.run();
}

void pass_constants(Module& m) {
    size_t ng = m.globals.size();
    if (ng == 0) return;
    std::vector<Node*> sv, gv;
    for (auto& f : m.funcs) collect(f->body, sv, gv);
    ByGlobal sets, gets;
    sets.build(sv, ng);
    gets.build(gv, ng);
    // Globals in ascending order: rewriting reads of g can turn a later
    // global's assignments `h := g` into literals.
    for (size_t g = 0; g < ng; ++g) {
        uint32_t sb = sets.off[g], se = sets.off[g + 1];
        if (m.globals[g].type.k != TK::Int || sb == se) continue;
        bool ok = true;
        int32_t v = 0;
        for (uint32_t i = sb; i < se; ++i) {
            Node* k = sets.by[i]->kids[0];
            if (k->op != Op::ConstInt || (i != sb && k->a != v)) ok = false;
            else v = k->a;
        }
        if (!ok) continue;
        for (uint32_t i = gets.off[g]; i < gets.off[g + 1]; ++i) {
            Node* r = gets.by[i];
            if (r->flags & F_CHECKED) continue;
            r->op = Op::ConstInt;
            r->a = v;
            r->type = Type::i32();
        }
        m.globals[g].constant = true;
        m.globals[g].value = v;
    }
}

void pass_demote_globals(Module& m) {
    size_t ng = m.globals.size();
    if (ng == 0) return;
    std::vector<uint8_t> blocked(ng, 0);
    std::vector<Node*> sets, gets;
    for (auto& f : m.funcs) {
        size_t s0 = sets.size(), g0 = gets.size();
        collect(f->body, sets, gets);
        if (!f->is_main) {
            for (size_t i = s0; i < sets.size(); ++i) blocked[(size_t)sets[i]->a] = 1;
            for (size_t i = g0; i < gets.size(); ++i) blocked[(size_t)gets[i]->a] = 1;
        }
        for (size_t i = g0; i < gets.size(); ++i)
            if (gets[i]->flags & F_CHECKED) blocked[(size_t)gets[i]->a] = 1;
    }
    Func* main = m.funcs[0].get();
    std::vector<int32_t> id_of(ng, -1);
    for (size_t g = 0; g < ng; ++g) {
        if (blocked[g] || m.globals[g].demoted) continue;
        Local l;
        l.type = m.globals[g].type;
        l.kind = LocalKind::Demoted;
        l.name = m.globals[g].name;
        l.global = (int32_t)g;
        main->locals.push_back(l);
        id_of[g] = (int32_t)main->locals.size() - 1;
        m.globals[g].demoted = true;
    }
    for (Node* s : sets) {
        int32_t id = id_of[(size_t)s->a];
        if (id < 0) continue;
        s->op = Op::SetLocal;
        s->a = id;
    }
    for (Node* r : gets) {
        int32_t id = id_of[(size_t)r->a];
        if (id < 0) continue;
        r->op = Op::Local;
        r->a = id;
        r->flags = 0;
    }
}

}  // namespace epq::tir
