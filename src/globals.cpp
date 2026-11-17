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
#include <vector>

#include "tir.hpp"

namespace epq::tir {

namespace {

struct Bits {
    bool top = true;  // unreachable / unconstrained: contains every global
    std::vector<uint64_t> w;
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
        for (size_t i = 0; i < w.size(); ++i) w[i] &= o.w[i];
    }
    bool operator==(const Bits& o) const { return top == o.top && (top || w == o.w); }
};

struct DA {
    Module& M;
    size_t nw;
    std::vector<Bits> S, Snew;
    Bits Sdyn;
    bool clear = false;

    explicit DA(Module& m) : M(m), nw((m.globals.size() + 63) / 64) {}

    Bits empty() {
        Bits b;
        b.top = false;
        b.w.assign(nw, 0);
        return b;
    }

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
        if (s.top && s.w.empty()) s.w.assign(nw, 0);
    }

    void round() {
        Snew.assign(M.funcs.size(), Bits{});
        for (auto& b : Snew) b.w.assign(nw, 0);
        Sdyn = Bits{};
        Sdyn.w.assign(nw, 0);
        for (auto& f : M.funcs) {
            Bits s = S[f->id];
            if (s.w.empty()) s.w.assign(nw, 0);
            walk(f->body, s);
        }
    }

    void run() {
        S.assign(M.funcs.size(), Bits{});
        for (auto& b : S) b.w.assign(nw, 0);
        S[0] = empty();
        for (int it = 0; it < 1000; ++it) {
            round();
            std::vector<Bits> next = Snew;
            next[0] = empty();
            for (auto& f : M.funcs)
                if (f->generic) next[f->id].meet(Sdyn);
            // Monotone decrease: intersect with the previous estimate.
            bool same = true;
            for (size_t i = 0; i < next.size(); ++i) {
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

void collect(Node* n, std::vector<std::vector<Node*>>& sets, std::vector<std::vector<Node*>>& gets) {
    if (n->op == Op::SetGlobal) sets[(size_t)n->a].push_back(n);
    if (n->op == Op::GetGlobal) gets[(size_t)n->a].push_back(n);
    for (uint32_t i = 0; i < n->nkids; ++i) collect(n->kids[i], sets, gets);
}

}  // namespace

void pass_definite_assignment(Module& m) {
    if (m.globals.empty()) return;
    DA d(m);
    d.run();
}

void pass_constants(Module& m) {
    size_t ng = m.globals.size();
    std::vector<std::vector<Node*>> sets(ng), gets(ng);
    for (auto& f : m.funcs) collect(f->body, sets, gets);
    for (size_t g = 0; g < ng; ++g) {
        if (m.globals[g].type.k != TK::Int || sets[g].empty()) continue;
        bool ok = true;
        int32_t v = 0;
        for (size_t i = 0; i < sets[g].size(); ++i) {
            Node* k = sets[g][i]->kids[0];
            if (k->op != Op::ConstInt || (i && k->a != v)) ok = false;
            else v = k->a;
        }
        if (!ok) continue;
        for (Node* r : gets[g]) {
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
    std::vector<uint8_t> blocked(ng, 0);
    std::vector<std::vector<Node*>> sets(ng), gets(ng);
    for (auto& f : m.funcs) {
        std::vector<std::vector<Node*>> s2(ng), g2(ng);
        collect(f->body, s2, g2);
        for (size_t g = 0; g < ng; ++g) {
            if (!f->is_main && (!s2[g].empty() || !g2[g].empty())) blocked[g] = 1;
            for (Node* r : g2[g])
                if (r->flags & F_CHECKED) blocked[g] = 1;
            sets[g].insert(sets[g].end(), s2[g].begin(), s2[g].end());
            gets[g].insert(gets[g].end(), g2[g].begin(), g2[g].end());
        }
    }
    Func* main = m.funcs[0].get();
    for (size_t g = 0; g < ng; ++g) {
        if (blocked[g] || m.globals[g].demoted) continue;
        Local l;
        l.type = m.globals[g].type;
        l.kind = LocalKind::Demoted;
        l.name = m.globals[g].name;
        l.global = (int32_t)g;
        main->locals.push_back(l);
        int32_t id = (int32_t)main->locals.size() - 1;
        for (Node* s : sets[g]) {
            s->op = Op::SetLocal;
            s->a = id;
        }
        for (Node* r : gets[g]) {
            r->op = Op::Local;
            r->a = id;
            r->flags = 0;
        }
        m.globals[g].demoted = true;
    }
}

}  // namespace epq::tir
