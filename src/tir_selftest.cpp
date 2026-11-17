// --tir-selftest=NAME|all: hand-built TIR programs exercising every extension
// op (src/tir.hpp) end to end through verify -> passes -> C backend -> run,
// without any optimisation pass existing. Each case runs in a forked child
// whose stdout / stderr / exit status are compared with the expectation
// computed here. Driven by tests/tir/run.py; also usable directly:
//   build/epq --engine=aot --tir-selftest=all
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "backend.hpp"
#include "tir.hpp"

namespace epq {

using namespace tir;

namespace {

using Node = tir::Node;
using Type = tir::Type;

struct B {
    Module& m;
    Func* f = nullptr;
    explicit B(Module& mm) : m(mm) {}

    Func* func(const char* name, Type ret, bool is_main) {
        auto fp = std::make_unique<Func>();
        fp->id = (uint32_t)m.funcs.size();
        fp->name = name;
        fp->ret = ret;
        fp->is_main = is_main;
        f = fp.get();
        m.funcs.push_back(std::move(fp));
        return f;
    }
    uint32_t local(Type t, LocalKind k = LocalKind::Temp) {
        Local l;
        l.type = t;
        l.kind = k;
        f->locals.push_back(l);
        return (uint32_t)f->locals.size() - 1;
    }
    uint32_t region(Type elem, bool pair = false) {
        Region r;
        r.elem = elem;
        r.pair = pair;
        m.regions.push_back(r);
        return (uint32_t)m.regions.size() - 1;
    }
    Node* n(Op op, Type t, std::initializer_list<Node*> k) { return m.node(op, t, k); }
    Node* I(int32_t v) {
        Node* x = m.node(Op::ConstInt, Type::i32(), 0);
        x->a = v;
        return x;
    }
    Node* L(uint32_t id) {
        Node* x = m.node(Op::Local, f->locals[id].type, 0);
        x->a = (int32_t)id;
        return x;
    }
    Node* set(uint32_t id, Node* v) {
        Node* x = m.node(Op::SetLocal, f->locals[id].type, {v});
        x->a = (int32_t)id;
        return x;
    }
    Node* seq(std::vector<Node*> k) {
        Node* x = m.node(Op::Seq, Type::bot(), (uint32_t)k.size());
        for (size_t i = 0; i < k.size(); ++i) x->kids[i] = k[i];
        return x;
    }
    Node* print(Node* v) {
        Node* x = n(Op::Print, Type::bot(), {v});
        x->a = (int32_t)m.desc_of(v->type);
        return x;
    }
    Node* add(Node* a, Node* b) { return n(Op::AddI32, Type::i32(), {a, b}); }
    Node* mul(Node* a, Node* b) { return n(Op::MulI32, Type::i32(), {a, b}); }
    Node* lt(Node* a, Node* b) { return n(Op::LtI32, Type::i32(), {a, b}); }
    Node* gt(Node* a, Node* b) { return n(Op::GtI32, Type::i32(), {a, b}); }
    Node* block(int32_t label, Node* body) {
        Node* x = n(Op::InlineBlock, Type::bot(), {body});
        x->a = label;
        return x;
    }
    Node* brk(int32_t label) {
        Node* x = m.node(Op::Break, Type::bot(), 0);
        x->a = label;
        return x;
    }
    Node* nop() { return m.node(Op::Nop, Type::bot(), 0); }
    Node* iff(Node* c, Node* t, Node* e) { return n(Op::If, Type::bot(), {c, t, e}); }
    // for (i = 0; i < hi; i++) body
    Node* loop(uint32_t i, Node* hi, Node* body) {
        return seq({set(i, I(0)), n(Op::While, Type::bot(), {lt(L(i), hi), body, set(i, add(L(i), I(1)))})});
    }
    Node* list(uint32_t r, std::vector<Node*> k) {
        Node* x = m.node(Op::ListLit, Type::list(r), (uint32_t)k.size());
        for (size_t i = 0; i < k.size(); ++i) x->kids[i] = k[i];
        return x;
    }
    Node* pmake(uint32_t pr, Node* a, Node* b) { return n(Op::PairMake, Type::pair(pr), {a, b}); }
    Node* eqd(Node* a, Node* b, bool ne = false) {
        Node* x = n(ne ? Op::Ne : Op::Eq, Type::i32(), {a, b});
        x->a = (int32_t)m.desc_of(a->type);
        x->b = (int32_t)m.desc_of(b->type);
        return x;
    }
    Node* cmpd(Op c, Node* a, Node* b) {
        Node* x = n(Op::Cmp, Type::i32(), {a, b});
        x->a = (int32_t)m.desc_of(a->type);
        x->b = (int32_t)m.desc_of(b->type);
        x->c = (int32_t)c;
        return x;
    }
};

struct Case {
    const char* name;
    std::function<void(B&, std::string& expect_out, std::string& expect_err, int& expect_rc)> build;
    const char* env = nullptr;  // NAME=VALUE set in the child
    bool expect_abort = false;  // verifier negative test: child must abort, stderr must contain expect_err
};

std::string pl(long long v) { return ">> " + std::to_string(v) + "\n"; }

const int32_t kVals[] = {INT_MIN, INT_MIN + 1, -5, -1, 0, 1, 7, INT_MAX - 1, INT_MAX};
constexpr int kNV = sizeof kVals / sizeof kVals[0];

std::vector<Case> cases() {
    std::vector<Case> cs;

    cs.push_back({"select", [](B& b, std::string& o, std::string&, int&) {
        uint32_t x = b.local(Type::i32()), y = b.local(Type::i32()), c = b.local(Type::i32());
        uint32_t pr = b.region(Type::i32(), true);
        uint32_t p = b.local(Type::pair(pr)), q = b.local(Type::pair(pr));
        b.f->body = b.seq({
            b.set(x, b.I(5)), b.set(y, b.I(7)), b.set(c, b.I(0)),
            b.print(b.n(Op::Select, Type::i32(), {b.lt(b.L(x), b.L(y)), b.L(x), b.L(y)})),
            b.print(b.n(Op::Select, Type::i32(), {b.gt(b.L(x), b.L(y)), b.L(x), b.L(y)})),
            b.print(b.n(Op::Select, Type::i32(), {b.L(c), b.I(1), b.I(2)})),
            b.set(x, b.n(Op::Select, Type::i32(), {b.I(-3), b.add(b.L(x), b.I(100)), b.L(x)})),
            b.print(b.L(x)),
            b.set(p, b.pmake(pr, b.I(1), b.I(2))), b.set(q, b.pmake(pr, b.I(-1), b.I(9))),
            b.print(b.n(Op::Select, Type::pair(pr), {b.L(c), b.L(p), b.L(q)})),
        });
        o = pl(5) + pl(7) + pl(2) + pl(105) + ">> [-1, 9]\n";
    }});

    cs.push_back({"inline", [](B& b, std::string& o, std::string&, int&) {
        uint32_t res = b.local(Type::i32()), x = b.local(Type::i32()), i = b.local(Type::i32()), acc = b.local(Type::i32());
        // Expression form: Comma[InlineBlock{if (x > 3) {res := 1; break}; res := 2; break; res := 3}, Local res]
        auto inl = [&](int32_t lab) {
            return b.n(Op::Comma, Type::i32(),
                       {b.block(lab, b.seq({b.iff(b.gt(b.L(x), b.I(3)), b.seq({b.set(res, b.I(1)), b.brk(lab)}), b.seq({})),
                                            b.set(res, b.I(2)), b.brk(lab), b.set(res, b.I(3))})),
                        b.L(res)});
        };
        int32_t l0 = (int32_t)new_label(*b.f), l1 = (int32_t)new_label(*b.f), l2 = (int32_t)new_label(*b.f),
                l3 = (int32_t)new_label(*b.f), l4 = (int32_t)new_label(*b.f);
        b.f->body = b.seq({
            b.set(x, b.I(5)), b.print(inl(l0)),
            b.set(x, b.I(1)), b.print(b.add(b.I(10), inl(l1))),
            // statement form, nested blocks, break to the outer block from the inner one
            b.set(acc, b.I(0)),
            b.block(l2, b.seq({b.block(l3, b.seq({b.set(acc, b.add(b.L(acc), b.I(1))), b.brk(l2),
                                                  b.set(acc, b.add(b.L(acc), b.I(100)))})),
                               b.set(acc, b.add(b.L(acc), b.I(1000)))})),
            b.print(b.L(acc)),
            // in a loop: break leaves the block, the loop continues (acc counts i >= 3 -> +i, else +0)
            b.set(acc, b.I(0)),
            b.loop(i, b.I(6), b.block(l4, b.seq({b.iff(b.lt(b.L(i), b.I(3)), b.brk(l4), b.nop()),
                                                 b.set(acc, b.add(b.L(acc), b.L(i)))}))),
            b.print(b.L(acc)),
        });
        o = pl(1) + pl(12) + pl(1) + pl(3 + 4 + 5);
    }});

    cs.push_back({"inline_fn", [](B& b, std::string& o, std::string&, int&) {
        // Inlined call inside a non-main function, with Return inside the block.
        Func* main = b.f;
        Func* g = b.func("f1_g", Type::i32(), false);
        uint32_t px = b.local(Type::i32(), LocalKind::Param);
        g->params.push_back(px);
        uint32_t r = b.local(Type::i32());
        int32_t lab = (int32_t)new_label(*g);
        b.f->body = b.seq({b.n(Op::Return, Type::bot(),
                               {b.add(b.n(Op::Comma, Type::i32(),
                                          {b.block(lab, b.seq({b.iff(b.lt(b.L(px), b.I(0)),
                                                                     b.n(Op::Return, Type::bot(), {b.I(-100)}), b.nop()),
                                                               b.set(r, b.mul(b.L(px), b.I(2))), b.brk(lab)})),
                                           b.L(r)}),
                                      b.I(1))})});
        b.f = main;
        b.m.defs.resize(1);
        auto call = [&](int32_t v) {
            Node* fk = b.m.node(Op::ConstFn, Type::fn(0), 0);
            Node* c = b.n(Op::CallDirect, Type::i32(), {b.I(v), fk});
            c->a = 1;
            return c;
        };
        b.f->body = b.seq({b.print(call(20)), b.print(call(-1))});
        o = pl(41) + pl(-100);
    }});

    cs.push_back({"pairs", [](B& b, std::string& o, std::string&, int&) {
        uint32_t pr = b.region(Type::i32(), true);
        uint32_t plr = b.region(Type::pair(pr));      // list of pairs
        uint32_t ir = b.region(Type::i32());          // int lists
        uint32_t llr = b.region(Type::list(ir));      // list of int lists
        uint32_t ps = b.local(Type::list(plr)), ls = b.local(Type::list(llr));
        uint32_t i = b.local(Type::i32()), j = b.local(Type::i32()), p = b.local(Type::pair(pr)), q = b.local(Type::pair(pr));
        uint32_t li = b.local(Type::list(ir));
        std::vector<Node*> pk, lk;
        for (int a = 0; a < kNV; ++a)
            for (int c = 0; c < kNV; ++c) {
                pk.push_back(b.pmake(pr, b.I(kVals[a]), b.I(kVals[c])));
                lk.push_back(b.list(ir, {b.I(kVals[a]), b.I(kVals[c])}));
            }
        int N = kNV * kNV;
        auto bit = [&](Node* v, int k) { return b.mul(v, b.I(1 << k)); };
        Node* code = b.add(b.add(b.add(bit(b.n(Op::PairLt, Type::i32(), {b.L(p), b.L(q)}), 0),
                                       bit(b.n(Op::PairLe, Type::i32(), {b.L(p), b.L(q)}), 1)),
                                 b.add(bit(b.n(Op::PairGt, Type::i32(), {b.L(p), b.L(q)}), 2),
                                       bit(b.n(Op::PairGe, Type::i32(), {b.L(p), b.L(q)}), 3))),
                           b.add(b.add(bit(b.n(Op::PairEq, Type::i32(), {b.L(p), b.L(q)}), 4),
                                       bit(b.n(Op::PairNe, Type::i32(), {b.L(p), b.L(q)}), 5)),
                                 // descriptor-driven: pair vs real list, both orders
                                 b.add(b.add(bit(b.cmpd(Op::LtI32, b.L(p), b.L(li)), 6), bit(b.cmpd(Op::GeI32, b.L(li), b.L(p)), 7)),
                                       b.add(bit(b.eqd(b.L(p), b.L(li)), 8), bit(b.eqd(b.L(li), b.L(p), true), 9)))));
        Node* body = b.seq({
            b.set(p, b.n(Op::Index, Type::pair(pr), {b.L(ps), b.L(i)})),
            b.loop(j, b.I(N), b.seq({
                b.set(q, b.n(Op::Index, Type::pair(pr), {b.L(ps), b.L(j)})),
                b.set(li, b.n(Op::Index, Type::list(ir), {b.L(ls), b.L(j)})),
                b.print(code),
            })),
        });
        Node* fields = b.seq({b.print(b.n(Op::PairFirst, Type::i32(), {b.L(p)})), b.print(b.n(Op::PairSecond, Type::i32(), {b.L(p)})),
                              b.print(b.n(Op::PairIndex, Type::i32(), {b.L(p), b.I(0)})),
                              b.print(b.n(Op::PairIndex, Type::i32(), {b.L(p), b.I(1)})), b.print(b.L(p))});
        b.f->body = b.seq({b.set(ps, b.list(plr, pk)), b.set(ls, b.list(llr, lk)),
                           b.loop(i, b.I(N), body),
                           b.loop(i, b.I(N), b.seq({b.set(p, b.n(Op::Index, Type::pair(pr), {b.L(ps), b.L(i)})), fields}))});
        auto lexlt = [](int a0, int a1, int b0, int b1) { return a0 != b0 ? a0 < b0 : a1 < b1; };
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y) {
                int a0 = kVals[x / kNV], a1 = kVals[x % kNV], b0 = kVals[y / kNV], b1 = kVals[y % kNV];
                bool l = lexlt(a0, a1, b0, b1), g = lexlt(b0, b1, a0, a1), e = a0 == b0 && a1 == b1;
                int c = l * 1 + (!g) * 2 + g * 4 + (!l) * 8 + e * 16 + (!e) * 32 + l * 64 + g * 128 + e * 256 + (!e) * 512;
                // Cmp(GE, li, p) == !(li < p) == !g ; bit 7
                c = c - g * 128 + (!g) * 128;
                o += pl(c);
            }
        for (int x = 0; x < N; ++x) {
            int a0 = kVals[x / kNV], a1 = kVals[x % kNV];
            o += pl(a0) + pl(a1) + pl(a0) + pl(a1) + ">> [" + std::to_string(a0) + ", " + std::to_string(a1) + "]\n";
        }
    }});

    cs.push_back({"pairlists", [](B& b, std::string& o, std::string&, int&) {
        uint32_t pr = b.region(Type::i32(), true);
        uint32_t plr = b.region(Type::pair(pr));
        uint32_t ir = b.region(Type::i32());
        uint32_t llr = b.region(Type::list(ir));
        uint32_t ps = b.local(Type::list(plr)), ls = b.local(Type::list(llr)), p = b.local(Type::pair(pr));
        b.f->body = b.seq({
            b.set(ps, b.list(plr, {b.pmake(pr, b.I(1), b.I(2)), b.pmake(pr, b.I(3), b.I(-4))})),
            b.print(b.L(ps)),
            b.n(Op::Append, Type::list(plr), {b.L(ps), b.pmake(pr, b.I(INT_MIN), b.I(INT_MAX))}),
            b.print(b.L(ps)), b.print(b.n(Op::Len, Type::i32(), {b.L(ps)})),
            b.set(p, b.n(Op::Pop, Type::pair(pr), {b.L(ps)})), b.print(b.L(p)),
            b.n(Op::SetIndex, Type::pair(pr), {b.L(ps), b.I(0), b.pmake(pr, b.I(3), b.I(-4))}),
            b.set(ls, b.list(llr, {b.list(ir, {b.I(3), b.I(-4)}), b.list(ir, {b.I(3), b.I(-4)})})),
            b.print(b.eqd(b.L(ps), b.L(ls))),       // [[3,-4],[3,-4]] == [[3,-4],[3,-4]] -> 1
            b.print(b.eqd(b.L(ls), b.L(ps))),
            b.n(Op::SetIndex, Type::list(ir), {b.L(ls), b.I(1), b.list(ir, {b.I(3), b.I(-4), b.I(0)})}),
            b.print(b.eqd(b.L(ps), b.L(ls))),       // 0
            b.print(b.cmpd(Op::LtI32, b.L(ps), b.L(ls))),  // [3,-4] < [3,-4,0] -> 1
            b.print(b.cmpd(Op::GtI32, b.L(ps), b.L(ls))),  // 0
            b.print(b.cmpd(Op::LtI32, b.L(p), b.list(ir, {b.I(INT_MIN), b.I(INT_MAX), b.I(1)}))),  // prefix -> 1
            b.print(b.cmpd(Op::LtI32, b.L(p), b.list(ir, {b.I(INT_MIN)}))),  // 0
            b.print(b.eqd(b.L(p), b.I(5))),  // pair vs int -> 0
            b.print(b.eqd(b.L(p), b.list(ir, {}))),  // 0
            b.iff(b.L(p), b.print(b.I(111)), b.print(b.I(222))),  // a pair is truthy
        });
        o = ">> [[1, 2], [3, -4]]\n>> [[1, 2], [3, -4], [-2147483648, 2147483647]]\n>> 3\n>> [-2147483648, 2147483647]\n" + pl(1) +
            pl(1) + pl(0) + pl(1) + pl(0) + pl(1) + pl(0) + pl(0) + pl(0) + pl(111);
    }});

    cs.push_back({"pair_index_oob", [](B& b, std::string& o, std::string& e, int& rc) {
        uint32_t pr = b.region(Type::i32(), true);
        uint32_t p = b.local(Type::pair(pr)), i = b.local(Type::i32());
        b.f->body = b.seq({b.set(p, b.pmake(pr, b.I(1), b.I(2))), b.set(i, b.I(2)), b.print(b.I(1)),
                           b.print(b.n(Op::PairIndex, Type::i32(), {b.L(p), b.L(i)}))});
        o = pl(1);
        e = "Index out of bounds\n";
        rc = 1;
    }});
    cs.push_back({"pair_cmp_int", [](B& b, std::string& o, std::string& e, int& rc) {
        uint32_t pr = b.region(Type::i32(), true);
        uint32_t p = b.local(Type::pair(pr));
        b.f->body = b.seq({b.set(p, b.pmake(pr, b.I(1), b.I(2))), b.print(b.cmpd(Op::LtI32, b.L(p), b.I(3)))});
        e = "Invalid comparison\n";
        rc = 1;
    }});
    cs.push_back({"pair_arith", [](B& b, std::string& o, std::string& e, int& rc) {
        uint32_t pr = b.region(Type::i32(), true);
        uint32_t p = b.local(Type::pair(pr));
        Node* d = b.n(Op::DynArith, Type::i32(), {b.L(p), b.I(1)});
        d->a = (int32_t)Op::AddI32;
        b.f->body = b.seq({b.set(p, b.pmake(pr, b.I(1), b.I(2))), b.print(d)});
        e = "Operands must be integers\n";
        rc = 1;
    }});

    cs.push_back({"elems", [](B& b, std::string& o, std::string&, int&) {
        uint32_t ir = b.region(Type::i32());
        uint32_t llr = b.region(Type::list(ir));
        uint32_t pr = b.region(Type::i32(), true);
        uint32_t plr = b.region(Type::pair(pr));
        uint32_t l = b.local(Type::list(ir)), ll = b.local(Type::list(llr)), pl_ = b.local(Type::list(plr));
        uint32_t d = b.local(Type::ptr(ir), LocalKind::Cache), dd = b.local(Type::ptr(llr), LocalKind::Cache),
                 dp = b.local(Type::ptr(plr), LocalKind::Cache), n = b.local(Type::i32(), LocalKind::Cache), i = b.local(Type::i32()),
                 s = b.local(Type::i32());
        b.f->body = b.seq({
            b.set(l, b.list(ir, {b.I(10), b.I(20), b.I(30)})),
            b.set(d, b.n(Op::ListData, Type::ptr(ir), {b.L(l)})),
            b.set(n, b.n(Op::Len, Type::i32(), {b.L(l)})),
            b.print(b.n(Op::ElemLoad, Type::i32(), {b.L(d), b.I(1)})),
            b.print(b.n(Op::ElemStore, Type::i32(), {b.L(d), b.n(Op::CheckIndex, Type::i32(), {b.I(2), b.L(n)}), b.I(99)})),
            b.print(b.L(l)),
            b.set(s, b.I(0)),
            b.loop(i, b.L(n), b.set(s, b.add(b.L(s), b.n(Op::ElemLoad, Type::i32(), {b.L(d), b.n(Op::CheckIndex, Type::i32(), {b.L(i), b.L(n)})})))),
            b.print(b.L(s)),
            b.set(ll, b.list(llr, {b.L(l), b.list(ir, {b.I(-7)})})),
            b.set(dd, b.n(Op::ListData, Type::ptr(llr), {b.L(ll)})),
            b.print(b.n(Op::ElemLoad, Type::list(ir), {b.L(dd), b.I(1)})),
            b.n(Op::ElemStore, Type::list(ir), {b.L(dd), b.I(1), b.L(l)}),
            b.print(b.L(ll)),
            b.print(b.n(Op::SameList, Type::i32(), {b.n(Op::ElemLoad, Type::list(ir), {b.L(dd), b.I(0)}), b.L(l)})),
            b.set(pl_, b.list(plr, {b.pmake(pr, b.I(4), b.I(-5))})),
            b.set(dp, b.n(Op::ListData, Type::ptr(plr), {b.L(pl_)})),
            b.print(b.n(Op::ElemLoad, Type::pair(pr), {b.L(dp), b.I(0)})),
            b.n(Op::ElemStore, Type::pair(pr), {b.L(dp), b.I(0), b.pmake(pr, b.I(-6), b.I(8))}),
            b.print(b.L(pl_)),
            b.print(b.n(Op::CheckIndex, Type::i32(), {b.I(0), b.I(1)})),
        });
        o = pl(20) + pl(99) + ">> [10, 20, 99]\n" + pl(129) + ">> [-7]\n>> [[10, 20, 99], [10, 20, 99]]\n" + pl(1) + ">> [4, -5]\n>> [[-6, 8]]\n" + pl(0);
    }});
    for (int which = 0; which < 3; ++which) {
        static const char* names[] = {"check_index_hi", "check_index_neg", "check_index_empty"};
        cs.push_back({names[which], [which](B& b, std::string& o, std::string& e, int& rc) {
            uint32_t i = b.local(Type::i32()), n = b.local(Type::i32());
            int iv = which == 0 ? 3 : which == 1 ? -1 : 0, nv = which == 2 ? 0 : 3;
            b.f->body = b.seq({b.set(i, b.I(iv)), b.set(n, b.I(nv)), b.print(b.n(Op::CheckIndex, Type::i32(), {b.I(0), b.I(1)})),
                               b.print(b.n(Op::CheckIndex, Type::i32(), {b.L(i), b.L(n)})), b.print(b.I(5))});
            o = pl(0);
            e = "Index out of bounds\n";
            rc = 1;
        }});
    }

    // Free + reuse: a freed list's block is handed out again for the next list of
    // the same size; grown buffers are recycled; contents stay correct.
    auto free_case = [](bool poison) {
        return [poison](B& b, std::string& o, std::string&, int&) {
            uint32_t ir = b.region(Type::i32());
            uint32_t a = b.local(Type::list(ir)), c = b.local(Type::list(ir)), k = b.local(Type::list(ir)), i = b.local(Type::i32()),
                     j = b.local(Type::i32()), s = b.local(Type::i32());
            Node* grow = b.seq({b.set(k, b.list(ir, {})),
                                b.loop(i, b.I(1000), b.n(Op::Append, Type::list(ir), {b.L(k), b.L(i)})),
                                b.set(s, b.I(0)),
                                b.loop(i, b.I(1000), b.set(s, b.add(b.L(s), b.n(Op::Index, Type::i32(), {b.L(k), b.L(i)})))),
                                b.print(b.L(s)),
                                b.n(Op::Free, Type::bot(), {b.L(k)})});
            b.f->body = b.seq({
                b.set(a, b.list(ir, {b.I(1), b.I(2), b.I(3)})),
                b.set(c, b.L(a)),
                b.print(b.n(Op::SameList, Type::i32(), {b.L(a), b.L(c)})),
                b.n(Op::Free, Type::bot(), {b.L(a)}),
                b.set(a, b.list(ir, {b.I(4), b.I(5), b.I(6)})),
                b.print(b.n(Op::SameList, Type::i32(), {b.L(a), b.L(c)})),  // reused: 1 (poison: 0)
                b.print(b.L(a)),
                b.loop(j, b.I(5), grow),
                b.print(b.L(a)),
            });
            o = pl(1) + pl(poison ? 0 : 1) + ">> [4, 5, 6]\n";
            for (int x = 0; x < 5; ++x) o += pl(499500);
            o += ">> [4, 5, 6]\n";
        };
    };
    cs.push_back({"free_reuse", free_case(false)});
    cs.push_back({"free_poison", free_case(true), "EPQ_POISON_FREED=1"});
    cs.push_back({"poison_header", [](B& b, std::string& o, std::string&, int&) {
        uint32_t ir = b.region(Type::i32());
        uint32_t a = b.local(Type::list(ir));
        b.f->body = b.seq({b.set(a, b.list(ir, {b.I(1), b.I(2)})), b.n(Op::Free, Type::bot(), {b.L(a)}),
                           b.print(b.n(Op::Len, Type::i32(), {b.L(a)}))});
        o = pl(INT_MAX);
    }, "EPQ_POISON_FREED=1"});
    cs.push_back({"samelist_nop", [](B& b, std::string& o, std::string&, int&) {
        uint32_t ir = b.region(Type::i32()), jr = b.region(Type::i32());
        uint32_t a = b.local(Type::list(ir)), c = b.local(Type::list(jr));
        b.f->body = b.seq({b.nop(), b.set(a, b.list(ir, {b.I(1)})), b.set(c, b.list(jr, {b.I(1)})), b.nop(),
                           b.print(b.n(Op::SameList, Type::i32(), {b.L(a), b.L(a)})),
                           b.print(b.n(Op::SameList, Type::i32(), {b.L(a), b.L(c)})), b.print(b.n(Op::Comma, Type::i32(), {b.nop(), b.I(3)}))});
        o = pl(1) + pl(0) + pl(3);
    }});

    // ---- verifier negative tests ----
    cs.push_back({"verify_break_outside", [](B& b, std::string&, std::string& e, int&) {
        b.f->body = b.seq({b.block(0, b.nop()), b.brk(0)});
        e = "not inside an InlineBlock";
    }, nullptr, true});
    cs.push_back({"verify_select_impure", [](B& b, std::string&, std::string& e, int&) {
        b.f->body = b.seq({b.print(b.n(Op::Select, Type::i32(), {b.I(1), b.m.node(Op::Input, Type::i32(), 0), b.I(2)}))});
        e = "is not pure";
    }, nullptr, true});
    cs.push_back({"verify_setlocal_type", [](B& b, std::string&, std::string& e, int&) {
        uint32_t ir = b.region(Type::i32());
        uint32_t x = b.local(Type::i32());
        Node* s = b.set(x, b.list(ir, {}));
        b.f->body = b.seq({s});
        e = "!= local type";
    }, nullptr, true});
    cs.push_back({"verify_pair_region", [](B& b, std::string&, std::string& e, int&) {
        uint32_t ir = b.region(Type::i32());  // not a pair region
        b.f->body = b.seq({b.print(b.n(Op::PairFirst, Type::i32(), {b.pmake(ir, b.I(1), b.I(2))}))});
        e = "not a pair region";
    }, nullptr, true});
    cs.push_back({"verify_elemload_type", [](B& b, std::string&, std::string& e, int&) {
        uint32_t ir = b.region(Type::i32());
        uint32_t d = b.local(Type::ptr(ir), LocalKind::Cache);
        b.f->body = b.seq({b.print(b.n(Op::ElemLoad, Type::any(), {b.L(d), b.I(0)}))});
        e = "result type Any, expected Int";
    }, nullptr, true});
    return cs;
}

int run_case(const Case& c, const Options& opts) {
    Module M;
    B b(M);
    b.func("f0_main", Type::bot(), true);
    std::string eo, ee;
    int erc = 0;
    c.build(b, eo, ee, erc);
    M.defs.resize(std::max<size_t>(M.defs.size(), 0));
    int po[2], pe[2];
    if (pipe(po) != 0 || pipe(pe) != 0) return 1;
    std::fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        dup2(po[1], 1);
        dup2(pe[1], 2);
        close(po[0]);
        close(pe[0]);
        if (c.env) {
            std::string kv = c.env;
            size_t eq = kv.find('=');
            setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
        }
        Options o = opts;
        o.verify_tir = true;
        tir::verify(M, "selftest-build");
        int rc = run_module(M, o);
        std::fflush(stdout);
        _exit(rc);
    }
    close(po[1]);
    close(pe[1]);
    std::string out, err;
    char buf[65536];
    // Read stdout fully, then stderr (stderr output is tiny).
    for (ssize_t k; (k = read(po[0], buf, sizeof buf)) > 0;) out.append(buf, (size_t)k);
    for (ssize_t k; (k = read(pe[0], buf, sizeof buf)) > 0;) err.append(buf, (size_t)k);
    close(po[0]);
    close(pe[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    bool ok;
    std::string why;
    if (c.expect_abort) {
        ok = WIFSIGNALED(st) && err.find(ee) != std::string::npos;
        if (!ok) why = "expected verifier abort with '" + ee + "'";
    } else {
        int rc = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
        ok = rc == erc && out == eo && err == ee;
        if (!ok) why = "rc " + std::to_string(rc) + " (want " + std::to_string(erc) + ")";
    }
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.name);
    if (!ok) {
        std::printf("  %s\n  --- stdout (%zu bytes):\n%.2000s  --- expected (%zu bytes):\n%.2000s  --- stderr:\n%.2000s\n", why.c_str(),
                    out.size(), out.c_str(), eo.size(), eo.c_str(), err.c_str());
    }
    std::fflush(stdout);
    return ok ? 0 : 1;
}

}  // namespace

int tir_selftest(const char* name, const Options& opts) {
    int fails = 0, n = 0;
    for (const Case& c : cases()) {
        if (std::strcmp(name, "all") != 0 && std::strcmp(name, c.name) != 0) {
            if (!std::strcmp(name, "list")) std::printf("%s\n", c.name);
            continue;
        }
        ++n;
        fails += run_case(c, opts);
    }
    if (!std::strcmp(name, "list")) return 0;
    if (n == 0) {
        std::fprintf(stderr, "epq: no TIR self-test named %s (try --tir-selftest=list)\n", name);
        return 2;
    }
    std::printf("%d/%d TIR self-tests passed\n", n - fails, n);
    return fails ? 1 : 0;
}

}  // namespace epq
