// Smoke test for jit::disassemble: two functions with symbols.
#include "jit/a64.hpp"
#include "jit/disasm.hpp"
#include "jit/jitmem.hpp"

#include <cstdio>

using namespace a64;

int main() {
    Assembler a;
    Label self = a.here();
    auto f = a.prologue(1u << 19 | 1u << 20, 0);
    Label base = a.new_label();
    a.mov(W(19), W(0));
    a.cmp(W(19), 2);
    a.b(LT, base);
    a.sub(W(0), W(19), 1); a.bl(self); a.mov(W(20), W(0));
    a.sub(W(0), W(19), 2); a.bl(self); a.add(W(19), W(0), W(20));
    a.bind(base);
    a.mov(W(0), W(19));
    a.epilogue(f);
    size_t loop_off = a.pos();
    Label loop = a.here();
    a.ldr(W(2), memr(X(1), W(3), UXTW, true));
    a.add(W(0), W(0), W(2));
    a.subs(W(3), W(3), 1);
    a.b(NE, loop);
    a.ret();
    jit::CodeRegion cr;
    if (!cr.init()) return 1;
    uint8_t* p = cr.install(a);
    std::string s = jit::disassemble(p, a.size(), {{0, "fib"}, {loop_off, "loop head"}}, reinterpret_cast<uint64_t>(p));
    fputs(s.c_str(), stdout);
    return s.rfind("disassembly failed", 0) == 0;
}
