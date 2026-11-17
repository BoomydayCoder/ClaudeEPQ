// JIT execution tests: emit -> install into MAP_JIT memory -> call.
// Prints one PASS/FAIL line per test; exit status = number of failures (capped).
// --asan-safe skips the big-stack test (ASan cannot follow an sp switch).
#include "jit/a64.hpp"
#include "jit/bigstack.hpp"
#include "jit/jitmem.hpp"

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace a64;

static jit::CodeRegion region;
static std::mt19937_64 rng(42);
static uint64_t R() { return rng(); }
static int failures = 0;
static bool g_exhaustive = false;
static long checks = 0;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { if (errs++ < 5) { printf("    "); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static void report(const char* name, int errs, const std::string& extra = "") {
    printf("%s %s%s%s\n", errs ? "FAIL" : "PASS", name, extra.empty() ? "" : " - ", extra.c_str());
    if (errs) failures++;
    fflush(stdout);
}

// A batch of functions in one buffer; fn<T>(i) returns the i-th entry point.
struct Batch {
    Assembler a;
    std::vector<size_t> entry;
    uint8_t* base = nullptr;
    void begin() { a.align(16); entry.push_back(a.pos()); }
    void done() { base = region.install(a); }
    template <class T> T fn(size_t i) const { return reinterpret_cast<T>(base + entry[i]); }
};

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// interesting 64-bit constants
static std::vector<int64_t> edge_constants() {
    std::vector<int64_t> v = {0, 1, -1, 2, -2, 4095, 4096, -4095, -4096, 4097, 0xfff000, 0xffffff, 0x1000000,
                              INT32_MAX, INT32_MIN, int64_t(UINT32_MAX), INT64_MAX, INT64_MIN, 0x5555555555555555,
                              0x00ff00ff00ff00ff, 0x1234567812345678, 0x0000ffff00000000, -0x10000, 0x7fff, 0x8000};
    for (int k = 0; k < 64; k++) {
        int64_t p = int64_t(uint64_t(1) << k);
        for (uint64_t d : {uint64_t(0), uint64_t(1), ~uint64_t(0)}) {
            v.push_back(int64_t(uint64_t(p) + d));
            v.push_back(int64_t(0 - (uint64_t(p) + d)));
        }
    }
    for (int a = 1; a < 20; a++) for (int b = 0; b < 20; b += 3) {
        v.push_back(int64_t(((uint64_t(1) << a) + 1) << b));
        v.push_back(int64_t(((uint64_t(1) << a) - 1) << b));
    }
    for (int i = 0; i < 3000; i++) {
        uint64_t r = R();
        switch (i % 6) {
        case 0: v.push_back(int64_t(r)); break;
        case 1: v.push_back(int64_t(r & 0xffffffff)); break;
        case 2: v.push_back(int64_t(int32_t(r))); break;
        case 3: v.push_back(int64_t(r & 0xffffff) * ((r >> 40) & 1 ? -1 : 1)); break;
        case 4: v.push_back(int64_t(r & 0xfff) << 12); break;
        default: v.push_back(int64_t(r % 20000) - 10000);
        }
    }
    return v;
}

static uint64_t inputs(int i) {
    static const uint64_t e[] = {0, 1, ~0ull, 0x7fffffffffffffff, 0x8000000000000000, 0x7fffffff, 0x80000000, 0xffffffff, 2, 0xfff};
    return i < 10 ? e[i] : R();
}

// ---------------------------------------------------------------- immediates
static void test_load_imm() {
    int errs = 0;
    auto cs = edge_constants();
    Batch b;
    std::vector<int> lens;
    for (int64_t c : cs) for (int x = 0; x < 2; x++) {
        b.begin();
        size_t p = b.a.pos();
        b.a.load_imm(x ? X(0) : W(0), c);
        int len = int(b.a.pos() - p) / 4;
        CHECK(len == Assembler::load_imm_len(c, x), "load_imm_len mismatch for %llx", (long long)c);
        CHECK(len <= (x ? 4 : 2), "load_imm too long (%d) for %llx", len, (long long)c);
        lens.push_back(len);
        b.a.ret();
    }
    b.done();
    for (size_t i = 0; i < cs.size(); i++) for (int x = 0; x < 2; x++) {
        uint64_t got = b.fn<uint64_t (*)()>(2 * i + x)();
        uint64_t want = x ? uint64_t(cs[i]) : uint64_t(uint32_t(cs[i]));
        CHECK(got == want, "load_imm(%s, %llx) gave %llx", x ? "x" : "w", (unsigned long long)cs[i], (unsigned long long)got);
    }
    double avg = 0;
    for (int l : lens) avg += l;
    report("load_imm: all edge + 3000 random constants, W and X", errs,
           std::to_string(cs.size() * 2) + " constants, avg " + std::to_string(avg / lens.size()).substr(0, 4) + " insns");
}

static uint32_t flags_sub(uint64_t a, uint64_t b, bool x) {
    if (!x) { a = uint32_t(a); b = uint32_t(b); }
    uint64_t r = a - b, sb = x ? 63 : 31;
    if (!x) r = uint32_t(r);
    uint32_t N = (r >> sb) & 1, Z = r == 0, C = a >= b, V = (((a ^ b) & (a ^ r)) >> sb) & 1;
    return N << 31 | Z << 30 | C << 29 | V << 28;
}
static uint32_t flags_add(uint64_t a, uint64_t b, bool x) {
    if (!x) { a = uint32_t(a); b = uint32_t(b); }
    uint64_t r = a + b, sb = x ? 63 : 31;
    uint32_t C;
    if (x) C = r < a; else { C = (r >> 32) & 1; r = uint32_t(r); }
    uint32_t N = (r >> sb) & 1, Z = r == 0, V = ((~(a ^ b) & (a ^ r)) >> sb) & 1;
    return N << 31 | Z << 30 | C << 29 | V << 28;
}

static void test_add_cmp_imm() {
    int errs = 0;
    auto cs = edge_constants();
    Batch b;
    // per constant: add_imm W/X, adds_imm W/X (+flags), cmp_imm W/X
    for (int64_t c : cs) for (int x = 0; x < 2; x++) {
        Reg r0 = x ? X(0) : W(0), r1 = x ? X(1) : W(1);
        b.begin(); b.a.add_imm(r1, r0, c); b.a.mov(r0, r1); b.a.ret();
        b.begin(); b.a.add_imm(r1, r0, c, true); b.a.mrs_nzcv(X(2)); b.a.str(X(2), mem(X(9))); b.a.mov(r0, r1); b.a.ret();
        b.begin(); b.a.cmp_imm(r0, c); b.a.mrs_nzcv(X(0)); b.a.ret();
    }
    b.done();
    uint64_t fl = 0;
    for (size_t i = 0; i < cs.size(); i++) for (int x = 0; x < 2; x++) {
        size_t k = (2 * i + x) * 3;
        auto add = b.fn<uint64_t (*)(uint64_t)>(k);
        auto adds = b.fn<uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*)>(k + 1);
        auto cmp = b.fn<uint64_t (*)(uint64_t)>(k + 2);
        for (int j = 0; j < 24; j++) {
            uint64_t a = inputs(j), c = uint64_t(cs[i]);
            uint64_t want = x ? a + c : uint32_t(a + c);
            uint64_t got = add(a);
            CHECK(got == want, "add_imm(%d, %llx) on %llx: %llx", x, (long long)c, (long long)a, (long long)got);
            // x9 carries the flag pointer (9th integer arg would be on the stack, so pass via asm-visible reg)
            uint64_t got2;
            asm volatile("mov x9, %1\n mov x0, %2\n blr %3\n mov %0, x0" : "=r"(got2) : "r"(&fl), "r"(a), "r"(adds)
                         : "x0", "x1", "x2", "x9", "x16", "x17", "x30", "memory", "cc");
            CHECK(got2 == want, "adds_imm value");
            CHECK(uint32_t(fl) == flags_add(a, c, x), "adds_imm(%d, %llx) on %llx: flags %x want %x", x, (long long)c,
                  (long long)a, unsigned(fl), flags_add(a, c, x));
            uint32_t f = uint32_t(cmp(a));
            CHECK(f == flags_sub(a, c, x), "cmp_imm(%d, %llx) on %llx: flags %x want %x", x, (long long)c, (long long)a, f,
                  flags_sub(a, c, x));
        }
    }
    // add_imm / sub_imm on sp
    Batch s;
    std::vector<int64_t> spc = {16, 4080, 4096, 65536, 0x123450, 0x1000000, 0x7fff0};
    for (int64_t c : spc) {
        s.begin(); s.a.mov(X(9), sp); s.a.sub_imm(sp, sp, c); s.a.mov(X(10), sp); s.a.add_imm(sp, sp, c);
        s.a.sub(X(0), X(9), X(10)); s.a.mov(X(11), sp); s.a.sub(X(11), X(11), X(9)); s.a.orr(X(0), X(0), X(11), LSL, 32); s.a.ret();
    }
    s.done();
    for (size_t i = 0; i < spc.size(); i++) {
        uint64_t r = s.fn<uint64_t (*)()>(i)();
        CHECK(r == uint64_t(spc[i]), "sp add/sub %lld gave %llx", (long long)spc[i], (unsigned long long)r);
    }
    report("add_imm / adds_imm (value+NZCV) / cmp_imm (NZCV) / sp arithmetic", errs, std::to_string(cs.size()) + " constants x 24 inputs x W/X");
}

static void test_mul_imm() {
    int errs = 0;
    auto cs = edge_constants();
    Batch b;
    for (int64_t c : cs) for (int x = 0; x < 2; x++) { b.begin(); b.a.mul_imm(x ? X(0) : W(0), x ? X(0) : W(0), c); b.a.ret(); }
    b.done();
    long n = 0;
    for (size_t i = 0; i < cs.size(); i++) for (int x = 0; x < 2; x++) {
        auto f = b.fn<uint64_t (*)(uint64_t)>(2 * i + x);
        for (int j = 0; j < 300; j++, n++) {
            uint64_t a = inputs(j);
            uint64_t want = x ? a * uint64_t(cs[i]) : uint32_t(uint32_t(a) * uint32_t(cs[i]));
            uint64_t got = x ? f(a) : uint32_t(f(a));  // W results: upper half unspecified
            CHECK(got == want, "mul_imm(%d, %lld) on %llx: %llx want %llx", x, (long long)cs[i], (long long)a, (long long)got, (long long)want);
        }
    }
    report("mul_imm vs C multiply", errs, std::to_string(n) + " products");
}

static void test_div_imm() {
    int errs = 0;
    std::vector<int32_t> ds;
    for (int d = -1100; d <= 1100; d++) if (d) ds.push_back(d);
    for (int k = 0; k < 31; k++) { ds.push_back(1 << k); ds.push_back(-(1 << k)); ds.push_back((1 << k) + 1); ds.push_back(-(1 << k) - 1); }
    ds.insert(ds.end(), {INT_MIN, INT_MAX, INT_MIN + 1, INT_MAX - 1, 641, 6700417, 1000000007});
    for (int i = 0; i < 2500; i++) { int32_t d = int32_t(R()); if (d) ds.push_back(d); }
    Batch b;
    for (int32_t d : ds) {
        b.begin(); b.a.sdiv_imm(W(0), W(0), d); b.a.ret();
        b.begin(); b.a.srem_imm(W(0), W(0), d); b.a.ret();
    }
    b.done();
    long n = 0;
    for (size_t i = 0; i < ds.size(); i++) {
        auto q = b.fn<int32_t (*)(int32_t)>(2 * i), r = b.fn<int32_t (*)(int32_t)>(2 * i + 1);
        int32_t d = ds[i];
        for (int j = 0; j < 1200; j++, n++) {
            int32_t v;
            switch (j) {
            case 0: v = INT_MIN; break; case 1: v = INT_MAX; break; case 2: v = 0; break; case 3: v = -1; break; case 4: v = 1; break;
            default:
                if (j < 40) v = int32_t(uint32_t(d) * uint32_t(j - 20) + uint32_t(j % 3) - 1);  // around multiples
                else v = int32_t(R());
            }
            int32_t wq = (d == -1) ? int32_t(0u - uint32_t(v)) : v / d;
            int32_t wr = (d == -1) ? 0 : v % d;
            int32_t gq = q(v), gr = r(v);
            CHECK(gq == wq, "sdiv_imm %d / %d = %d, want %d", v, d, gq, wq);
            CHECK(gr == wr, "srem_imm %d %% %d = %d, want %d", v, d, gr, wr);
        }
    }
    report("sdiv_imm / srem_imm vs C on random + edge inputs", errs, std::to_string(ds.size()) + " divisors, " + std::to_string(n) + " dividends each for / and %");

    // exhaustive over all 2^32 dividends for a few divisors, compared to hardware sdiv inside a JIT loop
    errs = 0;
    std::vector<int32_t> ex = {7, -3, 1000000007, INT_MIN, 1 << 12};
    int step = g_exhaustive ? 1 : 61;  // default: every 61st dividend (~70M per divisor)
    Batch e;
    for (int32_t d : ex) {
        Assembler& a = e.a;
        e.begin();
        Label loop = a.new_label(), bad = a.new_label();
        a.load_imm(W(1), INT_MIN);
        a.load_imm(W(2), d);
        a.bind(loop);
        a.sdiv_imm(W(3), W(1), d);
        a.sdiv(W(4), W(1), W(2));
        a.cmp(W(3), W(4));
        a.b(NE, bad);
        a.srem_imm(W(5), W(1), d);
        a.msub(W(6), W(4), W(2), W(1));
        a.cmp(W(5), W(6));
        a.b(NE, bad);
        a.adds(W(1), W(1), uint32_t(step));
        a.b(VC, loop);
        a.mov(X(0), xzr);
        a.ret();
        a.bind(bad);
        a.orr(X(0), X(1), uint64_t(1) << 40);
        a.ret();
    }
    e.done();
    double t0 = now_us();
    for (size_t i = 0; i < ex.size(); i++) {
        uint64_t r = e.fn<uint64_t (*)()>(i)();
        CHECK(r == 0, "exhaustive sdiv/srem by %d failed at dividend %d", ex[i], int32_t(r));
    }
    double t = (now_us() - t0) / 1e6;
    char buf[96];
    snprintf(buf, sizeof buf, "%zu divisors x 2^32/%d dividends in %.1fs", ex.size(), step, t);
    report("sdiv_imm / srem_imm exhaustive vs hardware sdiv", errs, buf);
}

// ---------------------------------------------------------------- control flow, calls, frames
static void test_fib() {
    int errs = 0;
    Batch b;
    Assembler& a = b.a;
    // iterative fib(w0)
    b.begin();
    {
        Label loop = a.new_label(), done = a.new_label();
        a.mov(W(1), wzr); a.load_imm(W(2), 1);
        a.cbz(W(0), done);
        a.bind(loop);
        a.add(W(3), W(1), W(2)); a.mov(W(1), W(2)); a.mov(W(2), W(3));
        a.subs(W(0), W(0), 1);
        a.b(NE, loop);
        a.bind(done);
        a.mov(W(0), W(1));
        a.ret();
    }
    // recursive fib
    b.begin();
    {
        Label self = a.here(), base = a.new_label(), out = a.new_label();
        auto f = a.prologue(1u << 19 | 1u << 20, 0);
        a.mov(W(19), W(0));
        a.cmp(W(19), 2);
        a.b(LT, base);
        a.sub(W(0), W(19), 1); a.bl(self); a.mov(W(20), W(0));
        a.sub(W(0), W(19), 2); a.bl(self); a.add(W(0), W(0), W(20));
        a.b(out);
        a.bind(base);
        a.mov(W(0), W(19));
        a.bind(out);
        a.epilogue(f);
    }
    b.done();
    auto it = b.fn<int32_t (*)(int32_t)>(0), rec = b.fn<int32_t (*)(int32_t)>(1);
    int64_t x = 0, y = 1;
    for (int n = 0; n <= 46; n++) {
        CHECK(it(n) == int32_t(x), "fib_iter(%d) = %d want %lld", n, it(n), (long long)x);
        if (n <= 25) CHECK(rec(n) == int32_t(x), "fib_rec(%d) = %d want %lld", n, rec(n), (long long)x);
        int64_t t = x + y; x = y; y = t;
    }
    double t0 = now_us();
    int32_t r32 = rec(32);
    double t = now_us() - t0;
    CHECK(r32 == 2178309, "fib_rec(32) = %d", r32);
    char buf[64]; snprintf(buf, sizeof buf, "fib_rec(32) in %.1f ms", t / 1000);
    report("iterative + recursive fibonacci (bl to self, prologue/epilogue)", errs, buf);
}

struct ListHdr { int32_t* data; int32_t len; int32_t cap; };

static void test_array() {
    int errs = 0;
    Batch b;
    Assembler& a = b.a;
    // sum32(hdr, n): sum of data[0..n) with unsigned bounds check against len; -1 if out of range
    for (int variant = 0; variant < 2; variant++) {
        b.begin();
        Label loop = a.new_label(), done = a.new_label(), oob = a.new_label();
        a.ldr(X(2), mem(X(0), 0));
        a.ldr(W(3), mem(X(0), 8));
        a.mov(W(4), wzr);
        a.mov(X(5), xzr);
        a.cbz(W(1), done);
        a.bind(loop);
        a.cmp(W(4), W(3));
        a.b(HS, oob);
        if (variant == 0) { a.ldr(W(6), memr(X(2), W(4), UXTW, true)); a.add(W(5), W(5), W(6)); }
        else { a.ldrsw(X(6), memr(X(2), W(4), SXTW, true)); a.add(X(5), X(5), X(6)); }
        a.add(W(4), W(4), 1);
        a.cmp(W(4), W(1));
        a.b(LO, loop);
        a.bind(done);
        a.mov(X(0), X(5));
        a.ret();
        a.bind(oob);  // cold path, out of line
        a.load_imm(X(0), -1);
        a.ret();
    }
    // store loop: data[i] = i * 3 via str w, [x, x, lsl #2]
    b.begin();
    {
        Label loop = a.new_label(), done = a.new_label();
        a.mov(X(2), xzr);
        a.cbz(X(1), done);
        a.bind(loop);
        a.add(W(3), W(2), W(2), LSL, 1);
        a.str(W(3), memr(X(0), X(2), UXTX, true));
        a.add(X(2), X(2), 1);
        a.cmp(X(2), X(1));
        a.b(NE, loop);
        a.bind(done);
        a.ret();
    }
    b.done();
    std::vector<int32_t> v(10000);
    for (auto& e : v) e = int32_t(R());
    ListHdr h{v.data(), int32_t(v.size()), int32_t(v.size())};
    auto s32 = b.fn<int64_t (*)(ListHdr*, uint32_t)>(0), s64 = b.fn<int64_t (*)(ListHdr*, uint32_t)>(1);
    for (uint32_t n : {0u, 1u, 7u, 9999u, 10000u}) {
        int32_t w = 0; int64_t w64 = 0;
        for (uint32_t i = 0; i < n; i++) { w = int32_t(uint32_t(w) + uint32_t(v[i])); w64 += v[i]; }
        CHECK(s32(&h, n) == int64_t(uint32_t(w)), "sum32 n=%u", n);
        CHECK(s64(&h, n) == w64, "sum64 n=%u", n);
    }
    CHECK(s32(&h, 10001) == -1, "out-of-bounds not detected");
    h.len = 5;
    CHECK(s64(&h, 6) == -1, "out-of-bounds (len 5) not detected");
    auto st = b.fn<void (*)(int32_t*, uint64_t)>(2);
    st(v.data(), v.size());
    bool ok = true;
    for (size_t i = 0; i < v.size(); i++) ok &= v[i] == int32_t(3 * i);
    CHECK(ok, "store loop");
    report("int32 array loops: scaled register addressing, unsigned bounds check, cold path", errs);
}

extern "C" int64_t helper8(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f, int64_t g, int64_t h) {
    char buf[256];
    snprintf(buf, sizeof buf, "%lld", (long long)a);  // touch libc, clobber caller-saved state
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h + (buf[0] == '-');
}

// abi_check(fn, arg, uint64_t* result): sets x19..x28 to sentinels, calls fn(arg),
// returns number of callee-saved registers (and sp) that fn failed to preserve.
static void emit_abi_check(Assembler& a) {
    auto f = a.prologue(0x1ff80000u, 16);
    a.str(X(2), mem(sp, f.locals_off));
    for (int r = 19; r <= 28; r++) a.load_imm(X(r), int64_t(0x9e3779b97f4a7c15ull * r));
    a.mov(X(16), X(0));
    a.mov(X(0), X(1));
    a.blr(X(16));
    a.ldr(X(3), mem(fp, f.locals_off));
    a.str(X(0), mem(X(3)));
    a.mov(X(1), xzr);
    for (int r = 19; r <= 28; r++) {
        a.load_imm(X(2), int64_t(0x9e3779b97f4a7c15ull * r));
        a.cmp(X(r), X(2));
        a.cinc(X(1), X(1), NE);
    }
    a.mov(X(2), sp);
    a.cmp(X(2), fp);
    a.cinc(X(1), X(1), NE);
    a.mov(X(0), X(1));
    a.epilogue(f);
}
using AbiCheck = uint64_t (*)(void*, uint64_t, uint64_t*);

static void test_calls_and_frames() {
    int errs = 0;
    Batch b;
    Assembler& a = b.a;
    b.begin();
    emit_abi_check(a);
    // 1: calls helper8 through blr, keeping values live in callee-saved regs; also
    //    calls a JIT "trasher" that clobbers every caller-saved register.
    Label trasher = a.new_label();
    b.begin();
    {
        auto f = a.prologue(1u << 19 | 1u << 20 | 1u << 21 | 1u << 25, 0);
        a.mov(X(19), X(0));
        a.add(X(20), X(0), 100);
        a.load_imm(X(21), 0x123456789);
        a.mov(X(25), xzr);
        a.bl(trasher);              // forward bl to a function emitted later
        for (int i = 0; i < 8; i++) a.add(X(i), X(19), i);
        a.call_abs(reinterpret_cast<void*>(&helper8));
        a.add(X(0), X(0), X(20));
        a.add(X(0), X(0), X(21));
        a.add(X(0), X(0), X(25));
        a.epilogue(f);
    }
    b.begin();
    a.bind(trasher);
    for (int r = 0; r <= 17; r++) a.load_imm(X(r), -1 - r);
    a.ret();
    // 3..: every subset of x19..x28 x several local sizes
    std::vector<std::pair<uint32_t, uint32_t>> fr;
    for (uint32_t m = 0; m < 1024; m++) for (uint32_t loc : {0u, 40u, 600u}) {
        uint32_t mask = m << 19;
        fr.push_back({mask, loc});
        b.begin();
        auto f = a.prologue(mask, loc);
        for (int r = 19; r <= 28; r++) if (mask >> r & 1) a.load_imm(X(r), r * 7 + 1);
        for (uint32_t o = 0; o + 8 <= loc; o += 8) a.str(X(0), mem(sp, f.locals_off + o));
        a.mov(X(1), xzr);
        for (uint32_t o = 0; o + 8 <= loc; o += 8) { a.ldr(X(2), mem(sp, f.locals_off + o)); a.add(X(1), X(1), X(2)); }
        a.add(X(0), X(0), X(1));
        for (int r = 19; r <= 28; r++) if (mask >> r & 1) a.add(X(0), X(0), X(r));
        a.epilogue(f);
    }
    b.done();
    auto chk = b.fn<AbiCheck>(0);
    uint64_t res = 0;
    for (int64_t arg : {0, 5, -77, 1 << 30}) {
        uint64_t bad = chk(b.base + b.entry[1], uint64_t(arg), &res);
        int64_t want = helper8(arg, arg + 1, arg + 2, arg + 3, arg + 4, arg + 5, arg + 6, arg + 7) + arg + 100 + 0x123456789;
        CHECK(bad == 0, "helper caller clobbered %llu callee-saved regs", (unsigned long long)bad);
        CHECK(int64_t(res) == want, "helper call result %lld want %lld", (long long)res, (long long)want);
    }
    for (size_t i = 0; i < fr.size(); i++) {
        auto [mask, loc] = fr[i];
        uint64_t arg = 1000 + i;
        uint64_t bad = chk(b.base + b.entry[3 + i], arg, &res);
        uint64_t want = arg + arg * (loc / 8);
        for (int r = 19; r <= 28; r++) if (mask >> r & 1) want += r * 7 + 1;
        CHECK(bad == 0, "prologue mask %x locals %u: %llu regs not preserved", mask, loc, (unsigned long long)bad);
        CHECK(res == want, "prologue mask %x locals %u: result %llu want %llu", mask, loc, (unsigned long long)res, (unsigned long long)want);
    }
    report("blr to C helper + bl to JIT clobberer with callee-saved discipline; prologue/epilogue for all 1024 subsets x 3 frame sizes", errs);
}

static void test_select() {
    int errs = 0;
    Batch b;
    Assembler& a = b.a;
    // 0: (a < b && b < c) ? a : c
    b.begin(); a.cmp(W(0), W(1)); a.ccmp(W(1), W(2), 0, LT); a.csel(W(0), W(0), W(2), LT); a.ret();
    // 1: (a == 5 || b == -3)
    b.begin(); a.cmp(W(0), 5); a.ccmn(W(1), 3, 4, NE); a.cset(W(0), EQ); a.ret();
    // 2: a > b ? a + 1 : -b   via cinc / cneg
    b.begin(); a.cmp(W(0), W(1)); a.cinc(W(3), W(0), GT); a.cneg(W(4), W(1), LE); a.csel(W(0), W(3), W(4), GT); a.ret();
    // 3: unsigned a < b ? -1 : 0 via csetm; 4: max via csel; 5: abs via cneg
    b.begin(); a.cmp(X(0), X(1)); a.csetm(X(0), LO); a.ret();
    b.begin(); a.cmp(W(0), W(1)); a.csel(W(0), W(0), W(1), GE); a.ret();
    b.begin(); a.cmp(W(0), 0); a.cneg(W(0), W(0), LT); a.ret();
    // 6: csinv / csneg / csinc direct
    b.begin(); a.cmp(W(0), W(1)); a.csinv(W(3), W(0), W(1), EQ); a.csneg(W(4), W(0), W(1), NE); a.csinc(W(5), W(0), W(1), HI);
           a.add(W(0), W(3), W(4)); a.add(W(0), W(0), W(5)); a.ret();
    b.done();
    auto f0 = b.fn<int32_t (*)(int32_t, int32_t, int32_t)>(0);
    auto f1 = b.fn<int32_t (*)(int32_t, int32_t)>(1);
    auto f2 = b.fn<int32_t (*)(int32_t, int32_t)>(2);
    auto f3 = b.fn<int64_t (*)(uint64_t, uint64_t)>(3);
    auto f4 = b.fn<int32_t (*)(int32_t, int32_t)>(4);
    auto f5 = b.fn<int32_t (*)(int32_t)>(5);
    auto f6 = b.fn<int32_t (*)(int32_t, int32_t)>(6);
    for (int i = 0; i < 200000; i++) {
        int32_t x = int32_t(R() % 13) - 6, y = int32_t(R() % 13) - 6, z = int32_t(R() % 13) - 6;
        if (i & 1) { x = int32_t(R()); y = int32_t(R()); z = int32_t(R()); }
        if (i % 7 == 0) x = 5;
        if (i % 11 == 0) y = -3;
        CHECK(f0(x, y, z) == ((x < y && y < z) ? x : z), "and-chain %d %d %d", x, y, z);
        CHECK(f1(x, y) == (x == 5 || y == -3), "or-chain %d %d", x, y);
        CHECK(f2(x, y) == (x > y ? int32_t(uint32_t(x) + 1) : int32_t(0u - uint32_t(y))), "cinc/cneg %d %d", x, y);
        CHECK(f3(uint64_t(x), uint64_t(y)) == (uint64_t(x) < uint64_t(y) ? -1 : 0), "csetm");
        CHECK(f4(x, y) == (x >= y ? x : y), "max");
        CHECK(f5(x) == int32_t(x < 0 ? 0u - uint32_t(x) : uint32_t(x)), "abs");
        int32_t w = int32_t(uint32_t(x == y ? x : ~y) + uint32_t(x != y ? x : int32_t(0u - uint32_t(y))) +
                            uint32_t(uint32_t(x) > uint32_t(y) ? x : int32_t(uint32_t(y) + 1)));
        CHECK(f6(x, y) == w, "csinv/csneg/csinc %d %d", x, y);
    }
    report("csel/csinc/csinv/csneg/cset/csetm/cinc/cneg/ccmp/ccmn sequences vs C", errs, "200000 random triples");
}

static void test_labels() {
    int errs = 0;
    std::string info;
    // far forward conditional branches of every kind, over > range of padding
    struct K { const char* name; int kind; int pad; };
    K ks[] = {{"b.cond", 0, 1100 * 1024}, {"cbz", 1, 1100 * 1024}, {"cbnz", 2, 1100 * 1024}, {"tbz", 3, 40 * 1024}, {"tbnz", 4, 40 * 1024}};
    for (const K& k : ks) for (int dir = 0; dir < 2; dir++) {
        Batch b;
        Assembler& a = b.a;
        Label target = a.new_label();
        auto branch = [&] {
            switch (k.kind) {
            case 0: a.cmp(X(0), 0); a.b(EQ, target); break;
            case 1: a.cbz(X(0), target); break;
            case 2: a.cbnz(X(1), target); break;
            case 3: a.tbz(X(0), 0, target); break;
            default: a.tbnz(X(1), 0, target);
            }
        };
        // x0 = arg, x1 = arg ^ 1: taken when arg is 0 (all kinds)
        if (dir == 0) {  // forward
            b.begin();
            a.eor(X(1), X(0), 1);
            for (int i = 0; i < 50; i++) branch();  // many pending branches to one far label
            a.load_imm(X(0), 1);
            a.ret();
            for (int i = 0; i < k.pad / 4; i++) a.brk(1);
            a.bind(target);
            a.load_imm(X(0), 2);
            a.ret();
        } else {  // backward
            b.begin();
            a.bind(target);
            a.load_imm(X(0), 2);
            a.ret();
            for (int i = 0; i < k.pad / 4; i++) a.brk(1);
            b.begin();
            a.eor(X(1), X(0), 1);
            branch();
            a.load_imm(X(0), 1);
            a.ret();
        }
        size_t islands = a.island_count();
        b.done();
        auto f = b.fn<uint64_t (*)(uint64_t)>(dir == 0 ? 0 : 1);
        CHECK(f(0) == 2, "%s %s: taken path wrong", k.name, dir ? "backward" : "forward");
        CHECK(f(1) == 1, "%s %s: fallthrough wrong", k.name, dir ? "backward" : "forward");
        if (dir == 0) CHECK(islands > 0, "%s forward: no island emitted", k.name);
    }
    // in-range forward branch: no veneer, single instruction
    {
        Batch b; Assembler& a = b.a;
        b.begin();
        Label l = a.new_label();
        a.cbz(X(0), l); a.load_imm(X(0), 1); a.ret(); a.bind(l); a.load_imm(X(0), 2); a.ret();
        CHECK(a.island_count() == 0 && a.size() == 5 * 4, "short branch grew");
        b.done();
        CHECK(b.fn<uint64_t (*)(uint64_t)>(0)(0) == 2, "short cbz");
    }
    // bl to a function emitted later and earlier; adr / adrp+add / literal address agree
    {
        Batch b; Assembler& a = b.a;
        Label later = a.new_label(), earlier = a.new_label(), data = a.new_label();
        b.begin();
        a.bind(earlier); a.add(X(0), X(0), 10); a.ret();
        b.begin();
        {
            auto f = a.prologue(0, 0);
            a.bl(later); a.bl(earlier);
            a.epilogue(f);
        }
        b.begin();
        a.bind(later); a.add(X(0), X(0), X(0)); a.ret();
        b.begin();  // 3: returns 0 iff adr == adrp_add == ldr_lit_addr
        a.adr(X(1), data); a.adrp_add(X(2), data); a.ldr_lit_addr(X(3), data);
        a.sub(X(0), X(1), X(2)); a.sub(X(4), X(1), X(3)); a.orr(X(0), X(0), X(4)); a.ret();
        b.begin();  // 4: returns address of data
        a.adr(X(0), data); a.ret();
        a.align(8); a.bind(data); a.emit_u64(0x1122334455667788ull);
        b.begin();  // 5: call `earlier` through a function pointer from the literal pool
        {
            auto f = a.prologue(0, 0);
            a.ldr_lit_addr(X(9), earlier); a.blr(X(9));
            a.epilogue(f);
        }
        b.done();
        CHECK(b.fn<uint64_t (*)(uint64_t)>(1)(5) == 20, "bl later/earlier");
        CHECK(b.fn<uint64_t (*)()>(3)() == 0, "adr/adrp/literal address disagree");
        CHECK(*reinterpret_cast<uint64_t*>(b.fn<uint64_t (*)()>(4)()) == 0x1122334455667788ull, "adr data");
        CHECK(b.fn<uint64_t (*)(uint64_t)>(5)(1) == 11, "call via literal function pointer");
    }
    // literal pool: many distinct values, and a literal whose use is > 1 MB before the end
    {
        Batch b; Assembler& a = b.a;
        std::vector<uint64_t> vals;
        for (int i = 0; i < 500; i++) vals.push_back(R());
        b.begin();
        a.mov(X(0), xzr);
        for (int rep = 0; rep < 2; rep++)   // second pass reuses the same pool slots
            for (uint64_t v : vals) { a.ldr_lit(X(1), v); a.eor(X(0), X(0), X(1)); }
        a.ldr_lit(W(2), 0xdeadbeef); a.add(X(0), X(0), X(2));
        a.ret();
        b.begin();
        Label end = a.new_label();
        a.ldr_lit(X(0), 0xfeedfacecafebeefull);
        a.b(end);
        for (int i = 0; i < 300000; i++) a.brk(2);
        a.bind(end);
        a.ret();
        b.done();
        CHECK(b.fn<uint64_t (*)()>(0)() == 0xdeadbeefull, "literal xor");
        CHECK(b.fn<uint64_t (*)()>(1)() == 0xfeedfacecafebeefull, "far literal");
        info = "islands in far-literal buffer: " + std::to_string(a.island_count());
    }
    report("labels: far forward/backward b.cond/cbz/cbnz/tbz/tbnz, veneers, bl across functions, adr/adrp/literal pool", errs, info);
}

static uint64_t on_big_stack(void* p) { return reinterpret_cast<int32_t (*)(int32_t)>(p)(27); }

static void test_bigstack_exec() {
    int errs = 0;
    Batch b; Assembler& a = b.a;
    b.begin();
    Label self = a.here(), base = a.new_label();
    auto f = a.prologue(1u << 19 | 1u << 20, 0);
    a.mov(W(19), W(0)); a.cmp(W(19), 2); a.b(LT, base);
    a.sub(W(0), W(19), 1); a.bl(self); a.mov(W(20), W(0));
    a.sub(W(0), W(19), 2); a.bl(self); a.add(W(19), W(0), W(20));
    a.bind(base); a.mov(W(0), W(19));
    a.epilogue(f);
    b.done();
    jit::BigStack bs;
    CHECK(bs.create(), "BigStack::create failed");
    uint64_t r = bs.run(on_big_stack, b.base);
    CHECK(r == 196418, "fib(27) on big stack = %llu", (unsigned long long)r);
    report("JIT code running on the big stack", errs, "stack " + std::to_string(bs.size() >> 30) + " GB");
}

int main(int argc, char** argv) {
    bool asan_safe = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--asan-safe")) asan_safe = true;
        if (!strcmp(argv[i], "--exhaustive")) g_exhaustive = true;
    }
    double t0 = now_us();
    if (!region.init()) { printf("FAIL region init\n"); return 1; }
    double t1 = now_us();
    printf("code region reserve: %.1f us\n", t1 - t0);
    test_load_imm();
    test_add_cmp_imm();
    test_mul_imm();
    test_div_imm();
    test_fib();
    test_array();
    test_calls_and_frames();
    test_select();
    test_labels();
    if (!asan_safe) test_bigstack_exec();
    printf("%ld checks, %d failed tests, code region used %.1f MB\n", checks, failures, region.used() / 1048576.0);
    return failures > 100 ? 100 : failures;
}
