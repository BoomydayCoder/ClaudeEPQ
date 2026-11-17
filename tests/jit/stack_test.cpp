// Big-stack tests; one scenario per process (run.py checks exit status/output).
//   deep         JIT recursion to depth 1e7 (64-byte frames) + libc calls at the bottom
//   overflow     runaway C recursion -> overflow callback ("Stack overflow", exit 1)
//   jitoverflow  runaway JIT recursion -> overflow callback
//   nullderef    unrelated SIGSEGV on the big stack -> normal crash
//   bench        start-up costs: code region, first install, stack reservation sizes
#include "jit/a64.hpp"
#include "jit/bigstack.hpp"
#include "jit/jitmem.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

using namespace a64;

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

static void on_overflow() {
    static const char m[] = "Stack overflow\n";
    (void)!write(2, m, sizeof m - 1);
    _exit(1);
}

static int libc_probe_ok = 0;
extern "C" void libc_probe() {
    char* p = static_cast<char*>(malloc(1 << 20));
    memset(p, 'x', 1 << 20);
    char* q = static_cast<char*>(malloc(1 << 20));
    memcpy(q, p, 1 << 20);
    char line[64];
    int n = snprintf(line, sizeof line, "libc at depth: %s\n", q[12345] == 'x' ? "ok" : "bad");
    (void)!write(1, line, size_t(n));
    libc_probe_ok = q[(1 << 20) - 1] == 'x';
    free(p);
    free(q);
}

// rec(n) = n == 0 ? (libc_probe(), 0) : rec(n - 1) + 1, 64-byte frames; `bottomless` drops the base case
static void* build_rec(jit::CodeRegion& cr, bool bottomless) {
    Assembler a;
    Label self = a.here(), base = a.new_label();
    auto f = a.prologue(1u << 19, 32);   // 16 (fp/lr) + 16 (x19) + 32 locals = 64 bytes
    a.mov(X(19), X(0));
    if (!bottomless) a.cbz(X(19), base);
    a.str(X(19), mem(sp, f.locals_off));  // touch the frame
    a.sub(X(0), X(19), 1);
    a.bl(self);
    a.add(X(0), X(0), 1);
    a.epilogue(f);
    a.bind(base);
    a.call_abs(reinterpret_cast<void*>(&libc_probe));
    a.mov(X(0), xzr);
    a.epilogue(f);
    return cr.install(a);
}

static void* g_fn;
static uint64_t call_rec(void* n) { return reinterpret_cast<uint64_t (*)(uint64_t)>(g_fn)(reinterpret_cast<uint64_t>(n)); }

__attribute__((noinline)) static uint64_t c_runaway(uint64_t n) {
    volatile char pad[48];
    pad[0] = char(n);
    return c_runaway(n + 1) + pad[0];
}
static uint64_t call_c_runaway(void*) { return c_runaway(0); }
static uint64_t null_deref(void*) { *reinterpret_cast<volatile int*>(16) = 1; return 0; }

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const char* mode = argv[1];
    jit::set_overflow_callback(on_overflow);
    jit::CodeRegion cr;
    if (!strcmp(mode, "bench")) {
        double t0 = now_us();
        bool ok = cr.init();
        double t1 = now_us();
        Assembler a;
        a.load_imm(X(0), 42);
        a.ret();
        double t2 = now_us();
        auto f = reinterpret_cast<uint64_t (*)()>(cr.install(a));
        uint64_t r = f();
        double t3 = now_us();
        Assembler a2;
        for (int i = 0; i < 4096; i++) a2.add(X(0), X(0), 1);
        a2.ret();
        double t4 = now_us();
        cr.install(a2);
        double t5 = now_us();
        printf("code region reserve (128 MB MAP_JIT): %.1f us\n", t1 - t0);
        printf("first install+call (W^X toggle, copy, icache): %.1f us\n", t3 - t2);
        printf("install 16 KB of code: %.1f us\n", t5 - t4);
        double c0 = now_us();
        jit::BigStack bs;
        bool sok = bs.create();
        double c1 = now_us();
        uint64_t rr = bs.run([](void*) -> uint64_t { return 7; }, nullptr);
        double c2 = now_us();
        uint64_t rr2 = bs.run([](void*) -> uint64_t { return 8; }, nullptr);
        double c3 = now_us();
        printf("BigStack::create: %zu GB in %.1f us\n", bs.size() >> 30, c1 - c0);
        printf("first run (sigaltstack + handlers): %.1f us; later run: %.2f us\n", c2 - c1, c3 - c2);
        // how far does the kernel go?
        size_t biggest = 0;
        for (size_t gb = 4; gb <= 16384; gb *= 2) {
            double s0 = now_us();
            void* p = mmap(nullptr, gb << 30, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
            double s1 = now_us();
            if (p == MAP_FAILED) { printf("mmap %zu GB: failed\n", gb); break; }
            printf("mmap %zu GB: %.1f us\n", gb, s1 - s0);
            biggest = gb;
            munmap(p, gb << 30);
        }
        printf("largest reservation: %zu GB\n", biggest);
        for (size_t gb = 4; gb <= 64; gb *= 2) {
            for (size_t guard_kb : {256, 1024, 16384}) {
                double s0 = now_us();
                jit::BigStack t;
                bool k = t.create(gb << 30, guard_kb << 10);
                double s1 = now_us();
                printf("BigStack::create(%zu GB, guard %zu KB): %.1f us%s\n", gb, guard_kb, s1 - s0, k ? "" : " FAILED");
            }
        }
        return ok && sok && r == 42 && rr == 7 && rr2 == 8 ? 0 : 1;
    }
    if (!cr.init()) { puts("region init failed"); return 3; }
    jit::BigStack bs;
    if (!strcmp(mode, "deep")) {
        if (!bs.create()) { puts("create failed"); return 3; }
        g_fn = build_rec(cr, false);
        double t0 = now_us();
        uint64_t r = bs.run(call_rec, reinterpret_cast<void*>(uint64_t(10000000)));
        double t1 = now_us();
        printf("depth %llu in %.0f ms\n", (unsigned long long)r, (t1 - t0) / 1000);
        uint64_t r2 = bs.run(call_rec, reinterpret_cast<void*>(uint64_t(10000000)));
        printf("again (pages committed) in %.0f ms\n", (now_us() - t1) / 1000);
        if (r == 10000000 && r2 == r && libc_probe_ok) puts("OK");
        return 0;
    }
    // smaller stack so overflow does not commit tens of GB
    if (!bs.create(size_t(1) << 30, size_t(1) << 20)) { puts("create failed"); return 3; }
    if (!strcmp(mode, "overflow")) bs.run(call_c_runaway, nullptr);
    else if (!strcmp(mode, "jitoverflow")) { g_fn = build_rec(cr, true); bs.run(call_rec, reinterpret_cast<void*>(uint64_t(5))); }
    else if (!strcmp(mode, "nullderef")) bs.run(null_deref, nullptr);
    puts("returned normally?!");
    return 4;
}
