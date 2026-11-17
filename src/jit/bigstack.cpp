#include "bigstack.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstring>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>

// x0 = fn, x1 = arg, x2 = new stack top. The old sp is saved on the new stack;
// x29 keeps pointing into the old stack so the frame chain stays walkable.
asm(R"(
    .text
    .globl _epq_bigstack_switch
    .p2align 2
_epq_bigstack_switch:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x9, sp
    mov sp, x2
    stp x9, xzr, [sp, #-16]!
    mov x16, x0
    mov x0, x1
    blr x16
    ldr x9, [sp]
    mov sp, x9
    ldp x29, x30, [sp], #16
    ret
)");

namespace jit {

namespace {
struct Range { std::atomic<uintptr_t> lo{0}, hi{0}; };
constexpr int kMaxStacks = 16;
Range g_guards[kMaxStacks];
std::atomic<OverflowCallback> g_cb{nullptr};
struct sigaction g_old_segv, g_old_bus;
std::atomic<bool> g_installed{false};
thread_local bool t_altstack = false;

void default_overflow() {
    static const char msg[] = "Stack overflow\n";
    (void)!write(2, msg, sizeof msg - 1);
    _exit(1);
}

void on_fault(int sig, siginfo_t* si, void* uctx) {
    uintptr_t a = reinterpret_cast<uintptr_t>(si->si_addr);
    uintptr_t spv = static_cast<ucontext_t*>(uctx)->uc_mcontext->__ss.__sp;
    for (Range& r : g_guards) {
        uintptr_t lo = r.lo.load(std::memory_order_relaxed), hi = r.hi.load(std::memory_order_relaxed);
        if (!hi) continue;
        if ((a >= lo && a < hi) || (spv >= lo && spv < hi)) {
            OverflowCallback cb = g_cb.load();
            (cb ? cb : default_overflow)();
            _exit(1);  // callback must not return
        }
    }
    // Not ours: restore the previous disposition and let the fault recur.
    sigaction(sig, sig == SIGSEGV ? &g_old_segv : &g_old_bus, nullptr);
}
} // namespace

void set_overflow_callback(OverflowCallback cb) { g_cb.store(cb); }

void install_overflow_handler() {
    if (!t_altstack) {
        size_t sz = 64 * 1024;
        void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p != MAP_FAILED) {
            stack_t ss{};
            ss.ss_sp = p;
            ss.ss_size = sz;
            sigaltstack(&ss, nullptr);
            t_altstack = true;
        }
    }
    bool expected = false;
    if (!g_installed.compare_exchange_strong(expected, true)) return;
    struct sigaction sa{};
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &g_old_segv);
    sigaction(SIGBUS, &sa, &g_old_bus);
}

bool BigStack::create(size_t max_bytes, size_t guard) {
    for (size_t sz = max_bytes, lo = std::min(max_bytes, size_t(4) << 30); sz >= lo; sz /= 2) {
        void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) continue;
        if (mprotect(p, guard, PROT_NONE) != 0) { munmap(p, sz); return false; }
        base_ = static_cast<uint8_t*>(p);
        size_ = sz;
        guard_ = guard;
        for (Range& r : g_guards) {
            uintptr_t z = 0;
            if (r.lo.load() == 0 && r.hi.load() == 0) {
                r.lo.store(reinterpret_cast<uintptr_t>(p));
                r.hi.compare_exchange_strong(z, reinterpret_cast<uintptr_t>(p) + guard);
                break;
            }
        }
        return true;
    }
    return false;
}

BigStack::~BigStack() {
    if (!base_) return;
    for (Range& r : g_guards)
        if (r.lo.load() == reinterpret_cast<uintptr_t>(base_)) { r.hi.store(0); r.lo.store(0); }
    munmap(base_, size_);
}

uint64_t BigStack::run(uint64_t (*fn)(void*), void* arg) {
    install_overflow_handler();
    return epq_bigstack_switch(fn, arg, top());
}

} // namespace jit
