// A very large, lazily committed stack for generated code.
//
// create() reserves the largest size it can (64 GB down to 4 GB) with an
// anonymous PROT_READ|WRITE mapping; the kernel commits pages only when touched,
// so the reservation is nearly free. The lowest `guard` bytes are PROT_NONE.
// run() switches sp to the top of the stack, calls fn(arg), switches back.
//
// Overflow: a SIGSEGV/SIGBUS handler running on an alternate signal stack
// checks whether the fault address (or the faulting sp) lies in a guard region
// of a live BigStack; if so it calls the overflow callback (which must not
// return: the runtime prints "Stack overflow" and _exit(1)s). Any other fault is
// handed to the previously installed handler / default action, so it crashes
// normally. Generated frames must be smaller than the guard (no stack probing).
#pragma once
#include <cstddef>
#include <cstdint>

namespace jit {

using OverflowCallback = void (*)();

class BigStack {
public:
    // Reserve the largest stack available up to `max_bytes`; false on failure.
    bool create(size_t max_bytes = size_t(64) << 30, size_t guard = size_t(16) << 20);
    ~BigStack();

    // Run fn(arg) on this stack (on the calling thread) and return its result.
    // Also installs the signal handler + alternate signal stack for this thread.
    uint64_t run(uint64_t (*fn)(void*), void* arg);

    size_t size() const { return size_; }
    uint8_t* lo() const { return base_; }
    uint8_t* top() const { return base_ + size_; }
    size_t guard() const { return guard_; }

private:
    uint8_t* base_ = nullptr;
    size_t size_ = 0, guard_ = 0;
};

// Called (on the signal stack) when a guard page of any BigStack is hit.
void set_overflow_callback(OverflowCallback cb);
// Install SIGSEGV/SIGBUS handler and a sigaltstack for the calling thread (idempotent).
void install_overflow_handler();

} // namespace jit

// Assembly trampoline: switches to `stack_top` (16-byte aligned), calls fn(arg).
extern "C" uint64_t epq_bigstack_switch(uint64_t (*fn)(void*), void* arg, void* stack_top);
