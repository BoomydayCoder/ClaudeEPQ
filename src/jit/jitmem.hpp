// Executable memory for the JIT (macOS / Apple Silicon).
//
// One MAP_JIT region is reserved up front (default 128 MB, the reach of b/bl)
// and committed lazily by the kernel as pages are touched. MAP_JIT pages are
// either writable or executable *per thread*: pthread_jit_write_protect_np(0)
// makes them writable for the calling thread, (1) executable again. Code must
// be made executable (and the icache invalidated) before it is called.
#pragma once
#include <cstddef>
#include <cstdint>

namespace a64 { class Assembler; }

namespace jit {

class CodeRegion {
public:
    // Reserve `reserve` bytes of MAP_JIT memory. Returns false on failure.
    bool init(size_t reserve = size_t(128) << 20);
    ~CodeRegion();

    // Bump-allocate `n` bytes, 4 KB aligned (adrp needs a 4 KB aligned base).
    uint8_t* alloc(size_t n);
    // Copy bytes into the region (handles write protection + icache).
    void write(uint8_t* dst, const void* src, size_t n);
    // Allocate, finalize the assembler against the address, copy, make executable.
    // Returns the start of the code.
    uint8_t* install(a64::Assembler& as);

    uint8_t* base() const { return base_; }
    size_t used() const { return used_; }
    size_t reserved() const { return reserved_; }

private:
    uint8_t* base_ = nullptr;
    size_t reserved_ = 0, used_ = 0;
};

// Toggle write access to MAP_JIT memory for the calling thread.
void jit_writable(bool writable);
void flush_icache(void* p, size_t n);

} // namespace jit
