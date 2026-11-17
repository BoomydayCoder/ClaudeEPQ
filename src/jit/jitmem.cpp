#include "jitmem.hpp"
#include "a64.hpp"

#include <cstring>
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>

namespace jit {

void jit_writable(bool writable) { pthread_jit_write_protect_np(writable ? 0 : 1); }
void flush_icache(void* p, size_t n) { sys_icache_invalidate(p, n); }

bool CodeRegion::init(size_t reserve) {
    void* p = mmap(nullptr, reserve, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (p == MAP_FAILED) return false;
    base_ = static_cast<uint8_t*>(p);
    reserved_ = reserve;
    used_ = 0;
    return true;
}

CodeRegion::~CodeRegion() {
    if (base_) munmap(base_, reserved_);
}

uint8_t* CodeRegion::alloc(size_t n) {
    size_t start = (used_ + 4095) & ~size_t(4095);
    if (start + n > reserved_) a64::fatal("JIT code region exhausted");
    used_ = start + n;
    return base_ + start;
}

void CodeRegion::write(uint8_t* dst, const void* src, size_t n) {
    jit_writable(true);
    std::memcpy(dst, src, n);
    jit_writable(false);
    sys_icache_invalidate(dst, n);
}

uint8_t* CodeRegion::install(a64::Assembler& as) {
    as.flush_pool(false);  // place pending literals so the size is final
    uint8_t* dst = alloc(as.size());
    as.finalize(reinterpret_cast<uint64_t>(dst));
    write(dst, as.data(), as.size());
    return dst;
}

} // namespace jit
