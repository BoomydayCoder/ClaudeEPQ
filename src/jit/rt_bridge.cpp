// Runtime entry points for JIT-generated code.
//
// This translation unit includes the shared runtime (runtime/epq_rt.h) with its
// default `static` linkage, so it owns one private copy of the runtime state
// (output buffer, input buffer, bump allocator, descriptor / function tables)
// used only by the JIT engine, and exports thin non-inline wrappers whose
// addresses generated code calls (through per-helper stubs in the code region).
// Semantics are exactly those of the helpers the C backend inlines.
#include <errno.h>
#include <poll.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma clang diagnostic ignored "-Wcast-qual"
#include "../../runtime/epq_rt.h"
#pragma clang diagnostic pop

#include "rt_bridge.hpp"

extern "C" {

void jrt_set_tables(const void* descs, const void* fns) {
    epq_rt_desc = static_cast<const EDesc*>(descs);
    epq_rt_fns = static_cast<const EFnInfo*>(fns);
}
void jrt_flush(void) { epq_flush(); }
void jrt_fail(int32_t code) { epq_fail(code); }
void jrt_print_int(int32_t v) { epq_print_int(v); }
void jrt_print(uint32_t d, uint64_t w) { epq_print(d, w); }
int32_t jrt_read_int(void) { return epq_read_int(); }
void* jrt_list_new(int32_t n, uint32_t esz) { return epq_list_new(n, esz); }
void jrt_list_grow(void* l, uint32_t esz) { epq_list_grow(static_cast<EList*>(l), esz); }
int32_t jrt_eq(uint32_t da, uint64_t a, uint32_t db, uint64_t b) { return epq_eq(da, a, db, b); }
int32_t jrt_lt(uint32_t da, uint64_t a, uint32_t db, uint64_t b) { return epq_lt(da, a, db, b); }
int32_t jrt_eq_li(void* a, void* b) { return epq_eq_li(static_cast<EList*>(a), static_cast<EList*>(b)); }
int32_t jrt_lt_li(void* a, void* b) { return epq_lt_li(static_cast<EList*>(a), static_cast<EList*>(b)); }
uint64_t jrt_dyn_neg(uint64_t w) { return epq_dyn_neg(w); }
int32_t jrt_dyn_abs(uint64_t w) { return epq_dyn_abs(w); }
uint64_t jrt_call_dyn(uint64_t f, int32_t n, const uint64_t* args) { return epq_call_dyn(f, n, args); }
void* jrt_alloc_slow(size_t n) { return epq_alloc_slow(n); }
char** jrt_bump_ptr(void) { return &epq_bump; }
char** jrt_bump_end_ptr(void) { return &epq_bump_end; }
void** jrt_fl_small_ptr(void) { return epq_fl_small; }
int32_t jrt_poison_on(void) { return epq_poison_on(); }
void jrt_list_free(void* l, uint32_t esz) { epq_list_free(static_cast<EList*>(l), esz); }
void jrt_list_reserve(void* l, int32_t n, uint32_t esz) { epq_list_reserve(static_cast<EList*>(l), n, esz); }
int32_t jrt_append_range(void* l, void* m, int32_t lo, int32_t hi, uint32_t esz) {
    return epq_append_range(static_cast<EList*>(l), static_cast<EList*>(m), lo, hi, esz);
}
int32_t jrt_eq_la(void* a, void* b) { return epq_eq_la(static_cast<EList*>(a), static_cast<EList*>(b)); }
int32_t jrt_cmp3_la(void* a, void* b) { return epq_cmp3_la(static_cast<EList*>(a), static_cast<EList*>(b)); }

}  // extern "C"
