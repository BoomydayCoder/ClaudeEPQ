// Exported runtime wrappers called by JIT code (see rt_bridge.cpp).
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
void jrt_set_tables(const void* descs, const void* fns);
void jrt_flush(void);
[[noreturn]] void jrt_fail(int32_t code);
void jrt_print_int(int32_t v);
void jrt_print(uint32_t d, uint64_t w);
int32_t jrt_read_int(void);
void* jrt_list_new(int32_t n, uint32_t esz);
void jrt_list_grow(void* l, uint32_t esz);
int32_t jrt_eq(uint32_t da, uint64_t a, uint32_t db, uint64_t b);
int32_t jrt_lt(uint32_t da, uint64_t a, uint32_t db, uint64_t b);
int32_t jrt_eq_li(void* a, void* b);
int32_t jrt_lt_li(void* a, void* b);
uint64_t jrt_dyn_neg(uint64_t w);
int32_t jrt_dyn_abs(uint64_t w);
uint64_t jrt_call_dyn(uint64_t f, int32_t n, const uint64_t* args);
void* jrt_alloc_slow(size_t n);
char** jrt_bump_ptr(void);
char** jrt_bump_end_ptr(void);
void** jrt_fl_small_ptr(void);
int32_t jrt_poison_on(void);
void jrt_list_free(void* l, uint32_t esz);
}
