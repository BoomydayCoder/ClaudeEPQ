/* The runtime compiled into the epq binary (for JIT-generated code).
 * Same source as the copy embedded into generated C; here the out-of-line
 * helpers get external linkage so machine code can call them by address. */
#define EPQ_RT_API __attribute__((visibility("default"), used))
#include "epq_rt.h"
