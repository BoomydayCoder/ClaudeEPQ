/* Exhaustive check of the runtime's integer formatter against a plain
 * digit loop, for every int32 value (about 20 s on one core).
 *   cc -O2 -I runtime tests/rtio/fmtcheck.c -o build/rtio/fmtcheck && build/rtio/fmtcheck
 * Optional argument: a stride (e.g. 7) to check a subset quickly. */
#include "epq_rt.h"
#include <stdio.h>
#include <stdlib.h>

static int ref_fmt(char* o, int32_t v) {
    char tmp[16];
    int p = 16, n = 0;
    uint32_t u = (uint32_t)v;
    if (v < 0) {
        o[n++] = '-';
        u = 0u - u;
    }
    do {
        tmp[--p] = (char)('0' + u % 10);
        u /= 10;
    } while (u);
    memcpy(o + n, tmp + p, (size_t)(16 - p));
    return n + 16 - p;
}

int main(int argc, char** argv) {
    uint64_t stride = argc > 1 ? strtoull(argv[1], 0, 10) : 1;
    char a[32], b[32];
    uint64_t bad = 0, count = 0;
    for (uint64_t k = 0; k < (1ull << 32); k += stride) {
        int32_t v = (int32_t)(uint32_t)k;
        memset(a, 'x', sizeof a);
        char* e = epq_fmt_i32(a, v);
        int n = ref_fmt(b, v);
        count++;
        if (e - a != n || memcmp(a, b, (size_t)n) != 0 || (n < 32 - 11 && e - a > 11)) {
            if (bad++ < 10) printf("MISMATCH %d: got %.*s want %.*s\n", v, (int)(e - a), a, n, b);
        }
    }
    /* digit counts at every power-of-ten boundary */
    for (int i = 0; i < 10; i++) {
        uint32_t p = epq_pow10[i];
        if (epq_ndigits(p) != (uint32_t)i + 1 || (p > 1 && epq_ndigits(p - 1) != (uint32_t)i)) bad++;
    }
    printf("%llu values checked, %llu mismatches\n", (unsigned long long)count, (unsigned long long)bad);
    return bad != 0;
}
