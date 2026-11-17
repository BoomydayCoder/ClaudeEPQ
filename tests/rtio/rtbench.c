/* Direct micro-benchmark of the runtime I/O helpers (no compiler in the loop).
 *   cc -O3 -fwrapv -mcpu=native -I runtime tests/rtio/rtbench.c -o build/rtio/rtbench
 *   build/rtio/rtbench > /dev/null       (input benchmarks read build/rtio/micro/readints.in)
 * Reports min of several runs per case on stderr. */
#include "epq_rt.h"
#include <stdio.h>
#include <fcntl.h>
#include <time.h>

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static const EDesc descs[] = {{0, 0}, {1, 0}, {3, 1}, {2, 0}, {3, 0}, {3, 4}, {3, 5}};
/* 4 = List(Int), 5 = List(List(Int)), 6 = List(List(List(Int))) */

static EList* il;   /* 10^6 ints */
static EList* nl;   /* 10^5 x [d, [d], []] as Any */

static void b_small(void) { for (int i = 0; i < 1000000; i++) epq_print_int(i % 100); }
static void b_big(void) { for (int i = 0; i < 1000000; i++) epq_print_int(i * 2147 - 1000000000); }
static void b_ilist(void) { epq_print(4, (EAny)(uintptr_t)il); }
static void b_nested(void) { epq_print(2, (EAny)(uintptr_t)nl); }
static void b_nestedN(void) {
    for (int i = 0; i < nl->len; i++) epq_print(1, ((EW64*)nl->data)[i].v);
}
static void b_prompt(void) { for (int i = 0; i < 1000000; i++) epq_out("<< ", 3); }

static int rfd = -1;
static void b_read(void) {
    lseek(rfd, 0, SEEK_SET);
#ifdef EPQ_OLD_RT
    epq_ipos = epq_ilen = 0;
#else
    epq_ip = epq_iend = epq_ibuf;
    epq_ibuf[0] = 0;
#endif
    epq_ieof = epq_ifail = 0;
    int64_t s = 0;
    for (int i = 0; i < 1000000; i++) s += epq_read_int();
    if (s == 42) write(2, "?", 1);
}

static char rawbuf[1 << 20];
static void b_rawread(void) {
    lseek(rfd, 0, SEEK_SET);
    while (read(rfd, rawbuf, EPQ_IBUF) > 0) {}
}

static void bench(const char* name, void (*f)(void)) {
    double best = 1e18;
    for (int r = 0; r < 15; r++) {
        double t0 = now_us();
        f();
        epq_flush();
        double t = now_us() - t0;
        if (t < best) best = t;
    }
    fprintf(stderr, "%-10s %8.0f us\n", name, best);
}

int main(int argc, char** argv) {
    epq_rt_desc = descs;
    il = epq_list_new(0, 4);
    for (int i = 0; i < 1000000; i++) epq_push_i32(il, i * 7 - (i / 3) * 5);
    nl = epq_list_new(0, 8);
    for (int i = 0; i < 100000; i++) {
        EList* a = epq_list_new(3, 8);
        EList* b = epq_list_new(1, 8);
        ((EW64*)b->data)[0].v = epq_box_int(i % 7);
        ((EW64*)a->data)[0].v = epq_box_int(i % 10);
        ((EW64*)a->data)[1].v = (EAny)(uintptr_t)b;
        ((EW64*)a->data)[2].v = (EAny)(uintptr_t)epq_list_new(0, 8);
        epq_push_w(nl, (EAny)(uintptr_t)a);
    }
    const char* only = argc > 1 ? argv[1] : "";
#define B(n, f) if (!*only || strstr(only, n)) bench(n, f)
    B("small", b_small);
    B("big", b_big);
    B("ilist", b_ilist);
    B("nested", b_nested);
    B("nestedN", b_nestedN);
    B("prompt", b_prompt);
    rfd = open("build/rtio/micro/readints.in", O_RDONLY);
    if (rfd >= 0) {
        dup2(rfd, 0);
        rfd = 0;
        B("rawread", b_rawread);
        B("read", b_read);
    }
    return 0;
}
