/* ClaudeEPQ runtime — shared by both engines.
 *
 * This single C source is (a) pasted verbatim at the top of every translation
 * unit the C backend generates (all helpers `static`, hot ones `static inline`,
 * so LLVM inlines them), and (b) compiled into the epq binary by
 * runtime/epq_rt_host.c with EPQ_RT_API defined to an exported linkage, so JIT
 * code can call exactly the same helpers.
 *
 * ---------------------------------------------------------------------------
 * ABI
 * ---------------------------------------------------------------------------
 * Static types (see src/tir.hpp) map to machine representations:
 *   Int, Bot     int32_t
 *   Fn(d)        nothing; a dummy int32_t 0 where C needs a value
 *   List(r)      EList* (never NULL once assigned)
 *   Any          EAny = uint64_t tagged word:
 *                  integer v  : ((uint64_t)(uint32_t)v << 32) | 1
 *                  list       : the EList* itself (16-byte aligned, non-zero)
 *                  function d : ((uint64_t)d << 32) | 2
 *                  0          : "no value" (undefined-global sentinel only)
 *                So: is_int = w & 1; truthy = (w != 1); two ints compare as signed
 *                64-bit words; a+b = a + (b - 1) keeps the tag.
 *
 * List header (16 bytes, 16-aligned):  { void* data; int32_t len; int32_t cap; }
 *   The element layout is fixed per region by the static element type:
 *     element type Int / Bot   -> data is EI32[]  (4 bytes per element)
 *     anything else            -> data is EW64[]  (8 bytes; always a valid EAny word:
 *                                 lists as pointers, functions as tagged words)
 *   Soundness invariant of the inference: a list reachable from an Any value
 *   always has Any elements, so generic (Any) code always sees EW64 slots.
 *   Elements are accessed through the one-member structs EI32 / EW64 so that
 *   type-based alias analysis knows element stores never touch a header.
 *   A list literal of n elements is allocated with its buffer directly behind
 *   the header; growth doubles the capacity into a fresh arena buffer.
 *
 * Descriptors (EDesc table, set in epq_rt_desc by the program's entry): static
 * type descriptions for deep equality / ordering / printing.
 *   index 0 = INT, 1 = ANY, 2 = LIST of ANY, 3 = FN; others LIST(elem index).
 *   Descriptor-driven helpers take values as EAny-sized "words":
 *     INT desc -> (uint32_t)value in the low bits, LIST -> pointer,
 *     FN desc  -> tagged function word, ANY -> tagged word.
 *
 * Dynamic calls: epq_rt_fns[DefId] = { arity, generic entry or NULL }; a generic
 * entry takes its arguments as an EAny array and returns EAny.
 *
 * Errors: epq_fail(code) flushes stdout, writes the original's message and a
 * newline to stderr, and exits with status 1.
 *
 * I/O: output is buffered (64 KB), flushed at exit, before an error, and before
 * any read that would block while output is pending. Input is read in blocks
 * and parsed by hand like C++ `cin >> int`; a failed read yields 0 and all
 * later reads yield 0 (LANGUAGE.md 9.2).
 */
#ifndef EPQ_RT_H
#define EPQ_RT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <poll.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifndef EPQ_RT_API
#define EPQ_RT_API static __attribute__((unused))
#endif
#define EPQ_INLINE static inline __attribute__((always_inline, unused))
#define EPQ_COLD __attribute__((cold, noreturn, noinline))
#define EPQ_LIKELY(x) __builtin_expect(!!(x), 1)
#define EPQ_UNLIKELY(x) __builtin_expect(!!(x), 0)

typedef uint64_t EAny;
typedef struct EI32 { int32_t v; } EI32;
typedef struct EW64 { EAny v; } EW64;
typedef struct __attribute__((aligned(16))) EList {
    void* data;
    int32_t len;
    int32_t cap;
} EList;

enum { EPQ_K_INT = 0, EPQ_K_ANY = 1, EPQ_K_FN = 2, EPQ_K_LIST = 3 };
enum { EPQ_D_INT = 0, EPQ_D_ANY = 1, EPQ_D_LISTANY = 2, EPQ_D_FN = 3 };
typedef struct EDesc {
    uint32_t kind; /* EPQ_K_* */
    uint32_t elem; /* LIST: element descriptor index */
} EDesc;

typedef EAny (*EGenericFn)(const EAny* args);
typedef struct EFnInfo {
    int32_t arity;
    EGenericFn fn;
} EFnInfo;

enum {
    EPQ_E_NONE = 0,
    EPQ_E_OPERANDS_INT,  /* Operands must be integers */
    EPQ_E_DIV_ZERO,      /* Division by zero */
    EPQ_E_LIST_EMPTY,    /* List is empty */
    EPQ_E_INT_OR_LIST,   /* Operand must be an integer or a list */
    EPQ_E_OPERAND_LIST,  /* Operand must be a list */
    EPQ_E_OPERAND_INT,   /* Operand must be an integer */
    EPQ_E_UNDEF_GLOBAL,  /* Undefined global */
    EPQ_E_INVALID_OP,    /* Invalid operation */
    EPQ_E_INDEX_OOB,     /* Index out of bounds */
    EPQ_E_OPERAND_FN,    /* Operand must be a function */
    EPQ_E_ARITY,         /* Invalid arity */
    EPQ_E_INVALID_CMP,   /* Invalid comparison */
    EPQ_E_STACK,         /* Stack overflow */
    EPQ_E_MEMORY,        /* Out of memory */
    EPQ_E_COUNT
};

static const char* const epq_err_msg[EPQ_E_COUNT] __attribute__((unused)) = {
    "",
    "Operands must be integers",
    "Division by zero",
    "List is empty",
    "Operand must be an integer or a list",
    "Operand must be a list",
    "Operand must be an integer",
    "Undefined global",
    "Invalid operation",
    "Index out of bounds",
    "Operand must be a function",
    "Invalid arity",
    "Invalid comparison",
    "Stack overflow",
    "Out of memory",
};

/* ---- program tables (set by the entry point) ---------------------------- */
static const EDesc* epq_rt_desc __attribute__((unused));
static const EFnInfo* epq_rt_fns __attribute__((unused));

/* ---- output ---------------------------------------------------------------- */
/* EPQ_OBUF bytes are flushed at a time; EPQ_OSLACK more bytes behind them let a
 * formatter write one whole bounded value after a single capacity check. */
#ifndef EPQ_OBUF
#define EPQ_OBUF (256 * 1024)
#endif
#define EPQ_OSLACK 64
static char epq_obuf[EPQ_OBUF + EPQ_OSLACK];
static uint32_t epq_olen;

EPQ_RT_API void epq_write_all(int fd, const char* p, size_t n) {
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return;
        p += w;
        n -= (size_t)w;
    }
}

EPQ_RT_API void epq_flush(void) {
    if (epq_olen) {
        epq_write_all(1, epq_obuf, epq_olen);
        epq_olen = 0;
    }
}

EPQ_RT_API EPQ_COLD void epq_fail(int code) {
    epq_flush();
    const char* m = epq_err_msg[code];
    char buf[64];
    size_t n = strlen(m);
    memcpy(buf, m, n);
    buf[n++] = '\n';
    epq_write_all(2, buf, n);
    _exit(1);
}

/* Output protocol for formatters: o = epq_oreserve(n) gives a write pointer
 * with at least n (<= EPQ_OSLACK) bytes of room; write, then epq_ocommit(o).
 * Values are emitted between a reserve and a commit without touching epq_olen. */
EPQ_INLINE char* epq_oreserve(uint32_t n) {
    if (EPQ_UNLIKELY(epq_olen > EPQ_OBUF - n)) epq_flush();
    return epq_obuf + epq_olen;
}
EPQ_INLINE void epq_ocommit(char* o) { epq_olen = (uint32_t)(o - epq_obuf); }
/* Within a loop that keeps the write pointer local: room for n more bytes. */
EPQ_INLINE char* epq_oroom(char* o, uint32_t n) {
    if (EPQ_UNLIKELY(o > epq_obuf + EPQ_OBUF - n)) {
        epq_ocommit(o);
        epq_flush();
        return epq_obuf;
    }
    return o;
}

EPQ_INLINE void epq_out(const char* s, uint32_t n) {
    if (EPQ_UNLIKELY(epq_olen > EPQ_OBUF - n)) {
        epq_flush();
        if (n > EPQ_OBUF) { /* never happens for the runtime's own strings */
            epq_write_all(1, s, n);
            return;
        }
    }
    memcpy(epq_obuf + epq_olen, s, n);
    epq_olen += n;
}

static const char epq_digits2[201] __attribute__((unused, aligned(2))) =
    "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
    "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";
static const uint32_t epq_pow10[10] __attribute__((unused)) = {
    1u, 10u, 100u, 1000u, 10000u, 100000u, 1000000u, 10000000u, 100000000u, 1000000000u};

/* Decimal digits of u (1..10). */
EPQ_INLINE uint32_t epq_ndigits(uint32_t u) {
    uint32_t t = ((32u - (uint32_t)__builtin_clz(u | 1)) * 1233u) >> 12;
    return t + 1 - (u < epq_pow10[t]);
}
/* Writes u in decimal at o (at most 10 bytes), returns the end. Digits are
 * produced two at a time, right to left, straight into the destination. */
EPQ_INLINE char* epq_fmt_u32(char* o, uint32_t u) {
    if (u < 10) {
        *o = (char)('0' + u);
        return o + 1;
    }
    if (u < 100) {
        memcpy(o, epq_digits2 + 2 * u, 2);
        return o + 2;
    }
    char* e = o + epq_ndigits(u);
    char* p = e;
    do {
        uint32_t q = u / 100, r = u - q * 100;
        p -= 2;
        memcpy(p, epq_digits2 + 2 * r, 2);
        u = q;
    } while (u >= 100);
    if (u >= 10) memcpy(p - 2, epq_digits2 + 2 * u, 2);
    else p[-1] = (char)('0' + u);
    return e;
}
/* Writes v in decimal at o (at most 11 bytes), returns the end. */
EPQ_INLINE char* epq_fmt_i32(char* o, int32_t v) {
    uint32_t u = (uint32_t)v;
    if (v < 0) {
        *o++ = '-';
        u = 0u - u;
    }
    return epq_fmt_u32(o, u);
}

EPQ_INLINE void epq_out_int(int32_t v) { epq_ocommit(epq_fmt_i32(epq_oreserve(11), v)); }

/* ---- input ----------------------------------------------------------------- */
/* The unread input is [epq_ip, epq_iend); *epq_iend is always a NUL sentinel
 * (neither whitespace, sign nor digit), so scanning loops test only the
 * character and look at the position only when they stop. The buffer has 16
 * spare bytes so an 8-byte word load at any position below epq_iend is in
 * bounds. */
#ifndef EPQ_IBUF
#define EPQ_IBUF (256 * 1024)
#endif
static char epq_ibuf[EPQ_IBUF + 16];
static const char* epq_ip = epq_ibuf;
static const char* epq_iend = epq_ibuf;
static int epq_ieof, epq_ifail;
/* Whether pending output must be flushed before reading stdin: decided once,
 * at the first refill. 1 = never (regular file, /dev/null or another non-tty
 * character device: data never waits for our output), 2 = when the read
 * would block (tty, pipe, socket: poll). */
static int epq_imode;

/* Refills the input buffer (all buffered input must have been consumed).
 * Returns 0 at end of input or on a read error (and stays at end). */
EPQ_RT_API int epq_fill(void) {
    if (epq_ieof) return 0;
    if (epq_olen) {
        if (EPQ_UNLIKELY(epq_imode == 0)) {
            struct stat st;
            epq_imode = 2;
            if (fstat(0, &st) == 0 &&
                (S_ISREG(st.st_mode) || (S_ISCHR(st.st_mode) && !isatty(0))))
                epq_imode = 1;
        }
        if (epq_imode == 2) {
            /* Flush pending output before a read that would block (interactive use). */
            struct pollfd p;
            p.fd = 0;
            p.events = POLLIN;
            p.revents = 0;
            if (poll(&p, 1, 0) <= 0) epq_flush();
        }
    }
    ssize_t n;
    do {
        n = read(0, epq_ibuf, EPQ_IBUF);
    } while (n < 0 && errno == EINTR);
    if (n <= 0) {
        epq_ieof = 1;
        epq_ip = epq_iend = epq_ibuf;
        epq_ibuf[0] = 0;
        return 0;
    }
    epq_ip = epq_ibuf;
    epq_iend = epq_ibuf + n;
    epq_ibuf[n] = 0;
    return 1;
}

EPQ_INLINE int epq_isdigit(unsigned c) { return c - '0' < 10u; }
EPQ_INLINE int epq_isspace(unsigned c) { return c == ' ' || c - 9u < 5u; } /* \t \n \v \f \r */

/* The general digit loop: numbers crossing a buffer boundary, long runs of
 * (leading-zero) digits, overflow. p points at the first digit (or at the
 * end of the buffer right after a sign). */
static __attribute__((noinline, unused)) int32_t epq_read_digits_slow(const char* p, int neg) {
    int64_t v = 0;
    int over = 0, any = 0;
    for (;;) {
        unsigned c = (unsigned char)*p;
        if (epq_isdigit(c)) {
            if (v < ((int64_t)1 << 40)) v = v * 10 + (c - '0');
            else over = 1;
            any = 1;
            p++;
            continue;
        }
        if (p == epq_iend) {
            if (!epq_fill()) {
                p = epq_ip;
                break;
            }
            p = epq_ip;
            continue;
        }
        break;
    }
    epq_ip = p;
    if (neg) v = -v;
    if (!any || over || v > 2147483647LL || v < -2147483648LL) {
        epq_ifail = 1;
        return 0;
    }
    return (int32_t)v;
}

/* Like C++ `cin >> int` (LANGUAGE.md 7): skip whitespace, optional sign,
 * decimal digits; any failure yields 0 now and for every later read. */
EPQ_INLINE int32_t epq_read_int_i(void) {
    epq_out("<< ", 3);
    if (EPQ_UNLIKELY(epq_ifail)) return 0;
    const char* p = epq_ip;
    unsigned c;
    for (;;) {
        c = (unsigned char)*p;
        if (epq_isspace(c)) {
            p++;
            continue;
        }
        if (EPQ_UNLIKELY(p == epq_iend)) {
            if (!epq_fill()) {
                epq_ifail = 1;
                return 0;
            }
            p = epq_ip;
            continue;
        }
        break;
    }
    /* Branch-free sign: input signs are often random. */
    int neg = c == '-';
    p += neg | (c == '+');
    if (EPQ_UNLIKELY(p == epq_iend)) return epq_read_digits_slow(p, neg);
    /* Up to 7 digits, terminated inside the buffer: one 8-byte load. Bytes
     * past the sentinel may be stale but the first non-digit is found at or
     * before it. */
    uint64_t x;
    memcpy(&x, p, 8);
    uint64_t nd = ((x + 0x4646464646464646ull) | (x - 0x3030303030303030ull)) & 0x8080808080808080ull;
    uint32_t len = nd ? (uint32_t)__builtin_ctzll(nd) >> 3 : 8;
    if (EPQ_LIKELY(len - 1 < 7u) && EPQ_LIKELY(p + len < epq_iend)) {
        uint64_t t = (x - 0x3030303030303030ull) << (8 * (8 - len));
        t = (t * 10 + (t >> 8)) & 0x00FF00FF00FF00FFull;
        t = (t * 100 + (t >> 16)) & 0x0000FFFF0000FFFFull;
        t = (t * 10000 + (t >> 32)) & 0xFFFFFFFFull;
        epq_ip = p + len;
        int32_t v = (int32_t)t;
        return neg ? -v : v;
    }
    if (len == 0) {
        if (p == epq_iend) return epq_read_digits_slow(p, neg);
        epq_ip = p;
        epq_ifail = 1;
        return 0;
    }
    return epq_read_digits_slow(p, neg);
}
EPQ_RT_API int32_t epq_read_int(void) { return epq_read_int_i(); }

/* ---- memory ------------------------------------------------------------------ */
static char* epq_bump;
static char* epq_bump_end;

EPQ_RT_API void* epq_alloc_slow(size_t n) {
    size_t chunk = (size_t)64 << 20;
    if (n > chunk / 4) chunk = (n + 4095) & ~(size_t)4095;
    void* p = mmap(0, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) epq_fail(EPQ_E_MEMORY);
    if (n > ((size_t)64 << 20) / 4) return p; /* big block: keep the current chunk */
    epq_bump = (char*)p + n;
    epq_bump_end = (char*)p + chunk;
    return p;
}

EPQ_INLINE void* epq_alloc(size_t n) {
    n = (n + 15) & ~(size_t)15;
    char* p = epq_bump;
    if (EPQ_UNLIKELY((size_t)(epq_bump_end - p) < n)) return epq_alloc_slow(n);
    epq_bump = p + n;
    return p;
}

/* A fresh list of n elements of esz bytes; the buffer follows the header.
 * An empty literal gets room for 16 bytes of elements. */
EPQ_INLINE EList* epq_list_new(int32_t n, uint32_t esz) {
    size_t bytes = (size_t)n * esz;
    if (bytes < 16) bytes = 16;
    bytes = (bytes + 15) & ~(size_t)15;
    EList* l = (EList*)epq_alloc(sizeof(EList) + bytes);
    l->data = (char*)l + sizeof(EList);
    l->len = n;
    l->cap = (int32_t)(bytes / esz);
    return l;
}

EPQ_RT_API void epq_list_grow(EList* l, uint32_t esz) {
    int64_t ncap = l->cap < 4 ? 8 : (int64_t)l->cap * 2;
    if (ncap > 0x7fffffff) ncap = 0x7fffffff;
    void* nd = epq_alloc((size_t)ncap * esz);
    memcpy(nd, l->data, (size_t)l->len * esz);
    l->data = nd;
    l->cap = (int32_t)ncap;
}

/* ---- typed list operations ------------------------------------------------- */
/* The bounds check also tests `data` (never NULL): that makes the data load
 * execute before the check's (noreturn) exit, so LLVM's LICM may hoist it out
 * of loops — a load after a conditional exit cannot be speculated. With both
 * header fields hoisted, the null test is loop-invariant and leaves the loop. */
EPQ_INLINE int32_t epq_ld_i32(EList* l, int32_t i) {
    EI32* d = (EI32*)l->data;
    if (EPQ_UNLIKELY(((uint32_t)i >= (uint32_t)l->len) | (d == 0))) epq_fail(EPQ_E_INDEX_OOB);
    return d[i].v;
}
EPQ_INLINE EAny epq_ld_w(EList* l, int32_t i) {
    EW64* d = (EW64*)l->data;
    if (EPQ_UNLIKELY(((uint32_t)i >= (uint32_t)l->len) | (d == 0))) epq_fail(EPQ_E_INDEX_OOB);
    return d[i].v;
}
EPQ_INLINE int32_t epq_st_i32(EList* l, int32_t i, int32_t v) {
    EI32* d = (EI32*)l->data;
    if (EPQ_UNLIKELY(((uint32_t)i >= (uint32_t)l->len) | (d == 0))) epq_fail(EPQ_E_INDEX_OOB);
    d[i].v = v;
    return v;
}
EPQ_INLINE EAny epq_st_w(EList* l, int32_t i, EAny v) {
    EW64* d = (EW64*)l->data;
    if (EPQ_UNLIKELY(((uint32_t)i >= (uint32_t)l->len) | (d == 0))) epq_fail(EPQ_E_INDEX_OOB);
    d[i].v = v;
    return v;
}
EPQ_INLINE EList* epq_push_i32(EList* l, int32_t v) {
    int32_t n = l->len;
    if (EPQ_UNLIKELY(n == l->cap)) epq_list_grow(l, 4);
    ((EI32*)l->data)[n].v = v;
    l->len = n + 1;
    return l;
}
EPQ_INLINE EList* epq_push_w(EList* l, EAny v) {
    int32_t n = l->len;
    if (EPQ_UNLIKELY(n == l->cap)) epq_list_grow(l, 8);
    ((EW64*)l->data)[n].v = v;
    l->len = n + 1;
    return l;
}
EPQ_INLINE int32_t epq_pop_i32(EList* l) {
    if (EPQ_UNLIKELY(l->len == 0)) epq_fail(EPQ_E_LIST_EMPTY);
    return ((EI32*)l->data)[--l->len].v;
}
EPQ_INLINE EAny epq_pop_w(EList* l) {
    if (EPQ_UNLIKELY(l->len == 0)) epq_fail(EPQ_E_LIST_EMPTY);
    return ((EW64*)l->data)[--l->len].v;
}

/* ---- integers --------------------------------------------------------------- */
EPQ_INLINE int32_t epq_div_i32(int32_t a, int32_t b) {
    if (EPQ_UNLIKELY(b == 0)) epq_fail(EPQ_E_DIV_ZERO);
    if (EPQ_UNLIKELY(b == -1)) return (int32_t)(0u - (uint32_t)a);
    return a / b;
}
/* Division after the divisor is known to be non-zero. */
EPQ_INLINE int32_t epq_div_nz(int32_t a, int32_t b) {
    if (EPQ_UNLIKELY(b == -1)) return (int32_t)(0u - (uint32_t)a);
    return a / b;
}
EPQ_INLINE int32_t epq_abs_i32(int32_t a) { return a < 0 ? (int32_t)(0u - (uint32_t)a) : a; }
EPQ_INLINE int32_t epq_neg_i32(int32_t a) { return (int32_t)(0u - (uint32_t)a); }

/* ---- dynamic values ------------------------------------------------------------ */
EPQ_INLINE EAny epq_box_int(int32_t v) { return ((EAny)(uint32_t)v << 32) | 1; }
EPQ_INLINE EAny epq_box_fn(uint32_t d) { return ((EAny)d << 32) | 2; }
EPQ_INLINE int32_t epq_int_of(EAny w) { return (int32_t)(w >> 32); }
EPQ_INLINE int epq_is_int(EAny w) { return (int)(w & 1); }
EPQ_INLINE int epq_is_fn(EAny w) { return (w & 3) == 2; }
EPQ_INLINE int epq_is_list(EAny w) { return (w & 3) == 0; }
EPQ_INLINE int epq_truthy(EAny w) { return w != 1; }
EPQ_INLINE int32_t epq_unbox_int(EAny w, int err) {
    if (EPQ_UNLIKELY(!(w & 1))) epq_fail(err);
    return (int32_t)(w >> 32);
}

EPQ_RT_API EAny epq_dyn_neg(EAny w) {
    if (w & 1) return epq_box_int(epq_neg_i32(epq_int_of(w)));
    if (epq_is_list(w)) return epq_pop_w((EList*)(uintptr_t)w);
    epq_fail(EPQ_E_INT_OR_LIST);
}
EPQ_RT_API int32_t epq_dyn_abs(EAny w) {
    if (w & 1) return epq_abs_i32(epq_int_of(w));
    if (epq_is_list(w)) return ((EList*)(uintptr_t)w)->len;
    epq_fail(EPQ_E_INT_OR_LIST);
}
EPQ_INLINE int32_t epq_dyn_not(EAny w) {
    if (EPQ_UNLIKELY(!(w & 1))) epq_fail(EPQ_E_OPERAND_INT);
    return w == 1;
}
EPQ_INLINE EAny epq_dyn_index(EAny l, EAny i) {
    if (EPQ_UNLIKELY(!epq_is_list(l) || !(i & 1))) epq_fail(EPQ_E_INVALID_OP);
    return epq_ld_w((EList*)(uintptr_t)l, epq_int_of(i));
}
EPQ_INLINE EAny epq_dyn_set_index(EAny l, EAny i, EAny v) {
    if (EPQ_UNLIKELY(!epq_is_list(l) || !(i & 1))) epq_fail(EPQ_E_INVALID_OP);
    return epq_st_w((EList*)(uintptr_t)l, epq_int_of(i), v);
}
EPQ_INLINE EAny epq_dyn_append(EAny l, EAny v) {
    if (EPQ_UNLIKELY(!epq_is_list(l))) epq_fail(EPQ_E_OPERAND_LIST);
    epq_push_w((EList*)(uintptr_t)l, v);
    return l;
}
EPQ_RT_API EAny epq_call_dyn(EAny f, int32_t n, const EAny* args) {
    if (!epq_is_fn(f)) epq_fail(EPQ_E_OPERAND_FN);
    uint32_t d = (uint32_t)(f >> 32);
    if (epq_rt_fns[d].arity != n) epq_fail(EPQ_E_ARITY);
    return epq_rt_fns[d].fn(args);
}

/* ---- descriptor-driven deep operations ------------------------------------- */
/* Normalise (desc, word): resolves ANY to the concrete kind. */
EPQ_INLINE void epq_norm(uint32_t* d, EAny* w) {
    if (*d == EPQ_D_ANY) {
        EAny x = *w;
        if (x & 1) {
            *d = EPQ_D_INT;
            *w = (uint32_t)(x >> 32);
        } else if ((x & 3) == 2) {
            *d = EPQ_D_FN;
        } else {
            *d = EPQ_D_LISTANY;
        }
    }
}
EPQ_INLINE EAny epq_elem_word(uint32_t ed, EList* l, int32_t k) {
    if (epq_rt_desc[ed].kind == EPQ_K_INT) return (uint32_t)((EI32*)l->data)[k].v;
    return ((EW64*)l->data)[k].v;
}

EPQ_RT_API int epq_eq(uint32_t da, EAny a, uint32_t db, EAny b) {
    epq_norm(&da, &a);
    epq_norm(&db, &b);
    uint32_t ka = epq_rt_desc[da].kind, kb = epq_rt_desc[db].kind;
    if (ka != kb) return 0;
    if (ka == EPQ_K_INT) return (int32_t)a == (int32_t)b;
    if (ka == EPQ_K_FN) return a == b;
    EList* la = (EList*)(uintptr_t)a;
    EList* lb = (EList*)(uintptr_t)b;
    if (la->len != lb->len) return 0;
    uint32_t ea = epq_rt_desc[da].elem, eb = epq_rt_desc[db].elem;
    if (epq_rt_desc[ea].kind == EPQ_K_INT && epq_rt_desc[eb].kind == EPQ_K_INT)
        return memcmp(la->data, lb->data, (size_t)la->len * 4) == 0;
    for (int32_t k = 0; k < la->len; k++)
        if (!epq_eq(ea, epq_elem_word(ea, la, k), eb, epq_elem_word(eb, lb, k))) return 0;
    return 1;
}

/* a < b  (LANGUAGE.md 6: ints by value, lists lexicographically, else error) */
EPQ_RT_API int epq_lt(uint32_t da, EAny a, uint32_t db, EAny b) {
    epq_norm(&da, &a);
    epq_norm(&db, &b);
    uint32_t ka = epq_rt_desc[da].kind, kb = epq_rt_desc[db].kind;
    if (ka == EPQ_K_INT && kb == EPQ_K_INT) return (int32_t)a < (int32_t)b;
    if (ka != EPQ_K_LIST || kb != EPQ_K_LIST) epq_fail(EPQ_E_INVALID_CMP);
    EList* la = (EList*)(uintptr_t)a;
    EList* lb = (EList*)(uintptr_t)b;
    uint32_t ea = epq_rt_desc[da].elem, eb = epq_rt_desc[db].elem;
    int32_t na = la->len, nb = lb->len;
    if (epq_rt_desc[ea].kind == EPQ_K_INT && epq_rt_desc[eb].kind == EPQ_K_INT) {
        const EI32* pa = (const EI32*)la->data;
        const EI32* pb = (const EI32*)lb->data;
        for (int32_t k = 0; k < nb; k++) {
            if (k == na) return 1;
            if (pa[k].v != pb[k].v) return pa[k].v < pb[k].v;
        }
        return 0;
    }
    for (int32_t k = 0; k < nb; k++) {
        if (k == na) return 1;
        EAny x = epq_elem_word(ea, la, k), y = epq_elem_word(eb, lb, k);
        if (epq_lt(ea, x, eb, y)) return 1;
        if (epq_lt(eb, y, ea, x)) return 0;
    }
    return 0;
}

/* Fast paths for two lists whose elements are statically Int. */
static inline __attribute__((unused)) int epq_lt_li(EList* a, EList* b) {
    const EI32* pa = (const EI32*)a->data;
    const EI32* pb = (const EI32*)b->data;
    int32_t na = a->len, nb = b->len;
    for (int32_t k = 0; k < nb; k++) {
        if (k == na) return 1;
        if (pa[k].v != pb[k].v) return pa[k].v < pb[k].v;
    }
    return 0;
}
static inline __attribute__((unused)) int epq_eq_li(EList* a, EList* b) {
    if (a->len != b->len) return 0;
    const EI32* pa = (const EI32*)a->data;
    const EI32* pb = (const EI32*)b->data;
    for (int32_t k = 0; k < a->len; k++)
        if (pa[k].v != pb[k].v) return 0;
    return 1;
}

/* ---- printing (LANGUAGE.md 5: `>> v` writes ">> ", the value, "\n") -------- */
/* Each element costs one capacity check (", " + up to 11 characters). */
#define EPQ_ELEM_MAX 13

/* [a, b, c] for an Int-element list. */
static __attribute__((noinline, unused)) void epq_print_ilist_val(const EList* l) {
    const EI32* p = (const EI32*)l->data;
    int32_t n = l->len;
    char* o = epq_oreserve(1 + 11 + 1);
    *o++ = '[';
    if (n > 0) {
        o = epq_fmt_i32(o, p[0].v);
        for (int32_t i = 1; i < n; i++) {
            o = epq_oroom(o, EPQ_ELEM_MAX);
            memcpy(o, ", ", 2);
            o = epq_fmt_i32(o + 2, p[i].v);
        }
        o = epq_oroom(o, 1);
    }
    *o++ = ']';
    epq_ocommit(o);
}

/* [ ... ] for an Any-element list (the elements are tagged words). */
static __attribute__((noinline, unused)) void epq_print_alist_val(const EList* l) {
    const EW64* p = (const EW64*)l->data;
    int32_t n = l->len;
    char* o = epq_oreserve(EPQ_ELEM_MAX);
    *o++ = '[';
    for (int32_t i = 0; i < n; i++) {
        EAny w = p[i].v;
        o = epq_oroom(o, EPQ_ELEM_MAX);
        if (i) {
            memcpy(o, ", ", 2);
            o += 2;
        }
        if (EPQ_LIKELY(w & 1)) {
            o = epq_fmt_i32(o, (int32_t)(w >> 32));
        } else if ((w & 3) == 2) {
            memcpy(o, "<function>", 10);
            o += 10;
        } else {
            epq_ocommit(o);
            epq_print_alist_val((const EList*)(uintptr_t)w);
            o = epq_obuf + epq_olen;
        }
    }
    o = epq_oroom(o, 1);
    *o++ = ']';
    epq_ocommit(o);
}

EPQ_RT_API void epq_print_val(uint32_t d, EAny w) {
    epq_norm(&d, &w);
    uint32_t k = epq_rt_desc[d].kind;
    if (k == EPQ_K_INT) {
        epq_out_int((int32_t)w);
    } else if (k == EPQ_K_FN) {
        epq_out("<function>", 10);
    } else {
        const EList* l = (const EList*)(uintptr_t)w;
        uint32_t e = epq_rt_desc[d].elem;
        uint32_t ek = epq_rt_desc[e].kind;
        if (ek == EPQ_K_INT) {
            epq_print_ilist_val(l);
        } else if (ek == EPQ_K_ANY) {
            epq_print_alist_val(l);
        } else {
            /* List or Fn elements: static element descriptor, word slots. */
            const EW64* p = (const EW64*)l->data;
            epq_out("[", 1);
            for (int32_t i = 0; i < l->len; i++) {
                if (i) epq_out(", ", 2);
                epq_print_val(e, p[i].v);
            }
            epq_out("]", 1);
        }
    }
}
EPQ_RT_API void epq_print(uint32_t d, EAny w) {
    epq_out(">> ", 3);
    epq_print_val(d, w);
    epq_out("\n", 1);
}
/* `>> v` for an Int: one capacity check for ">> ", the digits and "\n". */
EPQ_INLINE void epq_print_int(int32_t v) {
    char* o = epq_oreserve(3 + 11 + 1);
    memcpy(o, ">> ", 3);
    o = epq_fmt_i32(o + 3, v);
    *o++ = '\n';
    epq_ocommit(o);
}
/* Out-of-line entry points for JIT code (specialised forms of epq_print). */
EPQ_RT_API void epq_print_i32(int32_t v) { epq_print_int(v); }
EPQ_RT_API void epq_print_list_i32(const EList* l) {
    epq_out(">> ", 3);
    epq_print_ilist_val(l);
    epq_out("\n", 1);
}
EPQ_RT_API void epq_print_list_any(const EList* l) {
    epq_out(">> ", 3);
    epq_print_alist_val(l);
    epq_out("\n", 1);
}

#endif /* EPQ_RT_H */
