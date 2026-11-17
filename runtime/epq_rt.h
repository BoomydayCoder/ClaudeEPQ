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
#define EPQ_OBUF 65536
static char epq_obuf[EPQ_OBUF];
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

EPQ_INLINE void epq_out(const char* s, uint32_t n) {
    if (EPQ_UNLIKELY(epq_olen + n > EPQ_OBUF)) epq_flush();
    memcpy(epq_obuf + epq_olen, s, n);
    epq_olen += n;
}

static const char epq_digits2[201] __attribute__((unused)) =
    "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
    "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

EPQ_INLINE void epq_out_int(int32_t v) {
    if (EPQ_UNLIKELY(epq_olen + 16 > EPQ_OBUF)) epq_flush();
    char* o = epq_obuf + epq_olen;
    uint32_t u = (uint32_t)v;
    if (v < 0) {
        *o++ = '-';
        u = 0u - u;
    }
    char tmp[12];
    int p = 12;
    while (u >= 100) {
        uint32_t q = u / 100, r = u - q * 100;
        tmp[--p] = epq_digits2[2 * r + 1];
        tmp[--p] = epq_digits2[2 * r];
        u = q;
    }
    if (u >= 10) {
        tmp[--p] = epq_digits2[2 * u + 1];
        tmp[--p] = epq_digits2[2 * u];
    } else {
        tmp[--p] = (char)('0' + u);
    }
    memcpy(o, tmp + p, (size_t)(12 - p));
    o += 12 - p;
    epq_olen = (uint32_t)(o - epq_obuf);
}

/* ---- input ----------------------------------------------------------------- */
#define EPQ_IBUF 65536
static char epq_ibuf[EPQ_IBUF];
static uint32_t epq_ipos, epq_ilen;
static int epq_ieof, epq_ifail;

EPQ_RT_API int epq_fill(void) {
    if (epq_ieof) return 0;
    if (epq_olen) {
        /* Flush pending output before a read that would block (interactive use). */
        struct pollfd p;
        p.fd = 0;
        p.events = POLLIN;
        p.revents = 0;
        if (poll(&p, 1, 0) <= 0) epq_flush();
    }
    ssize_t n;
    do {
        n = read(0, epq_ibuf, EPQ_IBUF);
    } while (n < 0 && errno == EINTR);
    if (n <= 0) {
        epq_ieof = 1;
        return 0;
    }
    epq_ipos = 0;
    epq_ilen = (uint32_t)n;
    return 1;
}

EPQ_INLINE int epq_peekc(void) {
    if (epq_ipos == epq_ilen && !epq_fill()) return -1;
    return (unsigned char)epq_ibuf[epq_ipos];
}

EPQ_RT_API int32_t epq_read_int(void) {
    epq_out("<< ", 3);
    if (epq_ifail) return 0;
    int c;
    for (;;) {
        c = epq_peekc();
        if (c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r') {
            epq_ipos++;
            continue;
        }
        break;
    }
    int neg = 0;
    if (c == '+' || c == '-') {
        neg = (c == '-');
        epq_ipos++;
        c = epq_peekc();
    }
    if (c < '0' || c > '9') {
        epq_ifail = 1;
        return 0;
    }
    int64_t v = 0;
    int over = 0;
    while (c >= '0' && c <= '9') {
        if (v < ((int64_t)1 << 40)) v = v * 10 + (c - '0');
        else over = 1;
        epq_ipos++;
        c = epq_peekc();
    }
    if (neg) v = -v;
    if (over || v > 2147483647LL || v < -2147483648LL) {
        epq_ifail = 1;
        return 0;
    }
    return (int32_t)v;
}

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
EPQ_INLINE int32_t epq_ld_i32(EList* l, int32_t i) {
    if (EPQ_UNLIKELY((uint32_t)i >= (uint32_t)l->len)) epq_fail(EPQ_E_INDEX_OOB);
    return ((EI32*)l->data)[i].v;
}
EPQ_INLINE EAny epq_ld_w(EList* l, int32_t i) {
    if (EPQ_UNLIKELY((uint32_t)i >= (uint32_t)l->len)) epq_fail(EPQ_E_INDEX_OOB);
    return ((EW64*)l->data)[i].v;
}
EPQ_INLINE int32_t epq_st_i32(EList* l, int32_t i, int32_t v) {
    if (EPQ_UNLIKELY((uint32_t)i >= (uint32_t)l->len)) epq_fail(EPQ_E_INDEX_OOB);
    ((EI32*)l->data)[i].v = v;
    return v;
}
EPQ_INLINE EAny epq_st_w(EList* l, int32_t i, EAny v) {
    if (EPQ_UNLIKELY((uint32_t)i >= (uint32_t)l->len)) epq_fail(EPQ_E_INDEX_OOB);
    ((EW64*)l->data)[i].v = v;
    return v;
}
EPQ_INLINE EList* epq_push_i32(EList* l, int32_t v) {
    if (EPQ_UNLIKELY(l->len == l->cap)) epq_list_grow(l, 4);
    ((EI32*)l->data)[l->len++].v = v;
    return l;
}
EPQ_INLINE EList* epq_push_w(EList* l, EAny v) {
    if (EPQ_UNLIKELY(l->len == l->cap)) epq_list_grow(l, 8);
    ((EW64*)l->data)[l->len++].v = v;
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

EPQ_RT_API void epq_print_val(uint32_t d, EAny w) {
    epq_norm(&d, &w);
    uint32_t k = epq_rt_desc[d].kind;
    if (k == EPQ_K_INT) {
        epq_out_int((int32_t)w);
    } else if (k == EPQ_K_FN) {
        epq_out("<function>", 10);
    } else {
        EList* l = (EList*)(uintptr_t)w;
        uint32_t e = epq_rt_desc[d].elem;
        epq_out("[", 1);
        if (epq_rt_desc[e].kind == EPQ_K_INT) {
            const EI32* p = (const EI32*)l->data;
            for (int32_t i = 0; i < l->len; i++) {
                if (i) epq_out(", ", 2);
                epq_out_int(p[i].v);
            }
        } else {
            for (int32_t i = 0; i < l->len; i++) {
                if (i) epq_out(", ", 2);
                epq_print_val(e, epq_elem_word(e, l, i));
            }
        }
        epq_out("]", 1);
    }
}
EPQ_RT_API void epq_print(uint32_t d, EAny w) {
    epq_out(">> ", 3);
    epq_print_val(d, w);
    epq_out("\n", 1);
}
EPQ_INLINE void epq_print_int(int32_t v) {
    epq_out(">> ", 3);
    epq_out_int(v);
    epq_out("\n", 1);
}

#endif /* EPQ_RT_H */
