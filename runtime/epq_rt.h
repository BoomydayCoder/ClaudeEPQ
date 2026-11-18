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
 *     byte region (pass "bytes", DESIGN.md 5.12: Int elements all provably in
 *       0..255)                -> data is EU8[]   (1 byte per element)
 *     anything else            -> data is EW64[]  (8 bytes; always a valid EAny word:
 *                                 lists as pointers, functions as tagged words)
 *   Soundness invariant of the inference: a list reachable from an Any value
 *   always has Any elements, so generic (Any) code always sees EW64 slots.
 *   Elements are accessed through the one-member structs EI32 / EW64 so that
 *   type-based alias analysis knows element stores never touch a header.
 *   A list literal of n elements is allocated with its buffer directly behind
 *   the header; growth doubles the capacity, in place when the buffer is the
 *   most recent arena allocation (epq_list_grow_to), else into a fresh buffer.
 *
 * Descriptors (EDesc table, set in epq_rt_desc by the program's entry): static
 * type descriptions for deep equality / ordering / printing.
 *   index 0 = INT, 1 = ANY, 2 = LIST of ANY, 3 = FN; others LIST(elem index).
 *   INT8 = { EPQ_K_INT, 1 }: an integer stored in one byte. It only ever appears
 *   as the element descriptor of a byte list's LIST descriptor; used as the
 *   descriptor of a value it is exactly INT (every test looks at the kind).
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
 * I/O: output is buffered (64 KB to a pipe/tty, 1 MB to a file), flushed at
 * exit, before an error, and before any read that would block while output is
 * pending (never for file stdin). Input is read in 256 KB blocks and parsed by
 * hand like C++ `cin >> int`; a failed read yields 0 and all later reads
 * yield 0 (LANGUAGE.md 9.2).
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
#include <stdlib.h>
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
/* Byte-list elements (pass "bytes"). unsigned _BitInt(8) is not a character
 * type, so (unlike unsigned char) a byte store does not alias list headers for
 * type-based alias analysis. C++ (the JIT's runtime bridge) uses uint8_t. */
#if !defined(__cplusplus) && defined(__BITINT_MAXWIDTH__)
__extension__ typedef unsigned _BitInt(8) epq_u8_t;
#else
typedef uint8_t epq_u8_t;
#endif
typedef struct EU8 { epq_u8_t v; } EU8;
typedef struct __attribute__((aligned(16))) EList {
    void* data;
    int32_t len;
    int32_t cap;
} EList;

enum { EPQ_K_INT = 0, EPQ_K_ANY = 1, EPQ_K_FN = 2, EPQ_K_LIST = 3, EPQ_K_PAIR = 4 };
/* PAIR: an unboxed two-integer list (src/tir.hpp Pair); its word is the packed int64. */
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
/* Output is flushed whenever epq_olim bytes are pending. The limit starts at
 * EPQ_OBUF_SMALL and is decided at the first flush from the kind of stdout:
 * pipes, sockets and ttys keep 64 KB (the pipe's own capacity: each write
 * completes in one copy while the reader drains in parallel; measured fastest),
 * regular files and /dev/null get the whole EPQ_OBUF (fewer write calls). */
#ifndef EPQ_OBUF
#define EPQ_OBUF (1024 * 1024)
#endif
#if EPQ_OBUF < 65536
#define EPQ_OBUF_SMALL EPQ_OBUF
#else
#define EPQ_OBUF_SMALL 65536
#endif
#define EPQ_OSLACK 64
static char epq_obuf[EPQ_OBUF + EPQ_OSLACK];
static uint32_t epq_olen;
static uint32_t epq_olim = EPQ_OBUF_SMALL;
static int epq_odecided;

EPQ_RT_API void epq_write_all(int fd, const char* p, size_t n) {
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return;
        p += w;
        n -= (size_t)w;
    }
}

EPQ_RT_API void epq_flush(void) {
    if (epq_olen) {
        epq_write_all(1, epq_obuf, epq_olen);
        epq_olen = 0;
        if (EPQ_UNLIKELY(!epq_odecided)) {
            struct stat st;
            epq_odecided = 1;
            if (fstat(1, &st) == 0 &&
                (S_ISREG(st.st_mode) || (S_ISCHR(st.st_mode) && !isatty(1))))
                epq_olim = EPQ_OBUF;
        }
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
    if (EPQ_UNLIKELY(epq_olen > epq_olim - n)) epq_flush();
    return epq_obuf + epq_olen;
}
EPQ_INLINE void epq_ocommit(char* o) { epq_olen = (uint32_t)(o - epq_obuf); }
/* Within a loop that keeps the write pointer local: room for n more bytes. */
EPQ_INLINE char* epq_oroom(char* o, uint32_t n) {
    if (EPQ_UNLIKELY(o > epq_obuf + epq_olim - n)) {
        epq_ocommit(o);
        epq_flush();
        return epq_obuf;
    }
    return o;
}

EPQ_INLINE void epq_out(const char* s, uint32_t n) {
    if (EPQ_UNLIKELY(epq_olen > epq_olim - n)) {
        epq_flush();
        if (n > epq_olim) { /* never happens for the runtime's own strings */
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
    return t + 1 - ((u | 1) < epq_pow10[t]); /* u|1: 0 has one digit */
}
/* Writes u in decimal at o (at most 10 bytes), returns the end. Digits are
 * produced two at a time, right to left, straight into the destination.
 * (Measured against an all-digits-at-once SWAR formatter: this is as fast for
 * 7-digit values and faster for small and 10-digit ones; tests/rtio/rtbench.c.
 * Checked exhaustively over all int32 by tests/rtio/fmtcheck.c.) */
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

/* The bump arena is a sequence of large lazily-committed virtual reservations
 * (anonymous mmap: pages cost nothing until touched). A request that does not
 * fit opens a new reservation of max(1 GB, 2 x request) and bumps from it, so
 * even a huge list stays the most recent allocation and keeps growing in place
 * (epq_list_grow_to). The tail of the abandoned reservation is never touched. */
#define EPQ_ARENA_CHUNK ((size_t)1 << 30)
EPQ_RT_API void* epq_alloc_slow(size_t n) {
    size_t chunk = EPQ_ARENA_CHUNK;
    if (n > chunk / 2) chunk = ((n + 4095) & ~(size_t)4095) * 2;
    void* p = mmap(0, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        chunk = (n + 4095) & ~(size_t)4095; /* retry without the headroom */
        p = mmap(0, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) epq_fail(EPQ_E_MEMORY);
    }
    epq_bump = (char*)p + n;
    epq_bump_end = (char*)p + chunk;
    return p;
}

/* ---- free lists (static reclamation, DESIGN.md §5.7) -------------------------
 * Freed blocks (list headers with inline buffers, outgrown / freed buffers) go
 * to size-class free lists that epq_alloc consults before bumping. Classes:
 * exact multiples of 16 bytes up to EPQ_FL_SMALL_MAX (class n/16), then powers
 * of two (a block of size s goes to class floor(log2 s); a request of n is
 * served from class ceil(log2 n)). The first word of a free block links it.
 * EPQ_POISON_FREED=1: freed memory is filled with a garbage pattern (headers
 * get len 0x7fffffff and an unmapped data pointer) and never reused. */
#define EPQ_FL_SMALL_MAX 512
#ifndef EPQ_GROW_FACTOR
#define EPQ_GROW_FACTOR 2 /* list capacity growth factor (epq_list_grow) */
#endif
static void* epq_fl_small[EPQ_FL_SMALL_MAX / 16 + 1];
static void* epq_fl_big[64];
static int epq_poison_mode = -1; /* -1: not yet read from the environment */

EPQ_RT_API void* epq_alloc_big(size_t n) {
    unsigned c = 64 - (unsigned)__builtin_clzll((unsigned long long)(n - 1)); /* ceil(log2 n) */
    void* p = c < 64 ? epq_fl_big[c] : 0;
    if (p) {
        epq_fl_big[c] = *(void**)p;
        return p;
    }
    if ((size_t)(epq_bump_end - epq_bump) < n) return epq_alloc_slow(n);
    p = epq_bump;
    epq_bump += n;
    return p;
}

EPQ_INLINE void* epq_alloc(size_t n) {
    n = (n + 15) & ~(size_t)15;
    if (EPQ_UNLIKELY(n > EPQ_FL_SMALL_MAX)) return epq_alloc_big(n);
    void* f = epq_fl_small[n >> 4];
    if (f) {
        epq_fl_small[n >> 4] = *(void**)f;
        return f;
    }
    char* p = epq_bump;
    if (EPQ_UNLIKELY((size_t)(epq_bump_end - p) < n)) return epq_alloc_slow(n);
    epq_bump = p + n;
    return p;
}

EPQ_RT_API int epq_poison_on(void) {
    if (epq_poison_mode < 0) {
        const char* e = getenv("EPQ_POISON_FREED");
        epq_poison_mode = (e && e[0] == '1') ? 1 : 0;
    }
    return epq_poison_mode;
}
/* Returns a block of (at least) s bytes, 16-aligned, to the free lists. */
EPQ_RT_API void epq_free_block(void* p, size_t s) {
    if (epq_poison_on()) {
        memset(p, 0xA5, s);
        return;
    }
    s &= ~(size_t)15;
    if (s < 16) return;
    if (s <= EPQ_FL_SMALL_MAX) {
        *(void**)p = epq_fl_small[s >> 4];
        epq_fl_small[s >> 4] = p;
    } else {
        unsigned c = 63 - (unsigned)__builtin_clzll((unsigned long long)s); /* floor(log2 s) */
        *(void**)p = epq_fl_big[c];
        epq_fl_big[c] = p;
    }
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

/* Grows l's capacity to at least ncap elements (clamped to INT32_MAX).
 * Invariant used for in-place growth: a list buffer carved from the bump arena
 * occupies exactly r16(cap * esz) bytes from data (epq_alloc rounds to 16;
 * epq_list_new and this function keep cap * esz a multiple of 16 unless clamped).
 * So if data + r16(cap * esz) == epq_bump, the buffer (or the header block with
 * its inline buffer) is the most recent allocation and is extended by bumping.
 * A buffer taken from a free list may be larger than r16(cap * esz); then
 * data + r16(cap * esz) lies strictly inside that block, below epq_bump, and the
 * test fails safely. In-place growth frees nothing, so it is also correct with
 * EPQ_POISON_FREED=1 and for header blocks with inline buffers (epq_list_free
 * then returns sizeof(EList) + cap * esz, the extended block). */
EPQ_RT_API void epq_list_grow_to(EList* l, uint32_t esz, int64_t ncap) {
    if (ncap > 0x7fffffff) ncap = 0x7fffffff;
    if (ncap <= l->cap) return;
    size_t old = ((size_t)l->cap * esz + 15) & ~(size_t)15;
    size_t nb = ((size_t)ncap * esz + 15) & ~(size_t)15;
#ifndef EPQ_NO_GROW_IN_PLACE /* (measurement switch: EPQ_CFLAGS=-DEPQ_NO_GROW_IN_PLACE) */
    if ((char*)l->data + old == epq_bump && (size_t)(epq_bump_end - epq_bump) >= nb - old) {
        epq_bump += nb - old;
        int64_t c = (int64_t)(nb / esz);
        l->cap = (int32_t)(c > 0x7fffffff ? 0x7fffffff : c);
        return;
    }
#endif
    void* nd = epq_alloc((size_t)ncap * esz);
    memcpy(nd, l->data, (size_t)l->len * esz);
    /* An outgrown out-of-line buffer is only reachable through this header:
     * recycle it (an inline buffer is part of the header's block). */
    if (l->data != (void*)((char*)l + sizeof(EList))) epq_free_block(l->data, (size_t)l->cap * esz);
    l->data = nd;
    l->cap = (int32_t)ncap;
}

EPQ_RT_API void epq_list_grow(EList* l, uint32_t esz) {
    epq_list_grow_to(l, esz, l->cap < 4 ? 8 : (int64_t)l->cap * EPQ_GROW_FACTOR);
}

/* ---- capacity hints and bulk copy (appendopt, src/tir.hpp Reserve / AppendRange) ---- */
#define EPQ_RESERVE_MAX ((int32_t)1 << 26)
/* cap(l) >= len(l) + min(max(n, 0), 2^26); never changes len or contents. */
EPQ_RT_API void epq_list_reserve(EList* l, int32_t n, uint32_t esz) {
    if (!l || n <= 0) return;
    if (n > EPQ_RESERVE_MAX) n = EPQ_RESERVE_MAX;
    int64_t need = (int64_t)l->len + n;
    if (need > l->cap) {
        /* A hint repeated inside an outer loop must not defeat amortised growth
         * (exact-fit growth would copy the whole list every outer iteration). */
        if (l->len > 0 && (int64_t)l->cap * EPQ_GROW_FACTOR > need) need = (int64_t)l->cap * EPQ_GROW_FACTOR;
        epq_list_grow_to(l, esz, need);
    }
}
/* Appends m[lo..hi) to l if that is certainly what the element-by-element loop
 * would do without failing (value 1); otherwise does nothing (value 0). */
EPQ_RT_API int32_t epq_append_range(EList* l, EList* m, int32_t lo, int32_t hi, uint32_t esz) {
    if (hi <= lo) return 1;
    if (!l || !m || l == m || lo < 0 || hi > m->len) return 0;
    int64_t n = (int64_t)hi - lo;
    int64_t need = (int64_t)l->len + n;
    if (need > 0x7fffffff) return 0;
    if (need > l->cap) {
        int64_t c = need;
        if (l->len > 0 && (int64_t)l->cap * 2 > c) c = (int64_t)l->cap * 2; /* keep appends amortised */
        epq_list_grow_to(l, esz, c);
    }
    memcpy((char*)l->data + (size_t)l->len * esz, (const char*)m->data + (size_t)lo * esz, (size_t)n * esz);
    l->len = (int32_t)need;
    return 1;
}

/* Frees list l (element size esz): its header and its buffer, never what the
 * elements point to. The caller guarantees no other reference is ever used. */
EPQ_RT_API void epq_list_free(EList* l, uint32_t esz) {
    size_t buf = (size_t)l->cap * esz;
    int inl = l->data == (void*)((char*)l + sizeof(EList));
    if (epq_poison_on()) {
        if (!inl) memset(l->data, 0xA5, buf);
        else memset((char*)l + sizeof(EList), 0xA5, buf);
        l->data = (void*)(uintptr_t)0xDEAD0000DEAD0000ull; /* unmapped */
        l->len = 0x7fffffff;
        l->cap = 0x7fffffff;
        return;
    }
    if (inl) {
        epq_free_block(l, sizeof(EList) + buf);
    } else {
        epq_free_block(l->data, buf);
        epq_free_block(l, 2 * sizeof(EList)); /* header block: >= 16 + 16 bytes */
    }
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
/* byte lists (pass "bytes"): values are in 0..255; loads zero-extend */
EPQ_INLINE int32_t epq_ld_u8(EList* l, int32_t i) {
    EU8* d = (EU8*)l->data;
    if (EPQ_UNLIKELY(((uint32_t)i >= (uint32_t)l->len) | (d == 0))) epq_fail(EPQ_E_INDEX_OOB);
    return (int32_t)d[i].v;
}
EPQ_INLINE int32_t epq_st_u8(EList* l, int32_t i, int32_t v) {
    EU8* d = (EU8*)l->data;
    if (EPQ_UNLIKELY(((uint32_t)i >= (uint32_t)l->len) | (d == 0))) epq_fail(EPQ_E_INDEX_OOB);
    d[i].v = (epq_u8_t)v;
    return v;
}
EPQ_INLINE EList* epq_push_u8(EList* l, int32_t v) {
    int32_t n = l->len;
    if (EPQ_UNLIKELY(n == l->cap)) epq_list_grow(l, 1);
    ((EU8*)l->data)[n].v = (epq_u8_t)v;
    l->len = n + 1;
    return l;
}
EPQ_INLINE int32_t epq_pop_u8(EList* l) {
    if (EPQ_UNLIKELY(l->len == 0)) epq_fail(EPQ_E_LIST_EMPTY);
    return (int32_t)((EU8*)l->data)[--l->len].v;
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

/* ---- pairs (src/tir.hpp Pair) ------------------------------------------------
 * (int64)first << 32 | ((uint32)second ^ 0x80000000): signed 64-bit order is
 * lexicographic order, equality is bit equality. */
typedef int64_t EPair;
EPQ_INLINE EPair epq_pair_make(int32_t a, int32_t b) {
    return (EPair)(((uint64_t)(uint32_t)a << 32) | (uint64_t)((uint32_t)b ^ 0x80000000u));
}
EPQ_INLINE int32_t epq_pair_first(EPair p) { return (int32_t)((uint64_t)p >> 32); }
EPQ_INLINE int32_t epq_pair_second(EPair p) { return (int32_t)((uint32_t)(uint64_t)p ^ 0x80000000u); }
EPQ_RT_API EPQ_COLD void epq_fail_index(void) { epq_fail(EPQ_E_INDEX_OOB); }
EPQ_INLINE int32_t epq_pair_index(EPair p, int32_t i) {
    if (EPQ_UNLIKELY((uint32_t)i > 1u)) epq_fail_index();
    return i ? epq_pair_second(p) : epq_pair_first(p);
}
EPQ_INLINE int32_t epq_check_index(int32_t i, int32_t len) {
    if (EPQ_UNLIKELY((uint32_t)i >= (uint32_t)len)) epq_fail_index();
    return i;
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
    if (epq_rt_desc[ed].kind == EPQ_K_INT) {
        if (epq_rt_desc[ed].elem) return (uint32_t)((EU8*)l->data)[k].v; /* INT8: byte list */
        return (uint32_t)((EI32*)l->data)[k].v;
    }
    return ((EW64*)l->data)[k].v;
}
/* Element width in bytes of a list whose element descriptor is ed. */
EPQ_INLINE uint32_t epq_elem_width(uint32_t ed) {
    if (epq_rt_desc[ed].kind == EPQ_K_INT) return epq_rt_desc[ed].elem ? 1 : 4;
    return 8;
}

/* A list-like view of a LIST or PAIR (desc, word), for mixed comparisons. */
typedef struct EView {
    int32_t n;
    uint32_t ed;
    EList* l;
    int32_t p[2];
} EView;
EPQ_INLINE void epq_view(uint32_t d, EAny w, EView* v) {
    if (epq_rt_desc[d].kind == EPQ_K_PAIR) {
        v->n = 2;
        v->ed = EPQ_D_INT;
        v->l = 0;
        v->p[0] = epq_pair_first((EPair)w);
        v->p[1] = epq_pair_second((EPair)w);
    } else {
        v->l = (EList*)(uintptr_t)w;
        v->n = v->l->len;
        v->ed = epq_rt_desc[d].elem;
    }
}
EPQ_INLINE EAny epq_view_word(const EView* v, int32_t k) {
    if (!v->l) return (uint32_t)v->p[k];
    return epq_elem_word(v->ed, v->l, k);
}
EPQ_RT_API int epq_eq(uint32_t da, EAny a, uint32_t db, EAny b);
EPQ_RT_API int epq_lt(uint32_t da, EAny a, uint32_t db, EAny b);
/* records (block "records" at the end of this file) */
enum { EPQ_K_REC = 5, EPQ_K_FIELD = 6 };
EPQ_RT_API int epq_eq_rec(uint32_t da, EAny a, uint32_t db, EAny b);
EPQ_RT_API int epq_lt_rec(uint32_t da, EAny a, uint32_t db, EAny b);
EPQ_RT_API void epq_print_rec(uint32_t d, EAny w);
EPQ_RT_API int epq_cmp3_rec(uint32_t da, EAny a, uint32_t db, EAny b); /* integration: three-way record compare */
/* ---- litcmp: three-way deep comparison -------------------------------------------
 * epq_cmp3(a, b) < 0 / == 0 / > 0  <=>  a < b / neither / b < a, with the original's
 * `<` (std::lexicographical_compare: per element pair k, `x<y` then `y<x`).
 * It raises EPQ_E_INVALID_CMP in exactly the same situations: element pair k is
 * examined only if all earlier pairs compared equal; whether a pairing is valid is
 * symmetric in its two sides; and once `x<y` returned without raising, `y<x`
 * revisits only pairs `x<y` already visited (it stops at the first differing
 * pair, which `x<y` also stopped at). So the first invalid pairing reached is the
 * same, and each pair is examined once instead of twice per nesting level. */
EPQ_RT_API int epq_cmp3(uint32_t da, EAny a, uint32_t db, EAny b);
EPQ_INLINE int epq_cmp3_i32(int32_t x, int32_t y) { return (x > y) - (x < y); }
EPQ_INLINE int epq_cmp3_li(EList* a, EList* b) {
    const EI32* pa = (const EI32*)a->data;
    const EI32* pb = (const EI32*)b->data;
    int32_t na = a->len, nb = b->len, n = na < nb ? na : nb;
    for (int32_t k = 0; k < n; k++)
        if (pa[k].v != pb[k].v) return pa[k].v < pb[k].v ? -1 : 1;
    return (na > nb) - (na < nb);
}
/* Two byte lists: bytes are unsigned values 0..255, so memcmp order is value order. */
EPQ_INLINE int epq_cmp3_lb(EList* a, EList* b) {
    int32_t na = a->len, nb = b->len, n = na < nb ? na : nb;
    int c = n ? memcmp(a->data, b->data, (size_t)n) : 0;
    if (c) return c < 0 ? -1 : 1;
    return (na > nb) - (na < nb);
}
/* PAIR on at least one side (both already normalised). */
EPQ_RT_API int epq_cmp3_pair(uint32_t da, EAny a, uint32_t db, EAny b) {
    uint32_t ka = epq_rt_desc[da].kind, kb = epq_rt_desc[db].kind;
    if (ka == EPQ_K_PAIR && kb == EPQ_K_PAIR) return ((EPair)a > (EPair)b) - ((EPair)a < (EPair)b);
    if ((ka != EPQ_K_PAIR && ka != EPQ_K_LIST) || (kb != EPQ_K_PAIR && kb != EPQ_K_LIST)) epq_fail(EPQ_E_INVALID_CMP);
    EView va, vb;
    epq_view(da, a, &va);
    epq_view(db, b, &vb);
    int32_t n = va.n < vb.n ? va.n : vb.n;
    for (int32_t k = 0; k < n; k++) {
        int c = epq_cmp3(va.ed, epq_view_word(&va, k), vb.ed, epq_view_word(&vb, k));
        if (c) return c;
    }
    return (va.n > vb.n) - (va.n < vb.n);
}
EPQ_RT_API int epq_cmp3(uint32_t da, EAny a, uint32_t db, EAny b) {
    epq_norm(&da, &a);
    epq_norm(&db, &b);
    uint32_t ka = epq_rt_desc[da].kind, kb = epq_rt_desc[db].kind;
    if (ka == EPQ_K_INT && kb == EPQ_K_INT) return epq_cmp3_i32((int32_t)a, (int32_t)b);
    if (EPQ_UNLIKELY(ka >= EPQ_K_REC || kb >= EPQ_K_REC)) return epq_cmp3_rec(da, a, db, b); /* records */
    if (EPQ_UNLIKELY(ka == EPQ_K_PAIR || kb == EPQ_K_PAIR)) return epq_cmp3_pair(da, a, db, b);
    if (ka != EPQ_K_LIST || kb != EPQ_K_LIST) epq_fail(EPQ_E_INVALID_CMP);
    EList* la = (EList*)(uintptr_t)a;
    EList* lb = (EList*)(uintptr_t)b;
    uint32_t ea = epq_rt_desc[da].elem, eb = epq_rt_desc[db].elem;
    uint32_t kea = epq_rt_desc[ea].kind, keb = epq_rt_desc[eb].kind;
    if (kea == EPQ_K_INT && keb == EPQ_K_INT) {
        uint32_t wa = epq_elem_width(ea), wb = epq_elem_width(eb);
        if (wa == 4 && wb == 4) return epq_cmp3_li(la, lb);
        if (wa == 1 && wb == 1) return epq_cmp3_lb(la, lb);
    }
    int32_t na = la->len, nb = lb->len, n = na < nb ? na : nb;
    if (kea == EPQ_K_ANY && keb == EPQ_K_ANY) {
        /* Common dynamic case: inline the both-integer element test. */
        const EW64* pa = (const EW64*)la->data;
        const EW64* pb = (const EW64*)lb->data;
        for (int32_t k = 0; k < n; k++) {
            EAny x = pa[k].v, y = pb[k].v;
            int c;
            if (EPQ_LIKELY(x & y & 1)) c = ((int64_t)x > (int64_t)y) - ((int64_t)x < (int64_t)y);
            else c = epq_cmp3(EPQ_D_ANY, x, EPQ_D_ANY, y);
            if (c) return c;
        }
        return (na > nb) - (na < nb);
    }
    for (int32_t k = 0; k < n; k++) {
        int c = epq_cmp3(ea, epq_elem_word(ea, la, k), eb, epq_elem_word(eb, lb, k));
        if (c) return c;
    }
    return (na > nb) - (na < nb);
}
/* ---- end litcmp three-way comparison ------------------------------------------ */
/* PAIR on at least one side (both already normalised). */
EPQ_RT_API int epq_eq_pair(uint32_t da, EAny a, uint32_t db, EAny b) {
    uint32_t ka = epq_rt_desc[da].kind, kb = epq_rt_desc[db].kind;
    if (ka == EPQ_K_PAIR && kb == EPQ_K_PAIR) return a == b;
    if ((ka != EPQ_K_PAIR && ka != EPQ_K_LIST) || (kb != EPQ_K_PAIR && kb != EPQ_K_LIST)) return 0;
    EView va, vb;
    epq_view(da, a, &va);
    epq_view(db, b, &vb);
    if (va.n != vb.n) return 0;
    for (int32_t k = 0; k < va.n; k++)
        if (!epq_eq(va.ed, epq_view_word(&va, k), vb.ed, epq_view_word(&vb, k))) return 0;
    return 1;
}
EPQ_RT_API int epq_lt_pair(uint32_t da, EAny a, uint32_t db, EAny b) {
    return epq_cmp3_pair(da, a, db, b) < 0; /* litcmp: three-way */
}

EPQ_RT_API int epq_eq(uint32_t da, EAny a, uint32_t db, EAny b) {
    epq_norm(&da, &a);
    epq_norm(&db, &b);
    uint32_t ka = epq_rt_desc[da].kind, kb = epq_rt_desc[db].kind;
    if (EPQ_UNLIKELY(ka >= EPQ_K_REC || kb >= EPQ_K_REC)) return epq_eq_rec(da, a, db, b); /* records */
    if (EPQ_UNLIKELY(ka == EPQ_K_PAIR || kb == EPQ_K_PAIR)) return epq_eq_pair(da, a, db, b);
    if (ka != kb) return 0;
    if (ka == EPQ_K_INT) return (int32_t)a == (int32_t)b;
    if (ka == EPQ_K_FN) return a == b;
    EList* la = (EList*)(uintptr_t)a;
    EList* lb = (EList*)(uintptr_t)b;
    if (la->len != lb->len) return 0;
    uint32_t ea = epq_rt_desc[da].elem, eb = epq_rt_desc[db].elem;
    uint32_t kea = epq_rt_desc[ea].kind, keb = epq_rt_desc[eb].kind;
    if (kea == EPQ_K_INT && keb == EPQ_K_INT) {
        uint32_t wa = epq_elem_width(ea), wb = epq_elem_width(eb);
        if (wa == wb) return memcmp(la->data, lb->data, (size_t)la->len * wa) == 0;
        /* a byte list against an int list: by value, below */
    }
    if (kea == EPQ_K_ANY && keb == EPQ_K_ANY) {
        /* litcmp: words equal => values equal (same int, same list, same fn); two
         * ints that differ are unequal; only otherwise recurse. */
        const EW64* pa = (const EW64*)la->data;
        const EW64* pb = (const EW64*)lb->data;
        for (int32_t k = 0; k < la->len; k++) {
            EAny x = pa[k].v, y = pb[k].v;
            if (x == y) continue;
            if ((x & y & 1) || !epq_eq(EPQ_D_ANY, x, EPQ_D_ANY, y)) return 0;
        }
        return 1;
    }
    for (int32_t k = 0; k < la->len; k++)
        if (!epq_eq(ea, epq_elem_word(ea, la, k), eb, epq_elem_word(eb, lb, k))) return 0;
    return 1;
}

/* a < b  (LANGUAGE.md 6: ints by value, lists lexicographically, else error) */
EPQ_RT_API int epq_lt(uint32_t da, EAny a, uint32_t db, EAny b) {
    return epq_cmp3(da, a, db, b) < 0; /* litcmp: three-way, same results and errors */
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
/* litcmp: two lists with statically Any elements (LIST(ANY) on both sides):
 * inline loop, both-integer element pairs inline, the rest via epq_cmp3 / epq_eq. */
EPQ_INLINE int epq_cmp3_la(EList* la, EList* lb) {
    const EW64* pa = (const EW64*)la->data;
    const EW64* pb = (const EW64*)lb->data;
    int32_t na = la->len, nb = lb->len, n = na < nb ? na : nb;
    for (int32_t k = 0; k < n; k++) {
        EAny x = pa[k].v, y = pb[k].v;
        int c;
        if (EPQ_LIKELY(x & y & 1)) c = ((int64_t)x > (int64_t)y) - ((int64_t)x < (int64_t)y);
        else c = epq_cmp3(EPQ_D_ANY, x, EPQ_D_ANY, y);
        if (c) return c;
    }
    return (na > nb) - (na < nb);
}
EPQ_INLINE int epq_eq_la(EList* la, EList* lb) {
    if (la->len != lb->len) return 0;
    const EW64* pa = (const EW64*)la->data;
    const EW64* pb = (const EW64*)lb->data;
    for (int32_t k = 0; k < la->len; k++) {
        EAny x = pa[k].v, y = pb[k].v;
        if (x == y) continue;
        if ((x & y & 1) || !epq_eq(EPQ_D_ANY, x, EPQ_D_ANY, y)) return 0;
    }
    return 1;
}
/* litcmp: TIR DynIsEmptyList — x is a list of length 0. */
EPQ_INLINE int32_t epq_is_empty_list(EAny x) {
    return (x & 3) == 0 && ((EList*)(uintptr_t)x)->len == 0;
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

/* [a, b, c] for a byte list (pass "bytes"). */
static __attribute__((noinline, unused)) void epq_print_blist_val(const EList* l) {
    const EU8* p = (const EU8*)l->data;
    int32_t n = l->len;
    char* o = epq_oreserve(1 + 11 + 1);
    *o++ = '[';
    if (n > 0) {
        o = epq_fmt_i32(o, (int32_t)p[0].v);
        for (int32_t i = 1; i < n; i++) {
            o = epq_oroom(o, EPQ_ELEM_MAX);
            memcpy(o, ", ", 2);
            o = epq_fmt_i32(o + 2, (int32_t)p[i].v);
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
    } else if (k == EPQ_K_REC) { /* records */
        epq_print_rec(d, w);
    } else if (k == EPQ_K_PAIR) {
        epq_out("[", 1);
        epq_out_int(epq_pair_first((EPair)w));
        epq_out(", ", 2);
        epq_out_int(epq_pair_second((EPair)w));
        epq_out("]", 1);
    } else {
        const EList* l = (const EList*)(uintptr_t)w;
        uint32_t e = epq_rt_desc[d].elem;
        uint32_t ek = epq_rt_desc[e].kind;
        if (ek == EPQ_K_INT && epq_rt_desc[e].elem) {
            epq_print_blist_val(l); /* byte list */
        } else if (ek == EPQ_K_INT) {
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

/* ---- records (src/tir.hpp "Records"; DESIGN.md §5.11) ---------------------------
 * A record of a record region with k positions is a pointer to k 8-byte slots,
 * no header (k is static). Slot j holds the descriptor word of position j's
 * static type: Int as (uint64)(uint32)v, lists/records as pointers, pairs as the
 * packed int64, functions as tagged words, Any as tagged words. The empty list
 * of a nullable record region is NULL (word 0).
 * Descriptor RECORD (kind 5) at index d: elem = k | (nullable << 16); entries
 * d+1 .. d+k (kind FIELD) give each position's descriptor in their elem.
 * A record compares, equals and prints exactly like the list it stands for. */
typedef struct ERec { EAny v; } ERec;
EPQ_INLINE ERec* epq_rec_new(uint32_t k) { return (ERec*)epq_alloc((size_t)k * 8); }

/* (desc, word) of a LIST / PAIR / REC as a sequence; other kinds are not lists. */
EPQ_INLINE int epq_rq_listlike(uint32_t d) {
    uint32_t k = epq_rt_desc[d].kind;
    return k == EPQ_K_LIST || k == EPQ_K_PAIR || k == EPQ_K_REC;
}
EPQ_INLINE int32_t epq_rq_len(uint32_t d, EAny w) {
    uint32_t k = epq_rt_desc[d].kind;
    if (k == EPQ_K_REC) return w ? (int32_t)(epq_rt_desc[d].elem & 0xffff) : 0;
    if (k == EPQ_K_PAIR) return 2;
    return ((EList*)(uintptr_t)w)->len;
}
EPQ_INLINE void epq_rq_at(uint32_t d, EAny w, int32_t j, uint32_t* ed, EAny* ew) {
    uint32_t k = epq_rt_desc[d].kind;
    if (k == EPQ_K_REC) {
        *ed = epq_rt_desc[d + 1 + (uint32_t)j].elem;
        *ew = ((ERec*)(uintptr_t)w)[j].v;
    } else if (k == EPQ_K_PAIR) {
        *ed = EPQ_D_INT;
        *ew = (uint32_t)(j ? epq_pair_second((EPair)w) : epq_pair_first((EPair)w));
    } else {
        *ed = epq_rt_desc[d].elem;
        *ew = epq_elem_word(*ed, (EList*)(uintptr_t)w, j);
    }
}
/* At least one side is a REC (both normalised). */
EPQ_RT_API int epq_eq_rec(uint32_t da, EAny a, uint32_t db, EAny b) {
    if (!epq_rq_listlike(da) || !epq_rq_listlike(db)) return 0;
    int32_t na = epq_rq_len(da, a), nb = epq_rq_len(db, b);
    if (na != nb) return 0;
    for (int32_t k = 0; k < na; k++) {
        uint32_t ex, ey;
        EAny x, y;
        epq_rq_at(da, a, k, &ex, &x);
        epq_rq_at(db, b, k, &ey, &y);
        if (!epq_eq(ex, x, ey, y)) return 0;
    }
    return 1;
}
/* integration (litcmp x records): three-way, same element pairs and errors as
 * `x < y` followed by `y < x` (see epq_cmp3). */
EPQ_RT_API int epq_cmp3_rec(uint32_t da, EAny a, uint32_t db, EAny b) {
    if (!epq_rq_listlike(da) || !epq_rq_listlike(db)) epq_fail(EPQ_E_INVALID_CMP);
    int32_t na = epq_rq_len(da, a), nb = epq_rq_len(db, b), n = na < nb ? na : nb;
    for (int32_t k = 0; k < n; k++) {
        uint32_t ex, ey;
        EAny x, y;
        epq_rq_at(da, a, k, &ex, &x);
        epq_rq_at(db, b, k, &ey, &y);
        int c = epq_cmp3(ex, x, ey, y);
        if (c) return c;
    }
    return (na > nb) - (na < nb);
}
EPQ_RT_API int epq_lt_rec(uint32_t da, EAny a, uint32_t db, EAny b) { return epq_cmp3_rec(da, a, db, b) < 0; }
EPQ_RT_API void epq_print_rec(uint32_t d, EAny w) {
    epq_out("[", 1);
    if (w) {
        uint32_t k = epq_rt_desc[d].elem & 0xffff;
        const ERec* r = (const ERec*)(uintptr_t)w;
        for (uint32_t j = 0; j < k; j++) {
            if (j) epq_out(", ", 2);
            epq_print_val(epq_rt_desc[d + 1 + j].elem, r[j].v);
        }
    }
    epq_out("]", 1);
}
/* Helpers of the typed comparisons the C backend generates for records
 * (src/cgen_rec.inc): three-way results -1 / 0 / 1, raising exactly where
 * `<` raises (LANGUAGE.md 6: a three-way compare reaches the same element
 * pairs as `x < y` followed by `y < x`). */
EPQ_INLINE int epq_c3i(int32_t a, int32_t b) { return (a > b) - (a < b); }
EPQ_INLINE int epq_c3p(EPair a, EPair b) { return (a > b) - (a < b); }
EPQ_RT_API EPQ_COLD int epq_c3_invalid(void) { epq_fail(EPQ_E_INVALID_CMP); }
EPQ_RT_API int epq_c3_dyn(uint32_t da, EAny a, uint32_t db, EAny b) {
    int c = epq_cmp3(da, a, db, b);
    return (c > 0) - (c < 0);
}
static inline __attribute__((unused)) int epq_c3_li(EList* a, EList* b) {
    const EI32* pa = (const EI32*)a->data;
    const EI32* pb = (const EI32*)b->data;
    int32_t na = a->len, nb = b->len;
    int32_t n = na < nb ? na : nb;
    for (int32_t k = 0; k < n; k++)
        if (pa[k].v != pb[k].v) return pa[k].v < pb[k].v ? -1 : 1;
    return (na > nb) - (na < nb);
}

#endif /* EPQ_RT_H */
