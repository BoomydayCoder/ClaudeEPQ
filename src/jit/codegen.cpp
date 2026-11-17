// ARM64 code generator for TIR (DESIGN.md §6).
//
// One pass over each function instance's TIR tree, emitting straight into one
// a64::Assembler; the result is installed into a MAP_JIT region and main runs
// on a jit::BigStack. The C backend (src/cgen.cpp) is the executable
// specification of every op: evaluation order, which check fires first and the
// error raised are identical.
//
// Register use (macOS arm64 ABI):
//   x0..x8           expression temporaries (+ x9..x15 not given to locals)
//   x9..x15          locals whose live range contains no call (cold paths save them)
//   x19..x28         locals (heaviest first), then values held across calls
//   x16, x17         scratch for single instruction sequences (never live across
//                    anything that may use the assembler's *_imm helpers)
//   x18              never touched; x29 untouched (no frame chain); x30 link
// Arguments x0..x7 then stack; results in x0 (Int as w0).
//
// Frame (when needed): [sp, out) outgoing stack arguments; slots (8 bytes:
// spilled locals, held temporaries, CallDyn argument arrays); callee-saved
// registers; x29/x30. Each function is generated twice: a dry run into a
// scratch assembler measures the frame (slots, callee-saved registers used)
// and the real run emits the final code with an exact prologue.
//
// Local allocation: live interval = [first ref, last ref] in tree order,
// extended over each enclosing loop that the value may flow around (refs
// outside the loop, or the first ref inside it is not an unconditional write),
// and back to the entry for parameters and locals that may be read before being
// assigned (those are zero-initialised like the C backend's declarations).
// Greedy by weight = sum over refs of 10^loopdepth.
//
// Holding: when an operator evaluates kid i while the values of kids < i are
// pending, each pending value is kept in a temporary unless a later kid
// contains a call (then: a free callee-saved register or a frame slot), the
// temporaries run low, or (for a register borrowed from a local) a later kid
// may write that register.
#include "codegen.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sys/mman.h>
#include <sys/qos.h>
#include <time.h>
#include <unistd.h>

#include "a64.hpp"
#include "bigstack.hpp"
#include "disasm.hpp"
#include "jitmem.hpp"
#include "rt_bridge.hpp"

namespace epq::jitcg {

using namespace a64;
using tir::Err;
using tir::Func;
using tir::Module;
using tir::Node;
using tir::Op;
using tir::TK;
using tir::Type;

namespace {

uint64_t now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

// Runtime error codes (runtime/epq_rt.h EPQ_E_*): tir::Err values match.
constexpr int kErrCount = 15;

enum Helper : int {
    H_FAIL,
    H_PRINT_INT,
    H_PRINT,
    H_READ_INT,
    H_LIST_NEW,
    H_LIST_GROW,
    H_EQ,
    H_LT,
    H_EQ_LI,
    H_LT_LI,
    H_DYN_NEG,
    H_DYN_ABS,
    H_CALL_DYN,
    H_ALLOC_SLOW,
    H_LIST_FREE,
    H_RESERVE,
    H_APPEND_RANGE,
    H_EQ_LA,
    H_CMP3_LA,
    H_ALLOC,
    H_CMP3,
    H_CMP3_LI,
    H_COUNT
};

const void* helper_addr(int h) {
    switch (h) {
    case H_FAIL: return (const void*)&jrt_fail;
    case H_PRINT_INT: return (const void*)&jrt_print_int;
    case H_PRINT: return (const void*)&jrt_print;
    case H_READ_INT: return (const void*)&jrt_read_int;
    case H_LIST_NEW: return (const void*)&jrt_list_new;
    case H_LIST_GROW: return (const void*)&jrt_list_grow;
    case H_EQ: return (const void*)&jrt_eq;
    case H_LT: return (const void*)&jrt_lt;
    case H_EQ_LI: return (const void*)&jrt_eq_li;
    case H_LT_LI: return (const void*)&jrt_lt_li;
    case H_DYN_NEG: return (const void*)&jrt_dyn_neg;
    case H_DYN_ABS: return (const void*)&jrt_dyn_abs;
    case H_CALL_DYN: return (const void*)&jrt_call_dyn;
    case H_ALLOC_SLOW: return (const void*)&jrt_alloc_slow;
    case H_LIST_FREE: return (const void*)&jrt_list_free;
    case H_RESERVE: return (const void*)&jrt_list_reserve;
    case H_APPEND_RANGE: return (const void*)&jrt_append_range;
    case H_EQ_LA: return (const void*)&jrt_eq_la;
    case H_CMP3_LA: return (const void*)&jrt_cmp3_la;
    case H_ALLOC: return (const void*)&jrt_alloc;
    case H_CMP3: return (const void*)&jrt_cmp3;
    case H_CMP3_LI: return (const void*)&jrt_cmp3_li;
    }
    return nullptr;
}
const char* helper_name(int h) {
    static const char* n[] = {"rt_fail", "rt_print_int", "rt_print", "rt_read_int", "rt_list_new", "rt_list_grow",
                              "rt_eq", "rt_lt", "rt_eq_li", "rt_lt_li", "rt_dyn_neg", "rt_dyn_abs", "rt_call_dyn", "rt_alloc_slow", "rt_list_free",
                              "rt_list_reserve", "rt_append_range", "rt_eq_la", "rt_cmp3_la", "rt_alloc", "rt_cmp3", "rt_cmp3_li"};
    return n[h];
}

constexpr uint32_t bit(int r) { return 1u << r; }
constexpr uint32_t kCallerLocal = 0xfe00u;     // x9..x15
constexpr uint32_t kCallee = 0x1ff80000u;      // x19..x28
constexpr uint32_t kBaseTemps = 0x1ffu;        // x0..x8

// ---- values -------------------------------------------------------------------
struct V {
    enum K : uint8_t { NONE, IMM, REG, SLOT } k = NONE;
    uint8_t r = 0;
    bool own = false;
    int32_t slot = -1;
    int64_t imm = 0;
    static V none() { return V{}; }
    static V im(int64_t v) {
        V x;
        x.k = IMM;
        x.imm = v;
        return x;
    }
    static V reg(int r, bool own) {
        V x;
        x.k = REG;
        x.r = (uint8_t)r;
        x.own = own;
        return x;
    }
};

struct NInfo {
    uint32_t en = 0, lv = 0;
    bool call = false;
    bool cref = false;  // ConstInt read through its pseudo-local register
    int32_t hoist = -1; // loop-invariant arithmetic: read from this pseudo-local
};

struct Ref {
    uint32_t pos;
    uint32_t ctx;
    bool write;
};
struct LoopRec {
    uint32_t en, lv, cond_ctx, body_ctx;
    const tir::Node* node;
};

struct LocalInfo {
    bool fn = false;       // Fn-typed: no representation
    bool x64 = false;      // List / Any
    bool any = false;      // Any (zero-initialised to boxed 0)
    bool needs_init = false;
    bool param = false;
    int param_idx = -1;
    double weight = 0;
    uint32_t lo = UINT32_MAX, hi = 0;
    bool crosses_call = false;
    int reg = -1;   // assigned register
    int slot = -1;  // else frame slot
    bool is_const = false;  // pseudo-local holding a large constant used in loops
    int64_t cval = 0;
    int alias = -1;  // coalesced into this local (result temporaries, see coalesce())
    std::vector<Ref> refs;
};

bool is64(Type t) { return t.k == TK::List || t.k == TK::Any || t.k == TK::Pair || t.k == TK::Ptr || t.k == TK::Rec; }

Cond cond_of(Op o) {
    switch (o) {
    case Op::EqI32: return EQ;
    case Op::NeI32: return NE;
    case Op::LtI32: return LT;
    case Op::LeI32: return LE;
    case Op::GtI32: return GT;
    default: return GE;
    }
}
Cond swap_cond(Cond c) {
    switch (c) {
    case LT: return GT;
    case GT: return LT;
    case LE: return GE;
    case GE: return LE;
    default: return c;
    }
}


// ============================================================================
class Compiler {
public:
    const Module& M;
    explicit Compiler(const Module& m) : M(m) {}

    jit::CodeRegion region;
    Assembler main_as;
    Assembler* A = &main_as;
    std::unique_ptr<Assembler> scratch;
    bool dry = false;

    // data block (globals) near the code
    uint8_t* data = nullptr;
    size_t data_size = 0;
    std::vector<uint32_t> gval_off, gdef_off;
    bool data_near = false;
    uint32_t bump_off = 0;   // data: &epq_bump (runtime bump pointer; end follows it)
    uint32_t fntab_off = 0;  // data: per DefId {int32 arity, pad, u64 direct generic entry}
    bool bump_adjacent = false;
    uint32_t fl_off = 0;     // data: &epq_fl_small[0]
    bool poison = false;     // EPQ_POISON_FREED=1: never inline frees / free-list reuse

    bool getenv_always_dry_ = false;  // EPQ_JIT_ALWAYS_DRY: diagnostic, dry run for every function
    int single_pass_ = 0, redone_ = 0;
    Label helper_label[H_COUNT];
    bool helper_used[H_COUNT] = {};
    Label dummy;  // dry run: stands for every external label
    std::vector<std::pair<size_t, std::string>> syms;
    std::vector<size_t> func_off;
    std::vector<int32_t> wrapper_off;  // by DefId

    // ---- per function ----
    const Func* F = nullptr;
    // node -> NInfo: flat open-addressing table (pointer keys), reused per function
    struct InfoMap {
        struct E {
            const Node* k;
            NInfo v;
        };
        std::vector<E> t;
        size_t mask = 0, n = 0;
        void reset(size_t expect) {
            size_t cap = 64;
            while (cap < expect * 2) cap <<= 1;
            if (t.size() != cap) t.assign(cap, E{nullptr, {}});
            else std::fill(t.begin(), t.end(), E{nullptr, {}});
            mask = cap - 1;
            n = 0;
        }
        static size_t h(const Node* p) { return (size_t)(((uintptr_t)p >> 4) * 0x9E3779B97F4A7C15ull >> 20); }
        void put(const Node* k, NInfo v) {
            if (2 * (n + 1) > t.size()) grow();
            size_t i = h(k) & mask;
            while (t[i].k && t[i].k != k) i = (i + 1) & mask;
            if (!t[i].k) ++n;
            t[i] = E{k, v};
        }
        const NInfo* get(const Node* k) const {
            size_t i = h(k) & mask;
            while (t[i].k) {
                if (t[i].k == k) return &t[i].v;
                i = (i + 1) & mask;
            }
            return nullptr;
        }
        void grow() {
            std::vector<E> old;
            old.swap(t);
            t.assign(old.size() * 2, E{nullptr, {}});
            mask = t.size() - 1;
            n = 0;
            for (auto& e : old)
                if (e.k) put(e.k, e.v);
        }
    } info;
    static size_t count_nodes(const Node* n) {
        size_t c = 1;
        for (uint32_t i = 0; i < n->nkids; ++i) c += count_nodes(n->kids[i]);
        return c;
    }
    std::vector<LocalInfo> L;
    std::vector<LoopRec> loops;
    std::vector<uint32_t> call_pos;
    std::vector<std::vector<uint32_t>> reg_writes;  // per register: sorted write positions
    uint32_t pos_ = 0, ctx_ = 0, ctx_next_ = 0;
    std::vector<uint32_t> ctx_stack_;

    uint32_t temp_pool = 0, temp_free = 0;
    uint32_t spill_pool = 0, spill_free = 0;
    uint32_t used_callee = 0;
    uint32_t caller_locals = 0;  // x9..x15 assigned to locals
    int nfixed = 0;              // frame slots of spilled locals
    std::vector<uint8_t> slot_busy;
    int max_slots = 0;
    uint32_t out_bytes = 0;
    bool hot_calls = false;
    // frame (final)
    uint32_t frame_size = 0, slot_base = 0, saved_base = 0, lr_off = 0;
    bool has_frame = false;
    uint32_t saved_mask = 0;
    Label err_label[kErrCount];
    struct ColdPath {
        Label l;
        std::function<void()> f;
    };
    std::vector<ColdPath> colds;
    std::vector<std::pair<Label, uint32_t>> inline_breaks;  // (label id, InlineBlock a)

    // ------------------------------------------------------------------ utils
    [[noreturn]] void bug(const char* what) {
        std::fprintf(stderr, "epq: jit internal error: %s (function %s)\n", what, F ? F->name.c_str() : "?");
        std::abort();
    }
    const NInfo& I(const Node* n) {
        const NInfo* p = info.get(n);
        if (!p) bug("node without info");
        return *p;
    }
    Label ext_helper(int h) {
        if (dry) return dummy;
        if (!helper_used[h]) {
            helper_used[h] = true;
            helper_label[h] = main_as.new_label();
        }
        return helper_label[h];
    }
    void call_helper(int h) {
        A->bl(ext_helper(h));
    }
    Label err(int e) {
        if (!err_label[e].valid()) err_label[e] = A->new_label();
        return err_label[e];
    }
    Label err(Err e) { return err((int)e); }

    // ------------------------------------------------------------------ data block
    void data_addr(Reg t, uint32_t off, uint32_t* lo12) {
        uint64_t target = (uint64_t)(uintptr_t)(data + off);
        if (data_near && !dry) {
            uint64_t pc = (uint64_t)(uintptr_t)region.base() + A->pos();
            int64_t pages = (int64_t)(target >> 12) - (int64_t)(pc >> 12);
            size_t p0 = A->pos();
            A->adrp_off(t.xr(), pages);
            size_t p1 = A->pos() - 4;
            if (p1 != p0) {  // an island was inserted first: recompute
                uint64_t pc2 = (uint64_t)(uintptr_t)region.base() + p1;
                int64_t pg2 = (int64_t)(target >> 12) - (int64_t)(pc2 >> 12);
                Assembler tmp;
                tmp.adrp_off(t.xr(), pg2);
                A->patch_word(p1, tmp.word_at(0));
            }
            *lo12 = (uint32_t)(target & 0xfff);
        } else {
            li(t.xr(), (int64_t)target);
            *lo12 = 0;
        }
    }

    // constant into a register (zero as a plain movz)
    void li(Reg d, int64_t v) {
        if (d.x ? v == 0 : (int32_t)v == 0) A->movz(d, 0);
        else A->load_imm(d, v);
    }

    // ------------------------------------------------------------------ temporaries
    int talloc() {
        uint32_t f = temp_free;
        if (!f) bug("out of temporaries");
        int r = __builtin_ctz(f);
        temp_free &= ~bit(r);
        return r;
    }
    int talloc_specific(int r) {
        if (!(temp_free & bit(r))) bug("temporary not free");
        temp_free &= ~bit(r);
        return r;
    }
    int nfree() const { return __builtin_popcount(temp_free); }
    int salloc() {
        for (size_t i = 0; i < slot_busy.size(); ++i)
            if (!slot_busy[i]) {
                slot_busy[i] = 1;
                return (int)i;
            }
        slot_busy.push_back(1);
        max_slots = std::max(max_slots, (int)slot_busy.size() + nfixed);
        return (int)slot_busy.size() - 1;
    }
    // n contiguous dynamic slots; returns the first index
    int salloc_n(int n) {
        for (size_t i = 0;; ++i) {
            bool ok = true;
            for (int k = 0; k < n; ++k)
                if (i + k < slot_busy.size() && slot_busy[i + k]) {
                    ok = false;
                    break;
                }
            if (!ok) continue;
            if (slot_busy.size() < i + n) slot_busy.resize(i + n, 0);
            for (int k = 0; k < n; ++k) slot_busy[i + k] = 1;
            max_slots = std::max(max_slots, (int)slot_busy.size() + nfixed);
            return (int)i;
        }
    }
    void sfree(int s) { slot_busy[(size_t)s] = 0; }
    // byte offset (from sp) of dynamic slot s / fixed slot k
    uint32_t dyn_off(int s) const { return slot_base + 8u * (uint32_t)(nfixed + s); }
    uint32_t fix_off(int k) const { return slot_base + 8u * (uint32_t)k; }

    void release(V& v) {
        if (v.k == V::REG && v.own) {
            if (temp_pool & bit(v.r)) temp_free |= bit(v.r);
            else if (spill_pool & bit(v.r)) spill_free |= bit(v.r);
        } else if (v.k == V::SLOT && v.own) {
            sfree(v.slot);
        }
        v = V::none();
    }

    // Value in a register (x view if x64). Imm/slot values get a fresh temp.
    Reg R(V& v, bool x64) {
        switch (v.k) {
        case V::REG: return x64 ? X(v.r) : W(v.r);
        case V::IMM: {
            int r = talloc();
            if (x64) li(X(r), v.imm);
            else li(W(r), (int32_t)v.imm);
            v = V::reg(r, true);
            return x64 ? X(r) : W(r);
        }
        case V::SLOT: {
            int r = talloc();
            Reg t = x64 ? X(r) : W(r);
            A->ldr(t, mem(sp, dyn_off(v.slot)));
            sfree(v.slot);
            v = V::reg(r, true);
            return t;
        }
        default: {
            // no value (Fn / Bot): a zero register is fine
            int r = talloc();
            A->movz(W(r), 0);
            v = V::reg(r, true);
            return x64 ? X(r) : W(r);
        }
        }
    }
    // Ensure v is an owned temporary (so it may be overwritten).
    Reg own_reg(V& v, bool x64) {
        if (v.k == V::REG && v.own && (temp_pool & bit(v.r))) return x64 ? X(v.r) : W(v.r);
        Reg s = R(v, x64);
        if (v.own && (temp_pool & bit(v.r))) return s;
        int r = talloc();
        A->mov(X(r), s.xr());
        release(v);
        v = V::reg(r, true);
        return x64 ? X(r) : W(r);
    }
    // Destination register for an operation's result.
    int dest(int want) { return want >= 0 ? want : talloc(); }
    V result(int d, int want) { return V::reg(d, want < 0); }

    bool range_has_call(const Node* n, uint32_t from, uint32_t to) {
        for (uint32_t i = from; i <= to && i < n->nkids; ++i)
            if (I(n->kids[i]).call) return true;
        return false;
    }
    bool reg_written(int r, uint32_t a, uint32_t b) {
        auto& w = reg_writes[(size_t)r];
        auto it = std::lower_bound(w.begin(), w.end(), a);
        return it != w.end() && *it <= b;
    }
    // Move v into a location that survives a call.
    void to_safe(V& v) {
        bool x64 = true;  // copy all 64 bits
        if (spill_free) {
            int r = __builtin_ctz(spill_free);
            spill_free &= ~bit(r);
            used_callee |= bit(r);
            Reg s = R(v, x64);
            A->mov(X(r), s);
            release(v);
            v = V::reg(r, true);
            return;
        }
        int s = salloc();
        Reg src = R(v, x64);
        A->str(src, mem(sp, dyn_off(s)));
        release(v);
        v.k = V::SLOT;
        v.slot = s;
        v.own = true;
    }
    // v is pending while kids [from, to] of n are evaluated.
    void hold(V& v, const Node* n, uint32_t from, uint32_t to) {
        if (v.k != V::REG) return;
        if (early_) return;  // before the prologue: pure, call-free, no writes
        if (from > to || from >= n->nkids) return;
        bool across = range_has_call(n, from, to);
        if (!v.own) {
            bool clob = (across && (bit(v.r) & ~kCallee)) ||
                        reg_written(v.r, I(n->kids[from]).en, I(n->kids[std::min(to, n->nkids - 1)]).lv);
            if (!clob) return;
            if (across || nfree() < 6) {
                to_safe(v);
            } else {
                int r = talloc();
                A->mov(X(r), X(v.r));
                v = V::reg(r, true);
            }
            return;
        }
        if (spill_pool & bit(v.r)) return;  // already safe
        if (across || nfree() < 6) to_safe(v);
    }

    // ------------------------------------------------------------------ prepass
    double wdepth(int d) {
        static const double p[] = {1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
        return p[std::min(d, 8)];
    }
    void push_ctx() {
        ctx_stack_.push_back(ctx_);
        ctx_ = ++ctx_next_;
    }
    void pop_ctx() {
        ctx_ = ctx_stack_.back();
        ctx_stack_.pop_back();
    }
    bool cmp_inline(const Node* n) {
        Type ta = n->kids[0]->type, tb = n->kids[1]->type;
        auto intish = [](Type t) { return t.k == TK::Int || t.k == TK::Any; };
        return intish(ta) && intish(tb);
    }
    bool op_calls(const Node* n) {
        switch (n->op) {
        case Op::CallDirect:
        case Op::CallDyn:
        case Op::Print:
        case Op::Input:
        case Op::Reserve:
        case Op::AppendRange: return true;
        case Op::ListLit: return !lit_inline(n);
        case Op::Eq:
        case Op::Ne:
        case Op::Cmp: return !cmp_inline(n) && !pair_pair(n);
        default: return false;
        }
    }
    void prepass(const Node* n, int depth) {
        uint32_t en = pos_++;
        bool call = false;
        bool cref = false;
        int32_t self_hoist = -1;
        auto kid = [&](uint32_t i, int d) {
            prepass(n->kids[i], d);
            call |= info.get(n->kids[i])->call;
        };
        switch (n->op) {
        case Op::ConstInt: {
            // Constants needing more than one instruction (or a mov at their use),
            // used inside loops, compete for registers as pseudo-locals, loaded
            // at the entry of the outermost enclosing loop.
            if (depth > 0 && outer_loop_ && !no_const_reg_ && (Assembler::load_imm_len(n->a, false) > 1 || want_const_reg_)) {
                uint32_t id = const_local(n->a, n);
                L[id].refs.push_back(Ref{en, ctx_, false});
                L[id].weight += 0.5 * wdepth(depth);
                cref = true;
                self_hoist = (int32_t)id;
            }
            break;
        }
        case Op::DivI32:
        case Op::DynArith: {
            // a constant divisor keeps its immediate form (magic-number division)
            kid(0, depth);
            no_const_reg_ = true;
            kid(1, depth);
            no_const_reg_ = false;
            break;
        }
        case Op::Local: {
            LocalInfo& l = L[(size_t)n->a];
            if (!l.fn) {
                l.refs.push_back(Ref{en, ctx_, false});
                l.weight += wdepth(depth);
            }
            break;
        }
        case Op::SetLocal: {
            kid(0, depth);
            if (n->kids[0]->op == Op::Comma && n->kids[0]->kids[n->kids[0]->nkids - 1]->op == Op::Local)
                copies_.push_back(n);
            LocalInfo& l = L[(size_t)n->a];
            if (!l.fn) {
                l.refs.push_back(Ref{pos_, ctx_, true});
                l.weight += wdepth(depth);
            }
            break;
        }
        case Op::While: {
            LoopRec lr{en, 0, ctx_, 0, n};
            bool outermost = outer_loop_ == nullptr;
            if (outermost) outer_loop_ = n;
            size_t h0 = hoist_pending_.size();
            if (!no_hoist_) find_hoists(n);
            size_t h1 = hoist_pending_.size();
            kid(0, depth + 1);
            push_ctx();
            lr.body_ctx = ctx_;
            kid(1, depth + 1);
            kid(2, depth + 1);
            pop_ctx();
            lr.lv = pos_;
            loops.push_back(lr);
            // hoisted values: written at loop entry, live over the whole loop
            for (size_t h = h0; h < h1; ++h) {
                uint32_t id = hoist_pending_[h].second;
                L[id].refs.insert(L[id].refs.begin(), Ref{en, ctx_, true});
                L[id].refs.push_back(Ref{lr.lv, ctx_, false});
                hoists_[n].push_back(hoist_pending_[h]);
            }
            if (outermost) {
                outer_loop_ = nullptr;
                for (auto& c : const_nodes_)
                    if (c.first == n) {
                        uint32_t id = c.second.second;
                        L[id].refs.insert(L[id].refs.begin(), Ref{en, ctx_, true});
                        L[id].refs.push_back(Ref{lr.lv, ctx_, false});
                        hoists_[n].push_back(c.second);
                    }
            }
            break;
        }
        case Op::If:
            kid(0, depth);
            push_ctx();
            kid(1, depth);
            pop_ctx();
            push_ctx();
            kid(2, depth);
            pop_ctx();
            break;
        case Op::And:
        case Op::Or:
            kid(0, depth);
            push_ctx();
            kid(1, depth);
            pop_ctx();
            break;
        case Op::InlineBlock:
            push_ctx();
            kid(0, depth);
            pop_ctx();
            break;
        case Op::CallDirect: {
            uint32_t na = n->nkids - 1;
            if (na > 8) out_bytes = std::max(out_bytes, (uint32_t)((8 * (na - 8) + 15) & ~15u));
            for (uint32_t i = 0; i < n->nkids; ++i) kid(i, depth);
            break;
        }
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) {
                if (depth > 0 && const_wants_reg(n, i)) {
                    want_const_reg_ = true;
                    kid(i, depth);
                    want_const_reg_ = false;
                } else {
                    kid(i, depth);
                }
            }
            break;
        }
        uint32_t lv = pos_++;
        if (op_calls(n)) {
            call = true;
            call_pos.push_back(n->op == Op::ListLit ? en : lv);
        }
        if (!risky_ && n->nkids > 1) note_risk(n);
        NInfo ni{en, lv, call, cref, self_hoist};
        if (!hoist_mark_.empty()) {
            auto it = hoist_mark_.find(n);
            if (it != hoist_mark_.end()) {
                ni.hoist = (int32_t)it->second;
                LocalInfo& h = L[it->second];
                h.refs.push_back(Ref{en, ctx_, false});
                h.weight += wdepth(depth);
            }
        }
        info.put(n, ni);
    }
    // Could the code generator have to keep a value across a call here (which
    // needs a callee-saved register or a frame slot that only a dry run can
    // count)? Kids of an operator are pending while later kids are evaluated.
    bool risky_ = false;
    void note_risk(const Node* n) {
        switch (n->op) {
        case Op::Seq: case Op::If: case Op::While: case Op::Comma: case Op::Fail: case Op::And: case Op::Or:
        case Op::InlineBlock: case Op::Return: return;
        case Op::ListLit:  // the new list is pending while its elements are evaluated
            for (uint32_t i = 0; i < n->nkids; ++i)
                if (info.get(n->kids[i])->call) risky_ = true;
            return;
        case Op::CallDyn:
            if (n->nkids - 1 > 8) {  // argument array in frame slots
                risky_ = true;
                return;
            }
            break;
        default: break;
        }
        bool value_before = false;
        for (uint32_t i = 0; i < n->nkids; ++i) {
            const Node* k = n->kids[i];
            if (value_before && info.get(k)->call) {
                risky_ = true;
                return;
            }
            if (k->op != Op::ConstInt && k->op != Op::ConstFn) value_before = true;
        }
    }
    // Direct calls / branches to function entries: own fixups, patched once every
    // function is placed (a function may be regenerated, see compile_func).
    struct CallFix {
        uint32_t pos, func;
        bool is_b;
    };
    std::vector<CallFix> call_fixes;
    void call_func(uint32_t f, bool is_b = false) {
        A->emit(is_b ? 0x14000000u : 0x94000000u);
        if (!dry) call_fixes.push_back(CallFix{(uint32_t)(A->pos() - 4), f, is_b});
    }
    void patch_calls() {
        for (auto& c : call_fixes) {
            int64_t off = ((int64_t)func_off[c.func] - (int64_t)c.pos) / 4;
            main_as.patch_word(c.pos, (c.is_b ? 0x14000000u : 0x94000000u) | (uint32_t)(off & 0x3ffffff));
        }
    }

    // Definite assignment (for zero-initialisation): bits = assigned locals.
    // Fixed-size bit set (functions with more locals skip the analysis and
    // zero-initialise everything).
    static constexpr size_t kDaWords = 8;
    struct Bits {
        uint64_t w[kDaWords];
        uint64_t& operator[](size_t i) { return w[i]; }
        uint64_t operator[](size_t i) const { return w[i]; }
    };
    static void set_all(Bits& b) {
        for (auto& x : b.w) x = ~0ull;
    }
    static void band(Bits& a, const Bits& b) {
        for (size_t i = 0; i < kDaWords; ++i) a.w[i] &= b.w[i];
    }
    void da(const Node* n, Bits& b) {
        switch (n->op) {
        case Op::Local: {
            uint32_t id = (uint32_t)n->a;
            if (!(b[id >> 6] >> (id & 63) & 1)) L[id].needs_init = true;
            return;
        }
        case Op::SetLocal: {
            da(n->kids[0], b);
            uint32_t id = (uint32_t)n->a;
            b[id >> 6] |= 1ull << (id & 63);
            return;
        }
        case Op::If: {
            da(n->kids[0], b);
            Bits t = b, e = b;
            da(n->kids[1], t);
            da(n->kids[2], e);
            band(t, e);
            b = t;
            return;
        }
        case Op::While: {
            da(n->kids[0], b);
            Bits t = b;
            da(n->kids[1], t);
            da(n->kids[2], t);
            return;
        }
        case Op::And:
        case Op::Or: {
            da(n->kids[0], b);
            Bits t = b;
            da(n->kids[1], t);
            return;
        }
        case Op::Return:
        case Op::Fail:
        case Op::Coerce:
            for (uint32_t i = 0; i < n->nkids; ++i) da(n->kids[i], b);
            set_all(b);
            return;
        case Op::InlineBlock: {
            Bits ex;
            set_all(ex);
            da_blocks.push_back({n->a, &ex});
            da(n->kids[0], b);
            da_blocks.pop_back();
            band(b, ex);
            return;
        }
        case Op::Break:
            for (size_t i = da_blocks.size(); i-- > 0;)
                if (da_blocks[i].first == n->a) {
                    band(*da_blocks[i].second, b);
                    break;
                }
            set_all(b);
            return;
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) da(n->kids[i], b);
            return;
        }
    }
    std::vector<std::pair<int32_t, Bits*>> da_blocks;
    // Upward-exposed reads of one loop iteration (as da, over canonical ids):
    // expo_out_ gets every local read before being assigned in the iteration.
    Bits expo_out_;
    bool getenv_coarse_loops_ = false;
    std::vector<std::pair<int32_t, Bits*>> ex_blocks;
    void loop_exposed(const Node* w) {
        for (auto& x : expo_out_.w) x = 0;
        Bits b{};
        ex(w->kids[0], b);
        ex(w->kids[1], b);
        ex(w->kids[2], b);
    }
    void ex(const Node* n, Bits& b) {
        switch (n->op) {
        case Op::Local: {
            uint32_t id = canon((uint32_t)n->a);
            if (!(b[id >> 6] >> (id & 63) & 1)) expo_out_[id >> 6] |= 1ull << (id & 63);
            return;
        }
        case Op::SetLocal: {
            ex(n->kids[0], b);
            uint32_t id = canon((uint32_t)n->a);
            b[id >> 6] |= 1ull << (id & 63);
            return;
        }
        case Op::If: {
            ex(n->kids[0], b);
            Bits t = b, e = b;
            ex(n->kids[1], t);
            ex(n->kids[2], e);
            band(t, e);
            b = t;
            return;
        }
        case Op::While: {
            ex(n->kids[0], b);
            Bits t = b;
            ex(n->kids[1], t);
            ex(n->kids[2], t);
            return;
        }
        case Op::And:
        case Op::Or: {
            ex(n->kids[0], b);
            Bits t = b;
            ex(n->kids[1], t);
            return;
        }
        case Op::Return:
        case Op::Fail:
        case Op::Coerce:
            for (uint32_t i = 0; i < n->nkids; ++i) ex(n->kids[i], b);
            set_all(b);
            return;
        case Op::InlineBlock: {
            Bits exb;
            set_all(exb);
            ex_blocks.push_back({n->a, &exb});
            ex(n->kids[0], b);
            ex_blocks.pop_back();
            band(b, exb);
            return;
        }
        case Op::Break:
            for (size_t i = ex_blocks.size(); i-- > 0;)
                if (ex_blocks[i].first == n->a) {
                    band(*ex_blocks[i].second, b);
                    break;
                }
            set_all(b);
            return;
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) ex(n->kids[i], b);
            return;
        }
    }

    bool no_const_reg_ = false;
    bool want_const_reg_ = false;  // the parent would materialise this constant with a mov
    static bool imm12ish(int64_t v) {
        int32_t x = (int32_t)v;
        int64_t a = x < 0 ? -(int64_t)x : x;
        return a < 4096 || ((a & 0xfff) == 0 && a < (1 << 24));
    }
    static bool mul_cheap(int64_t c) {  // Assembler::mul_imm without a constant load
        uint32_t u = (uint32_t)c, nu = 0u - u;
        auto p2 = [](uint32_t x) { return x && !(x & (x - 1)); };
        if (u == 0 || u == 1 || nu == 1 || p2(u) || p2(nu) || p2(u - 1) || p2(nu + 1) || p2(u + 1) || p2(nu - 1)) return true;
        uint32_t odd = u >> __builtin_ctz(u);
        return p2(odd - 1) || p2(odd + 1);
    }
    // kid i of n is a constant the code for n would load with a mov
    bool const_wants_reg(const Node* n, uint32_t i) {
        const Node* k = n->kids[i];
        if (k->op != Op::ConstInt) return false;
        switch (n->op) {
        case Op::AddI32: case Op::SubI32: return i == 1 && !imm12ish(k->a);
        case Op::MulI32: return !mul_cheap(k->a);
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
            if (k->a == 0) return false;
            return !imm12ish(k->a);
        case Op::CheckIndex: return !imm12ish(k->a);
        case Op::Append: return i == 1 && k->a != 0;  // stored value: str needs a register
        case Op::ElemStore: case Op::SetIndex: return i == 2 && k->a != 0;
        default: return false;
        }
    }
    // constant pseudo-locals: (value, outermost loop) -> id; loaded at that loop's entry
    struct ConstId {
        int64_t v;
        const Node* loop;
        uint32_t id;
    };
    std::vector<ConstId> const_ids;
    const Node* outer_loop_ = nullptr;
    uint32_t const_local(int64_t v, const Node* node) {
        for (auto& c : const_ids)
            if (c.v == v && c.loop == outer_loop_) return c.id;
        LocalInfo l;
        l.is_const = true;
        l.cval = v;
        L.push_back(l);
        uint32_t id = (uint32_t)L.size() - 1;
        const_ids.push_back({v, outer_loop_, id});
        const_nodes_.push_back({outer_loop_, {node, id}});
        return id;
    }
    std::vector<std::pair<const Node*, std::pair<const Node*, uint32_t>>> const_nodes_;
    // Result temporaries of inlined bodies: x := Comma[k.., Local t] where every
    // ref of t lies inside the Comma S (and t is written before it is read on
    // every path through S), x is not written in S, every ref of x in S precedes
    // the first ref of t, no loop in S holds refs of both, and no Break leaves S:
    // then t can live in x's location (x is dead from t's first ref until the
    // copy, t is dead after it), so the copy disappears.
    std::vector<const Node*> copies_;
    bool getenv_no_coalesce_ = false;
    bool break_escapes(const Node* n, std::vector<int32_t>& inner) {
        if (n->op == Op::Break) {
            for (int32_t l : inner)
                if (l == n->a) return false;
            return true;
        }
        if (n->op == Op::InlineBlock) inner.push_back(n->a);
        bool r = false;
        for (uint32_t i = 0; i < n->nkids && !r; ++i) r = break_escapes(n->kids[i], inner);
        if (n->op == Op::InlineBlock) inner.pop_back();
        return r;
    }
    void coalesce() {
        std::vector<int32_t> inner;
        for (size_t ci = copies_.size(); ci-- > 0;) {  // outermost (later in post-order) first
            const Node* C = copies_[ci];
            const Node* S = C->kids[0];
            uint32_t x = (uint32_t)C->a, t = (uint32_t)S->kids[S->nkids - 1]->a;
            while (L[x].alias >= 0) x = (uint32_t)L[x].alias;
            if (x == t || L[t].alias >= 0) continue;
            LocalInfo &lx = L[x], &lt = L[t];
            if (lx.fn || lt.fn || lt.param || lt.needs_init || lt.is_const || lx.is_const) continue;
            if (lx.x64 != lt.x64 || lx.any != lt.any || lt.refs.empty()) continue;
            uint32_t en = I(S).en, lv = I(S).lv;
            uint32_t tfirst = UINT32_MAX;
            bool ok = true;
            for (auto& r : lt.refs) {
                if (r.pos < en || r.pos > lv) ok = false;
                tfirst = std::min(tfirst, r.pos);
            }
            if (!ok) continue;
            for (auto& r : lx.refs)
                if (r.pos >= en && r.pos <= lv && (r.write || r.pos >= tfirst)) ok = false;
            if (!ok) continue;
            for (auto& lp : loops) {
                if (lp.en < en || lp.lv > lv) continue;
                bool hx = false, ht = false;
                for (auto& r : lx.refs) hx |= r.pos >= lp.en && r.pos <= lp.lv;
                for (auto& r : lt.refs) ht |= r.pos >= lp.en && r.pos <= lp.lv;
                if (hx && ht) ok = false;
            }
            if (!ok) continue;
            inner.clear();
            if (break_escapes(S, inner)) continue;
            // t definitely assigned before every read inside S
            Bits b;
            set_all(b);
            b[t >> 6] &= ~(1ull << (t & 63));
            da(S, b);
            if (lt.needs_init) {
                lt.needs_init = false;
                continue;
            }
            lt.alias = (int)x;
            lx.weight += lt.weight;
            lx.refs.insert(lx.refs.end(), lt.refs.begin(), lt.refs.end());
            std::sort(lx.refs.begin(), lx.refs.end(), [](const Ref& a, const Ref& b) { return a.pos < b.pos; });
            lt.refs.clear();
            lt.weight = 0;
        }
    }
    uint32_t canon(uint32_t id) {
        while (L[id].alias >= 0) id = (uint32_t)L[id].alias;
        return id;
    }
    // ---- loop-invariant arithmetic (hoisted to the loop entry) -------------
    // A subtree of a loop made of AddI32/SubI32/MulI32/NegI32 (at least two
    // operations) over constants and Int locals that nothing in the loop writes
    // cannot fail and has the same value in every iteration: it is computed
    // once before the loop (first test included) into a pseudo-local.
    bool no_hoist_ = false;
    std::vector<std::pair<const Node*, uint32_t>> hoist_pending_;
    std::unordered_map<const Node*, uint32_t> hoist_mark_;
    std::unordered_map<const Node*, std::vector<std::pair<const Node*, uint32_t>>> hoists_;
    bool hoist_eval_ = false;
    std::vector<uint8_t> written_;
    void mark_written(const Node* n) {
        if (n->op == Op::SetLocal) written_[(size_t)n->a] = 1;
        for (uint32_t i = 0; i < n->nkids; ++i) mark_written(n->kids[i]);
    }
    // number of operations if n is invariant arithmetic, else -1
    int inv_ops(const Node* n) {
        switch (n->op) {
        case Op::ConstInt: return 0;
        case Op::Local: {
            size_t id = (size_t)n->a;
            if (id >= F->locals.size() || written_[id] || F->locals[id].type.k != TK::Int) return -1;
            return 0;
        }
        case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::NegI32: {
            int c = 1;
            for (uint32_t i = 0; i < n->nkids; ++i) {
                int k = inv_ops(n->kids[i]);
                if (k < 0) return -1;
                c += k;
            }
            return c;
        }
        default: return -1;
        }
    }
    void hoist_walk(const Node* n) {
        if (hoist_mark_.count(n)) return;
        if (n->op == Op::AddI32 || n->op == Op::SubI32 || n->op == Op::MulI32 || n->op == Op::NegI32) {
            int c = inv_ops(n);
            if (c >= 2) {
                LocalInfo l;
                L.push_back(l);
                uint32_t id = (uint32_t)L.size() - 1;
                hoist_mark_[n] = id;
                hoist_pending_.push_back({n, id});
                return;
            }
        }
        if (n->op == Op::While) return;  // an inner loop decides for itself (it runs per entry)
        for (uint32_t i = 0; i < n->nkids; ++i) hoist_walk(n->kids[i]);
    }
    void find_hoists(const Node* w) {
        written_.assign(F->locals.size(), 0);
        mark_written(w);
        for (uint32_t i = 0; i < w->nkids; ++i) hoist_walk(w->kids[i]);
    }
    void analyze() {
        info.reset(count_nodes(F->body));
        hoist_pending_.clear();
        hoist_mark_.clear();
        hoists_.clear();
        const_ids.clear();
        const_nodes_.clear();
        outer_loop_ = nullptr;
        copies_.clear();
        loops.clear();
        call_pos.clear();
        pos_ = 0;
        ctx_ = 0;
        ctx_next_ = 0;
        ctx_stack_.clear();
        out_bytes = 0;
        L.clear();
        L.reserve(F->locals.size() + 32);
        L.resize(F->locals.size());
        for (size_t i = 0; i < F->locals.size(); ++i) {
            Type t = F->locals[i].type;
            L[i].fn = t.k == TK::Fn;
            L[i].x64 = is64(t);
            L[i].any = t.k == TK::Any;
        }
        for (size_t i = 0; i < F->params.size(); ++i) {
            L[F->params[i]].param = true;
            L[F->params[i]].param_idx = (int)i;
        }
        pos_ = 1;  // 0 = function entry
        prepass(F->body, 0);
        std::sort(call_pos.begin(), call_pos.end());
        hot_calls = !call_pos.empty();
        if (F->locals.size() <= 64 * kDaWords) {
            Bits b{};
            for (uint32_t p : F->params) b[p >> 6] |= 1ull << (p & 63);
            da(F->body, b);
        } else {
            for (size_t i = 0; i < F->locals.size(); ++i) L[i].needs_init = true;
        }
        if (F->locals.size() <= 64 * kDaWords && !getenv_no_coalesce_) coalesce();
        // intervals
        for (auto& l : L) {
            if (l.fn || (l.refs.empty() && !l.param)) continue;
            for (auto& r : l.refs) {
                l.lo = std::min(l.lo, r.pos);
                l.hi = std::max(l.hi, r.pos);
            }
            if (l.param || l.needs_init) l.lo = 0;
            if (l.refs.empty()) l.hi = 0;
        }
        bool precise = F->locals.size() <= 64 * kDaWords && !getenv_coarse_loops_;
        for (auto& lp : loops) {
            if (precise) loop_exposed(lp.node);
            for (size_t li = 0; li < L.size(); ++li) {
                LocalInfo& l = L[li];
                if (l.fn || l.refs.empty()) continue;
                // refs inside the loop?
                const Ref* first = nullptr;
                bool outside = l.lo < lp.en;  // includes entry-defined values
                for (auto& r : l.refs) {
                    if (r.pos >= lp.en && r.pos <= lp.lv) {
                        if (!first) first = &r;
                    } else {
                        outside = true;
                    }
                }
                if (!first) continue;
                // Does the loop read the value an earlier iteration (or the code before
                // the loop) left behind? Precisely: a read of the local reachable from
                // the loop top without an assignment in between (upward exposed).
                bool uncond_def;
                if (precise && li < F->locals.size()) uncond_def = !(expo_out_[li >> 6] >> (li & 63) & 1);
                else uncond_def = first->write && (first->ctx == lp.body_ctx || first->ctx == lp.cond_ctx);
                if (outside || !uncond_def) {
                    l.lo = std::min(l.lo, lp.en);
                    l.hi = std::max(l.hi, lp.lv);
                }
            }
        }
        for (auto& l : L) {
            if (l.lo == UINT32_MAX) continue;
            auto it = std::upper_bound(call_pos.begin(), call_pos.end(), l.lo);
            l.crosses_call = it != call_pos.end() && *it < l.hi;
        }
        allocate();
    }

    void allocate() {
        std::vector<uint32_t> order;
        for (uint32_t i = 0; i < L.size(); ++i)
            if (L[i].lo != UINT32_MAX) order.push_back(i);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (L[a].weight != L[b].weight) return L[a].weight > L[b].weight;
            return L[a].lo < L[b].lo;
        });
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> busy(32);
        auto fits = [&](int r, uint32_t lo, uint32_t hi) {
            for (auto& iv : busy[(size_t)r])
                if (!(hi < iv.first || lo > iv.second)) return false;
            return true;
        };
        uint32_t callee_used = 0;
        caller_locals = 0;
        nfixed = 0;
        for (uint32_t id : order) {
            LocalInfo& l = L[id];
            int pick = -1;
            if (!l.crosses_call) {
                for (int r = 9; r <= 15 && pick < 0; ++r)
                    if (fits(r, l.lo, l.hi)) pick = r;
            }
            // a constant is rematerialised (mov) rather than kept in a callee-saved
            // register across calls (that register costs a save/restore and is
            // better used for a variable)
            if (l.is_const && pick < 0 && l.crosses_call) continue;
            // prefer callee-saved registers already in use
            for (int r = 19; r <= 28 && pick < 0; ++r)
                if ((callee_used & bit(r)) && fits(r, l.lo, l.hi)) pick = r;
            for (int r = 19; r <= 28 && pick < 0; ++r)
                if (fits(r, l.lo, l.hi)) pick = r;
            if (pick >= 0) {
                l.reg = pick;
                busy[(size_t)pick].push_back({l.lo, l.hi});
                if (pick >= 19) callee_used |= bit(pick);
                else caller_locals |= bit(pick);
            } else if (!l.is_const) {
                l.slot = nfixed++;
            }
        }
        reg_writes.assign(32, {});
        for (auto& l : L) {
            if (l.reg < 0) continue;
            for (auto& r : l.refs)
                if (r.write) reg_writes[(size_t)l.reg].push_back(r.pos);
        }
        for (auto& w : reg_writes) std::sort(w.begin(), w.end());
        temp_pool = kBaseTemps | (kCallerLocal & ~caller_locals);
        spill_pool = kCallee & ~callee_used;
        used_callee = callee_used;
    }

    // ------------------------------------------------------------------ locals
    V read_local(uint32_t id) {
        id = canon(id);
        LocalInfo& l = L[id];
        if (l.fn) return V::none();
        if (early_) return V::reg(l.param_idx, false);
        if (l.reg >= 0) return V::reg(l.reg, false);
        if (l.slot >= 0) {
            int r = talloc();
            A->ldr(l.x64 ? X(r) : W(r), mem(sp, fix_off(l.slot)));
            return V::reg(r, true);
        }
        // never referenced elsewhere (cannot happen: this is a reference)
        return V::im(l.any ? 1 : 0);
    }

    // ------------------------------------------------------------------ calls
    struct Move {
        int dst;  // register, or -1-k for outgoing stack slot k
        V src;
        bool x64;
    };
    // Emits the moves and releases every source value.
    std::vector<V> pm_srcs_;
    std::vector<Move*> pm_pend_;
    std::vector<Move> hc_moves_;
    void parallel_move(std::vector<Move>& mv) {
        pm_srcs_.clear();
        for (auto& m : mv) pm_srcs_.push_back(m.src);
        parallel_move_(mv);
        for (auto& v : pm_srcs_) release(v);
    }
    void parallel_move_(std::vector<Move>& mv) {
        // stack destinations first (they read sources, write memory)
        for (auto& m : mv) {
            if (m.dst >= 0) continue;
            uint32_t off = 8u * (uint32_t)(-1 - m.dst);
            switch (m.src.k) {
            case V::REG: A->str(X(m.src.r), mem(sp, off)); break;
            case V::IMM:
                li(X(16), m.src.imm);
                A->str(X(16), mem(sp, off));
                break;
            case V::SLOT:
                A->ldr(X(16), mem(sp, dyn_off(m.src.slot)));
                A->str(X(16), mem(sp, off));
                break;
            default: break;
            }
        }
        // register-to-register moves
        std::vector<Move*>& pend = pm_pend_;
        pend.clear();
        for (auto& m : mv)
            if (m.dst >= 0 && m.src.k == V::REG && m.src.r != m.dst) pend.push_back(&m);
        while (!pend.empty()) {
            bool progress = false;
            for (size_t i = 0; i < pend.size(); ++i) {
                int d = pend[i]->dst;
                bool used = false;
                for (size_t j = 0; j < pend.size(); ++j)
                    if (j != i && pend[j]->src.r == d) used = true;
                if (!used) {
                    A->mov(X(d), X(pend[i]->src.r));
                    pend.erase(pend.begin() + (long)i);
                    progress = true;
                    break;
                }
            }
            if (!progress) {
                int d = pend[0]->dst;
                A->mov(X(16), X(d));
                for (auto* p : pend)
                    if (p->src.r == d) p->src.r = 16;
            }
        }
        for (auto& m : mv) {
            if (m.dst < 0) continue;
            if (m.src.k == V::IMM) {
                if (m.x64) li(X(m.dst), m.src.imm);
                else li(W(m.dst), (int32_t)m.src.imm);
            } else if (m.src.k == V::SLOT) {
                A->ldr(X(m.dst), mem(sp, dyn_off(m.src.slot)));
            }
        }
    }
    // Before a hot call: no temporary may be live.
    void check_no_temps(const char* where) {
        if ((temp_free & temp_pool) != temp_pool) {
            std::fprintf(stderr, "epq: jit: live temporaries at call (%s): %x\n", where, temp_pool & ~temp_free);
            bug("live temporaries across a call");
        }
    }
    // Call a helper with arguments (released), result register x0 (allocated) or none.
    V helper_call(int h, std::initializer_list<Move> args, bool has_result) {
        hc_moves_.assign(args.begin(), args.end());
        parallel_move(hc_moves_);
        check_no_temps(helper_name(h));
        call_helper(h);
        if (!has_result) return V::none();
        return V::reg(talloc_specific(0), true);
    }

    // Registers a cold path must preserve around a call.
    uint32_t live_caller_saved(uint32_t except) {
        uint32_t m = ((temp_pool & ~temp_free) | caller_locals) & ~except;
        return m;
    }
    void save_regs(uint32_t m) {
        m |= bit(30);
        int r[32];
        size_t n = 0;
        for (int i = 0; i < 31; ++i)
            if (m & bit(i)) r[n++] = i;
        for (size_t i = 0; i < n; i += 2) {
            if (i + 1 < n) A->stp(X(r[i]), X(r[i + 1]), pre(sp, -16));
            else A->str(X(r[i]), pre(sp, -16));
        }
    }
    void restore_regs(uint32_t m) {
        m |= bit(30);
        int r[32];
        size_t n = 0;
        for (int i = 0; i < 31; ++i)
            if (m & bit(i)) r[n++] = i;
        size_t last = (n - 1) & ~(size_t)1;
        for (size_t i = last + 2; i-- > 0;) {
            if (i & 1) continue;
            if (i >= n) continue;
            if (i + 1 < n) A->ldp(X(r[i]), X(r[i + 1]), post(sp, 16));
            else A->ldr(X(r[i]), post(sp, 16));
        }
    }
    void cold(Label l, std::function<void()> f) { colds.push_back(ColdPath{l, std::move(f)}); }

    // ------------------------------------------------------------------ expressions
    // Evaluate kids [from, to) of n, holding earlier values while later ones run.
    void eval_kids(const Node* n, uint32_t from, uint32_t to, V* out) {
        for (uint32_t i = from; i < to; ++i) {
            out[i - from] = expr(n->kids[i]);
            if (i + 1 < to) hold(out[i - from], n, i + 1, to - 1);
        }
    }
    // hold all earlier values over kid range — eval_kids holds each value once
    // over every later kid, which covers all of them.

    void discard(const Node* n) {
        if (tir::is_pure(n)) return;
        stmt(n);
    }

    V expr(const Node* n, int want = -1) {
        switch (n->op) {
        case Op::ConstInt: {
            if (!const_ids.empty() && !early_ && !hoist_eval_ && I(n).cref) {
                int r = L[(size_t)I(n).hoist].reg;
                if (r >= 0) return V::reg(r, false);
            }
            return V::im(n->a);
        }
        case Op::ConstFn: return V::none();
        case Op::Local: return read_local((uint32_t)n->a);
        case Op::GetGlobal: return get_global(n, want);
        case Op::Input: {
            check_no_temps("input");
            call_helper(H_READ_INT);
            return V::reg(talloc_specific(0), true);
        }
        case Op::SetLocal: return set_local(n);
        case Op::SetGlobal: return set_global(n);
        case Op::Box: return box(n, want);
        case Op::UnboxInt: {
            V v = expr(n->kids[0]);
            if (v.k == V::IMM && (v.imm & 1)) return V::im((int32_t)((uint64_t)v.imm >> 32));
            Reg s = R(v, true);
            A->tbz(s, 0, err((int)n->a));
            release(v);
            int d = dest(want);
            A->lsr(X(d), s, 32);
            return result(d, want);
        }
        case Op::Coerce: {
            discard_value(n->kids[0]);
            return V::im(0);
        }
        case Op::Comma: {
            for (uint32_t i = 0; i + 1 < n->nkids; ++i) discard(n->kids[i]);
            return expr(n->kids[n->nkids - 1], want);
        }
        case Op::Fail: {
            for (uint32_t i = 0; i < n->nkids; ++i) discard(n->kids[i]);
            A->b(err((int)n->a));
            return V::im(0);
        }
        case Op::AddI32:
        case Op::SubI32:
        case Op::MulI32: return arith(n, want);
        case Op::DivI32: return div(n, want);
        case Op::NegI32: {
            if (!hoist_mark_.empty() && !hoist_eval_ && I(n).hoist >= 0) return read_local((uint32_t)I(n).hoist);
            V v = expr(n->kids[0]);
            if (v.k == V::IMM) return V::im((int32_t)(0u - (uint32_t)v.imm));
            Reg s = R(v, false);
            release(v);
            int d = dest(want);
            A->neg(W(d), s);
            return result(d, want);
        }
        case Op::AbsI32: {
            V v = expr(n->kids[0]);
            if (v.k == V::IMM) {
                int32_t x = (int32_t)v.imm;
                return V::im(x < 0 ? (int32_t)(0u - (uint32_t)x) : x);
            }
            Reg s = R(v, false);
            release(v);
            int d = dest(want);
            A->cmp(s, 0u);
            A->cneg(W(d), s, LT);
            return result(d, want);
        }
        case Op::NotI32: {
            V v = expr(n->kids[0]);
            if (v.k == V::IMM) return V::im((int32_t)v.imm == 0);
            Reg s = R(v, false);
            release(v);
            int d = dest(want);
            A->cmp(s, 0u);
            A->cset(W(d), EQ);
            return result(d, want);
        }
        case Op::EqI32:
        case Op::NeI32:
        case Op::LtI32:
        case Op::LeI32:
        case Op::GtI32:
        case Op::GeI32: {
            V fold;
            Cond c;
            if (cmp_i32(n, &c, &fold)) return fold;
            int d = dest(want);
            A->cset(W(d), c);
            return result(d, want);
        }
        case Op::DynArith: return dyn_arith(n, want);
        case Op::DynNeg:
        case Op::DynAbs: return dyn_neg_abs(n, want);
        case Op::DynNot: {
            V v = expr(n->kids[0]);
            Reg s = R(v, true);
            A->tbz(s, 0, err(Err::OperandInt));
            release(v);
            int d = dest(want);
            A->cmp(s, 1u);
            A->cset(W(d), EQ);
            return result(d, want);
        }
        case Op::Eq:
        case Op::Ne:
        case Op::Cmp: return deep_cmp_value(n, want);
        case Op::ListLit: return list_lit(n);
        case Op::Len: {
            V l = expr(n->kids[0]);
            Reg x = R(l, true);
            release(l);
            int d = dest(want);
            A->ldr(W(d), mem(x, 8));
            return result(d, want);
        }
        case Op::Index: return index(n, want);
        case Op::SetIndex: return set_index(n);
        case Op::Append: return append(n);
        case Op::Pop: return pop(n, want);
        case Op::DynIndex: return dyn_index(n, want);
        case Op::DynSetIndex: return dyn_set_index(n);
        case Op::DynAppend: return dyn_append(n);
        case Op::And:
        case Op::Or: return and_or_value(n);
        case Op::CallDirect: return call_direct(n);
        case Op::CallDyn: return call_dyn(n);
        // statements in expression position
        case Op::Seq:
        case Op::If:
        case Op::While:
        case Op::Return:
        case Op::Print:
        case Op::InlineBlock:
        case Op::Break:
        case Op::Free:
        case Op::Nop: stmt(n); return V::none();
        case Op::Select: return select(n, want);
        case Op::PairMake: return pair_make(n, want);
        case Op::PairFirst: {
            V v = expr(n->kids[0]);
            if (v.k == V::IMM) return V::im((int32_t)((uint64_t)v.imm >> 32));
            Reg s = R(v, true);
            release(v);
            int d = dest(want);
            A->lsr(X(d), s, 32);
            return result(d, want);
        }
        case Op::PairSecond: {
            V v = expr(n->kids[0]);
            if (v.k == V::IMM) return V::im((int32_t)((uint32_t)v.imm ^ 0x80000000u));
            Reg s = R(v, false);
            release(v);
            int d = dest(want);
            A->eor(W(d), s, (uint64_t)0x80000000u);
            return result(d, want);
        }
        case Op::PairIndex: return pair_index(n, want);
        case Op::PairEq:
        case Op::PairNe:
        case Op::PairLt:
        case Op::PairLe:
        case Op::PairGt:
        case Op::PairGe:
        case Op::SameList: {
            Cond c = cmp64(n);
            int d = dest(want);
            A->cset(W(d), c);
            return result(d, want);
        }
        case Op::ListData: {
            V l = expr(n->kids[0]);
            Reg x = R(l, true);
            release(l);
            int d = dest(want);
            A->ldr(X(d), mem(x, 0));
            return result(d, want);
        }
        case Op::ElemLoad: return elem_load(n, want);
        case Op::ElemStore: return elem_store(n);
        case Op::CheckIndex: return check_index(n);
        case Op::RecLit: return rec_lit(n);
        case Op::RecNull: return V::im(0);
        case Op::RecGet: {  // one load at a constant offset (null check first when b)
            V v = expr(n->kids[0]);
            Reg x = R(v, true);
            if (n->b) A->cbz(x, err(Err::IndexOOB));
            release(v);
            if (n->type.k == TK::Fn) return V::none();
            int d = dest(want);
            A->ldr(is64(n->type) ? X(d) : W(d), mem(x, 8 * (int64_t)n->a));
            return result(d, want);
        }
        case Op::RecSet: {  // both kids, then the null check, then the store; value v
            V k[2];
            eval_kids(n, 0, 2, k);
            Type vt = n->kids[1]->type;
            V w = word_of(vt, k[1]);
            Reg x = R(k[0], true);
            if (n->b) A->cbz(x, err(Err::IndexOOB));
            Reg s = (w.k == V::IMM && w.imm == 0) ? xzr : R(w, true);
            A->str(s, mem(x, 8 * (int64_t)n->a));
            release(k[0]);
            if (vt.k == TK::Fn) {
                release(w);
                return V::none();
            }
            return w;  // for an Int, the zero-extended word is the value
        }
        case Op::RecLen: {  // rec ? k : 0
            V v = expr(n->kids[0]);
            uint32_t k = M.regions[n->kids[0]->type.a].rec_len;
            if (v.k == V::IMM) return V::im(v.imm ? k : 0);
            Reg x = R(v, true);
            release(v);
            int d = dest(want);
            A->cmp(x, 0u);
            if (k == 1) {
                A->cset(W(d), NE);
            } else {
                A->movz(W(16), k);
                A->csel(W(d), W(16), wzr, NE);
            }
            return result(d, want);
        }
        case Op::Reserve: {  // appendopt: epq_list_reserve(l, n, esz); value 0
            V k[2];
            eval_kids(n, 0, 2, k);
            std::initializer_list<Move> mv{{0, k[0], true}, {1, k[1], false}, {2, V::im(esz_of(n->kids[0]->type.a)), false}};
            helper_call(H_RESERVE, mv, false);
            return V::im(0);
        }
        case Op::AppendRange: {  // appendopt: epq_append_range(l, m, lo, hi, esz)
            V k[4];
            eval_kids(n, 0, 4, k);
            std::initializer_list<Move> mv{{0, k[0], true}, {1, k[1], true}, {2, k[2], false}, {3, k[3], false},
                                           {4, V::im(esz_of(n->kids[0]->type.a)), false}};
            return helper_call(H_APPEND_RANGE, mv, true);
        }
        case Op::DynIsEmptyList: {
            // (x & 3) == 0 && ((EList*)x)->len == 0  (epq_is_empty_list)
            V v = expr(n->kids[0]);
            if (v.k == V::IMM) return V::im(0);  // a constant word is an int / fn, never a list
            Reg x = R(v, true);
            Label no = A->new_label();
            A->movz(W(16), 1);
            A->tst(x, (uint64_t)3);
            A->b(NE, no);
            A->ldr(W(16), mem(x, 8));
            A->bind(no);
            release(v);
            int d = dest(want);
            A->cmp(W(16), 0u);
            A->cset(W(d), EQ);
            return result(d, want);
        }
        }
        bug("unknown op");
    }

    // Evaluate and drop (expression known to have no useful value).
    void discard_value(const Node* n) {
        V v = expr(n);
        release(v);
    }

    V get_global(const Node* n, int want) {
        uint32_t g = (uint32_t)n->a;
        Type t = M.globals[g].type;
        uint32_t lo;
        if (n->flags & tir::F_CHECKED) {
            data_addr(X(16), gdef_off[g], &lo);
            A->ldrb(W(16), mem(X(16), lo));
            A->cbz(W(16), err(Err::UndefGlobal));
        }
        if (t.k == TK::Fn) return V::none();
        int d = dest(want);
        data_addr(X(16), gval_off[g], &lo);
        A->ldr(is64(t) ? X(d) : W(d), mem(X(16), lo));
        return result(d, want);
    }

    V set_local(const Node* n) {
        uint32_t id = canon((uint32_t)n->a);
        LocalInfo& l = L[id];
        if (l.fn) {
            discard_value(n->kids[0]);
            return V::none();
        }
        if (l.reg >= 0) {
            V v = expr(n->kids[0], l.reg);
            if (!(v.k == V::REG && v.r == l.reg)) {
                if (v.k == V::IMM) {
                    if (l.x64) li(X(l.reg), v.imm);
                    else li(W(l.reg), (int32_t)v.imm);
                } else if (v.k == V::NONE) {
                    A->movz(W(l.reg), 0);
                } else {
                    Reg s = R(v, l.x64);
                    A->mov(l.x64 ? X(l.reg) : W(l.reg), s);
                }
                release(v);
            }
            return V::reg(l.reg, false);
        }
        V v = expr(n->kids[0]);
        if (l.slot < 0) {  // unreferenced local (only written): value only
            return v;
        }
        Reg s = R(v, l.x64);
        A->str(s, mem(sp, fix_off(l.slot)));
        return v;
    }

    void store_local(uint32_t id, V v) {
        LocalInfo& l = L[id];
        if (l.reg >= 0) {
            if (v.k == V::IMM) li(l.x64 ? X(l.reg) : W(l.reg), l.x64 ? v.imm : (int32_t)v.imm);
            else if (!(v.k == V::REG && v.r == l.reg)) A->mov(l.x64 ? X(l.reg) : W(l.reg), R(v, l.x64));
        } else if (l.slot >= 0) {
            A->str(R(v, l.x64), mem(sp, fix_off(l.slot)));
        }
        release(v);
    }
    V set_global(const Node* n) {
        uint32_t g = (uint32_t)n->a;
        Type t = M.globals[g].type;
        V v = expr(n->kids[0]);
        uint32_t lo;
        if (t.k != TK::Fn) {
            Reg s = R(v, is64(t));
            data_addr(X(16), gval_off[g], &lo);
            A->str(s, mem(X(16), lo));
        }
        if (M.globals[g].ever_checked) {
            data_addr(X(16), gdef_off[g], &lo);
            A->movz(W(17), 1);
            A->strb(W(17), mem(X(16), lo));
        }
        return v;
    }

    V box(const Node* n, int want) {
        const Node* k = n->kids[0];
        switch (k->type.k) {
        case TK::Int: {
            V v = expr(k);
            if (v.k == V::IMM) return V::im((int64_t)(((uint64_t)(uint32_t)v.imm << 32) | 1));
            Reg s = R(v, true);
            release(v);
            int d = dest(want);
            A->lsl(X(d), s, 32);
            A->orr(X(d), X(d), (uint64_t)1);
            return result(d, want);
        }
        case TK::Fn: {
            discard_value(k);
            return V::im((int64_t)(((uint64_t)k->type.a << 32) | 2));
        }
        case TK::Bot: discard_value(k); return V::im(1);
        default: return expr(k, want);
        }
    }

    // ---- integer arithmetic
    static int32_t wrap(int64_t v) { return (int32_t)(uint32_t)(uint64_t)v; }
    // x - (x / c) * c (the inlined `mod`) with x a local and c a non-zero constant
    static bool is_srem(const Node* n, const Node** x, int32_t* c) {
        if (n->op != Op::SubI32) return false;
        const Node* a = n->kids[0];
        const Node* m = n->kids[1];
        if (a->op != Op::Local || m->op != Op::MulI32) return false;
        const Node* d = m->kids[0];
        const Node* k = m->kids[1];
        if (d->op != Op::DivI32) std::swap(d, k);
        if (d->op != Op::DivI32 || k->op != Op::ConstInt) return false;
        if (d->kids[0]->op != Op::Local || d->kids[0]->a != a->a) return false;
        if (d->kids[1]->op != Op::ConstInt || d->kids[1]->a != k->a || (int32_t)k->a == 0) return false;
        *x = a;
        *c = (int32_t)k->a;
        return true;
    }
    V arith(const Node* n, int want) {
        if (!hoist_mark_.empty() && !hoist_eval_ && I(n).hoist >= 0) return read_local((uint32_t)I(n).hoist);
        {
            const Node* x;
            int32_t c;
            if (is_srem(n, &x, &c)) {
                V v = expr(x);
                if (v.k == V::IMM) {
                    int32_t a = (int32_t)v.imm;
                    return V::im(c == -1 ? 0 : a % c);
                }
                Reg r = R(v, false);
                release(v);
                int d = dest(want);
                A->srem_imm(W(d), r, c);
                return result(d, want);
            }
        }
        V k[2];
        eval_kids(n, 0, 2, k);
        Op op = n->op;
        if (k[0].k == V::IMM && k[1].k == V::IMM) {
            uint32_t a = (uint32_t)k[0].imm, b = (uint32_t)k[1].imm;
            return V::im((int32_t)(op == Op::AddI32 ? a + b : op == Op::SubI32 ? a - b : a * b));
        }
        if (op == Op::AddI32 || op == Op::MulI32) {
            if (k[0].k == V::IMM) std::swap(k[0], k[1]);
        }
        if (k[1].k == V::IMM) {
            int32_t c = (int32_t)k[1].imm;
            Reg a = R(k[0], false);
            release(k[0]);
            int d = dest(want);
            if (op == Op::AddI32) A->add_imm(W(d), a, c);
            else if (op == Op::SubI32) A->add_imm(W(d), a, (int32_t)(0u - (uint32_t)c));
            else A->mul_imm(W(d), a, c);
            return result(d, want);
        }
        if (k[0].k == V::IMM && op == Op::SubI32 && k[0].imm == 0) {
            Reg b = R(k[1], false);
            release(k[1]);
            int d = dest(want);
            A->neg(W(d), b);
            return result(d, want);
        }
        Reg a = R(k[0], false);
        Reg b = R(k[1], false);
        release(k[0]);
        release(k[1]);
        int d = dest(want);
        if (op == Op::AddI32) A->add(W(d), a, b);
        else if (op == Op::SubI32) A->sub(W(d), a, b);
        else A->mul(W(d), a, b);
        return result(d, want);
    }

    V div(const Node* n, int want) {
        V k[2];
        eval_kids(n, 0, 2, k);
        if (k[1].k == V::IMM) {
            int32_t c = (int32_t)k[1].imm;
            if (c == 0) {
                release(k[0]);
                A->b(err(Err::DivZero));
                return V::im(0);
            }
            if (k[0].k == V::IMM) {
                int32_t a = (int32_t)k[0].imm;
                if (c == -1) return V::im((int32_t)(0u - (uint32_t)a));
                return V::im(a / c);
            }
            Reg a = R(k[0], false);
            release(k[0]);
            int d = dest(want);
            if (c == 1) A->mov(W(d), a);
            else if (c == -1) A->neg(W(d), a);
            else A->sdiv_imm(W(d), a, c);
            return result(d, want);
        }
        Reg a = R(k[0], false);
        Reg b = R(k[1], false);
        A->cbz(b, err(Err::DivZero));
        release(k[0]);
        release(k[1]);
        int d = dest(want);
        A->sdiv(W(d), a, b);
        return result(d, want);
    }

    // Int comparison: sets flags, returns condition. Folds constants (then *fold
    // gets the value and the function returns true).
    bool cmp_i32(const Node* n, Cond* c, V* fold) {
        V k[2];
        eval_kids(n, 0, 2, k);
        Cond cc = cond_of(n->op);
        if (k[0].k == V::IMM && k[1].k == V::IMM) {
            int32_t a = (int32_t)k[0].imm, b = (int32_t)k[1].imm;
            bool r = cc == EQ ? a == b : cc == NE ? a != b : cc == LT ? a < b : cc == LE ? a <= b : cc == GT ? a > b : a >= b;
            *fold = V::im(r);
            return true;
        }
        if (k[0].k == V::IMM) {
            std::swap(k[0], k[1]);
            cc = swap_cond(cc);
        }
        Reg a = R(k[0], false);
        if (k[1].k == V::IMM) {
            A->cmp_imm(a, (int32_t)k[1].imm);
        } else {
            Reg b = R(k[1], false);
            A->cmp(a, b);
        }
        release(k[0]);
        release(k[1]);
        *c = cc;
        return false;
    }

    V dyn_arith(const Node* n, int want) {
        const Node* x = n->kids[0];
        const Node* y = n->kids[1];
        V k[2];
        eval_kids(n, 0, 2, k);
        Op op = (Op)n->a;
        TK tx = x->type.k, ty = y->type.k;
        if (op == Op::DivI32) {
            if (ty == TK::Int) {
                if (k[1].k == V::IMM) {
                    if ((int32_t)k[1].imm == 0) A->b(err(Err::DivZero));
                } else {
                    A->cbz(R(k[1], false), err(Err::DivZero));
                }
            } else if (ty == TK::Any) {
                if (k[1].k == V::IMM) {
                    if (k[1].imm == 1) A->b(err(Err::DivZero));
                } else {
                    Reg b = R(k[1], true);
                    A->cmp(b, 1u);
                    A->b(EQ, err(Err::DivZero));
                }
            }
        }
        bool xs = tx == TK::List || tx == TK::Fn || tx == TK::Pair || tx == TK::Rec,
             ys = ty == TK::List || ty == TK::Fn || ty == TK::Pair || ty == TK::Rec;
        if (xs || ys) {
            release(k[0]);
            release(k[1]);
            A->b(err(Err::OperandsInt));
            return V::im(0);
        }
        bool xa = tx == TK::Any, ya = ty == TK::Any;
        // integer operand values in w registers
        Reg ra = xa ? R(k[0], true) : R(k[0], false);
        Reg rb = ya ? R(k[1], true) : R(k[1], false);
        if (xa && ya) {
            A->and_(X(16), ra, rb);
            A->tbz(X(16), 0, err(Err::OperandsInt));
        } else if (xa) {
            A->tbz(ra, 0, err(Err::OperandsInt));
        } else if (ya) {
            A->tbz(rb, 0, err(Err::OperandsInt));
        }
        int d;
        if (xa && ya && (op == Op::AddI32 || op == Op::SubI32)) {
            release(k[0]);
            release(k[1]);
            d = dest(want);
            if (op == Op::AddI32) A->add(X(16), ra, rb);
            else A->sub(X(16), ra, rb);
            A->lsr(X(d), X(16), 32);
            return result(d, want);
        }
        Reg ia = ra, ib = rb;
        if (xa) {
            A->lsr(X(16), ra, 32);
            ia = W(16);
        }
        if (ya) {
            A->lsr(X(17), rb, 32);
            ib = W(17);
        }
        release(k[0]);
        release(k[1]);
        d = dest(want);
        switch (op) {
        case Op::AddI32: A->add(W(d), ia, ib); break;
        case Op::SubI32: A->sub(W(d), ia, ib); break;
        case Op::MulI32: A->mul(W(d), ia, ib); break;
        default: A->sdiv(W(d), ia, ib); break;  // divisor known non-zero; INT_MIN/-1 = INT_MIN
        }
        return result(d, want);
    }

    // Word (EAny-sized) of a value of static type t, for descriptor helpers.
    // Returns a V that is an IMM or a register value with the 64-bit word.
    V word_of(Type t, V v) {
        switch (t.k) {
        case TK::Int:
        case TK::Bot:
            if (v.k == V::IMM) return V::im((int64_t)(uint32_t)v.imm);
            {
                Reg s = R(v, false);
                int d = talloc();
                A->mov(W(d), s);  // zero-extend
                release(v);
                return V::reg(d, true);
            }
        case TK::Fn: release(v); return V::im((int64_t)(((uint64_t)t.a << 32) | 2));
        default: return v;
        }
    }
    V boxed_of(Type t, V v) {  // Int|Any operand -> Any word
        if (t.k == TK::Any) return v;
        if (v.k == V::IMM) return V::im((int64_t)(((uint64_t)(uint32_t)v.imm << 32) | 1));
        Reg s = R(v, false);
        int d = talloc();
        A->lsl(X(d), s.xr(), 32);
        A->orr(X(d), X(d), (uint64_t)1);
        release(v);
        return V::reg(d, true);
    }

    // ---- Eq / Ne / Cmp
    struct DeepCmp {
        bool fast;
        Cond c;           // fast path: condition on the flags
        Label slow, cont; // fast path: slow path entry, continuation
        V a, b;           // fast path: boxed words (kept for the slow path)
    };
    // Helper call computing the boolean result of n (descriptor driven) into w0.
    // Arguments are the words (registers / imms) a, b of the two kids.
    void deep_call(const Node* n, Reg a, Reg b, bool fast_words) {
        // fast_words: a/b are boxed Any words and descriptors are ANY (1).
        uint32_t da = fast_words ? 1 : (uint32_t)n->a, db = fast_words ? 1 : (uint32_t)n->b;
        Op c = n->op == Op::Cmp ? (Op)n->c : n->op;
        bool swap = c == Op::GtI32 || c == Op::LeI32;
        bool neg = n->op == Op::Ne || c == Op::LeI32 || c == Op::GeI32;
        int h = (n->op == Op::Eq || n->op == Op::Ne) ? H_EQ : H_LT;
        // x0 = da, x1 = a, x2 = db, x3 = b  (swapped for > and <=)
        Reg p = swap ? b : a, q = swap ? a : b;
        uint32_t dp = swap ? db : da, dq = swap ? da : db;
        // move p -> x1, q -> x3 safely
        if (q.id == 1) {
            A->mov(X(16), q.xr());
            q = X(16);
        }
        A->mov(X(1), p.xr());
        A->mov(X(3), q.xr());
        A->movz(W(0), dp);
        A->movz(W(2), dq);
        call_helper(h);
        if (neg) A->eor(W(0), W(0), (uint64_t)1);
    }
    bool li_fast(const Node* n) {
        Type ta = n->kids[0]->type, tb = n->kids[1]->type;
        return ta.k == TK::List && tb.k == TK::List && M.elem_is_i32(ta.a) && M.elem_is_i32(tb.a);
    }
    bool rec_rec(const Node* n) {
        Type ta = n->kids[0]->type, tb = n->kids[1]->type;
        return ta.k == TK::Rec && tb.k == TK::Rec && (size_t)n->a < M.descs.size() && (size_t)n->b < M.descs.size() &&
               M.descs[n->a].kind == 5 && M.descs[n->b].kind == 5;
    }
    // ---- typed record comparison functions (out of line, emitted after the
    // program's functions): x0 = a, x1 = b -> w0 (eq: 0/1; three-way: -1/0/1).
    struct RecStub {
        bool eq;
        Type a, b;
        uint32_t da, db;
        Label l;
    };
    std::vector<RecStub> rec_stubs_;
    Label rec_stub(bool eq, Type a, Type b, uint32_t da, uint32_t db) {
        if (dry) return dummy;
        for (auto& s : rec_stubs_)
            if (s.eq == eq && s.a == a && s.b == b && s.da == da && s.db == db) return s.l;
        rec_stubs_.push_back(RecStub{eq, a, b, da, db, main_as.new_label()});
        return rec_stubs_.back().l;
    }
    void emit_rec_stubs() {
        for (size_t i = 0; i < rec_stubs_.size(); ++i) {
            RecStub st = rec_stubs_[i];
            A->align(8);
            A->bind(st.l);
            syms.push_back({A->pos(), std::string(st.eq ? "rec_eq" : "rec_cmp3") + std::to_string(i)});
            emit_rec_stub(st);
        }
    }
    void emit_rec_stub(const RecStub& st) {
        const tir::Region& ra = M.regions[st.a.a];
        const tir::Region& rb = M.regions[st.b.a];
        uint32_t ka = ra.rec_len, kb = rb.rec_len;
        bool dyn = ra.nullable || rb.nullable;
        Label lneg = A->new_label(), lpos = A->new_label(), lzero = A->new_label(), lone = A->new_label(),
              lret = A->new_label(), lanull = A->new_label(), linv = A->new_label();
        A->stp(X(29), X(30), pre(sp, -32));
        A->stp(X(19), X(20), mem(sp, 16));
        A->mov(X(19), X(0));
        A->mov(X(20), X(1));
        if (dyn) {
            A->cbz(X(19), lanull);
            A->cbz(X(20), st.eq ? lzero : lpos);
        }
        uint32_t k = std::min(ka, kb);
        bool skip_fields = st.eq && ka != kb;
        auto generic = [&](int h, uint32_t fda, uint32_t fdb, uint32_t off) {
            A->ldr(X(1), mem(X(19), off));
            A->ldr(X(3), mem(X(20), off));
            li(W(0), fda);
            li(W(2), fdb);
            call_helper(h);
        };
        for (uint32_t j = 0; j < k && !skip_fields; ++j) {
            Type fa = ra.fields[j], fb = rb.fields[j];
            uint32_t fda = M.descs[st.da + 1 + j].elem, fdb = M.descs[st.db + 1 + j].elem;
            uint32_t off = 8 * j;
            bool ia = fa.is_int_like(), ib = fb.is_int_like();
            if (ia && ib) {
                A->ldr(W(0), mem(X(19), off));
                A->ldr(W(1), mem(X(20), off));
                A->cmp(W(0), W(1));
                if (st.eq) {
                    A->b(NE, lzero);
                } else {
                    A->b(LT, lneg);
                    A->b(GT, lpos);
                }
            } else if (fa.k == TK::Any || fb.k == TK::Any) {
                generic(st.eq ? H_EQ : H_CMP3, fda, fdb, off);
                if (st.eq) A->cbz(W(0), lzero);
                else A->cbnz(W(0), lret);
            } else if (fa.k == TK::Fn || fb.k == TK::Fn) {
                if (!st.eq) A->b(linv);
                else if (!(fa.k == TK::Fn && fb.k == TK::Fn && fa.a == fb.a)) A->b(lzero);
            } else if (fa.k == TK::Pair && fb.k == TK::Pair) {
                A->ldr(X(0), mem(X(19), off));
                A->ldr(X(1), mem(X(20), off));
                A->cmp(X(0), X(1));
                if (st.eq) {
                    A->b(NE, lzero);
                } else {
                    A->b(LT, lneg);
                    A->b(GT, lpos);
                }
            } else if (ia || ib) {  // an int against a list
                A->b(st.eq ? lzero : linv);
            } else if (fa.k == TK::Pair || fb.k == TK::Pair) {
                generic(st.eq ? H_EQ : H_CMP3, fda, fdb, off);
                if (st.eq) A->cbz(W(0), lzero);
                else A->cbnz(W(0), lret);
            } else if (fa.k == TK::List && fb.k == TK::List && M.elem_is_i32(fa.a) && M.elem_is_i32(fb.a)) {
                A->ldr(X(0), mem(X(19), off));
                A->ldr(X(1), mem(X(20), off));
                call_helper(st.eq ? H_EQ_LI : H_CMP3_LI);
                if (st.eq) A->cbz(W(0), lzero);
                else A->cbnz(W(0), lret);
            } else if (fa.k == TK::Rec && fb.k == TK::Rec && M.descs[fda].kind == 5 && M.descs[fdb].kind == 5) {
                Label sub = rec_stub(st.eq, fa, fb, fda, fdb);
                A->ldr(X(0), mem(X(19), off));
                A->ldr(X(1), mem(X(20), off));
                A->bl(sub);
                if (st.eq) A->cbz(W(0), lzero);
                else A->cbnz(W(0), lret);
            } else {
                generic(st.eq ? H_EQ : H_CMP3, fda, fdb, off);
                if (st.eq) A->cbz(W(0), lzero);
                else A->cbnz(W(0), lret);
            }
        }
        // all compared positions equal
        if (st.eq) A->b(ka == kb ? lone : lzero);
        else A->b(ka > kb ? lpos : ka < kb ? lneg : lzero);
        if (dyn) {  // a is null (the empty list)
            A->bind(lanull);
            A->cbz(X(20), st.eq ? lone : lzero);
            A->b(st.eq ? lzero : lneg);
        }
        A->bind(lneg);
        A->movn(W(0), 0);
        A->b(lret);
        A->bind(lpos);
        A->bind(lone);
        A->movz(W(0), 1);
        A->b(lret);
        A->bind(lzero);
        A->movz(W(0), 0);
        A->bind(lret);
        A->ldp(X(19), X(20), mem(sp, 16));
        A->ldp(X(29), X(30), post(sp, 32));
        A->ret();
        A->bind(linv);
        A->movz(W(0), (uint32_t)Err::InvalidCmp);
        A->b(ext_helper(H_FAIL));
    }
    bool la_fast(const Node* n) {
        Type ta = n->kids[0]->type, tb = n->kids[1]->type;
        return ta.k == TK::List && tb.k == TK::List && M.regions[ta.a].elem.k == TK::Any && M.regions[tb.a].elem.k == TK::Any;
    }
    // Non-inline Eq/Ne/Cmp: result in a fresh register (helper call).
    V deep_cmp_call(const Node* n) {
        V k[2];
        eval_kids(n, 0, 2, k);
        Op c = n->op == Op::Cmp ? (Op)n->c : n->op;
        bool swap = c == Op::GtI32 || c == Op::LeI32;
        bool neg = n->op == Op::Ne || c == Op::LeI32 || c == Op::GeI32;
        if (rec_rec(n)) {  // records: typed comparison function (as the C backend's epq_rq / epq_rc)
            bool eq = n->op == Op::Eq || n->op == Op::Ne;
            Label st = rec_stub(eq, n->kids[0]->type, n->kids[1]->type, (uint32_t)n->a, (uint32_t)n->b);
            std::initializer_list<Move> mv{{0, k[0], true}, {1, k[1], true}};
            hc_moves_.assign(mv.begin(), mv.end());
            parallel_move(hc_moves_);
            check_no_temps("record compare");
            A->bl(st);
            V r = V::reg(talloc_specific(0), true);
            if (eq) {
                if (n->op == Op::Ne) A->eor(W(0), W(0), (uint64_t)1);
            } else {
                A->cmp(W(0), 0u);
                A->cset(W(0), cond_of(c));
            }
            return r;
        }
        if (la_fast(n)) {  // litcmp: epq_eq_la / epq_cmp3_la(a, b) <op> 0, as the C backend
            if (n->op == Op::Eq || n->op == Op::Ne) {
                std::initializer_list<Move> mv{{0, k[0], true}, {1, k[1], true}};
                V r = helper_call(H_EQ_LA, mv, true);
                if (n->op == Op::Ne) A->eor(W(r.r), W(r.r), (uint64_t)1);
                return r;
            }
            std::initializer_list<Move> mv{{0, k[0], true}, {1, k[1], true}};
            V r = helper_call(H_CMP3_LA, mv, true);
            A->cmp(W(r.r), 0u);
            A->cset(W(r.r), cond_of(c));
            return r;
        }
        if (li_fast(n)) {
            int h = (n->op == Op::Eq || n->op == Op::Ne) ? H_EQ_LI : H_LT_LI;
            std::initializer_list<Move> mv{{swap ? 1 : 0, k[0], true}, {swap ? 0 : 1, k[1], true}};
            V r = helper_call(h, mv, true);
            if (neg) A->eor(W(r.r), W(r.r), (uint64_t)1);
            return r;
        }
        V wa = word_of(n->kids[0]->type, k[0]);
        V wb = word_of(n->kids[1]->type, k[1]);
        int h = (n->op == Op::Eq || n->op == Op::Ne) ? H_EQ : H_LT;
        uint32_t da = (uint32_t)n->a, db = (uint32_t)n->b;
        std::initializer_list<Move> mv{{0, V::im(swap ? db : da), false},
                             {1, swap ? wb : wa, true},
                             {2, V::im(swap ? da : db), false},
                             {3, swap ? wa : wb, true}};
        V r = helper_call(h, mv, true);
        if (neg) A->eor(W(r.r), W(r.r), (uint64_t)1);
        return r;
    }
    // Inline (Int|Any) comparison: evaluates kids, emits the int check + cmp.
    // Returns the condition; slow path label for non-int operands.
    struct FastCmp {
        Cond c;
        Label slow, cont;
        V a, b;  // boxed words, still allocated (caller releases after the slow path is set up)
    };
    // A kid statically known to be an integer (Int, or the Box of an Int).
    static bool static_int(const Node* k) {
        return k->type.k == TK::Int || (k->op == Op::Box && k->kids[0]->type.k == TK::Int);
    }
    // What the non-integer path of an inline (Int|Any) comparison does:
    //   SLOW_CALL  both sides dynamic: descriptor helper
    //   SLOW_FALSE / SLOW_TRUE  one side is an integer, the other is not: Eq / Ne
    //              are decided (different kinds are never equal)
    //   SLOW_ERR   ... and Cmp raises Invalid comparison
    //   SLOW_NONE  both integers: no slow path
    enum { SLOW_NONE, SLOW_CALL, SLOW_FALSE, SLOW_TRUE, SLOW_ERR };
    int slow_kind(const Node* n) {
        bool sa = static_int(n->kids[0]), sb = static_int(n->kids[1]);
        if (sa && sb) return SLOW_NONE;
        if (!sa && !sb) return SLOW_CALL;
        if (n->op == Op::Eq) return SLOW_FALSE;
        if (n->op == Op::Ne) return SLOW_TRUE;
        return SLOW_ERR;
    }
    // Evaluates the kids, branches to `slow` when an operand is not an integer,
    // then compares the tagged words (flags for f.c).
    FastCmp fast_cmp(const Node* n, Label slow) {
        V k[2];
        eval_kids(n, 0, 2, k);
        Type ta = n->kids[0]->type, tb = n->kids[1]->type;
        FastCmp f;
        Op c = n->op == Op::Cmp ? (Op)n->c : n->op == Op::Eq ? Op::EqI32 : Op::NeI32;
        f.c = cond_of(c);
        f.a = boxed_of(ta, k[0]);
        f.b = boxed_of(tb, k[1]);
        f.slow = slow;
        bool ca = !static_int(n->kids[0]), cb = !static_int(n->kids[1]);
        if (f.a.k == V::IMM && !f.a.own) {
            // keep the constant on the right-hand side
        }
        Reg ra = R(f.a, true);
        bool bimm = f.b.k == V::IMM && f.b.imm >= 0 && f.b.imm < 4096;
        Reg rb = bimm ? xzr : R(f.b, true);
        if (ca && cb) {
            A->and_(X(16), ra, rb);
            A->tbz(X(16), 0, slow);
        } else if (ca) {
            A->tbz(ra, 0, slow);
        } else if (cb) {
            A->tbz(rb, 0, slow);
        }
        if (bimm) A->cmp(ra, (uint32_t)f.b.imm);
        else A->cmp(ra, rb);
        return f;
    }
    // Slow path of a fast comparison: compute the result into w0 with saved regs.
    void fast_cmp_slow_call(const Node* n, Reg a, Reg b, uint32_t save) {
        save_regs(save);
        // a, b may live in saved registers: still valid (save does not clobber)
        deep_call(n, a, b, true);
        A->mov(W(16), W(0));
        restore_regs(save);
    }
    bool pair_pair(const Node* n) { return n->kids[0]->type.k == TK::Pair && n->kids[1]->type.k == TK::Pair; }
    V deep_cmp_value(const Node* n, int want) {
        if (pair_pair(n)) {
            V k[2];
            eval_kids(n, 0, 2, k);
            Reg a = R(k[0], true), b = R(k[1], true);
            A->cmp(a, b);
            release(k[0]);
            release(k[1]);
            int d = dest(want);
            A->cset(W(d), cond_of(n->op == Op::Cmp ? (Op)n->c : n->op == Op::Eq ? Op::EqI32 : Op::NeI32));
            return result(d, want);
        }
        if (!cmp_inline(n)) {
            V r = deep_cmp_call(n);
            return r;
        }
        int sk = slow_kind(n);
        Label slow = sk == SLOW_ERR ? err(Err::InvalidCmp) : A->new_label();
        Label cont = A->new_label();
        FastCmp f = fast_cmp(n, slow);
        Reg ra = R(f.a, true);
        Reg rb = f.b.k == V::IMM ? X(17) : R(f.b, true);
        int64_t bimm = f.b.k == V::IMM ? f.b.imm : 0;
        bool bimm_p = f.b.k == V::IMM;
        int d = dest(want);
        A->cset(W(d), f.c);
        A->bind(cont);
        uint32_t save = live_caller_saved(bit(d));
        save &= ~bit(d);
        if (sk == SLOW_CALL) {
            cold(slow, [=, this]() {
                if (bimm_p) li(X(17), bimm);
                fast_cmp_slow_call(n, ra, rb, save);
                A->mov(W(d), W(16));
                A->b(cont);
            });
        } else if (sk == SLOW_FALSE || sk == SLOW_TRUE) {
            cold(slow, [=, this]() {
                A->movz(W(d), sk == SLOW_TRUE ? 1 : 0);
                A->b(cont);
            });
        }
        release(f.a);
        release(f.b);
        return result(d, want);
    }

    // ---- lists
    uint32_t esz_of(uint32_t rg) { return M.elem_is_i32(rg) ? 4 : 8; }

    // epq_list_new(n, esz) sizes (runtime/epq_rt.h)
    static uint32_t lit_bytes(uint32_t n, uint32_t esz) {
        uint32_t b = n * esz;
        if (b < 16) b = 16;
        return (b + 15) & ~15u;
    }
    bool lit_inline(const Node* n) {
        return bump_adjacent && 16 + lit_bytes(n->nkids, esz_of(n->type.a)) <= 512;
    }
    V list_lit(const Node* n) {
        uint32_t r = n->type.a;
        uint32_t esz = esz_of(r);
        V lst;
        if (!lit_inline(n)) {
            check_no_temps("listlit");
            std::initializer_list<Move> mv{{0, V::im(n->nkids), false}, {1, V::im(esz), false}};
            lst = helper_call(H_LIST_NEW, mv, true);
        } else {
            // inline bump allocation (epq_alloc + epq_list_new)
            uint32_t bytes = lit_bytes(n->nkids, esz), total = 16 + bytes;
            uint32_t cap = bytes / esz, cnt = n->nkids;
            int d = talloc();
            uint32_t lo;
            Label slow = A->new_label(), cont = A->new_label(), bump = A->new_label();
            // size-class free list first (epq_alloc), then the bump pointer
            data_addr(X(16), fl_off, &lo);
            A->ldr(X(16), mem(X(16), lo));
            A->ldr(X(d), mem(X(16), 8 * (total >> 4)));
            A->cbz(X(d), bump);
            A->ldr(X(17), mem(X(d), 0));
            A->str(X(17), mem(X(16), 8 * (total >> 4)));
            A->b(cont);
            A->bind(bump);
            data_addr(X(16), bump_off, &lo);
            A->ldr(X(16), mem(X(16), lo));
            A->ldp(X(d), X(17), mem(X(16), 0));
            A->sub(X(17), X(17), X(d));
            A->cmp(X(17), total);
            A->b(LO, slow);
            A->add(X(17), X(d), total);
            A->str(X(17), mem(X(16), 0));
            A->bind(cont);
            A->add(X(17), X(d), 16u);
            li(X(16), (int64_t)(((uint64_t)cap << 32) | cnt));
            A->stp(X(17), X(16), mem(X(d), 0));
            uint32_t save = live_caller_saved(bit(d));
            cold(slow, [=, this]() {
                save_regs(save);
                A->movz(W(0), total);
                call_helper(H_ALLOC_SLOW);
                A->mov(X(16), X(0));
                restore_regs(save);
                A->mov(X(d), X(16));
                A->b(cont);
            });
            lst = V::reg(d, true);
        }
        for (uint32_t i = 0; i < n->nkids; ++i) {
            hold(lst, n, i, n->nkids - 1);
            const Node* k = n->kids[i];
            V v = expr(k);
            if (esz == 8) v = word_of(k->type, v);
            Reg l = R(lst, true);
            uint32_t off = 16 + i * esz;
            if (v.k == V::IMM && v.imm == 0) {
                A->str(esz == 8 ? xzr : wzr, mem(l, off));
            } else {
                Reg s = R(v, esz == 8);
                A->str(s, mem(l, off));
            }
            release(v);
        }
        return lst;
    }

    // Bounds check of index (w reg or imm) against l's length; then x16 = data.
    // On return *imm_ok tells whether the index is a (valid, non-negative) imm.
    void bounds(Reg l, V& i, Label oob) {
        A->ldr(W(17), mem(l, 8));
        if (i.k == V::IMM) {
            int32_t c = (int32_t)i.imm;
            if (c < 0) {
                A->b(oob);
            } else if (c == 0) {
                A->cbz(W(17), oob);
            } else {
                A->cmp_imm(W(17), c);
                A->b(LS, oob);
            }
        } else {
            A->cmp(R(i, false), W(17));
            A->b(HS, oob);
        }
        A->ldr(X(16), mem(l, 0));
    }
    Mem elem_mem(V& i, uint32_t esz) {
        if (i.k == V::IMM) return mem(X(16), (int64_t)(uint32_t)(int32_t)i.imm * esz);
        return memr(X(16), W(i.r), UXTW, true);
    }
    V from_word(Type t, int d, int want) {
        if (t.k == TK::Fn) {
            if (want < 0) {
                temp_free |= bit(d);
            }
            return V::none();
        }
        return result(d, want);
    }

    V index(const Node* n, int want) {
        V k[2];
        eval_kids(n, 0, 2, k);
        uint32_t esz = esz_of(n->kids[0]->type.a);
        Reg l = R(k[0], true);
        if (k[1].k != V::IMM) R(k[1], false);
        bounds(l, k[1], err(Err::IndexOOB));
        Mem m = elem_mem(k[1], esz);
        release(k[0]);
        release(k[1]);
        int d = dest(want);
        A->ldr(esz == 4 ? W(d) : X(d), m);
        return from_word(n->type, d, want);
    }

    V set_index(const Node* n) {
        V k[3];
        eval_kids(n, 0, 3, k);
        uint32_t esz = esz_of(n->kids[0]->type.a);
        V w = esz == 8 ? word_of(n->kids[2]->type, k[2]) : k[2];
        Reg l = R(k[0], true);
        if (k[1].k != V::IMM) R(k[1], false);
        Reg s = (w.k == V::IMM && w.imm == 0) ? (esz == 8 ? xzr : wzr) : R(w, esz == 8);
        bounds(l, k[1], err(Err::IndexOOB));
        A->str(s, elem_mem(k[1], esz));
        release(k[0]);
        release(k[1]);
        // value of SetIndex is v: for 8-byte slots its word (word_of only rewrites Int/Fn)
        if (esz == 8 && n->kids[2]->type.k == TK::Fn) {
            release(w);
            return V::none();
        }
        return w;
    }

    // push: l (x reg), v (reg or zr). Grow out of line.
    void push(Reg l, Reg v, uint32_t esz) {
        Label top = A->here();
        Label grow = A->new_label();
        A->ldp(W(16), W(17), mem(l, 8));
        A->cmp(W(16), W(17));
        A->b(EQ, grow);
        A->ldr(X(17), mem(l, 0));
        A->str(v, memr(X(17), W(16), UXTW, true));
        A->add(W(16), W(16), 1u);
        A->str(W(16), mem(l, 8));
        uint32_t save = live_caller_saved(0);
        cold(grow, [=, this]() {
            save_regs(save);
            A->mov(X(0), l.xr());
            A->movz(W(1), esz);
            call_helper(H_LIST_GROW);
            restore_regs(save);
            A->b(top);
        });
    }

    V append(const Node* n) {
        V k[2];
        eval_kids(n, 0, 2, k);
        uint32_t esz = esz_of(n->kids[0]->type.a);
        V w = esz == 8 ? word_of(n->kids[1]->type, k[1]) : k[1];
        Reg l = R(k[0], true);
        Reg v = (w.k == V::IMM && w.imm == 0) ? (esz == 8 ? xzr : wzr) : R(w, esz == 8);
        push(l, v, esz);
        release(w);
        return k[0];
    }

    V pop(const Node* n, int want) {
        V k = expr(n->kids[0]);
        uint32_t esz = esz_of(n->kids[0]->type.a);
        Reg l = R(k, true);
        A->ldr(W(17), mem(l, 8));
        A->cbz(W(17), err(Err::ListEmpty));
        A->sub(W(17), W(17), 1u);
        A->str(W(17), mem(l, 8));
        A->ldr(X(16), mem(l, 0));
        release(k);
        int d = dest(want);
        A->ldr(esz == 4 ? W(d) : X(d), memr(X(16), W(17), UXTW, true));
        return from_word(n->type, d, want);
    }

    // Any list + Int|Any index checks -> index in a w register (or imm)
    void dyn_list_index(Reg l, V& i, Type it, int scratch_idx) {
        A->tst(l, (uint64_t)3);
        A->b(NE, err(Err::InvalidOp));
        if (it.k == TK::Any) {
            Reg x = R(i, true);
            A->tbz(x, 0, err(Err::InvalidOp));
            A->lsr(X(scratch_idx), x, 32);
            release(i);
            i = V::reg(scratch_idx, true);
        }
    }

    V dyn_index(const Node* n, int want) {
        V k[2];
        eval_kids(n, 0, 2, k);
        Reg l = R(k[0], true);
        if (n->kids[1]->type.k == TK::Any) {
            int t = talloc();
            dyn_list_index(l, k[1], n->kids[1]->type, t);
        } else {
            dyn_list_index(l, k[1], n->kids[1]->type, 0);
            if (k[1].k != V::IMM) R(k[1], false);
        }
        bounds(l, k[1], err(Err::IndexOOB));
        Mem m = elem_mem(k[1], 8);
        release(k[0]);
        release(k[1]);
        int d = dest(want);
        A->ldr(X(d), m);
        return result(d, want);
    }

    V dyn_set_index(const Node* n) {
        V k[3];
        eval_kids(n, 0, 3, k);
        Reg l = R(k[0], true);
        Reg s = R(k[2], true);
        if (n->kids[1]->type.k == TK::Any) {
            int t = talloc();
            dyn_list_index(l, k[1], n->kids[1]->type, t);
        } else {
            dyn_list_index(l, k[1], n->kids[1]->type, 0);
            if (k[1].k != V::IMM) R(k[1], false);
        }
        bounds(l, k[1], err(Err::IndexOOB));
        A->str(s, elem_mem(k[1], 8));
        release(k[0]);
        release(k[1]);
        return k[2];
    }

    V dyn_append(const Node* n) {
        V k[2];
        eval_kids(n, 0, 2, k);
        Reg l = R(k[0], true);
        A->tst(l, (uint64_t)3);
        A->b(NE, err(Err::OperandList));
        Reg v = R(k[1], true);
        push(l, v, 8);
        release(k[1]);
        return k[0];
    }

    // ---- And / Or value form
    void truthy_branch(Type t, Reg r, bool sense, Label target) {
        // jump to target if truthy(r) == sense
        switch (t.k) {
        case TK::Int:
            if (sense) A->cbnz(r.w(), target);
            else A->cbz(r.w(), target);
            break;
        case TK::Any:
            A->cmp(r.xr(), 1u);
            A->b(sense ? NE : EQ, target);
            break;
        case TK::Bot: break;
        default:
            if (sense) A->b(target);
            break;
        }
    }
    V and_or_value(const Node* n) {
        if (n->flags & tir::F_COND) {
            Label f = A->new_label(), e = A->new_label();
            branch(n, false, f);
            int d = talloc();
            A->movz(W(d), 1);
            A->b(e);
            A->bind(f);
            A->movz(W(d), 0);
            A->bind(e);
            return V::reg(d, true);
        }
        Type t = n->type;
        bool x64 = is64(t);
        if (t.k == TK::Fn) {
            discard_value(n->kids[0]);
            if (n->op == Op::And) discard_value(n->kids[1]);
            return V::none();
        }
        V a = expr(n->kids[0]);
        Reg d = own_reg(a, x64);
        int dr = d.id;
        Label end = A->new_label();
        truthy_branch(t, d, n->op == Op::Or, end);
        // second operand overwrites d: the value of kid 0 is dead on this path
        temp_free |= bit(dr);
        V b = expr(n->kids[1], -1);
        if (!(b.k == V::REG && b.r == dr)) {
            if (temp_free & bit(dr)) temp_free &= ~bit(dr);
            else bug("and/or: result register taken");
            if (b.k == V::IMM) {
                if (x64) li(X(dr), b.imm);
                else li(W(dr), (int32_t)b.imm);
            } else if (b.k != V::NONE) {
                Reg s = R(b, x64);
                A->mov(x64 ? X(dr) : W(dr), s);
            }
            release(b);
        }
        A->bind(end);
        return V::reg(dr, true);
    }

    // ---- extension ops
    // 64-bit compare of two values (Pair order / list identity); returns the condition.
    Cond cmp64(const Node* n) {
        V k[2];
        eval_kids(n, 0, 2, k);
        Cond c;
        switch (n->op) {
        case Op::PairEq: c = EQ; break;
        case Op::PairNe: c = NE; break;
        case Op::PairLt: c = LT; break;
        case Op::PairLe: c = LE; break;
        case Op::PairGt: c = GT; break;
        case Op::PairGe: c = GE; break;
        default: c = EQ; break;  // SameList
        }
        if (k[0].k == V::IMM && k[1].k != V::IMM) {
            std::swap(k[0], k[1]);
            c = swap_cond(c);
        }
        Reg a = R(k[0], true);
        if (k[1].k == V::IMM && k[1].imm >= 0 && k[1].imm < 4096) {
            A->cmp(a, (uint32_t)k[1].imm);
        } else {
            Reg b = R(k[1], true);
            A->cmp(a, b);
        }
        release(k[0]);
        release(k[1]);
        return c;
    }
    // Pure condition -> flags (returns the condition that means "true").
    Cond cond_flags(const Node* c) {
        switch (c->op) {
        case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
        case Op::SameList: return cmp64(c);
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32: {
            V fold;
            Cond cc;
            if (!cmp_i32(c, &cc, &fold)) return cc;
            A->cmp(wzr, 0u);  // Z set
            return fold.imm ? EQ : NE;
        }
        case Op::NotI32: return invert(cond_flags(c->kids[0]));
        default: break;
        }
        V v = expr(c);
        Reg r = R(v, false);
        A->cmp(r, 0u);
        release(v);
        return NE;
    }
    // `t := <compare>; x := Select(t, a, b)` with t read nowhere else: the
    // comparison sets the flags for the csel directly (no 0/1 in between).
    // The comparison is evaluated after a and b, which is safe because a and b
    // are pure (Select kids) and the comparison writes no local and calls nothing.
    const Node* select_cond_ = nullptr;
    static bool flag_cond_op(Op o) {
        switch (o) {
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
        case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
        case Op::SameList: case Op::NotI32: return true;
        default: return false;
        }
    }
    static bool calm(const Node* n) {  // no local / global writes, no calls, no allocation
        switch (n->op) {
        case Op::ConstInt: case Op::Local: case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::NegI32:
        case Op::DivI32: case Op::AbsI32: case Op::NotI32:
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
        case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
        case Op::PairFirst: case Op::PairSecond: case Op::PairIndex: case Op::SameList:
        case Op::ElemLoad: case Op::CheckIndex: case Op::Index: case Op::Len: case Op::ListData: case Op::RecGet:
            for (uint32_t i = 0; i < n->nkids; ++i)
                if (!calm(n->kids[i])) return false;
            return true;
        default: return false;
        }
    }
    // Returns how many statements after kids[i] were consumed (0: no fusion).
    // One select: its a/b may be any pure expressions (evaluated before the
    // comparison). Several selects on the same t: a/b must be locals or
    // constants (their evaluation leaves the flags alone); one comparison.
    static bool is_sel_of(const Node* b, int32_t t, bool simple) {
        if (b->op != Op::SetLocal) return false;
        const Node* sel = b->kids[0];
        if (sel->op != Op::Select || sel->kids[0]->op != Op::Local || sel->kids[0]->a != t || sel->type.k == TK::Fn) return false;
        if (b->a == t) return false;
        if (!simple) return true;
        for (int j = 1; j <= 2; ++j)
            if (sel->kids[j]->op != Op::Local && sel->kids[j]->op != Op::ConstInt) return false;
        return true;
    }
    uint32_t fuse_cond_select(const Node* seq, uint32_t i) {
        const Node* a = seq->kids[i];
        if (a->op != Op::SetLocal) return 0;
        const Node* c = a->kids[0];
        if (!flag_cond_op(c->op) || !calm(c)) return 0;
        uint32_t t = (uint32_t)a->a;
        const LocalInfo& l = L[t];
        if (l.alias >= 0 || l.param || l.needs_init || l.fn) return 0;
        uint32_t cnt = 0;
        while (i + 1 + cnt < seq->nkids && is_sel_of(seq->kids[i + 1 + cnt], (int32_t)t, true)) ++cnt;
        if (cnt >= 2 && l.refs.size() == 1 + cnt) {
            for (uint32_t j = 0; j < cnt; ++j)
                if (canon((uint32_t)seq->kids[i + 1 + j]->a) == t) return 0;
            Cond cc = cond_flags(c);
            for (uint32_t j = 0; j < cnt; ++j) {
                const Node* b = seq->kids[i + 1 + j];
                const Node* sel = b->kids[0];
                bool x64 = is64(sel->type);
                uint32_t x = canon((uint32_t)b->a);
                V k[2] = {expr(sel->kids[1]), expr(sel->kids[2])};
                auto reg_or_zr = [&](V& v) -> Reg {
                    if (v.k == V::IMM && v.imm == 0) return x64 ? xzr : wzr;
                    return R(v, x64);
                };
                Reg ra = reg_or_zr(k[0]);
                Reg rb = reg_or_zr(k[1]);
                release(k[0]);
                release(k[1]);
                if (L[x].fn) continue;
                int d = L[x].reg >= 0 ? L[x].reg : talloc();
                A->csel(x64 ? X(d) : W(d), ra, rb, cc);
                if (L[x].reg < 0) store_local(x, V::reg(d, true));
            }
            return cnt;
        }
        if (i + 1 < seq->nkids && is_sel_of(seq->kids[i + 1], (int32_t)t, false) && l.refs.size() == 2) {
            const Node* b = seq->kids[i + 1];
            if (canon((uint32_t)b->a) == t) return 0;
            select_cond_ = c;
            stmt(b);
            select_cond_ = nullptr;
            return 1;
        }
        return 0;
    }
    V select(const Node* n, int want) {
        Type t = n->type;
        bool x64 = is64(t);
        const Node* c = n->kids[0];
        if (select_cond_) {
            c = select_cond_;
            select_cond_ = nullptr;
        }
        if (c->op == Op::ConstInt) return expr(c->a ? n->kids[1] : n->kids[2], want);
        if (t.k == TK::Fn) return V::none();
        V k[2];
        k[0] = expr(n->kids[1]);
        k[1] = expr(n->kids[2]);
        auto reg_or_zr = [&](V& v) -> Reg {
            if (v.k == V::IMM && v.imm == 0) return x64 ? xzr : wzr;
            return R(v, x64);
        };
        Reg a = reg_or_zr(k[0]);
        Reg b = reg_or_zr(k[1]);
        Cond cc = cond_flags(c);
        release(k[0]);
        release(k[1]);
        int d = dest(want);
        A->csel(x64 ? X(d) : W(d), a, b, cc);
        return result(d, want);
    }
    V pair_make(const Node* n, int want) {
        V k[2];
        eval_kids(n, 0, 2, k);
        if (k[0].k == V::IMM && k[1].k == V::IMM)
            return V::im((int64_t)(((uint64_t)(uint32_t)k[0].imm << 32) | ((uint32_t)k[1].imm ^ 0x80000000u)));
        if (k[1].k == V::IMM) {
            Reg a = R(k[0], true);
            release(k[0]);
            int d = dest(want);
            A->lsl(X(d), a, 32);
            uint32_t lo = (uint32_t)k[1].imm ^ 0x80000000u;
            if (lo) {
                li(X(16), lo);
                A->orr(X(d), X(d), X(16));
            }
            return result(d, want);
        }
        Reg b = R(k[1], false);
        A->eor(W(16), b, (uint64_t)0x80000000u);
        if (k[0].k == V::IMM) {
            release(k[1]);
            int d = dest(want);
            li(X(17), (int64_t)((uint64_t)(uint32_t)k[0].imm << 32));
            A->orr(X(d), X(16), X(17));
            return result(d, want);
        }
        Reg a = R(k[0], true);
        release(k[0]);
        release(k[1]);
        int d = dest(want);
        A->orr(X(d), X(16), a, LSL, 32);
        return result(d, want);
    }
    V pair_index(const Node* n, int want) {
        V k[2];
        eval_kids(n, 0, 2, k);
        if (k[1].k == V::IMM) {
            int32_t i = (int32_t)k[1].imm;
            if (i != 0 && i != 1) {
                release(k[0]);
                A->b(err(Err::IndexOOB));
                return V::im(0);
            }
            if (k[0].k == V::IMM)
                return V::im(i ? (int32_t)((uint32_t)k[0].imm ^ 0x80000000u) : (int32_t)((uint64_t)k[0].imm >> 32));
            Reg p = R(k[0], true);
            release(k[0]);
            int d = dest(want);
            if (i) A->eor(W(d), p.w(), (uint64_t)0x80000000u);
            else A->lsr(X(d), p, 32);
            return result(d, want);
        }
        Reg p = R(k[0], true);
        Reg i = R(k[1], false);
        A->cmp(i, 1u);
        A->b(HI, err(Err::IndexOOB));
        A->lsr(X(16), p, 32);
        A->eor(W(17), p.w(), (uint64_t)0x80000000u);
        release(k[0]);
        release(k[1]);
        int d = dest(want);
        A->csel(W(d), W(17), W(16), EQ);
        return result(d, want);
    }
    Mem elem_at(Reg base, V& i, uint32_t esz) {
        if (i.k == V::IMM) return mem(base, (int64_t)(int32_t)i.imm * esz);
        return memr(base, W(i.r), SXTW, true);
    }
    V elem_load(const Node* n, int want) {
        V k[2];
        eval_kids(n, 0, 2, k);
        uint32_t esz = esz_of(n->kids[0]->type.a);
        Reg d0 = R(k[0], true);
        if (k[1].k != V::IMM) R(k[1], false);
        Mem m = elem_at(d0, k[1], esz);
        release(k[0]);
        release(k[1]);
        if (n->type.k == TK::Fn) return V::none();
        int d = dest(want);
        A->ldr(esz == 4 ? W(d) : X(d), m);
        return result(d, want);
    }
    V elem_store(const Node* n) {
        V k[3];
        eval_kids(n, 0, 3, k);
        uint32_t esz = esz_of(n->kids[0]->type.a);
        Type vt = n->kids[2]->type;
        V w = esz == 8 ? word_of(vt, k[2]) : k[2];
        Reg d0 = R(k[0], true);
        if (k[1].k != V::IMM) R(k[1], false);
        Reg s = (w.k == V::IMM && w.imm == 0) ? (esz == 8 ? xzr : wzr) : R(w, esz == 8);
        A->str(s, elem_at(d0, k[1], esz));
        release(k[0]);
        release(k[1]);
        if (vt.k == TK::Fn) {
            release(w);
            return V::none();
        }
        return w;  // Int (esz 4) or List/Any/Pair: the word is the value
    }
    V check_index(const Node* n) {
        V k[2];
        eval_kids(n, 0, 2, k);
        if (k[0].k == V::IMM && k[1].k == V::IMM) {
            if ((uint32_t)k[0].imm >= (uint32_t)k[1].imm) A->b(err(Err::IndexOOB));
            return k[0];
        }
        if (k[0].k == V::IMM) {
            int32_t c = (int32_t)k[0].imm;
            Reg len = R(k[1], false);
            if (c < 0) {
                A->b(err(Err::IndexOOB));
            } else {
                A->cmp_imm(len, c);
                A->b(LS, err(Err::IndexOOB));
            }
            release(k[1]);
            return k[0];
        }
        Reg i = R(k[0], false);
        if (k[1].k == V::IMM) A->cmp_imm(i, (int32_t)k[1].imm);
        else A->cmp(i, R(k[1], false));
        A->b(HS, err(Err::IndexOOB));
        release(k[1]);
        return k[0];
    }
    // ---- records
    // Inline epq_alloc(bytes) (size-class free list, then the bump pointer, slow
    // path out of line) into register d. bytes: a multiple of 16, <= 512.
    void inline_alloc(int d, uint32_t total) {
        uint32_t lo;
        Label slow = A->new_label(), cont = A->new_label(), bump = A->new_label();
        data_addr(X(16), fl_off, &lo);
        A->ldr(X(16), mem(X(16), lo));
        A->ldr(X(d), mem(X(16), 8 * (total >> 4)));
        A->cbz(X(d), bump);
        A->ldr(X(17), mem(X(d), 0));
        A->str(X(17), mem(X(16), 8 * (total >> 4)));
        A->b(cont);
        A->bind(bump);
        data_addr(X(16), bump_off, &lo);
        A->ldr(X(16), mem(X(16), lo));
        A->ldp(X(d), X(17), mem(X(16), 0));
        A->sub(X(17), X(17), X(d));
        A->cmp(X(17), total);
        A->b(LO, slow);
        A->add(X(17), X(d), total);
        A->str(X(17), mem(X(16), 0));
        A->bind(cont);
        uint32_t save = live_caller_saved(bit(d));
        cold(slow, [=, this]() {
            save_regs(save);
            A->movz(W(0), total);
            call_helper(H_ALLOC_SLOW);
            A->mov(X(16), X(0));
            restore_regs(save);
            A->mov(X(d), X(16));
            A->b(cont);
        });
    }
    // RecLit: every kid first (in order), then epq_rec_new(k) and k slot stores.
    V rec_lit(const Node* n) {
        uint32_t k = n->nkids;
        std::vector<V> v(k);
        eval_kids(n, 0, k, v.data());
        for (uint32_t i = 0; i < k; ++i) v[i] = word_of(n->kids[i]->type, v[i]);
        uint32_t total = (8 * k + 15) & ~15u;
        if (total == 0) total = 16;
        int d = talloc();
        if (bump_adjacent && total <= 512) {
            inline_alloc(d, total);
        } else {  // epq_alloc (big block) with every live register saved
            uint32_t save = live_caller_saved(bit(d));
            save_regs(save);
            li(X(0), total);
            call_helper(H_ALLOC);
            A->mov(X(16), X(0));
            restore_regs(save);
            A->mov(X(d), X(16));
        }
        for (uint32_t i = 0; i < k; ++i) {
            Reg s = (v[i].k == V::IMM && v[i].imm == 0) ? xzr : R(v[i], true);
            A->str(s, mem(X(d), 8 * (int64_t)i));
            release(v[i]);
        }
        return V::reg(d, true);
    }

    void free_list(const Node* n) {
        V v = expr(n->kids[0]);
        uint32_t esz = esz_of(n->kids[0]->type.a);
        Reg l = R(v, true);
        Label slow = A->new_label(), cont = A->new_label();
        if (!poison) {
            // inline-buffer list of a small size class: push its block on the free list
            A->ldr(X(16), mem(l, 0));
            A->add(X(17), l, 16u);
            A->cmp(X(16), X(17));
            A->b(NE, slow);
            A->ldr(W(16), mem(l, 12));
            A->lsl(W(16), W(16), esz == 4 ? 2 : 3);
            A->add(W(16), W(16), 16u);
            A->and_(W(16), W(16), (uint64_t)~15u);
            A->cmp(W(16), 512u);
            A->b(HI, slow);
            A->lsr(W(16), W(16), 1);  // class * 8
            uint32_t lo;
            data_addr(X(17), fl_off, &lo);
            A->ldr(X(17), mem(X(17), lo));
            A->add(X(17), X(17), X(16));
            A->ldr(X(16), mem(X(17), 0));
            A->str(X(16), mem(l, 0));
            A->str(l, mem(X(17), 0));
        } else {
            A->b(slow);
        }
        A->bind(cont);
        uint32_t save = live_caller_saved(0);
        cold(slow, [=, this]() {
            save_regs(save);
            A->mov(X(0), l);
            A->movz(W(1), esz);
            call_helper(H_LIST_FREE);
            restore_regs(save);
            A->b(cont);
        });
        release(v);
    }

    V dyn_neg_abs(const Node* n, int want) {
        bool neg = n->op == Op::DynNeg;
        V v = expr(n->kids[0]);
        Reg x = R(v, true);
        release(v);
        int d = dest(want);
        Label slow = A->new_label(), cont = A->new_label();
        A->tbz(x, 0, slow);
        if (neg) {
            A->and_(X(16), x, ~(uint64_t)1);
            A->neg(X(16), X(16));
            A->orr(X(d), X(16), (uint64_t)1);
        } else {
            A->asr(X(16), x, 32);
            A->cmp(W(16), 0u);
            A->cneg(W(d), W(16), LT);
        }
        A->bind(cont);
        // not an integer: a list (length / pop, as epq_dyn_abs / epq_dyn_neg), else error
        cold(slow, [=, this]() {
            A->tst(x, (uint64_t)3);
            A->b(NE, err(Err::IntOrList));
            if (neg) {
                A->ldr(W(16), mem(x, 8));
                A->cbz(W(16), err(Err::ListEmpty));
                A->sub(W(16), W(16), 1u);
                A->str(W(16), mem(x, 8));
                A->ldr(X(17), mem(x, 0));
                A->ldr(X(d), memr(X(17), W(16), UXTW, true));
            } else {
                A->ldr(W(d), mem(x, 8));
            }
            A->b(cont);
        });
        return result(d, want);
    }

    // ---- calls
    V call_direct(const Node* n) {
        const Func* callee = M.funcs[(size_t)n->a].get();
        uint32_t na = n->nkids - 1;
        std::vector<V> v(n->nkids);
        eval_kids(n, 0, n->nkids, v.data());
        release(v[na]);  // callee expression (Fn): evaluated for effects only
        std::vector<Move> mv;
        for (uint32_t i = 0; i < na; ++i) {
            Type t = callee->locals[callee->params[i]].type;
            if (t.k == TK::Fn) {
                release(v[i]);
                continue;
            }
            int dst = i < 8 ? (int)i : -1 - (int)(i - 8);
            mv.push_back(Move{dst, v[i], is64(t)});
        }
        parallel_move(mv);
        check_no_temps("call");
        call_func(callee->id);
        Type rt = callee->ret;
        if (rt.k == TK::Fn) return V::none();
        if (rt.k == TK::Bot) return V::im(0);
        return V::reg(talloc_specific(0), true);
    }

    V call_dyn(const Node* n) {
        uint32_t na = n->nkids - 1;
        if (na <= 8) {
            std::vector<V> v(n->nkids);
            eval_kids(n, 0, n->nkids, v.data());
            Reg f = R(v[na], true);
            A->and_(X(16), f, (uint64_t)3);
            A->cmp(X(16), 2u);
            A->b(NE, err(Err::OperandFn));
            A->lsr(X(16), f, 32);
            uint32_t lo;
            data_addr(X(17), fntab_off, &lo);
            if (lo) A->add(X(17), X(17), lo);
            A->add(X(17), X(17), X(16), LSL, 4);
            A->ldr(W(16), mem(X(17), 0));
            A->cmp(W(16), na);
            A->b(NE, err(Err::Arity));
            A->ldr(X(17), mem(X(17), 8));
            release(v[na]);
            std::vector<Move> mv;
            for (uint32_t i = 0; i < na; ++i) mv.push_back(Move{(int)i, v[i], true});
            parallel_move(mv);
            check_no_temps("calldyn");
            A->blr(X(17));
            return V::reg(talloc_specific(0), true);
        }
        int base = na ? salloc_n((int)na) : 0;
        V f;
        for (uint32_t i = 0; i < n->nkids; ++i) {
            V v = expr(n->kids[i]);
            if (i < na) {
                if (v.k == V::IMM) {
                    li(X(16), v.imm);
                    A->str(X(16), mem(sp, dyn_off(base + (int)i)));
                } else {
                    Reg s = R(v, true);
                    A->str(s, mem(sp, dyn_off(base + (int)i)));
                }
                release(v);
            } else {
                f = v;
            }
        }
        std::vector<Move> mv{{0, f, true}, {1, V::im(na), false}};
        parallel_move(mv);
        if (na) A->add_imm(X(2), sp, dyn_off(base));
        else A->movz(X(2), 0);
        check_no_temps("calldyn");
        call_helper(H_CALL_DYN);
        for (uint32_t i = 0; i < na; ++i) sfree(base + (int)i);
        return V::reg(talloc_specific(0), true);
    }

    // ------------------------------------------------------------------ conditions
    // Jump to target if truthy(n) == sense; else fall through.
    void branch(const Node* n, bool sense, Label target) {
        switch (n->op) {
        case Op::ConstInt:
            if ((n->a != 0) == sense) A->b(target);
            return;
        case Op::And:
        case Op::Or:
            if (n->flags & tir::F_COND) {
                bool is_and = n->op == Op::And;
                if (is_and != sense) {
                    // And, false: either false -> target.  Or, true: either true -> target.
                    branch(n->kids[0], sense, target);
                    branch(n->kids[1], sense, target);
                } else {
                    Label skip = A->new_label();
                    branch(n->kids[0], !sense, skip);
                    branch(n->kids[1], sense, target);
                    A->bind(skip);
                }
                return;
            }
            break;
        case Op::NotI32: branch(n->kids[0], !sense, target); return;
        case Op::DynIsEmptyList: {
            V v = expr(n->kids[0]);
            if (v.k == V::IMM) {  // a constant word is an int / fn, never a list
                if (!sense) A->b(target);
                return;
            }
            Reg x = R(v, true);
            release(v);
            if (sense) {
                Label no = A->new_label();
                A->tst(x, (uint64_t)3);
                A->b(NE, no);
                A->ldr(W(16), mem(x, 8));
                A->cbz(W(16), target);
                A->bind(no);
            } else {
                A->tst(x, (uint64_t)3);
                A->b(NE, target);
                A->ldr(W(16), mem(x, 8));
                A->cbnz(W(16), target);
            }
            return;
        }
        case Op::EqI32:
        case Op::NeI32:
        case Op::LtI32:
        case Op::LeI32:
        case Op::GtI32:
        case Op::GeI32: {
            // cbz/cbnz for comparisons with 0 (eq/ne); tbz/tbnz on the sign bit (lt/ge)
            const Node* a = n->kids[0];
            const Node* b = n->kids[1];
            if ((n->op == Op::LtI32 || n->op == Op::GeI32) && (b->op == Op::ConstInt && b->a == 0)) {
                V v = expr(a);
                if (v.k == V::IMM) {
                    bool r = ((int32_t)v.imm < 0) == (n->op == Op::LtI32);
                    if (r == sense) A->b(target);
                    return;
                }
                Reg r = R(v, false);
                bool jneg = (n->op == Op::LtI32) == sense;
                if (jneg) A->tbnz(r, 31, target);
                else A->tbz(r, 31, target);
                release(v);
                return;
            }
            if ((n->op == Op::EqI32 || n->op == Op::NeI32) && (b->op == Op::ConstInt && b->a == 0) && a->op == Op::RecLen) {
                // records: `t = []` is RecLen(t) == 0, i.e. t == null
                V v = expr(a->kids[0]);
                bool jz = (n->op == Op::EqI32) == sense;
                if (v.k == V::IMM) {
                    if ((v.imm == 0) == jz) A->b(target);
                    return;
                }
                Reg r = R(v, true);
                if (jz) A->cbz(r, target);
                else A->cbnz(r, target);
                release(v);
                return;
            }
            const Node* sx;
            int32_t sc;
            if ((n->op == Op::EqI32 || n->op == Op::NeI32) && (b->op == Op::ConstInt && b->a == 0) && is_srem(a, &sx, &sc) &&
                sc != INT32_MIN && ((sc < 0 ? -sc : sc) & ((sc < 0 ? -sc : sc) - 1)) == 0 && (sc < 0 ? -sc : sc) > 1) {
                // x % 2^k == 0  <=>  low k bits of x are 0 (also for negative x)
                V v = expr(sx);
                uint32_t mask = (uint32_t)(sc < 0 ? -sc : sc) - 1;
                if (v.k == V::IMM) {
                    bool z = ((uint32_t)v.imm & mask) == 0;
                    if (z == ((n->op == Op::EqI32) == sense)) A->b(target);
                    return;
                }
                Reg r = R(v, false);
                bool jz = (n->op == Op::EqI32) == sense;
                if (mask == 1) {
                    if (jz) A->tbz(r, 0, target);
                    else A->tbnz(r, 0, target);
                } else {
                    A->tst(r, (uint64_t)mask);
                    A->b(jz ? EQ : NE, target);
                }
                release(v);
                return;
            }
            if ((n->op == Op::EqI32 || n->op == Op::NeI32) && (b->op == Op::ConstInt && b->a == 0)) {
                V v = expr(a);
                if (v.k == V::IMM) {
                    bool r = ((int32_t)v.imm == 0) == (n->op == Op::EqI32);
                    if (r == sense) A->b(target);
                    return;
                }
                Reg r = R(v, false);
                bool jz = (n->op == Op::EqI32) == sense;
                if (jz) A->cbz(r, target);
                else A->cbnz(r, target);
                release(v);
                return;
            }
            V fold;
            Cond c;
            if (cmp_i32(n, &c, &fold)) {
                if ((fold.imm != 0) == sense) A->b(target);
                return;
            }
            A->b(sense ? c : invert(c), target);
            return;
        }
        case Op::PairEq:
        case Op::PairNe:
        case Op::PairLt:
        case Op::PairLe:
        case Op::PairGt:
        case Op::PairGe:
        case Op::SameList: {
            Cond c = cmp64(n);
            A->b(sense ? c : invert(c), target);
            return;
        }
        case Op::Eq:
        case Op::Ne:
        case Op::Cmp:
            if (pair_pair(n)) {
                V k[2];
                eval_kids(n, 0, 2, k);
                Reg a = R(k[0], true), b = R(k[1], true);
                A->cmp(a, b);
                release(k[0]);
                release(k[1]);
                Cond c = cond_of(n->op == Op::Cmp ? (Op)n->c : n->op == Op::Eq ? Op::EqI32 : Op::NeI32);
                A->b(sense ? c : invert(c), target);
                return;
            }
            if (cmp_inline(n)) {
                int sk = slow_kind(n);
                Label cont = A->new_label();
                Label slow = sk == SLOW_ERR ? err(Err::InvalidCmp)
                           : sk == SLOW_FALSE ? (sense ? cont : target)
                           : sk == SLOW_TRUE ? (sense ? target : cont)
                           : A->new_label();
                FastCmp f = fast_cmp(n, slow);
                A->b(sense ? f.c : invert(f.c), target);
                A->bind(cont);
                Reg ra = R(f.a, true);
                Reg rb = f.b.k == V::IMM ? X(17) : R(f.b, true);
                int64_t bimm = f.b.k == V::IMM ? f.b.imm : 0;
                bool bimm_p = f.b.k == V::IMM;
                uint32_t save = live_caller_saved(0);
                if (sk == SLOW_CALL) {
                    cold(slow, [=, this]() {
                        if (bimm_p) li(X(17), bimm);
                        fast_cmp_slow_call(n, ra, rb, save);
                        if (sense) A->cbnz(W(16), target);
                        else A->cbz(W(16), target);
                        A->b(cont);
                    });
                }
                release(f.a);
                release(f.b);
                return;
            }
            break;
        default: break;
        }
        Type t = n->type;
        if (t.k != TK::Int && t.k != TK::Any) {
            // List / Fn: always true after evaluation; Bot: never returns
            if (!tir::is_pure(n)) discard_value(n);
            if (t.k != TK::Bot && sense) A->b(target);
            return;
        }
        V v = expr(n);
        if (v.k == V::IMM) {
            bool tr = t.k == TK::Any ? v.imm != 1 : (int32_t)v.imm != 0;
            if (tr == sense) A->b(target);
            return;
        }
        Reg r = R(v, t.k == TK::Any);
        truthy_branch(t, r, sense, target);
        release(v);
    }

    // ------------------------------------------------------------------ statements
    void emit_epilogue() {
        if (!has_frame) {
            A->ret();
            return;
        }
        restore_callee();
        if (lr_off == 0 && frame_size <= 504) {
            A->ldp(X(29), X(30), post(sp, frame_size));
        } else {
            if (lr_off <= 504) A->ldp(X(29), X(30), mem(sp, lr_off));
            else {
                A->ldr(X(29), mem(sp, lr_off));
                A->ldr(X(30), mem(sp, lr_off + 8));
            }
            A->add_imm(sp, sp, frame_size);
        }
        A->ret();
    }

    void stmt(const Node* n) {
        switch (n->op) {
        case Op::Seq:
            for (uint32_t i = 0; i < n->nkids; ++i) {
                if (i + 1 < n->nkids) {
                    uint32_t k = fuse_cond_select(n, i);
                    if (k) {
                        i += k;
                        continue;
                    }
                }
                stmt(n->kids[i]);
            }
            return;
        case Op::If: {
            const Node* e = n->kids[2];
            bool has_else = !(e->op == Op::Seq && e->nkids == 0);
            if (!has_else) {  // `if (c) break`: one conditional branch to the block's end
                const Node* t = n->kids[1];
                while (t->op == Op::Seq && t->nkids == 1) t = t->kids[0];
                if (t->op == Op::Break) {
                    for (size_t i = blocks.size(); i-- > 0;)
                        if (blocks[i].first == t->a) {
                            branch(n->kids[0], true, blocks[i].second);
                            return;
                        }
                }
            }
            Label lelse = A->new_label();
            branch(n->kids[0], false, lelse);
            stmt(n->kids[1]);
            if (has_else) {
                Label lend = A->new_label();
                A->b(lend);
                A->bind(lelse);
                stmt(e);
                A->bind(lend);
            } else {
                A->bind(lelse);
            }
            return;
        }
        case Op::While: {
            Label top = A->new_label(), exit = A->new_label();
            if (!hoists_.empty()) {
                auto it = hoists_.find(n);
                if (it != hoists_.end())
                    for (auto& h : it->second) {
                        hoist_eval_ = true;
                        V v = expr(h.first, L[h.second].reg);
                        hoist_eval_ = false;
                        store_local(h.second, v);
                    }
            }
            branch(n->kids[0], false, exit);
            A->bind(top);
            const Node* body = n->kids[1];
            while (body->op == Op::Seq && body->nkids == 1) body = body->kids[0];
            const Node* step = n->kids[2];
            if (n->kids[0]->op == Op::ConstInt && n->kids[0]->a != 0 && body->op == Op::InlineBlock &&
                step->op == Op::Seq && step->nkids == 0) {
                // While(1) { InlineBlock L {...} } (tail calls as loops): Break L is the back-edge
                blocks.push_back({body->a, top});
                stmt(body->kids[0]);
                blocks.pop_back();
                A->b(top);
                A->bind(exit);
                return;
            }
            stmt(n->kids[1]);
            stmt(n->kids[2]);
            branch(n->kids[0], true, top);
            A->bind(exit);
            return;
        }
        case Op::Return: {
            const Node* v = n->kids[0];
            if (F->is_main) {
                discard(v);
            } else {
                Type t = F->ret;
                if (t.k == TK::Fn || t.k == TK::Bot) {
                    discard(v);
                } else {
                    bool x64 = is64(t);
                    V r = expr(v, -1);
                    if (r.k == V::IMM) {
                        if (x64) li(X(0), r.imm);
                        else li(W(0), (int32_t)r.imm);
                    } else if (r.k == V::NONE) {
                        A->movz(W(0), 0);
                    } else {
                        Reg s = R(r, x64);
                        if (s.id != 0) A->mov(x64 ? X(0) : W(0), s);
                    }
                    release(r);
                }
            }
            emit_epilogue();
            return;
        }
        case Op::Print: {
            const Node* v = n->kids[0];
            Type t = v->type;
            V x = expr(v);
            if (t.k == TK::Bot) {
                release(x);
                return;
            }
            if (t.k == TK::Int) {
                std::initializer_list<Move> mv{{0, x, false}};
                helper_call(H_PRINT_INT, mv, false);
            } else {
                V w = word_of(t, x);
                std::initializer_list<Move> mv{{0, V::im(n->a), false}, {1, w, true}};
                helper_call(H_PRINT, mv, false);
            }
            return;
        }
        case Op::InlineBlock: {
            Label end = A->new_label();
            blocks.push_back({n->a, end});
            stmt(n->kids[0]);
            blocks.pop_back();
            A->bind(end);
            return;
        }
        case Op::Break: {
            for (size_t i = blocks.size(); i-- > 0;)
                if (blocks[i].first == n->a) {
                    A->b(blocks[i].second);
                    return;
                }
            bug("Break outside its InlineBlock");
        }
        case Op::Free: free_list(n); return;
        case Op::Reserve: discard_value(n); return;
        case Op::Nop: return;
        default:
            if (tir::is_pure(n)) return;
            discard_value(n);
            return;
        }
    }
    std::vector<std::pair<int32_t, Label>> blocks;

    // ------------------------------------------------------------------ functions
    void reset_body_state() {
        temp_free = temp_pool;
        spill_free = spill_pool;
        slot_busy.clear();
        max_slots = nfixed;
        colds.clear();
        colds.reserve(32);
        for (auto& e : err_label) e = Label{};
    }
    void save_pair_at(int r1, int r2, uint32_t off, bool load) {
        if (r2 >= 0 && off <= 504) {
            if (load) A->ldp(X(r1), X(r2), mem(sp, off));
            else A->stp(X(r1), X(r2), mem(sp, off));
            return;
        }
        if (load) A->ldr(X(r1), mem(sp, off));
        else A->str(X(r1), mem(sp, off));
        if (r2 >= 0) {
            if (load) A->ldr(X(r2), mem(sp, off + 8));
            else A->str(X(r2), mem(sp, off + 8));
        }
    }
    void callee_regs(std::vector<int>& r) {
        for (int i = 19; i <= 28; ++i)
            if (saved_mask & bit(i)) r.push_back(i);
    }
    void save_callee() {
        std::vector<int> r;
        callee_regs(r);
        for (size_t i = 0; i < r.size(); i += 2)
            save_pair_at(r[i], i + 1 < r.size() ? r[i + 1] : -1, saved_base + 8 * (uint32_t)i, false);
    }
    void restore_callee() {
        std::vector<int> r;
        callee_regs(r);
        for (size_t i = 0; i < r.size(); i += 2)
            save_pair_at(r[i], i + 1 < r.size() ? r[i + 1] : -1, saved_base + 8 * (uint32_t)i, true);
    }

    void emit_prologue() {
        if (!has_frame) return;
        if (lr_off == 0 && frame_size <= 504) {
            A->stp(X(29), X(30), pre(sp, -(int64_t)frame_size));
        } else {
            A->sub_imm(sp, sp, frame_size);
            save_pair_at(29, 30, lr_off, false);
        }
        save_callee();
    }

    void entry_moves() {
        // parameters
        for (size_t i = 0; i < F->params.size(); ++i) {
            LocalInfo& l = L[F->params[i]];
            if (l.fn || (l.reg < 0 && l.slot < 0)) continue;
            Reg dst = l.x64 ? X(l.reg >= 0 ? l.reg : 16) : W(l.reg >= 0 ? l.reg : 16);
            if (i < 8) {
                Reg src = l.x64 ? X((int)i) : W((int)i);
                if (l.reg >= 0) A->mov(dst, src);
                else A->str(src, mem(sp, fix_off(l.slot)));
            } else {
                uint32_t off = frame_size + 8 * (uint32_t)(i - 8);
                if (l.reg >= 0) A->ldr(dst, mem(sp, off));
                else {
                    A->ldr(X(16), mem(sp, off));
                    A->str(X(16), mem(sp, fix_off(l.slot)));
                }
            }
        }
        for (auto& l : L) {
            if (l.param || l.fn || !l.needs_init) continue;
            if (l.is_const) continue;  // (loaded at its loop's entry)
            if (l.reg >= 0) {
                if (l.any) A->movz(X(l.reg), 1);
                else A->movz(X(l.reg), 0);
            } else if (l.slot >= 0) {
                if (l.any) {
                    A->movz(X(16), 1);
                    A->str(X(16), mem(sp, fix_off(l.slot)));
                } else {
                    A->str(xzr, mem(sp, fix_off(l.slot)));
                }
            }
        }
    }

    // Shrink-wrapping for leading `If(c, Return(v))` statements whose condition
    // and value only read parameters (still in x0..x7) through pure integer
    // operations: they run before the prologue and return without a frame.
    bool early_ = false;
    bool simple_param_expr(const Node* n, bool cond) {
        switch (n->op) {
        case Op::ConstInt: return true;
        case Op::Local: {
            const LocalInfo& l = L[(size_t)n->a];
            return l.param && !l.fn && l.param_idx < 8;
        }
        case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::NegI32: case Op::NotI32:
        case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
            for (uint32_t i = 0; i < n->nkids; ++i)
                if (!simple_param_expr(n->kids[i], false)) return false;
            return true;
        case Op::And:
        case Op::Or:
            if (!cond || !(n->flags & tir::F_COND)) return false;
            return simple_param_expr(n->kids[0], true) && simple_param_expr(n->kids[1], true);
        default: return false;
        }
    }
    const Node* early_return_value(const Node* s) {
        if (s->op != Op::If) return nullptr;
        const Node* e = s->kids[2];
        if (!(e->op == Op::Seq && e->nkids == 0)) return nullptr;
        const Node* t = s->kids[1];
        while (t->op == Op::Seq && t->nkids == 1) t = t->kids[0];
        if (t->op != Op::Return) return nullptr;
        if (!simple_param_expr(s->kids[0], true)) return nullptr;
        const Node* v = t->kids[0];
        if (F->ret.k == TK::Fn || F->ret.k == TK::Bot) return nullptr;
        if (!simple_param_expr(v, false)) return nullptr;
        if (v->op == Op::Local && L[(size_t)v->a].x64 != is64(F->ret)) return nullptr;
        return v;
    }
    // Body tail `If(c, T, {}); Return(v)` with simple c and v: `if (!c) return v`
    // runs before the prologue, and the body then starts with T (c known true).
    const Node* tail_return_value(uint32_t k) {
        const Node* b = F->body;
        if (b->nkids != k + 2) return nullptr;
        const Node* s = b->kids[k];
        const Node* r = b->kids[k + 1];
        if (s->op != Op::If || r->op != Op::Return) return nullptr;
        const Node* e = s->kids[2];
        if (!(e->op == Op::Seq && e->nkids == 0)) return nullptr;
        if (!simple_param_expr(s->kids[0], true)) return nullptr;
        if (F->ret.k == TK::Fn || F->ret.k == TK::Bot) return nullptr;
        const Node* v = r->kids[0];
        if (!simple_param_expr(v, false)) return nullptr;
        if (v->op == Op::Local && L[(size_t)v->a].x64 != is64(F->ret)) return nullptr;
        return v;
    }
    bool tail_if_true_ = false;  // body statement `skip` is an If whose condition is known true
    void early_ret(V r, bool x64) {
        if (r.k == V::IMM) {
            if (x64) li(X(0), r.imm);
            else li(W(0), (int32_t)r.imm);
        } else {
            Reg src = R(r, x64);
            if (src.id != 0) A->mov(x64 ? X(0) : W(0), src);
        }
        release(r);
        A->ret();
    }
    uint32_t early_returns() {
        tail_if_true_ = false;
        if (F->is_main || !hot_calls || F->body->op != Op::Seq) return 0;
        uint32_t k = 0;
        while (k < F->body->nkids && early_return_value(F->body->kids[k])) ++k;
        const Node* tail = tail_return_value(k);
        if (!k && !tail) return 0;
        early_ = true;
        uint32_t argregs = 0;
        for (size_t i = 0; i < F->params.size() && i < 8; ++i) argregs |= bit((int)i);
        temp_free = temp_pool & ~argregs;
        if (nfree() < 5) {
            early_ = false;
            temp_free = temp_pool;
            return 0;
        }
        bool x64 = is64(F->ret);
        for (uint32_t i = 0; i < k; ++i) {
            const Node* s = F->body->kids[i];
            Label skip = A->new_label();
            branch(s->kids[0], false, skip);
            V r = expr(early_return_value(s));
            if (r.k == V::IMM) {
                if (x64) li(X(0), r.imm);
                else li(W(0), (int32_t)r.imm);
            } else {
                Reg src = R(r, x64);
                if (src.id != 0) A->mov(x64 ? X(0) : W(0), src);
            }
            release(r);
            A->ret();
            A->bind(skip);
        }
        if (tail) {
            Label go = A->new_label();
            branch(F->body->kids[k]->kids[0], true, go);
            early_ret(expr(tail), x64);
            A->bind(go);
            tail_if_true_ = true;
        }
        early_ = false;
        temp_free = temp_pool;
        return k;
    }

    void gen_body() {
        reset_body_state();
        uint32_t skip = early_returns();
        emit_prologue();
        entry_moves();
        if (tail_if_true_) {
            stmt(F->body->kids[skip]->kids[1]);
            for (uint32_t i = skip + 1; i < F->body->nkids; ++i) stmt(F->body->kids[i]);
        } else if (skip) {
            for (uint32_t i = skip; i < F->body->nkids; ++i) stmt(F->body->kids[i]);
        } else {
            stmt(F->body);
        }
        if (F->is_main) emit_epilogue();
        // a non-main function always ends in Return (builder guarantee); guard:
        else A->brk(1);
        for (size_t i = 0; i < colds.size(); ++i) {
            A->bind(colds[i].l);
            colds[i].f();
        }
        for (int e = 0; e < kErrCount; ++e) {
            if (!err_label[e].valid()) continue;
            A->bind(err_label[e]);
            A->movz(W(0), (uint32_t)e);
            A->b(ext_helper(H_FAIL));
        }
    }

    void layout_frame() {
        saved_mask = used_callee;
        uint32_t nsaved = (uint32_t)__builtin_popcount(saved_mask);
        uint32_t slots_bytes = (8u * (uint32_t)max_slots + 15) & ~15u;
        uint32_t saved_bytes = (8u * nsaved + 15) & ~15u;
        has_frame = hot_calls || max_slots > 0 || nsaved > 0 || out_bytes > 0;
        if (!has_frame) {
            frame_size = 0;
            return;
        }
        frame_size = out_bytes + slots_bytes + saved_bytes + 16;
        if (out_bytes == 0) {
            lr_off = 0;
            slot_base = 16;
            saved_base = 16 + slots_bytes;
        } else {
            slot_base = out_bytes;
            saved_base = out_bytes + slots_bytes;
            lr_off = saved_base + saved_bytes;
        }
    }

    // A dry run into a scratch assembler measures the frame (dynamic slots,
    // callee-saved registers used for held values) before the real run. When
    // the prepass sees no value held across a call (risky_ false), the frame
    // from register allocation is tried directly; if pressure spills prove it
    // wrong, the attempt is abandoned (left as dead code) and redone.
    void compile_func(const Func& f) {
        F = &f;
        risky_ = false;
        analyze();
        if (!risky_ && !getenv_always_dry_) {
            dry = false;
            A = &main_as;
            uint32_t guess_callee = used_callee;
            max_slots = nfixed;
            layout_frame();
            func_off[f.id] = A->pos();
            size_t sym = syms.size();
            syms.push_back({A->pos(), f.name});
            gen_body();
            if (used_callee == guess_callee && max_slots <= nfixed) {
                ++single_pass_;
                return;
            }
            // redo with the measured frame
            syms[sym].second = f.name + "_abandoned";
            uint32_t mc = used_callee;
            int ms = max_slots;
            used_callee = mc;
            max_slots = ms;
            layout_frame();
            uint32_t fixed_callee = used_callee;
            int fixed_slots = max_slots;
            func_off[f.id] = A->pos();
            syms.push_back({A->pos(), f.name});
            gen_body();
            if (used_callee != fixed_callee || max_slots > fixed_slots) bug("redo mismatch");
            ++redone_;
            return;
        }
        // dry run: measure the frame
        dry = true;
        scratch = std::make_unique<Assembler>();
        A = scratch.get();
        dummy = A->new_label();
        slot_base = 16;
        // provisional layout for offsets (values are irrelevant in the dry run)
        has_frame = true;
        frame_size = 1024;
        lr_off = 0;
        saved_mask = 0;
        saved_base = 16;
        if (out_bytes) slot_base = out_bytes;
        gen_body();
        uint32_t measured_callee = used_callee;
        int measured_slots = max_slots;
        // real run
        dry = false;
        A = &main_as;
        used_callee = measured_callee;
        max_slots = measured_slots;
        layout_frame();
        uint32_t fixed_callee = used_callee;
        int fixed_slots = max_slots;
        func_off[f.id] = A->pos();
        syms.push_back({A->pos(), f.name});
        gen_body();
        if (used_callee != fixed_callee || max_slots > fixed_slots) bug("dry run mismatch");
        used_callee = fixed_callee;
    }

    void emit_wrappers() {
        wrapper_off.assign(M.defs.size(), -1);
        for (size_t d = 0; d < M.defs.size(); ++d) {
            int32_t gf = M.defs[d].generic_func;
            if (gf < 0) continue;
            const Func& f = *M.funcs[(size_t)gf];
            size_t n = f.params.size();
            A->align(8);
            wrapper_off[d] = (int32_t)A->pos();
            syms.push_back({A->pos(), "gw_" + f.name});
            if (n <= 8) {
                A->mov(X(16), X(0));
                for (size_t i = 0; i < n; ++i) A->ldr(X((int)i), mem(X(16), 8 * (int64_t)i));
                call_func(f.id, true);
            } else {
                uint32_t out = (uint32_t)((8 * (n - 8) + 15) & ~(size_t)15);
                A->stp(X(29), X(30), pre(sp, -16));
                A->sub_imm(sp, sp, out);
                A->mov(X(16), X(0));
                for (size_t i = 8; i < n; ++i) {
                    A->ldr(X(17), mem(X(16), 8 * (int64_t)i));
                    A->str(X(17), mem(sp, 8 * (int64_t)(i - 8)));
                }
                for (size_t i = 0; i < 8; ++i) A->ldr(X((int)i), mem(X(16), 8 * (int64_t)i));
                call_func(f.id);
                A->add_imm(sp, sp, out);
                A->ldp(X(29), X(30), post(sp, 16));
                A->ret();
            }
        }
    }

    void emit_helper_stubs() {
        for (int h = 0; h < H_COUNT; ++h) {
            if (!helper_used[h]) continue;
            A->align(8);
            A->bind(helper_label[h]);
            syms.push_back({A->pos(), helper_name(h)});
            A->ldr_lit_off(X(16), 8);
            A->br(X(16));
            A->emit_u64((uint64_t)(uintptr_t)helper_addr(h));
        }
    }

    bool setup_data() {
        size_t G = M.globals.size();
        gval_off.assign(G, 0);
        gdef_off.assign(G, 0);
        size_t off = 0;
        for (size_t g = 0; g < G; ++g) {
            gval_off[g] = (uint32_t)off;
            off += 8;
        }
        for (size_t g = 0; g < G; ++g) gdef_off[g] = (uint32_t)(off++);
        off = (off + 15) & ~(size_t)15;
        bump_off = (uint32_t)off;
        off += 8;
        fl_off = (uint32_t)off;
        off += 8;
        fntab_off = (uint32_t)off;
        off += 16 * M.defs.size();
        data_size = (off + 16383) & ~(size_t)16383;
        if (data_size == 0) data_size = 16384;
        void* hint = region.base() + region.reserved();
        void* p = mmap(hint, data_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) return false;
        data = (uint8_t*)p;
        int64_t dist = (int64_t)((uintptr_t)data >> 12) - (int64_t)((uintptr_t)region.base() >> 12);
        int64_t span = (int64_t)(region.reserved() >> 12);
        data_near = dist - span > -(1 << 20) + 16 && dist + span < (1 << 20) - 16;
        char** bp = jrt_bump_ptr();
        bump_adjacent = jrt_bump_end_ptr() == bp + 1;
        std::memcpy(data + bump_off, &bp, 8);
        void** flp = jrt_fl_small_ptr();
        std::memcpy(data + fl_off, &flp, 8);
        poison = jrt_poison_on() != 0;
        for (size_t g = 0; g < G; ++g) {
            if (M.globals[g].type.k == TK::Any) {
                uint64_t one = 1;
                std::memcpy(data + gval_off[g], &one, 8);
            }
        }
        return true;
    }

    uint8_t* code = nullptr;
    size_t code_size = 0;
    std::vector<tir::Desc> descs_rt;
    struct FnInfo {
        int32_t arity;
        uint64_t fn;
    };
    std::vector<FnInfo> fns_rt;

    uint64_t tm[6] = {};
    bool compile() {
        tm[0] = now_us();
        if (!region.init()) return false;
        tm[1] = now_us();
        if (!setup_data()) return false;
        tm[2] = now_us();
        {
            size_t nn = 0;
            for (auto& f : M.funcs) nn += count_nodes(f->body);
            main_as.reserve(std::max<size_t>(4096, nn * 24));
        }
        func_off.assign(M.funcs.size(), 0);
        getenv_always_dry_ = getenv("EPQ_JIT_ALWAYS_DRY") != nullptr;
        getenv_no_coalesce_ = getenv("EPQ_JIT_NO_COALESCE") != nullptr;
        no_hoist_ = getenv("EPQ_JIT_NO_HOIST") != nullptr;
        getenv_coarse_loops_ = getenv("EPQ_JIT_COARSE_LOOPS") != nullptr;
        for (auto& f : M.funcs) compile_func(*f);
        F = nullptr;
        emit_rec_stubs();
        emit_wrappers();
        emit_helper_stubs();
        patch_calls();
        tm[3] = now_us();
        code = region.install(main_as);
        tm[4] = now_us();
        code_size = main_as.size();
        if (code != region.base()) bug("code not at region base");
        // runtime tables
        descs_rt = M.descs;
        fns_rt.assign(M.defs.size(), FnInfo{0, 0});
        for (size_t d = 0; d < M.defs.size(); ++d) {
            fns_rt[d].arity = (int32_t)M.defs[d].arity;
            if (wrapper_off[d] >= 0) fns_rt[d].fn = (uint64_t)(uintptr_t)(code + wrapper_off[d]);
        }
        jrt_set_tables(descs_rt.data(), fns_rt.data());
        for (size_t d = 0; d < M.defs.size(); ++d) {
            int32_t ar = (int32_t)M.defs[d].arity;
            uint64_t e = M.defs[d].generic_func >= 0 ? (uint64_t)(uintptr_t)(code + func_off[(size_t)M.defs[d].generic_func]) : 0;
            std::memcpy(data + fntab_off + 16 * d, &ar, 4);
            std::memcpy(data + fntab_off + 16 * d + 8, &e, 8);
        }
        return true;
    }
};

uint64_t run_main(void* p) {
    reinterpret_cast<void (*)(void)>(p)();
    return 0;
}

void on_overflow() {
    jrt_flush();
    static const char msg[] = "Stack overflow\n";
    (void)!::write(2, msg, sizeof msg - 1);
    _exit(1);
}

}  // namespace

// ---- support check ---------------------------------------------------------------
// Whitelists of what this code generator implements. A TIR op or type kind
// added later lands in `default` and makes the module unsupported: the driver
// then reports it (--engine=jit) or falls back to the C engine (--engine=auto),
// never a silent miscompilation. To add an op: list it here, give it a case in
// Compiler::expr (values) and/or Compiler::stmt (statements), in the prepass
// (Compiler::kid / op_calls if it may call) and in cond() if it has a branch form.
namespace {
bool op_supported(Op o) {
    switch (o) {
    case Op::Seq: case Op::If: case Op::While: case Op::Return: case Op::Print:
    case Op::ConstInt: case Op::ConstFn: case Op::Local: case Op::GetGlobal: case Op::Input:
    case Op::SetLocal: case Op::SetGlobal:
    case Op::Box: case Op::UnboxInt: case Op::Coerce: case Op::Comma: case Op::Fail:
    case Op::AddI32: case Op::SubI32: case Op::MulI32: case Op::DivI32: case Op::NegI32: case Op::AbsI32:
    case Op::NotI32: case Op::EqI32: case Op::NeI32: case Op::LtI32: case Op::LeI32: case Op::GtI32: case Op::GeI32:
    case Op::DynArith: case Op::DynNeg: case Op::DynAbs: case Op::DynNot:
    case Op::Eq: case Op::Ne: case Op::Cmp:
    case Op::ListLit: case Op::Len: case Op::Index: case Op::SetIndex: case Op::Append: case Op::Pop:
    case Op::DynIndex: case Op::DynSetIndex: case Op::DynAppend:
    case Op::And: case Op::Or: case Op::CallDirect: case Op::CallDyn:
    case Op::Select: case Op::InlineBlock: case Op::Break:
    case Op::PairMake: case Op::PairFirst: case Op::PairSecond: case Op::PairIndex:
    case Op::PairEq: case Op::PairNe: case Op::PairLt: case Op::PairLe: case Op::PairGt: case Op::PairGe:
    case Op::ListData: case Op::ElemLoad: case Op::ElemStore: case Op::CheckIndex:
    case Op::Free: case Op::SameList: case Op::Nop:
    case Op::DynIsEmptyList:  // litcmp
    case Op::Reserve: case Op::AppendRange:  // appendopt
    case Op::RecLit: case Op::RecNull: case Op::RecGet: case Op::RecSet: case Op::RecLen:  // records
        return true;
    default: return false;
    }
}
bool type_supported(Type t) {
    switch (t.k) {
    case TK::Bot: case TK::Int: case TK::List: case TK::Fn: case TK::Any: case TK::Pair: case TK::Ptr: case TK::Rec: return true;
    default: return false;
    }
}
bool sub_op_ok(const Node* n) {
    if (n->op == Op::DynArith) {
        Op s = (Op)n->a;
        return s == Op::AddI32 || s == Op::SubI32 || s == Op::MulI32 || s == Op::DivI32;
    }
    if (n->op == Op::Cmp) {
        Op s = (Op)n->c;
        return s == Op::LtI32 || s == Op::LeI32 || s == Op::GtI32 || s == Op::GeI32;
    }
    return true;
}
bool check_tree(const Node* n, std::string& why) {
    if (!op_supported(n->op)) {
        why = std::string("TIR op ") + tir::op_name(n->op);
        return false;
    }
    if (!sub_op_ok(n)) {
        why = std::string("operator of TIR op ") + tir::op_name(n->op);
        return false;
    }
    if (!type_supported(n->type)) {
        why = std::string("type kind ") + std::to_string((int)n->type.k) + " (op " + tir::op_name(n->op) + ")";
        return false;
    }
    for (uint32_t i = 0; i < n->nkids; ++i)
        if (!check_tree(n->kids[i], why)) return false;
    return true;
}
}  // namespace

// EPQ_JIT_PRETEND_UNSUPPORTED=OPNAME (test hook): treat that op as unsupported.
static const char* pretend_unsupported_ = nullptr;
static bool check_tree_pretend(const Node* n) {
    if (!std::strcmp(tir::op_name(n->op), pretend_unsupported_)) return false;
    for (uint32_t i = 0; i < n->nkids; ++i)
        if (!check_tree_pretend(n->kids[i])) return false;
    return true;
}

std::string unsupported(const Module& m) {
    std::string why;
    pretend_unsupported_ = getenv("EPQ_JIT_PRETEND_UNSUPPORTED");
    if (pretend_unsupported_)
        for (auto& f : m.funcs)
            if (f->body && !check_tree_pretend(f->body))
                return std::string("TIR op ") + pretend_unsupported_ + " in function " + f->name + " (pretend)";
    for (auto& r : m.regions)
        if (!type_supported(r.elem)) return "region element type kind " + std::to_string((int)r.elem.k);
    for (auto& g : m.globals)
        if (!type_supported(g.type)) return "global type kind " + std::to_string((int)g.type.k);
    for (auto& f : m.funcs) {
        if (!type_supported(f->ret)) return "return type kind of " + f->name;
        for (auto& l : f->locals)
            if (!type_supported(l.type)) return "local type kind in " + f->name;
        if (f->body && !check_tree(f->body, why)) return why + " in function " + f->name;
    }
    return "";
}

int run(const Module& m, const JitConfig& cfg) {
    static_assert(sizeof(tir::Desc) == 8, "EDesc layout");
    if (getenv("EPQ_JIT_QOS")) pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint64_t t0 = now_us();
    if (const char* b = getenv("EPQ_JIT_BENCH")) {  // diagnostic: repeated compiles
        int n = atoi(b);
        uint64_t best = UINT64_MAX;
        for (int i = 0; i < n; ++i) {
            uint64_t a = now_us();
            Compiler* X = new Compiler(m);
            X->compile();
            best = std::min(best, now_us() - a);
            delete X;
        }
        std::fprintf(stderr, "jit: best of %d compiles: %llu us\n", n, (unsigned long long)best);
    }
    Compiler* C = new Compiler(m);  // never freed: the process exits
    if (!C->compile()) {
        std::fprintf(stderr, "epq: jit: cannot allocate code memory\n");
        return 2;
    }
    uint64_t t1 = now_us();
    if (cfg.stats) {
        std::fprintf(stderr, "jit: compile %llu us, code %zu bytes, %zu functions\n", (unsigned long long)(t1 - t0),
                     C->code_size, m.funcs.size());
        if (cfg.verbose)
            std::fprintf(stderr,
                         "jit: region %llu us, data %llu us, codegen %llu us, install %llu us; "
                         "%d single-pass, %d redone, %d with dry run\n",
                         (unsigned long long)(C->tm[1] - C->tm[0]), (unsigned long long)(C->tm[2] - C->tm[1]),
                         (unsigned long long)(C->tm[3] - C->tm[2]), (unsigned long long)(C->tm[4] - C->tm[3]),
                         C->single_pass_, C->redone_, (int)m.funcs.size() - C->single_pass_ - C->redone_);
    }
    if (cfg.dump_asm) {
        std::string s = jit::disassemble(C->code, C->code_size, C->syms, (uint64_t)(uintptr_t)C->code);
        (void)!::write(1, s.data(), s.size());
        return 0;
    }
    jit::BigStack* st = new jit::BigStack();
    jit::set_overflow_callback(on_overflow);
    void* entry = C->code + C->func_off[0];
    uint64_t e0 = now_us();
    if (!getenv("EPQ_JIT_NOSTACK") && st->create(size_t(8) << 30)) {
        e0 = now_us();
        st->run(run_main, entry);
    } else {
        run_main(entry);
    }
    uint64_t e1 = now_us();
    jrt_flush();
    if (cfg.time) {
        char buf[80];
        int n = std::snprintf(buf, sizeof buf, "Execution time: %llu microseconds\n", (unsigned long long)(e1 - e0));
        (void)!::write(2, buf, (size_t)n);
    }
    return 0;
}

}  // namespace epq::jitcg
