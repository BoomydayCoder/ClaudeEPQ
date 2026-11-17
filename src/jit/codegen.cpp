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

#include <sys/mman.h>
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
    }
    return nullptr;
}
const char* helper_name(int h) {
    static const char* n[] = {"rt_fail", "rt_print_int", "rt_print", "rt_read_int", "rt_list_new", "rt_list_grow",
                              "rt_eq", "rt_lt", "rt_eq_li", "rt_lt_li", "rt_dyn_neg", "rt_dyn_abs", "rt_call_dyn"};
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
};

struct Ref {
    uint32_t pos;
    uint32_t ctx;
    bool write;
};
struct LoopRec {
    uint32_t en, lv, cond_ctx, body_ctx;
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
    std::vector<Ref> refs;
};

bool is64(Type t) { return t.k == TK::List || t.k == TK::Any; }

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

    std::vector<Label> func_label;
    Label helper_label[H_COUNT];
    bool helper_used[H_COUNT] = {};
    Label dummy;  // dry run: stands for every external label
    std::vector<std::pair<size_t, std::string>> syms;
    std::vector<size_t> func_off;
    std::vector<int32_t> wrapper_off;  // by DefId

    // ---- per function ----
    const Func* F = nullptr;
    std::unordered_map<const Node*, NInfo> info;
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
        auto it = info.find(n);
        if (it == info.end()) bug("node without info");
        return it->second;
    }
    Label ext_func(uint32_t f) { return dry ? dummy : func_label[f]; }
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
            A->load_imm(t.xr(), (int64_t)target);
            *lo12 = 0;
        }
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
            if (x64) A->load_imm(X(r), v.imm);
            else A->load_imm(W(r), (int32_t)v.imm);
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
        case Op::ListLit:
        case Op::DynNeg:
        case Op::DynAbs: return true;
        case Op::Eq:
        case Op::Ne:
        case Op::Cmp: return !cmp_inline(n);
        default: return false;
        }
    }
    void prepass(const Node* n, int depth) {
        uint32_t en = pos_++;
        bool call = false;
        auto kid = [&](uint32_t i, int d) {
            prepass(n->kids[i], d);
            call |= info[n->kids[i]].call;
        };
        switch (n->op) {
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
            LocalInfo& l = L[(size_t)n->a];
            if (!l.fn) {
                l.refs.push_back(Ref{pos_, ctx_, true});
                l.weight += wdepth(depth);
            }
            break;
        }
        case Op::While: {
            LoopRec lr{en, 0, ctx_, 0};
            kid(0, depth + 1);
            push_ctx();
            lr.body_ctx = ctx_;
            kid(1, depth + 1);
            kid(2, depth + 1);
            pop_ctx();
            lr.lv = pos_;
            loops.push_back(lr);
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
        case Op::CallDirect: {
            uint32_t na = n->nkids - 1;
            if (na > 8) out_bytes = std::max(out_bytes, (uint32_t)((8 * (na - 8) + 15) & ~15u));
            for (uint32_t i = 0; i < n->nkids; ++i) kid(i, depth);
            break;
        }
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) kid(i, depth);
            break;
        }
        uint32_t lv = pos_++;
        if (op_calls(n)) {
            call = true;
            call_pos.push_back(n->op == Op::ListLit ? en : lv);
        }
        info[n] = NInfo{en, lv, call};
    }

    // Definite assignment (for zero-initialisation): bits = assigned locals.
    using Bits = std::vector<uint64_t>;
    static void set_all(Bits& b) { std::fill(b.begin(), b.end(), ~0ull); }
    static void band(Bits& a, const Bits& b) {
        for (size_t i = 0; i < a.size(); ++i) a[i] &= b[i];
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
        default:
            for (uint32_t i = 0; i < n->nkids; ++i) da(n->kids[i], b);
            return;
        }
    }

    void analyze() {
        info.clear();
        info.reserve(1024);
        loops.clear();
        call_pos.clear();
        pos_ = 0;
        ctx_ = 0;
        ctx_next_ = 0;
        ctx_stack_.clear();
        out_bytes = 0;
        L.assign(F->locals.size(), LocalInfo{});
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
        {
            Bits b((F->locals.size() + 63) / 64 + 1, 0);
            for (uint32_t p : F->params) b[p >> 6] |= 1ull << (p & 63);
            da(F->body, b);
        }
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
        for (auto& lp : loops) {
            for (auto& l : L) {
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
                bool uncond_def = first->write && (first->ctx == lp.body_ctx || first->ctx == lp.cond_ctx);
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
            } else {
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
        LocalInfo& l = L[id];
        if (l.fn) return V::none();
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
    void parallel_move(std::vector<Move> mv) {
        std::vector<V> srcs;
        for (auto& m : mv) srcs.push_back(m.src);
        parallel_move_(mv);
        for (auto& v : srcs) release(v);
    }
    void parallel_move_(std::vector<Move>& mv) {
        // stack destinations first (they read sources, write memory)
        for (auto& m : mv) {
            if (m.dst >= 0) continue;
            uint32_t off = 8u * (uint32_t)(-1 - m.dst);
            switch (m.src.k) {
            case V::REG: A->str(X(m.src.r), mem(sp, off)); break;
            case V::IMM:
                A->load_imm(X(16), m.src.imm);
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
        std::vector<Move*> pend;
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
                if (m.x64) A->load_imm(X(m.dst), m.src.imm);
                else A->load_imm(W(m.dst), (int32_t)m.src.imm);
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
    V helper_call(int h, std::vector<Move> args, bool has_result) {
        parallel_move(args);
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
        std::vector<int> r;
        for (int i = 0; i < 31; ++i)
            if (m & bit(i)) r.push_back(i);
        for (size_t i = 0; i < r.size(); i += 2) {
            if (i + 1 < r.size()) A->stp(X(r[i]), X(r[i + 1]), pre(sp, -16));
            else A->str(X(r[i]), pre(sp, -16));
        }
    }
    void restore_regs(uint32_t m) {
        m |= bit(30);
        std::vector<int> r;
        for (int i = 0; i < 31; ++i)
            if (m & bit(i)) r.push_back(i);
        size_t n = r.size();
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
        case Op::ConstInt: return V::im(n->a);
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
        case Op::DynNeg: {
            V v = expr(n->kids[0]);
            std::vector<Move> mv{{0, v, true}};
            return helper_call(H_DYN_NEG, mv, true);
        }
        case Op::DynAbs: {
            V v = expr(n->kids[0]);
            std::vector<Move> mv{{0, v, true}};
            return helper_call(H_DYN_ABS, mv, true);
        }
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
        case Op::Print: stmt(n); return V::none();
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
        uint32_t id = (uint32_t)n->a;
        LocalInfo& l = L[id];
        if (l.fn) {
            discard_value(n->kids[0]);
            return V::none();
        }
        if (l.reg >= 0) {
            V v = expr(n->kids[0], l.reg);
            if (!(v.k == V::REG && v.r == l.reg)) {
                if (v.k == V::IMM) {
                    if (l.x64) A->load_imm(X(l.reg), v.imm);
                    else A->load_imm(W(l.reg), (int32_t)v.imm);
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
    V arith(const Node* n, int want) {
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
        bool xs = tx == TK::List || tx == TK::Fn, ys = ty == TK::List || ty == TK::Fn;
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
    // Non-inline Eq/Ne/Cmp: result in a fresh register (helper call).
    V deep_cmp_call(const Node* n) {
        V k[2];
        eval_kids(n, 0, 2, k);
        Op c = n->op == Op::Cmp ? (Op)n->c : n->op;
        bool swap = c == Op::GtI32 || c == Op::LeI32;
        bool neg = n->op == Op::Ne || c == Op::LeI32 || c == Op::GeI32;
        if (li_fast(n)) {
            int h = (n->op == Op::Eq || n->op == Op::Ne) ? H_EQ_LI : H_LT_LI;
            std::vector<Move> mv{{swap ? 1 : 0, k[0], true}, {swap ? 0 : 1, k[1], true}};
            V r = helper_call(h, mv, true);
            if (neg) A->eor(W(r.r), W(r.r), (uint64_t)1);
            return r;
        }
        V wa = word_of(n->kids[0]->type, k[0]);
        V wb = word_of(n->kids[1]->type, k[1]);
        int h = (n->op == Op::Eq || n->op == Op::Ne) ? H_EQ : H_LT;
        uint32_t da = (uint32_t)n->a, db = (uint32_t)n->b;
        std::vector<Move> mv{{0, V::im(swap ? db : da), false},
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
    FastCmp fast_cmp(const Node* n) {
        V k[2];
        eval_kids(n, 0, 2, k);
        Type ta = n->kids[0]->type, tb = n->kids[1]->type;
        FastCmp f;
        Op c = n->op == Op::Cmp ? (Op)n->c : n->op == Op::Eq ? Op::EqI32 : Op::NeI32;
        f.c = cond_of(c);
        f.a = boxed_of(ta, k[0]);
        f.b = boxed_of(tb, k[1]);
        f.slow = A->new_label();
        f.cont = A->new_label();
        bool aa = ta.k == TK::Any, ab = tb.k == TK::Any;
        if (f.a.k == V::IMM && f.b.k == V::IMM) {
            // both constant ints (cannot happen for Any): compare words directly
        }
        Reg ra = R(f.a, true);
        Reg rb = f.b.k == V::IMM ? xzr : R(f.b, true);
        if (aa && ab) {
            A->and_(X(16), ra, rb);
            A->tbz(X(16), 0, f.slow);
        } else if (aa) {
            A->tbz(ra, 0, f.slow);
        } else if (ab) {
            A->tbz(rb, 0, f.slow);
        }
        if (f.b.k == V::IMM) A->cmp_imm(ra, f.b.imm);
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
    V deep_cmp_value(const Node* n, int want) {
        if (!cmp_inline(n)) {
            V r = deep_cmp_call(n);
            return r;
        }
        FastCmp f = fast_cmp(n);
        Reg ra = R(f.a, true);
        Reg rb = f.b.k == V::IMM ? X(17) : R(f.b, true);
        int64_t bimm = f.b.k == V::IMM ? f.b.imm : 0;
        bool bimm_p = f.b.k == V::IMM;
        int d = dest(want);
        A->cset(W(d), f.c);
        A->bind(f.cont);
        uint32_t save = live_caller_saved(bit(d));
        save &= ~bit(d);
        Label slow = f.slow, cont = f.cont;
        cold(slow, [=, this]() {
            if (bimm_p) A->load_imm(X(17), bimm);
            fast_cmp_slow_call(n, ra, rb, save);
            A->mov(W(d), W(16));
            A->b(cont);
        });
        release(f.a);
        release(f.b);
        return result(d, want);
    }

    // ---- lists
    uint32_t esz_of(uint32_t rg) { return M.elem_is_i32(rg) ? 4 : 8; }

    V list_lit(const Node* n) {
        uint32_t r = n->type.a;
        uint32_t esz = esz_of(r);
        check_no_temps("listlit");
        std::vector<Move> mv{{0, V::im(n->nkids), false}, {1, V::im(esz), false}};
        V lst = helper_call(H_LIST_NEW, mv, true);
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
        if (esz == 8 && n->kids[2]->type.k != TK::Any && n->kids[2]->type.k != TK::List) {
            // value of SetIndex is v (of its static type); word_of made a copy
            release(w);
            return n->kids[2]->type.k == TK::Fn ? V::none() : k[2];
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
                if (x64) A->load_imm(X(dr), b.imm);
                else A->load_imm(W(dr), (int32_t)b.imm);
            } else if (b.k != V::NONE) {
                Reg s = R(b, x64);
                A->mov(x64 ? X(dr) : W(dr), s);
            }
            release(b);
        }
        A->bind(end);
        return V::reg(dr, true);
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
        A->bl(ext_func(callee->id));
        Type rt = callee->ret;
        if (rt.k == TK::Fn) return V::none();
        if (rt.k == TK::Bot) return V::im(0);
        return V::reg(talloc_specific(0), true);
    }

    V call_dyn(const Node* n) {
        uint32_t na = n->nkids - 1;
        int base = na ? salloc_n((int)na) : 0;
        V f;
        for (uint32_t i = 0; i < n->nkids; ++i) {
            V v = expr(n->kids[i]);
            if (i < na) {
                if (v.k == V::IMM) {
                    A->load_imm(X(16), v.imm);
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
        case Op::EqI32:
        case Op::NeI32:
        case Op::LtI32:
        case Op::LeI32:
        case Op::GtI32:
        case Op::GeI32: {
            // cbz/cbnz for comparisons with 0 (eq/ne)
            const Node* a = n->kids[0];
            const Node* b = n->kids[1];
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
        case Op::Eq:
        case Op::Ne:
        case Op::Cmp:
            if (cmp_inline(n)) {
                FastCmp f = fast_cmp(n);
                A->b(sense ? f.c : invert(f.c), target);
                A->bind(f.cont);
                Reg ra = R(f.a, true);
                Reg rb = f.b.k == V::IMM ? X(17) : R(f.b, true);
                int64_t bimm = f.b.k == V::IMM ? f.b.imm : 0;
                bool bimm_p = f.b.k == V::IMM;
                uint32_t save = live_caller_saved(0);
                Label cont = f.cont;
                cold(f.slow, [=, this]() {
                    if (bimm_p) A->load_imm(X(17), bimm);
                    fast_cmp_slow_call(n, ra, rb, save);
                    if (sense) A->cbnz(W(16), target);
                    else A->cbz(W(16), target);
                    A->b(cont);
                });
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
            for (uint32_t i = 0; i < n->nkids; ++i) stmt(n->kids[i]);
            return;
        case Op::If: {
            const Node* e = n->kids[2];
            bool has_else = !(e->op == Op::Seq && e->nkids == 0);
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
            branch(n->kids[0], false, exit);
            A->bind(top);
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
                        if (x64) A->load_imm(X(0), r.imm);
                        else A->load_imm(W(0), (int32_t)r.imm);
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
                std::vector<Move> mv{{0, x, false}};
                helper_call(H_PRINT_INT, mv, false);
            } else {
                V w = word_of(t, x);
                std::vector<Move> mv{{0, V::im(n->a), false}, {1, w, true}};
                helper_call(H_PRINT, mv, false);
            }
            return;
        }
        default:
            if (tir::is_pure(n)) return;
            discard_value(n);
            return;
        }
    }

    // ------------------------------------------------------------------ functions
    void reset_body_state() {
        temp_free = temp_pool;
        spill_free = spill_pool;
        slot_busy.clear();
        max_slots = nfixed;
        colds.clear();
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

    void gen_body() {
        reset_body_state();
        emit_prologue();
        entry_moves();
        stmt(F->body);
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

    void compile_func(const Func& f) {
        F = &f;
        analyze();
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
        A->bind(func_label[f.id]);
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
                A->b(func_label[f.id]);
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
                A->bl(func_label[f.id]);
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
        data_size = (off + 16383) & ~(size_t)16383;
        if (data_size == 0) data_size = 16384;
        void* hint = region.base() + region.reserved();
        void* p = mmap(hint, data_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) return false;
        data = (uint8_t*)p;
        int64_t dist = (int64_t)((uintptr_t)data >> 12) - (int64_t)((uintptr_t)region.base() >> 12);
        int64_t span = (int64_t)(region.reserved() >> 12);
        data_near = dist - span > -(1 << 20) + 16 && dist + span < (1 << 20) - 16;
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

    bool compile() {
        if (!region.init()) return false;
        if (!setup_data()) return false;
        main_as.reserve(1 << 16);
        func_label.resize(M.funcs.size());
        func_off.assign(M.funcs.size(), 0);
        for (auto& f : M.funcs) func_label[f->id] = main_as.new_label();
        for (auto& f : M.funcs) compile_func(*f);
        F = nullptr;
        emit_wrappers();
        emit_helper_stubs();
        code = region.install(main_as);
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

int run(const Module& m, const JitConfig& cfg) {
    static_assert(sizeof(tir::Desc) == 8, "EDesc layout");
    uint64_t t0 = now_us();
    Compiler* C = new Compiler(m);  // never freed: the process exits
    if (!C->compile()) {
        std::fprintf(stderr, "epq: jit: cannot allocate code memory\n");
        return 2;
    }
    uint64_t t1 = now_us();
    if (cfg.stats)
        std::fprintf(stderr, "jit: compile %llu us, code %zu bytes, %zu functions\n", (unsigned long long)(t1 - t0),
                     C->code_size, m.funcs.size());
    if (cfg.dump_asm) {
        std::string s = jit::disassemble(C->code, C->code_size, C->syms, (uint64_t)(uintptr_t)C->code);
        (void)!::write(1, s.data(), s.size());
        return 0;
    }
    jit::BigStack* st = new jit::BigStack();
    jit::set_overflow_callback(on_overflow);
    void* entry = C->code + C->func_off[0];
    uint64_t e0 = now_us();
    if (st->create(size_t(8) << 30)) {
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
