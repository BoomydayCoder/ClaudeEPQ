// ARM64 assembler implementation. Encodings follow the Arm ARM (DDI 0487);
// every form is cross-checked byte-for-byte against clang by tests/jit.
#include "a64.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace a64 {

void fatal(const char* msg) {
    std::fprintf(stderr, "a64 assembler: %s\n", msg);
    std::abort();
}

const char* cond_name(Cond c) {
    static const char* n[16] = {"eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc",
                                "hi", "ls", "ge", "lt", "gt", "le", "al", "nv"};
    return n[c & 15];
}

// ---------------------------------------------------------------- logical immediates
static inline uint64_t ones(unsigned n) { return n >= 64 ? ~0ull : (1ull << n) - 1; }

bool decode_logical_imm(uint32_t enc, unsigned width, uint64_t* out) {
    unsigned N = (enc >> 12) & 1, immr = (enc >> 6) & 63, imms = enc & 63;
    if (width == 32 && N) return false;
    unsigned combined = (N << 6) | (~imms & 63);
    if (!combined) return false;
    int len = 31 - __builtin_clz(combined);
    unsigned size = 1u << len;
    if (len < 1) return false;
    unsigned R = immr & (size - 1), S = imms & (size - 1);
    if (S == size - 1) return false;
    uint64_t elem = ones(S + 1);
    if (R) elem = ((elem >> R) | (elem << (size - R))) & ones(size);
    uint64_t v = 0;
    for (unsigned i = 0; i < width; i += size) v |= elem << i;
    *out = v & ones(width);
    return true;
}

bool encode_logical_imm(uint64_t imm, unsigned width, uint32_t* enc) {
    imm &= ones(width);
    if (imm == 0 || imm == ones(width)) return false;
    // smallest repeating element
    unsigned size = width;
    while (size > 2) {
        unsigned half = size / 2;
        uint64_t m = ones(half);
        if ((imm & m) != ((imm >> half) & m)) break;
        size = half;
    }
    uint64_t mask = ones(size), val = imm & mask;
    // val must be a rotated run of ones: find rotation r with rotr(val, r) = 0..01..1
    unsigned r, cto;
    auto is_shifted_mask = [](uint64_t v) { return v && (((v | (v - 1)) + 1) & (v | (v - 1))) == 0; };
    if (is_shifted_mask(val)) {
        r = __builtin_ctzll(val);
        cto = __builtin_ctzll(~(val >> r));
    } else {
        uint64_t inv = ~val & mask;
        if (!is_shifted_mask(inv)) return false;
        // ones wrap around: leading ones at top of element, trailing at bottom
        unsigned lo = __builtin_ctzll(~val);                 // trailing ones
        unsigned hi = size - (64 - __builtin_clzll(inv));     // leading ones within element
        r = size - hi;
        cto = lo + hi;
    }
    unsigned immr = (size - r) & (size - 1);
    unsigned nimms = (~(size - 1) << 1) | (cto - 1);
    unsigned N = ((nimms >> 6) & 1) ^ 1;
    *enc = (N << 12) | (immr << 6) | (nimms & 63);
    return true;
}

DivMagic sdiv_magic(int32_t d) {  // Hacker's Delight, figure 10-1
    const uint32_t two31 = 0x80000000u;
    uint32_t ad = d < 0 ? 0u - uint32_t(d) : uint32_t(d);
    uint32_t t = two31 + (uint32_t(d) >> 31);
    uint32_t anc = t - 1 - t % ad;
    int p = 31;
    uint32_t q1 = two31 / anc, r1 = two31 - q1 * anc, q2 = two31 / ad, r2 = two31 - q2 * ad, delta;
    do {
        p++;
        q1 *= 2; r1 *= 2; if (r1 >= anc) { q1++; r1 -= anc; }
        q2 *= 2; r2 *= 2; if (r2 >= ad) { q2++; r2 -= ad; }
        delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    uint32_t m = q2 + 1;
    if (d < 0) m = 0u - m;
    return DivMagic{int32_t(m), p - 32};
}

// ---------------------------------------------------------------- basics
Assembler::Assembler() { buf_.reserve(1024); }

uint32_t Assembler::rz(Reg r) {
    if (r.is_sp) fatal("sp not allowed here (31 encodes zr)");
    return r.id;
}
uint32_t Assembler::rs(Reg r) {
    if (r.is_zr()) fatal("zr not allowed here (31 encodes sp)");
    return r.id;
}

uint32_t Assembler::imm_field(int64_t off, int bits, int scale) {
    if (off & ((1 << scale) - 1)) fatal("misaligned pc-relative offset");
    int64_t v = off >> scale;
    if (v < -(int64_t(1) << (bits - 1)) || v >= (int64_t(1) << (bits - 1))) fatal("pc-relative offset out of range");
    return uint32_t(v) & uint32_t(ones(bits));
}

void Assembler::align(size_t bytes) {
    while (pos() % bytes) nop();
}

// ---------------------------------------------------------------- data processing
void Assembler::addsub_imm(int op, int s, Reg d, Reg n, uint32_t imm12, bool lsl12) {
    same_width(d, n);
    if (imm12 > 4095) fatal("add/sub imm12 out of range");
    uint32_t rd = s ? rz(d) : rs(d);
    emit(uint32_t(d.x) << 31 | op << 30 | s << 29 | 0x11000000u | uint32_t(lsl12) << 22 | imm12 << 10 | rs(n) << 5 | rd);
}

void Assembler::addsub_sr(int op, int s, Reg d, Reg n, Reg m, Shift sh, int amt) {
    same_width(d, n); same_width(d, m);
    if (sh == ROR || amt < 0 || amt >= (d.x ? 64 : 32)) fatal("bad add/sub shift");
    emit(uint32_t(d.x) << 31 | op << 30 | s << 29 | 0x0B000000u | uint32_t(sh) << 22 | rz(m) << 16 | amt << 10 | rz(n) << 5 | rz(d));
}

void Assembler::addsub_er(int op, int s, Reg d, Reg n, Reg m, Extend e, int amt) {
    same_width(d, n);
    if (amt < 0 || amt > 4) fatal("bad extend amount");
    bool mx = d.x && (e & 3) == 3;
    if (m.x != mx) fatal("extended-register operand width mismatch");
    uint32_t rd = s ? rz(d) : rs(d);
    emit(uint32_t(d.x) << 31 | op << 30 | s << 29 | 0x0B200000u | rz(m) << 16 | uint32_t(e) << 13 | amt << 10 | rs(n) << 5 | rd);
}

void Assembler::log_sr(int opc, int neg, Reg d, Reg n, Reg m, Shift sh, int amt) {
    same_width(d, n); same_width(d, m);
    if (amt < 0 || amt >= (d.x ? 64 : 32)) fatal("bad logical shift");
    emit(uint32_t(d.x) << 31 | opc << 29 | 0x0A000000u | uint32_t(sh) << 22 | neg << 21 | rz(m) << 16 | amt << 10 | rz(n) << 5 | rz(d));
}

void Assembler::log_imm(int opc, Reg d, Reg n, uint64_t imm) {
    same_width(d, n);
    uint32_t enc;
    if (!encode_logical_imm(imm, d.x ? 64 : 32, &enc)) fatal("logical immediate not encodable");
    uint32_t rd = opc == 3 ? rz(d) : rs(d);
    emit(uint32_t(d.x) << 31 | opc << 29 | 0x12000000u | enc << 10 | rz(n) << 5 | rd);
}

void Assembler::log_any(int opc, Reg d, Reg n, uint64_t imm) {
    uint32_t enc;
    unsigned w = d.x ? 64 : 32;
    uint64_t v = imm & ones(w);
    if (encode_logical_imm(v, w, &enc)) { log_imm(opc, d, n, v); return; }
    if (n.id == 16) fatal("*_imm: source may not be x16");
    // and #0 / orr #0 / eor #0 / and #~0 etc. degenerate cases
    Reg t = d.x ? X(16) : W(16);
    if (v == 0) t = zr(d);
    else load_imm(t, int64_t(v));
    log_sr(opc, 0, d, n, t, LSL, 0);
}

void Assembler::bitfield(int opc, Reg d, Reg n, int immr, int imms) {
    same_width(d, n);
    int w = d.x ? 64 : 32;
    if (immr < 0 || immr >= w || imms < 0 || imms >= w) fatal("bitfield immediate out of range");
    emit(uint32_t(d.x) << 31 | opc << 29 | 0x13000000u | uint32_t(d.x) << 22 | immr << 16 | imms << 10 | rz(n) << 5 | rz(d));
}

void Assembler::extr(Reg d, Reg n, Reg m, int lsb) {
    same_width(d, n); same_width(d, m);
    if (lsb < 0 || lsb >= (d.x ? 64 : 32)) fatal("extr lsb out of range");
    emit(uint32_t(d.x) << 31 | 0x13800000u | uint32_t(d.x) << 22 | rz(m) << 16 | lsb << 10 | rz(n) << 5 | rz(d));
}

void Assembler::dp1(int opcode, Reg d, Reg n) {
    same_width(d, n);
    emit(uint32_t(d.x) << 31 | 0x5AC00000u | opcode << 10 | rz(n) << 5 | rz(d));
}

void Assembler::dp2(int opcode, Reg d, Reg n, Reg m) {
    same_width(d, n); same_width(d, m);
    emit(uint32_t(d.x) << 31 | 0x1AC00000u | rz(m) << 16 | opcode << 10 | rz(n) << 5 | rz(d));
}

void Assembler::dp3(int op31, int o0, Reg d, Reg n, Reg m, Reg a) {
    if (op31 == 0) { same_width(d, n); same_width(d, m); same_width(d, a); }
    else if (op31 == 2 || op31 == 6) { if (!d.x || !n.x || !m.x) fatal("smulh/umulh need x registers"); }
    else if (!d.x || n.x || m.x || !a.x) fatal("long multiply needs Xd, Wn, Wm, Xa");
    uint32_t sf = op31 ? 1 : d.x;
    emit(sf << 31 | 0x1B000000u | op31 << 21 | rz(m) << 16 | o0 << 15 | rz(a) << 10 | rz(n) << 5 | rz(d));
}

void Assembler::condsel(int op, int op2, Reg d, Reg n, Reg m, Cond c) {
    same_width(d, n); same_width(d, m);
    emit(uint32_t(d.x) << 31 | op << 30 | 0x1A800000u | rz(m) << 16 | uint32_t(c) << 12 | op2 << 10 | rz(n) << 5 | rz(d));
}

void Assembler::condcmp(int op, Reg n, int rm_or_imm, bool imm, int nzcv, Cond c, Reg m) {
    same_width(n, m);
    if (nzcv < 0 || nzcv > 15) fatal("bad nzcv");
    uint32_t f;
    if (imm) { if (rm_or_imm < 0 || rm_or_imm > 31) fatal("ccmp imm5 out of range"); f = rm_or_imm; }
    else f = rz(m);
    emit(uint32_t(n.x) << 31 | op << 30 | 0x3A400000u | f << 16 | uint32_t(c) << 12 | uint32_t(imm) << 11 | rz(n) << 5 | nzcv);
}

// ---------------------------------------------------------------- moves
void Assembler::wide(int opc, Reg d, uint32_t imm16, int shift) {
    if (imm16 > 0xffff || shift % 16 || shift < 0 || shift >= (d.x ? 64 : 32)) fatal("bad move-wide operand");
    emit(uint32_t(d.x) << 31 | opc << 29 | 0x12800000u | (shift / 16) << 21 | imm16 << 5 | rz(d));
}
void Assembler::movz(Reg d, uint32_t imm16, int shift) { wide(2, d, imm16, shift); }
void Assembler::movn(Reg d, uint32_t imm16, int shift) { wide(0, d, imm16, shift); }
void Assembler::movk(Reg d, uint32_t imm16, int shift) { wide(3, d, imm16, shift); }

void Assembler::mov(Reg d, Reg n) {
    same_width(d, n);
    if (d.is_sp || n.is_sp) addsub_imm(0, 0, d, n, 0, false);
    else orr(d, zr(d), n);
}

// Plan for materialising a constant: kind 0 = movz/movk, 1 = movn/movk, 2 = orr,
// 3 = orr + movk of one half.
struct ImmPlan { int len; int kind; uint64_t base; int half; };

static ImmPlan plan_imm(uint64_t u, bool x) {
    int n = x ? 4 : 2;
    uint64_t full = ones(n * 16);
    u &= full;
    int z = 0, o = 0;
    for (int i = 0; i < n; i++) {
        uint32_t h = (u >> (16 * i)) & 0xffff;
        z += h == 0; o += h == 0xffff;
    }
    int cz = std::max(1, n - z), cn = std::max(1, n - o);
    if (cz == 1) return {1, 0, u, 0};
    if (cn == 1) return {1, 1, u, 0};
    uint32_t enc;
    if (encode_logical_imm(u, n * 16, &enc)) return {1, 2, u, 0};
    int best = std::min(cz, cn);
    if (best > 2) {
        // orr of an encodable pattern, then movk one halfword
        for (int i = 0; i < n; i++) {
            uint64_t hm = 0xffffull << (16 * i);
            uint64_t cands[4] = {u & ~hm, u | hm, 0, 0};
            uint64_t nb = (u >> (16 * ((i + 1) % n))) & 0xffff, pb = (u >> (16 * ((i + n - 1) % n))) & 0xffff;
            cands[2] = (u & ~hm) | (nb << (16 * i));
            cands[3] = (u & ~hm) | (pb << (16 * i));
            for (uint64_t c : cands)
                if (encode_logical_imm(c, n * 16, &enc)) return {2, 3, c, i};
        }
    }
    return cz <= cn ? ImmPlan{cz, 0, u, 0} : ImmPlan{cn, 1, u, 0};
}

int Assembler::load_imm_len(int64_t v, bool x) { return plan_imm(uint64_t(v), x).len; }

void Assembler::load_imm(Reg d, int64_t v) {
    if (d.is_sp) fatal("load_imm into sp");
    int n = d.x ? 4 : 2;
    uint64_t u = uint64_t(v) & ones(n * 16);
    ImmPlan p = plan_imm(u, d.x);
    auto half = [&](int i) { return uint32_t(u >> (16 * i)) & 0xffff; };
    switch (p.kind) {
    case 2: log_imm(1, d, zr(d), u); return;
    case 3: log_imm(1, d, zr(d), p.base); movk(d, half(p.half), 16 * p.half); return;
    case 0: {
        bool first = true;
        for (int i = 0; i < n; i++) {
            if (half(i) == 0 && !(first && i == n - 1)) continue;
            if (first) { movz(d, half(i), 16 * i); first = false; }
            else movk(d, half(i), 16 * i);
        }
        return;
    }
    default: {
        bool first = true;
        for (int i = 0; i < n; i++) {
            if (half(i) == 0xffff && !(first && i == n - 1)) continue;
            if (first) { movn(d, ~half(i) & 0xffff, 16 * i); first = false; }
            else movk(d, half(i), 16 * i);
        }
    }
    }
}

void Assembler::add_imm(Reg d, Reg n, int64_t v, bool set_flags) {
    same_width(d, n);
    if (!d.x) v = int32_t(uint32_t(v));
    if (v == 0 && !set_flags) { if (!(d == n)) mov(d, n); return; }
    int op = v < 0;
    uint64_t a = op ? 0 - uint64_t(v) : uint64_t(v);
    int s = set_flags;
    if (a < 4096) { addsub_imm(op, s, d, n, uint32_t(a), false); return; }
    if ((a & 0xfff) == 0 && a < (1u << 24)) { addsub_imm(op, s, d, n, uint32_t(a >> 12), true); return; }
    if (a < (1u << 24) && !s) {
        addsub_imm(op, 0, d, n, uint32_t(a >> 12), true);
        addsub_imm(op, 0, d, d, uint32_t(a & 0xfff), false);
        return;
    }
    if (n.id == 16 && !n.is_sp) fatal("add_imm: source may not be x16");
    Reg t = d.x ? X(16) : W(16);
    load_imm(t, v);
    if (d.is_sp || n.is_sp) addsub_er(0, s, d, n, t, d.x ? UXTX : UXTW, 0);
    else addsub_sr(0, s, d, n, t, LSL, 0);
}

void Assembler::cmp_imm(Reg n, int64_t v) {
    if (!n.x) v = int32_t(uint32_t(v));
    Reg z = zr(n);
    if (v >= 0 && v < 4096) { addsub_imm(1, 1, z, n, uint32_t(v), false); return; }
    if (v >= 0 && (v & 0xfff) == 0 && v < (1 << 24)) { addsub_imm(1, 1, z, n, uint32_t(v >> 12), true); return; }
    // cmn would give identical N/Z/V but a different C flag, so only for v != min
    // and only through the register form below when the caller needs carry; we
    // keep it exact: use cmn only when -v is small (C differs only for v == 0).
    if (v < 0 && v > -4096) { addsub_imm(0, 1, z, n, uint32_t(-v), false); return; }
    if (n.id == 16 && !n.is_sp) fatal("cmp_imm: source may not be x16");
    Reg t = n.x ? X(16) : W(16);
    load_imm(t, v);
    if (n.is_sp) addsub_er(1, 1, z, n, t, n.x ? UXTX : UXTW, 0);
    else addsub_sr(1, 1, z, n, t, LSL, 0);
}

// ---------------------------------------------------------------- division / multiplication by constants
void Assembler::sdiv_core(Reg q, Reg n, int32_t dv) {
    if (n.x || q.x) fatal("sdiv_imm is 32-bit");
    if (n.id == 16 || n.id == 17) fatal("sdiv_imm: source may not be w16/w17");
    uint32_t ad = dv < 0 ? 0u - uint32_t(dv) : uint32_t(dv);
    Reg t = W(17);
    if ((ad & (ad - 1)) == 0) {
        int k = __builtin_ctz(ad);
        if (k == 1) add(t, n, n, LSR, 31);
        else { asr(t, n, 31); add(t, n, t, LSR, 32 - k); }
        if (dv > 0) asr(q, t, k);
        else neg(q, t, ASR, k);
        return;
    }
    DivMagic mg = sdiv_magic(dv);
    load_imm(W(16), mg.m);
    smull(X(17), n, W(16));
    if (dv > 0 && mg.m < 0) { asr(X(17), X(17), 32); add(t, t, n); if (mg.s) asr(t, t, mg.s); }
    else if (dv < 0 && mg.m > 0) { asr(X(17), X(17), 32); sub(t, t, n); if (mg.s) asr(t, t, mg.s); }
    else asr(X(17), X(17), 32 + mg.s);
    add(q, t, t, LSR, 31);
}

void Assembler::sdiv_imm(Reg d, Reg n, int32_t dv) {
    if (dv == 0) fatal("sdiv_imm by zero");
    if (dv == 1) { mov(d, n); return; }
    if (dv == -1) { neg(d, n); return; }
    sdiv_core(d, n, dv);
}

void Assembler::srem_imm(Reg d, Reg n, int32_t dv) {
    if (dv == 0) fatal("srem_imm by zero");
    if (dv == 1 || dv == -1) { mov(d, wzr); return; }
    sdiv_core(W(17), n, dv);
    uint32_t ad = dv < 0 ? 0u - uint32_t(dv) : uint32_t(dv);
    if ((ad & (ad - 1)) == 0) {
        int k = __builtin_ctz(ad);
        if (dv > 0) sub(d, n, W(17), LSL, k);
        else add(d, n, W(17), LSL, k);
        return;
    }
    load_imm(W(16), dv);
    msub(d, W(17), W(16), n);
}

void Assembler::mul_imm(Reg d, Reg n, int64_t c) {
    same_width(d, n);
    int w = d.x ? 64 : 32;
    uint64_t u = uint64_t(c) & ones(w), nu = (0 - u) & ones(w);
    auto pow2 = [](uint64_t x) { return x && !(x & (x - 1)); };
    auto lg = [](uint64_t x) { return __builtin_ctzll(x); };
    Reg t = d.x ? X(16) : W(16);
    if (u == 0) { mov(d, zr(d)); return; }
    if (u == 1) { if (!(d == n)) mov(d, n); return; }
    if (nu == 1) { neg(d, n); return; }
    if (pow2(u)) { lsl(d, n, lg(u)); return; }
    if (pow2(nu)) { neg(d, n, LSL, lg(nu)); return; }
    if (pow2(u - 1)) { add(d, n, n, LSL, lg(u - 1)); return; }        // 2^k + 1
    if (pow2(nu + 1)) { sub(d, n, n, LSL, lg(nu + 1)); return; }      // 1 - 2^k
    if (n.id == 16) fatal("mul_imm: source may not be x16");
    if (pow2(u + 1)) { lsl(t, n, lg(u + 1)); sub(d, t, n); return; }  // 2^k - 1
    if (pow2(nu - 1)) { add(t, n, n, LSL, lg(nu - 1)); neg(d, t); return; } // -(2^k + 1)
    int tz = lg(u);
    uint64_t odd = u >> tz;
    if (pow2(odd - 1)) { add(t, n, n, LSL, lg(odd - 1)); lsl(d, t, tz); return; }   // (2^a+1) << b
    if (pow2(odd + 1)) { lsl(t, n, lg(odd + 1)); sub(t, t, n); lsl(d, t, tz); return; } // (2^a-1) << b
    load_imm(t, int64_t(u));
    mul(d, n, t);
}

// ---------------------------------------------------------------- loads / stores
void Assembler::ldst_unscaled(int size, int opc, Reg t, Reg n, int off, int mode) {
    if (off < -256 || off > 255) fatal("imm9 offset out of range");
    if (!n.x) fatal("base must be an x register");
    emit(uint32_t(size) << 30 | 0x38000000u | opc << 22 | (uint32_t(off) & 0x1ff) << 12 | mode << 10 | rs(n) << 5 | rz(t));
}

void Assembler::ldst(int size, int opc, Reg t, const Mem& m) {
    if (!m.base.x) fatal("base must be an x register");
    switch (m.mode) {
    case Mem::Pre: ldst_unscaled(size, opc, t, m.base, int(m.off), 3); return;
    case Mem::Post: ldst_unscaled(size, opc, t, m.base, int(m.off), 1); return;
    case Mem::RegOff: {
        uint32_t opt;
        switch (m.ext) {
        case UXTW: opt = 2; break;
        case UXTX: opt = 3; break;
        case SXTW: opt = 6; break;
        case SXTX: opt = 7; break;
        default: fatal("bad index extend");
        }
        if (m.index.x != ((opt & 1) == 1)) fatal("index register width does not match extend");
        emit(uint32_t(size) << 30 | 0x38200800u | opc << 22 | rz(m.index) << 16 | opt << 13 |
             uint32_t(m.scaled) << 12 | rs(m.base) << 5 | rz(t));
        return;
    }
    case Mem::Off: {
        int64_t off = m.off;
        if (off >= 0 && (off & ((1 << size) - 1)) == 0 && (off >> size) < 4096) {
            emit(uint32_t(size) << 30 | 0x39000000u | opc << 22 | uint32_t(off >> size) << 10 | rs(m.base) << 5 | rz(t));
            return;
        }
        if (off >= -256 && off <= 255) { ldst_unscaled(size, opc, t, m.base, int(off), 0); return; }
        if (t.id == 17 || m.base.id == 17) fatal("large offset: x17 in use");
        load_imm(X(17), off);
        ldst(size, opc, t, memr(m.base, X(17)));
        return;
    }
    }
}

void Assembler::pair(int opc, int load, Reg t1, Reg t2, const Mem& m) {
    same_width(t1, t2);
    if (!m.base.x) fatal("base must be an x register");
    int scale = opc == 2 ? 3 : 2;
    if (m.off & ((1 << scale) - 1)) fatal("ldp/stp offset misaligned");
    int64_t v = m.off >> scale;
    if (v < -64 || v > 63) fatal("ldp/stp offset out of range");
    uint32_t type = m.mode == Mem::Post ? 1 : m.mode == Mem::Off ? 2 : m.mode == Mem::Pre ? 3 : 0;
    if (!type) fatal("ldp/stp: register offset not supported");
    emit(uint32_t(opc) << 30 | 0x28000000u | type << 23 | load << 22 | (uint32_t(v) & 0x7f) << 15 |
         rz(t2) << 10 | rs(m.base) << 5 | rz(t1));
}

// ---------------------------------------------------------------- labels, fixups, islands
static constexpr size_t kSlack = 64;            // bytes emitted between island checks, at most
static constexpr size_t kVeneerWindow = 1 << 15; // veneer pending branches due within this

Label Assembler::new_label() {
    labels_.push_back(LabelInfo{});
    return Label{uint32_t(labels_.size() - 1)};
}

size_t Assembler::label_offset(Label l) const {
    if (labels_[l.id].pos < 0) fatal("label not bound");
    return size_t(labels_[l.id].pos);
}

int64_t Assembler::reach(Kind k) {
    switch (k) {
    case KB14: return (1 << 15) - 4;
    case KB26: return (1 << 27) - 4;
    default: return (1 << 20) - 4;
    }
}

void Assembler::apply(uint32_t p, int64_t target, Kind k) {
    int64_t off = target - int64_t(p);
    uint32_t w = buf_[p / 4];
    switch (k) {
    case KB26: w = (w & ~0x3ffffffu) | imm_field(off, 26, 2); break;
    case KB19: case KLit: w = (w & ~(0x7ffffu << 5)) | imm_field(off, 19, 2) << 5; break;
    case KB14: w = (w & ~(0x3fffu << 5)) | imm_field(off, 14, 2) << 5; break;
    case KAdr: {
        uint32_t f = imm_field(off, 21, 0);
        w = (w & 0x9f00001fu) | (f & 3) << 29 | (f >> 2) << 5;
        break;
    }
    case KAdrp: {
        int64_t pages = (target >> 12) - (int64_t(p) >> 12);
        uint32_t f = imm_field(pages, 21, 0);
        w = (w & 0x9f00001fu) | (f & 3) << 29 | (f >> 2) << 5;
        break;
    }
    case KAddLo12: w = (w & ~(0xfffu << 10)) | uint32_t(target & 0xfff) << 10; break;
    }
    buf_[p / 4] = w;
}

void Assembler::add_fixup(uint32_t p, Label l, Kind k) {
    LabelInfo& li = labels_[l.id];
    fixups_.push_back(Fixup{p, l.id, li.first, k, false});
    li.first = int32_t(fixups_.size() - 1);
    if (k == KB19 || k == KB14) {
        pending_.push_back(uint32_t(fixups_.size() - 1));
        size_t dl = p + size_t(reach(k));
        size_t lim = dl > 8 + kSlack + 4 * pending_.size() + 8 * lits_.size()
                         ? dl - 8 - kSlack - 4 * pending_.size() - 8 * lits_.size() : 0;
        island_check_ = std::min(island_check_ >= 4 ? island_check_ - 4 : 0, lim);
    }
}

void Assembler::bind(Label l) {
    LabelInfo& li = labels_[l.id];
    if (li.pos >= 0) fatal("label bound twice");
    li.pos = int64_t(pos());
    for (int32_t i = li.first; i >= 0; i = fixups_[i].next) {
        Fixup& f = fixups_[i];
        if (f.done) continue;
        apply(f.pos, li.pos, f.kind);
        f.done = true;
    }
}

void Assembler::recompute_check() {
    size_t n = 0, dl = SIZE_MAX;
    for (uint32_t i : pending_)
        if (!fixups_[i].done) { pending_[n++] = i; dl = std::min(dl, fixups_[i].pos + size_t(reach(fixups_[i].kind))); }
    pending_.resize(n);
    for (const Lit& lt : lits_) dl = std::min(dl, lt.uses.front() + size_t(reach(KLit)));
    size_t est = 8 + 4 * pending_.size() + 8 * lits_.size() + kSlack;
    island_check_ = dl == SIZE_MAX ? SIZE_MAX : dl > est ? dl - est : 0;
}

void Assembler::check_island() {
    recompute_check();
    if (pos() >= island_check_) emit_island(true);
}

void Assembler::emit_island(bool jump_over) {
    islands_++;
    size_t skip = pos();
    if (jump_over) buf_.push_back(0x14000000u);
    size_t window = pos() + 8 + 4 * pending_.size() + 8 * lits_.size() + kSlack + kVeneerWindow;
    std::vector<uint32_t> todo(pending_);
    for (uint32_t i : todo) {
        Fixup f = fixups_[i];
        if (f.done || f.pos + size_t(reach(f.kind)) >= window) continue;
        uint32_t v = uint32_t(pos());
        buf_.push_back(0x14000000u);  // veneer: b label
        apply(f.pos, v, f.kind);
        fixups_[i].done = true;
        add_fixup(v, Label{f.label}, KB26);
    }
    if (!lits_.empty()) {
        if (pos() % 8) buf_.push_back(0xD503201Fu);
        for (Lit& lt : lits_) {
            lt.slot = pos();
            emit_u64(lt.value);
            for (uint32_t u : lt.uses) apply(u, int64_t(lt.slot), KLit);
            if (lt.label >= 0) placed_lits_.push_back(lt);
        }
        lits_.clear();
    }
    if (jump_over) buf_[skip / 4] = 0x14000000u | imm_field(int64_t(pos() - skip), 26, 2);
    recompute_check();
}

void Assembler::flush_pool(bool jump_over) {
    recompute_check();
    if (lits_.empty() && pending_.empty()) return;
    emit_island(jump_over);
}

void Assembler::finalize(uint64_t base) {
    if (base & 0xfff) fatal("code base must be 4 KB aligned");
    if (!lits_.empty()) emit_island(false);
    for (const Fixup& f : fixups_)
        if (!f.done) fatal("branch to a label that was never bound");
    for (const Lit& lt : placed_lits_) {
        uint64_t a = base + label_offset(Label{uint32_t(lt.label)});
        buf_[lt.slot / 4] = uint32_t(a);
        buf_[lt.slot / 4 + 1] = uint32_t(a >> 32);
    }
}

// ---------------------------------------------------------------- branches
void Assembler::branch_like(uint32_t word, Reg t, Label l, Kind k) {
    if (pos() >= island_check_) check_island();
    uint32_t p = uint32_t(pos());
    buf_.push_back(word | rz(t));
    if (bound(l)) apply(p, labels_[l.id].pos, k);
    else add_fixup(p, l, k);
}

void Assembler::b(Label l) { branch_like(0x14000000u, X(0), l, KB26); }
void Assembler::bl(Label l) { branch_like(0x94000000u, X(0), l, KB26); }

uint32_t Assembler::tb_word(uint32_t op, Reg t, int bit) {
    if (bit < 0 || bit >= (t.x ? 64 : 32)) fatal("tbz bit out of range");
    return op | uint32_t(bit >> 5) << 31 | uint32_t(bit & 31) << 19 | rz(t);
}

void Assembler::b(Cond c, Label l) {
    if (c >= AL) { b(l); return; }
    condbr(0x54000000u | c, xzr, -1, l, KB19);
}

// Conditional branch with automatic range handling. Inversion: b.cond flips
// bit 0 (the condition), cbz/cbnz and tbz/tbnz flip bit 24.
void Assembler::condbr(uint32_t op, Reg t, int bit, Label l, Kind k) {
    uint32_t word;
    uint32_t flip;
    if (bit < 0) { word = op; flip = 1; }  // b.cond
    else if (k == KB14) { word = tb_word(op, t, bit); flip = 1u << 24; }
    else { word = op | uint32_t(t.x) << 31 | rz(t); flip = 1u << 24; }
    if (pos() >= island_check_) check_island();
    uint32_t p = uint32_t(pos());
    if (!bound(l)) { buf_.push_back(word); add_fixup(p, l, k); return; }
    int64_t off = labels_[l.id].pos - int64_t(p);
    if (off >= -reach(k) - 4) { buf_.push_back(word); apply(p, labels_[l.id].pos, k); return; }
    // too far back: inverted short branch over an unconditional b
    buf_.push_back(word ^ flip);
    apply(p, p + 8, k);
    buf_.push_back(0x14000000u | imm_field(labels_[l.id].pos - int64_t(p + 4), 26, 2));
}

void Assembler::adr_off(Reg d, int64_t off) {
    uint32_t f = imm_field(off, 21, 0);
    emit(0x10000000u | (f & 3) << 29 | (f >> 2) << 5 | rz(d.xr()));
}
void Assembler::adrp_off(Reg d, int64_t pages) {
    uint32_t f = imm_field(pages, 21, 0);
    emit(0x90000000u | (f & 3) << 29 | (f >> 2) << 5 | rz(d.xr()));
}

void Assembler::adrp_add(Reg d, Label l) {
    Reg x = d.xr();
    adrp(x, l);
    branch_like(0x91000000u | rs(x) << 5, x, l, KAddLo12);  // add x, x, #:lo12:l
}

void Assembler::ldr_lit(Reg t, uint64_t value) {
    if (pos() >= island_check_) check_island();
    uint32_t p = uint32_t(pos());
    buf_.push_back((t.x ? 0x58000000u : 0x18000000u) | rz(t));
    if (!t.x) value &= 0xffffffffu;
    for (Lit& lt : lits_)
        if (lt.label < 0 && lt.value == value) { lt.uses.push_back(p); return; }
    lits_.push_back(Lit{value, -1, {p}, 0});
    recompute_check();
}

void Assembler::ldr_lit_addr(Reg t, Label target) {
    if (pos() >= island_check_) check_island();
    uint32_t p = uint32_t(pos());
    buf_.push_back(0x58000000u | rz(t.xr()));
    lits_.push_back(Lit{0, int32_t(target.id), {p}, 0});
    recompute_check();
}

void Assembler::call_abs(const void* fn) {
    load_addr(X(16), fn);
    blr(X(16));
}

// ---------------------------------------------------------------- frames
Assembler::Frame Assembler::prologue(uint32_t saved_mask, uint32_t local_bytes) {
    if (saved_mask & ~0x1ff80000u) fatal("prologue: only x19..x28 may be listed");
    int regs[10], n = 0;
    for (int r = 19; r <= 28; r++) if (saved_mask >> r & 1) regs[n++] = r;
    uint32_t save = (8 * n + 15) & ~15u, locals = (local_bytes + 15) & ~15u;
    Frame f{saved_mask, 16 + save + locals, 16 + save};
    if (f.size <= 504) stp(fp, lr, pre(sp, -int64_t(f.size)));
    else { sub_imm(sp, sp, f.size); stp(fp, lr, mem(sp, 0)); }
    mov(fp, sp);
    for (int i = 0; i + 1 < n; i += 2) stp(X(regs[i]), X(regs[i + 1]), mem(sp, 16 + 8 * i));
    if (n & 1) str(X(regs[n - 1]), mem(sp, 16 + 8 * (n - 1)));
    return f;
}

void Assembler::epilogue(const Frame& f, bool emit_ret) {
    int regs[10], n = 0;
    for (int r = 19; r <= 28; r++) if (f.saved >> r & 1) regs[n++] = r;
    for (int i = 0; i + 1 < n; i += 2) ldp(X(regs[i]), X(regs[i + 1]), mem(sp, 16 + 8 * i));
    if (n & 1) ldr(X(regs[n - 1]), mem(sp, 16 + 8 * (n - 1)));
    if (f.size <= 504) ldp(fp, lr, post(sp, f.size));
    else { ldp(fp, lr, mem(sp, 0)); add_imm(sp, sp, f.size); }
    if (emit_ret) ret();
}

} // namespace a64
