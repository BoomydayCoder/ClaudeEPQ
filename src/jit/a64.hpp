// ARM64 (AArch64) assembler for the ClaudeEPQ JIT.
//
// Emits into a growable word buffer that is position independent: every branch,
// adr, adrp and literal load is PC-relative (adrp assumes the final base is
// 4 KB aligned, which JitMem guarantees). Absolute addresses (literal-pool slots
// holding label addresses) are patched by finalize(base).
//
// Register conventions the code generator must respect (macOS arm64 ABI):
//   x16, x17  assembler scratch (IP0/IP1): used by load_imm-backed helpers,
//             veneers, call_abs, *_imm helpers. Never allocate them.
//   x18       platform register, reserved by Darwin. Never touch it.
//   x29/x30   frame pointer / link register; sp must stay 16-byte aligned at
//             every memory access through sp and at every call. Darwin has no
//             red zone for user code that can be relied upon below sp: never
//             store below sp.
//
// Long-range handling: forward b.cond/cbz/cbnz/tbz/tbnz to an unbound label
// are emitted short; if the label is still unbound when the branch is about to
// go out of range, an "island" (b over; veneers `b label`; pending literals)
// is inserted automatically at the next instruction boundary. Backward
// conditional branches out of range are emitted as inverted-branch + b.
// Unconditional b/bl reach +-128 MB, which bounds the size of one buffer.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace a64 {

[[noreturn]] void fatal(const char* msg);

// ---------------------------------------------------------------- registers
struct Reg {
    uint8_t id;    // 0..31 (31 = zr or sp depending on is_sp)
    uint8_t x;     // 1 = 64-bit view
    uint8_t is_sp; // 1 = stack pointer
    constexpr bool operator==(const Reg& o) const { return id == o.id && x == o.x && is_sp == o.is_sp; }
    constexpr bool is_zr() const { return id == 31 && !is_sp; }
    constexpr Reg w() const { return Reg{id, 0, is_sp}; }
    constexpr Reg xr() const { return Reg{id, 1, is_sp}; }
};
constexpr Reg W(int n) { return Reg{uint8_t(n), 0, 0}; } // n in 0..30
constexpr Reg X(int n) { return Reg{uint8_t(n), 1, 0}; }
constexpr Reg wzr{31, 0, 0}, xzr{31, 1, 0}, wsp{31, 0, 1}, sp{31, 1, 1};
constexpr Reg fp = X(29), lr = X(30), ip0 = X(16), ip1 = X(17);

// ---------------------------------------------------------------- conditions
enum Cond : uint8_t { EQ, NE, HS, LO, MI, PL, VS, VC, HI, LS, GE, LT, GT, LE, AL, NV };
constexpr Cond CS = HS, CC = LO;
constexpr Cond invert(Cond c) { return Cond(c ^ 1); }
const char* cond_name(Cond c);

enum Shift : uint8_t { LSL, LSR, ASR, ROR };
enum Extend : uint8_t { UXTB, UXTH, UXTW, UXTX, SXTB, SXTH, SXTW, SXTX };

// ---------------------------------------------------------------- memory operands
struct Mem {
    enum Mode : uint8_t { Off, Pre, Post, RegOff };
    Reg base;
    int64_t off = 0;
    Reg index{0, 1, 0};
    Mode mode = Off;
    Extend ext = UXTX; // RegOff: UXTX means "lsl"
    bool scaled = false;  // RegOff: shift index by access size
};
inline Mem mem(Reg base, int64_t off = 0) { Mem m; m.base = base; m.off = off; return m; }
inline Mem pre(Reg base, int64_t off) { Mem m; m.base = base; m.off = off; m.mode = Mem::Pre; return m; }
inline Mem post(Reg base, int64_t off) { Mem m; m.base = base; m.off = off; m.mode = Mem::Post; return m; }
// [base, index{, ext}{ #scale}]: ext = UXTX (lsl, X index), UXTW/SXTW (W index), SXTX (X index)
inline Mem memr(Reg base, Reg index, Extend ext = UXTX, bool scaled = false) {
    Mem m; m.base = base; m.index = index; m.mode = Mem::RegOff; m.ext = ext; m.scaled = scaled; return m;
}

// ---------------------------------------------------------------- logical immediates
// Encode imm (of `width` 32 or 64 bits) as N:immr:imms (13 bits). False if not encodable.
bool encode_logical_imm(uint64_t imm, unsigned width, uint32_t* enc);
// Decode N:immr:imms; false for reserved encodings.
bool decode_logical_imm(uint32_t enc, unsigned width, uint64_t* imm);

// Signed division magic (Hacker's Delight 10-1) for d not in {0, 1, -1, +-2^k}.
struct DivMagic { int32_t m; int s; };
DivMagic sdiv_magic(int32_t d);

struct Label { uint32_t id = UINT32_MAX; bool valid() const { return id != UINT32_MAX; } };

// ---------------------------------------------------------------- assembler
class Assembler {
public:
    Assembler();

    // --- buffer
    size_t size() const { return buf_.size() * 4; }          // bytes
    size_t pos() const { return buf_.size() * 4; }
    const uint32_t* words() const { return buf_.data(); }
    const uint8_t* data() const { return reinterpret_cast<const uint8_t*>(buf_.data()); }
    uint32_t word_at(size_t byte_off) const { return buf_[byte_off / 4]; }
    void patch_word(size_t byte_off, uint32_t w) { buf_[byte_off / 4] = w; }
    void emit(uint32_t insn) { if (pos() >= island_check_) check_island(); buf_.push_back(insn); }
    void emit_u64(uint64_t v) { buf_.push_back(uint32_t(v)); buf_.push_back(uint32_t(v >> 32)); }
    void align(size_t bytes);     // pads with nop
    void reserve(size_t bytes) { buf_.reserve(bytes / 4); }

    // --- labels
    Label new_label();
    void bind(Label l);
    bool bound(Label l) const { return labels_[l.id].pos >= 0; }
    size_t label_offset(Label l) const;        // byte offset of a bound label
    Label here() { Label l = new_label(); bind(l); return l; }

    // Emits an island now (veneers for all pending short branches whose label is
    // unbound, plus all pending literals). `jump_over`=false after ret/b.
    void flush_pool(bool jump_over = true);
    // Final pass: resolves absolute-address literals against `base` (4 KB aligned)
    // and checks that every referenced label is bound. Flushes pending literals.
    void finalize(uint64_t base);
    size_t island_count() const { return islands_; }

    // --- moves and immediates
    void mov(Reg d, Reg n);                      // reg-reg; handles sp via add #0
    void movz(Reg d, uint32_t imm16, int shift = 0);
    void movn(Reg d, uint32_t imm16, int shift = 0);
    void movk(Reg d, uint32_t imm16, int shift = 0);
    void load_imm(Reg d, int64_t v);             // shortest movz/movn/movk/orr sequence
    void load_addr(Reg d, const void* p) { load_imm(d.xr(), int64_t(reinterpret_cast<uintptr_t>(p))); }
    static int load_imm_len(int64_t v, bool x);  // instruction count load_imm would use

    // --- add/sub
    void add(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { addsub_sr(0, 0, d, n, m, sh, amt); }
    void adds(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { addsub_sr(0, 1, d, n, m, sh, amt); }
    void sub(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { addsub_sr(1, 0, d, n, m, sh, amt); }
    void subs(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { addsub_sr(1, 1, d, n, m, sh, amt); }
    void add(Reg d, Reg n, Reg m, Extend e, int amt) { addsub_er(0, 0, d, n, m, e, amt); }
    void adds(Reg d, Reg n, Reg m, Extend e, int amt) { addsub_er(0, 1, d, n, m, e, amt); }
    void sub(Reg d, Reg n, Reg m, Extend e, int amt) { addsub_er(1, 0, d, n, m, e, amt); }
    void subs(Reg d, Reg n, Reg m, Extend e, int amt) { addsub_er(1, 1, d, n, m, e, amt); }
    // imm12, optionally lsl #12 (raw encodings; see add_imm for any constant)
    void add(Reg d, Reg n, uint32_t imm12, bool lsl12 = false) { addsub_imm(0, 0, d, n, imm12, lsl12); }
    void adds(Reg d, Reg n, uint32_t imm12, bool lsl12 = false) { addsub_imm(0, 1, d, n, imm12, lsl12); }
    void sub(Reg d, Reg n, uint32_t imm12, bool lsl12 = false) { addsub_imm(1, 0, d, n, imm12, lsl12); }
    void subs(Reg d, Reg n, uint32_t imm12, bool lsl12 = false) { addsub_imm(1, 1, d, n, imm12, lsl12); }
    void cmp(Reg n, Reg m, Shift sh = LSL, int amt = 0) { subs(zr(n), n, m, sh, amt); }
    void cmn(Reg n, Reg m, Shift sh = LSL, int amt = 0) { adds(zr(n), n, m, sh, amt); }
    void cmp(Reg n, Reg m, Extend e, int amt) { subs(zr(n), n, m, e, amt); }
    void cmp(Reg n, uint32_t imm12, bool lsl12 = false) { subs(zr(n), n, imm12, lsl12); }
    void cmn(Reg n, uint32_t imm12, bool lsl12 = false) { adds(zr(n), n, imm12, lsl12); }
    void neg(Reg d, Reg m, Shift sh = LSL, int amt = 0) { sub(d, zr(d), m, sh, amt); }
    void negs(Reg d, Reg m, Shift sh = LSL, int amt = 0) { subs(d, zr(d), m, sh, amt); }
    // Any constant (mod 2^width); uses x16/w16 when it cannot be encoded directly.
    // With set_flags / cmp_imm, NZCV is exactly that of adds/subs by the constant.
    void add_imm(Reg d, Reg n, int64_t v, bool set_flags = false);
    void sub_imm(Reg d, Reg n, int64_t v, bool set_flags = false) { add_imm(d, n, -v, set_flags); }
    void cmp_imm(Reg n, int64_t v);

    // --- multiply / divide
    void madd(Reg d, Reg n, Reg m, Reg a) { dp3(0, 0, d, n, m, a); }
    void msub(Reg d, Reg n, Reg m, Reg a) { dp3(0, 1, d, n, m, a); }
    void mul(Reg d, Reg n, Reg m) { madd(d, n, m, zr(d)); }
    void mneg(Reg d, Reg n, Reg m) { msub(d, n, m, zr(d)); }
    void smaddl(Reg d, Reg n, Reg m, Reg a) { dp3(1, 0, d, n, m, a); }
    void umaddl(Reg d, Reg n, Reg m, Reg a) { dp3(5, 0, d, n, m, a); }
    void smull(Reg d, Reg n, Reg m) { smaddl(d, n, m, xzr); }
    void umull(Reg d, Reg n, Reg m) { umaddl(d, n, m, xzr); }
    void smulh(Reg d, Reg n, Reg m) { dp3(2, 0, d, n, m, xzr); }
    void umulh(Reg d, Reg n, Reg m) { dp3(6, 0, d, n, m, xzr); }
    void sdiv(Reg d, Reg n, Reg m) { dp2(3, d, n, m); }
    void udiv(Reg d, Reg n, Reg m) { dp2(2, d, n, m); }
    // 32-bit signed division / remainder by a constant (C semantics: truncation;
    // INT_MIN / -1 == INT_MIN like sdiv). d != 0. Clobbers w16, w17.
    void sdiv_imm(Reg d, Reg n, int32_t divisor);
    void srem_imm(Reg d, Reg n, int32_t divisor);
    // 32- or 64-bit multiply by constant with shift/add strength reduction. Clobbers x16.
    void mul_imm(Reg d, Reg n, int64_t c);

    // --- logical
    void and_(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { log_sr(0, 0, d, n, m, sh, amt); }
    void orr(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { log_sr(1, 0, d, n, m, sh, amt); }
    void eor(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { log_sr(2, 0, d, n, m, sh, amt); }
    void ands(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { log_sr(3, 0, d, n, m, sh, amt); }
    void bic(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { log_sr(0, 1, d, n, m, sh, amt); }
    void orn(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { log_sr(1, 1, d, n, m, sh, amt); }
    void eon(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { log_sr(2, 1, d, n, m, sh, amt); }
    void bics(Reg d, Reg n, Reg m, Shift sh = LSL, int amt = 0) { log_sr(3, 1, d, n, m, sh, amt); }
    void mvn(Reg d, Reg m, Shift sh = LSL, int amt = 0) { orn(d, zr(d), m, sh, amt); }
    void tst(Reg n, Reg m, Shift sh = LSL, int amt = 0) { ands(zr(n), n, m, sh, amt); }
    // Logical immediate: must be encodable (fatal otherwise); see *_imm for any value.
    void and_(Reg d, Reg n, uint64_t imm) { log_imm(0, d, n, imm); }
    void orr(Reg d, Reg n, uint64_t imm) { log_imm(1, d, n, imm); }
    void eor(Reg d, Reg n, uint64_t imm) { log_imm(2, d, n, imm); }
    void ands(Reg d, Reg n, uint64_t imm) { log_imm(3, d, n, imm); }
    void tst(Reg n, uint64_t imm) { ands(zr(n), n, imm); }
    void and_imm(Reg d, Reg n, uint64_t imm) { log_any(0, d, n, imm); }
    void orr_imm(Reg d, Reg n, uint64_t imm) { log_any(1, d, n, imm); }
    void eor_imm(Reg d, Reg n, uint64_t imm) { log_any(2, d, n, imm); }
    void ands_imm(Reg d, Reg n, uint64_t imm) { log_any(3, d, n, imm); }
    void tst_imm(Reg n, uint64_t imm) { log_any(3, zr(n), n, imm); }

    // --- shifts and bitfields
    void sbfm(Reg d, Reg n, int immr, int imms) { bitfield(0, d, n, immr, imms); }
    void bfm(Reg d, Reg n, int immr, int imms) { bitfield(1, d, n, immr, imms); }
    void ubfm(Reg d, Reg n, int immr, int imms) { bitfield(2, d, n, immr, imms); }
    void extr(Reg d, Reg n, Reg m, int lsb);
    void lsl(Reg d, Reg n, int s) { int w = d.x ? 64 : 32; ubfm(d, n, (w - s) % w, w - 1 - s); }
    void lsr(Reg d, Reg n, int s) { ubfm(d, n, s, d.x ? 63 : 31); }
    void asr(Reg d, Reg n, int s) { sbfm(d, n, s, d.x ? 63 : 31); }
    void ror(Reg d, Reg n, int s) { extr(d, n, n, s); }
    void lsl(Reg d, Reg n, Reg m) { dp2(8, d, n, m); }
    void lsr(Reg d, Reg n, Reg m) { dp2(9, d, n, m); }
    void asr(Reg d, Reg n, Reg m) { dp2(10, d, n, m); }
    void ror(Reg d, Reg n, Reg m) { dp2(11, d, n, m); }
    void sbfx(Reg d, Reg n, int lsb, int width) { sbfm(d, n, lsb, lsb + width - 1); }
    void ubfx(Reg d, Reg n, int lsb, int width) { ubfm(d, n, lsb, lsb + width - 1); }
    void bfi(Reg d, Reg n, int lsb, int width) { int w = d.x ? 64 : 32; bfm(d, n, (w - lsb) % w, width - 1); }
    void bfxil(Reg d, Reg n, int lsb, int width) { bfm(d, n, lsb, lsb + width - 1); }
    void sxtb(Reg d, Reg n) { sbfm(d, n.x == d.x ? n : Reg{n.id, d.x, 0}, 0, 7); }
    void sxth(Reg d, Reg n) { sbfm(d, n.x == d.x ? n : Reg{n.id, d.x, 0}, 0, 15); }
    void sxtw(Reg d, Reg n) { sbfm(d, n.xr(), 0, 31); }       // sxtw xd, wn
    void uxtb(Reg d, Reg n) { ubfm(d.w(), n.w(), 0, 7); }
    void uxth(Reg d, Reg n) { ubfm(d.w(), n.w(), 0, 15); }
    void uxtw(Reg d, Reg n) { mov(d.w(), n.w()); }            // 32-bit mov zero-extends

    // --- conditional select / compare
    void csel(Reg d, Reg n, Reg m, Cond c) { condsel(0, 0, d, n, m, c); }
    void csinc(Reg d, Reg n, Reg m, Cond c) { condsel(0, 1, d, n, m, c); }
    void csinv(Reg d, Reg n, Reg m, Cond c) { condsel(1, 0, d, n, m, c); }
    void csneg(Reg d, Reg n, Reg m, Cond c) { condsel(1, 1, d, n, m, c); }
    void cset(Reg d, Cond c) { check_al(c); csinc(d, zr(d), zr(d), invert(c)); }
    void csetm(Reg d, Cond c) { check_al(c); csinv(d, zr(d), zr(d), invert(c)); }
    void cinc(Reg d, Reg n, Cond c) { check_al(c); csinc(d, n, n, invert(c)); }
    void cinv(Reg d, Reg n, Cond c) { check_al(c); csinv(d, n, n, invert(c)); }
    void cneg(Reg d, Reg n, Cond c) { check_al(c); csneg(d, n, n, invert(c)); }
    void ccmp(Reg n, Reg m, int nzcv, Cond c) { condcmp(1, n, m.id, false, nzcv, c, m); }
    void ccmn(Reg n, Reg m, int nzcv, Cond c) { condcmp(0, n, m.id, false, nzcv, c, m); }
    void ccmp(Reg n, int imm5, int nzcv, Cond c) { condcmp(1, n, imm5, true, nzcv, c, n); }
    void ccmn(Reg n, int imm5, int nzcv, Cond c) { condcmp(0, n, imm5, true, nzcv, c, n); }

    // --- misc data processing
    void clz(Reg d, Reg n) { dp1(4, d, n); }
    void cls(Reg d, Reg n) { dp1(5, d, n); }
    void rbit(Reg d, Reg n) { dp1(0, d, n); }
    void rev(Reg d, Reg n) { dp1(d.x ? 3 : 2, d, n); }
    void rev16(Reg d, Reg n) { dp1(1, d, n); }

    // --- loads / stores. Off-mode offsets pick scaled imm12, else ldur/stur, else
    // materialise into x17 (register form). Pre/Post need imm9.
    void ldr(Reg t, const Mem& m) { ldst(t.x ? 3 : 2, 1, t, m); }
    void str(Reg t, const Mem& m) { ldst(t.x ? 3 : 2, 0, t, m); }
    void ldrb(Reg t, const Mem& m) { ldst(0, 1, t.w(), m); }
    void strb(Reg t, const Mem& m) { ldst(0, 0, t.w(), m); }
    void ldrh(Reg t, const Mem& m) { ldst(1, 1, t.w(), m); }
    void strh(Reg t, const Mem& m) { ldst(1, 0, t.w(), m); }
    void ldrsb(Reg t, const Mem& m) { ldst(0, t.x ? 2 : 3, t, m); }
    void ldrsh(Reg t, const Mem& m) { ldst(1, t.x ? 2 : 3, t, m); }
    void ldrsw(Reg t, const Mem& m) { ldst(2, 2, t.xr(), m); }
    // explicit unscaled forms (imm9)
    void ldur(Reg t, Reg n, int off) { ldst_unscaled(t.x ? 3 : 2, 1, t, n, off, 0); }
    void stur(Reg t, Reg n, int off) { ldst_unscaled(t.x ? 3 : 2, 0, t, n, off, 0); }
    void ldurb(Reg t, Reg n, int off) { ldst_unscaled(0, 1, t.w(), n, off, 0); }
    void sturb(Reg t, Reg n, int off) { ldst_unscaled(0, 0, t.w(), n, off, 0); }
    void ldurh(Reg t, Reg n, int off) { ldst_unscaled(1, 1, t.w(), n, off, 0); }
    void sturh(Reg t, Reg n, int off) { ldst_unscaled(1, 0, t.w(), n, off, 0); }
    void ldursb(Reg t, Reg n, int off) { ldst_unscaled(0, t.x ? 2 : 3, t, n, off, 0); }
    void ldursh(Reg t, Reg n, int off) { ldst_unscaled(1, t.x ? 2 : 3, t, n, off, 0); }
    void ldursw(Reg t, Reg n, int off) { ldst_unscaled(2, 2, t.xr(), n, off, 0); }
    // pairs: Off (signed scaled imm7), Pre, Post
    void ldp(Reg t1, Reg t2, const Mem& m) { pair(t1.x ? 2 : 0, 1, t1, t2, m); }
    void stp(Reg t1, Reg t2, const Mem& m) { pair(t1.x ? 2 : 0, 0, t1, t2, m); }
    void ldpsw(Reg t1, Reg t2, const Mem& m) { pair(1, 1, t1.xr(), t2.xr(), m); }
    // PC-relative literal loads
    void ldr_lit(Reg t, Label l) { branch_like(t.x ? 0x58000000u : 0x18000000u, t, l, KLit); }
    void ldrsw_lit(Reg t, Label l) { branch_like(0x98000000u, t.xr(), l, KLit); }
    void ldr_lit(Reg t, uint64_t value);            // via the pool (X or W)
    void ldr_lit_addr(Reg t, Label target);         // 64-bit absolute address of a label

    // --- branches. All label forms handle range automatically.
    void b(Label l);
    void bl(Label l);
    void b(Cond c, Label l);
    void cbz(Reg t, Label l) { condbr(0x34000000u, t, 0, l, KB19); }
    void cbnz(Reg t, Label l) { condbr(0x35000000u, t, 0, l, KB19); }
    void tbz(Reg t, int bit, Label l) { condbr(0x36000000u, t, bit, l, KB14); }
    void tbnz(Reg t, int bit, Label l) { condbr(0x37000000u, t, bit, l, KB14); }
    void br(Reg n) { emit(0xD61F0000u | rz(n.xr()) << 5); }
    void blr(Reg n) { emit(0xD63F0000u | rz(n.xr()) << 5); }
    void ret(Reg n = lr) { emit(0xD65F0000u | rz(n.xr()) << 5); }
    void adr(Reg d, Label l) { branch_like(0x10000000u, d.xr(), l, KAdr); }
    void adrp(Reg d, Label l) { branch_like(0x90000000u, d.xr(), l, KAdrp); }
    void adrp_add(Reg d, Label l);  // +-4 GB address of label (adrp + add :lo12:)
    // raw offset forms (byte offsets relative to this instruction; for tests)
    void b_off(int64_t off) { emit(0x14000000u | imm_field(off, 26, 2)); }
    void bl_off(int64_t off) { emit(0x94000000u | imm_field(off, 26, 2)); }
    void b_off(Cond c, int64_t off) { emit(0x54000000u | imm_field(off, 19, 2) << 5 | c); }
    void cbz_off(Reg t, int64_t off) { emit(0x34000000u | uint32_t(t.x) << 31 | imm_field(off, 19, 2) << 5 | rz(t)); }
    void cbnz_off(Reg t, int64_t off) { emit(0x35000000u | uint32_t(t.x) << 31 | imm_field(off, 19, 2) << 5 | rz(t)); }
    void tbz_off(Reg t, int bit, int64_t off) { emit(tb_word(0x36000000u, t, bit) | imm_field(off, 14, 2) << 5); }
    void tbnz_off(Reg t, int bit, int64_t off) { emit(tb_word(0x37000000u, t, bit) | imm_field(off, 14, 2) << 5); }
    void adr_off(Reg d, int64_t off);
    void adrp_off(Reg d, int64_t pages);
    void ldr_lit_off(Reg t, int64_t off) { emit((t.x ? 0x58000000u : 0x18000000u) | imm_field(off, 19, 2) << 5 | rz(t)); }
    void ldrsw_lit_off(Reg t, int64_t off) { emit(0x98000000u | imm_field(off, 19, 2) << 5 | rz(t.xr())); }
    void call_abs(const void* fn);  // movz/movk x16 + blr x16

    void nop() { emit(0xD503201Fu); }
    void mrs_nzcv(Reg t) { emit(0xD53B4200u | rz(t.xr())); }
    void msr_nzcv(Reg t) { emit(0xD51B4200u | rz(t.xr())); }
    void brk(uint32_t imm16) { emit(0xD4200000u | (imm16 & 0xffff) << 5); }
    void udf(uint32_t imm16) { emit(imm16 & 0xffff); }

    // --- frames. `saved` is a bitmask of callee-saved GPRs (bits 19..28). Layout
    // (sp-relative after prologue): [0]=x29,x30  [16..)=saved regs in pairs
    // [locals_offset()..)=locals. x29 = sp. Total frame is 16-byte aligned.
    struct Frame { uint32_t saved; uint32_t size; uint32_t locals_off; };
    Frame prologue(uint32_t saved_mask, uint32_t local_bytes);
    void epilogue(const Frame& f, bool emit_ret = true);

    static uint32_t imm_field(int64_t off, int bits, int scale); // range-checked

private:
    enum Kind : uint8_t { KB26, KB19, KB14, KAdr, KAdrp, KAddLo12, KLit };
    struct LabelInfo { int64_t pos = -1; int32_t first = -1; };
    struct Fixup { uint32_t pos; uint32_t label; int32_t next; Kind kind; bool done; };
    struct Lit { uint64_t value; int32_t label; std::vector<uint32_t> uses; size_t slot; };

    std::vector<uint32_t> buf_;
    std::vector<LabelInfo> labels_;
    std::vector<Fixup> fixups_;
    std::vector<uint32_t> pending_;     // short-range fixups with unbound labels
    std::vector<Lit> lits_;             // literals not yet placed
    std::vector<Lit> placed_lits_;      // literal slots holding label addresses
    size_t island_check_ = SIZE_MAX;
    size_t islands_ = 0;

    static uint32_t rz(Reg r);  // register where 31 = zr
    static uint32_t rs(Reg r);  // register where 31 = sp
    static Reg zr(Reg like) { return like.x ? xzr : wzr; }
    static void same_width(Reg a, Reg b) { if (a.x != b.x) fatal("register width mismatch"); }
    static void check_al(Cond c) { if (c >= AL) fatal("al/nv not allowed"); }
    uint32_t tb_word(uint32_t op, Reg t, int bit);

    void addsub_sr(int op, int s, Reg d, Reg n, Reg m, Shift sh, int amt);
    void addsub_er(int op, int s, Reg d, Reg n, Reg m, Extend e, int amt);
    void addsub_imm(int op, int s, Reg d, Reg n, uint32_t imm12, bool lsl12);
    void log_sr(int opc, int neg, Reg d, Reg n, Reg m, Shift sh, int amt);
    void log_imm(int opc, Reg d, Reg n, uint64_t imm);
    void log_any(int opc, Reg d, Reg n, uint64_t imm);
    void bitfield(int opc, Reg d, Reg n, int immr, int imms);
    void dp1(int opcode, Reg d, Reg n);
    void dp2(int opcode, Reg d, Reg n, Reg m);
    void dp3(int op31, int o0, Reg d, Reg n, Reg m, Reg a);
    void condsel(int op, int op2, Reg d, Reg n, Reg m, Cond c);
    void condcmp(int op, Reg n, int rm_or_imm, bool imm, int nzcv, Cond c, Reg m);
    void ldst(int size, int opc, Reg t, const Mem& m);
    void ldst_unscaled(int size, int opc, Reg t, Reg n, int off, int mode);
    void pair(int opc, int load, Reg t1, Reg t2, const Mem& m);
    void wide(int opc, Reg d, uint32_t imm16, int shift);
    void sdiv_core(Reg q, Reg n, int32_t divisor);

    void condbr(uint32_t op, Reg t, int bit, Label l, Kind k);
    void branch_like(uint32_t word, Reg t, Label l, Kind k);
    void add_fixup(uint32_t pos, Label l, Kind k);
    void apply(uint32_t pos, int64_t target, Kind k);
    static int64_t reach(Kind k);
    void check_island();
    void emit_island(bool jump_over);
    void recompute_check();
};

} // namespace a64
