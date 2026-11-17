// Encoder cross-check generator. For every instruction form the assembler can
// emit, generates random operands (biased to edge cases), emits the word with
// a64::Assembler and writes the equivalent assembly text. run.py assembles the
// text with clang and compares byte-for-byte.
//
// usage: encode_gen OUTDIR [cases_per_form]
//   OUTDIR/cases.s    one instruction per line
//   OUTDIR/cases.bin  our encodings (little-endian words)
//   OUTDIR/cases.tsv  form \t text  (one line per case)
#include "../../src/jit/a64.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace a64;

static std::mt19937_64 rng(0xC0FFEE);
static uint64_t R() { return rng(); }
static int64_t ri(int64_t lo, int64_t hi) { return lo + int64_t(R() % uint64_t(hi - lo + 1)); }
// edge-biased integer in [lo, hi]
static int64_t re(int64_t lo, int64_t hi) {
    switch (R() % 8) {
    case 0: return lo;
    case 1: return hi;
    case 2: return lo + 1 <= hi ? lo + 1 : lo;
    case 3: return hi - 1 >= lo ? hi - 1 : hi;
    case 4: return (lo <= 0 && 0 <= hi) ? 0 : lo;
    default: return ri(lo, hi);
    }
}
static bool coin() { return R() & 1; }

enum R31 { AS_ZR, AS_SP };
static Reg rnd(bool x, R31 k) {
    int n = (R() % 6 == 0) ? 31 : int(ri(0, 30));
    if (n == 31) return k == AS_SP ? (x ? sp : wsp) : (x ? xzr : wzr);
    return x ? X(n) : W(n);
}
static std::string nm(Reg r) {
    if (r.id == 31) return r.is_sp ? (r.x ? "sp" : "wsp") : (r.x ? "xzr" : "wzr");
    return (r.x ? "x" : "w") + std::to_string(r.id);
}
static std::string I(int64_t v) { return "#" + std::to_string(v); }
static std::string H(uint64_t v) { char b[32]; snprintf(b, sizeof b, "#0x%llx", (unsigned long long)v); return b; }

struct Case { std::string form, text; uint32_t word; };
static std::vector<Case> cases;
static Assembler A;
static int N = 1500;

template <class F>
static void put(const std::string& form, const std::string& text, F&& f) {
    size_t b = A.size();
    f();
    if (A.size() - b != 4) { fprintf(stderr, "form %s: emitted %zu bytes for '%s'\n", form.c_str(), A.size() - b, text.c_str()); exit(1); }
    cases.push_back({form, text, A.word_at(b)});
}

static const char* shname[4] = {"lsl", "lsr", "asr", "ror"};
static const char* extname[8] = {"uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"};

static void gen_arith() {
    const char* ops[4] = {"add", "adds", "sub", "subs"};
    for (int op = 0; op < 4; op++) for (int x = 0; x < 2; x++) {
        bool S = op & 1;
        std::string f = std::string(ops[op]) + (x ? "_x" : "_w");
        for (int i = 0; i < N; i++) {  // immediate
            Reg d = rnd(x, S ? AS_ZR : AS_SP), n = rnd(x, AS_SP);
            uint32_t imm = uint32_t(re(0, 4095)); bool sh = coin();
            put(f + "_imm", std::string(ops[op]) + " " + nm(d) + ", " + nm(n) + ", " + I(imm) + (sh ? ", lsl #12" : ""), [&] {
                switch (op) { case 0: A.add(d, n, imm, sh); break; case 1: A.adds(d, n, imm, sh); break;
                              case 2: A.sub(d, n, imm, sh); break; default: A.subs(d, n, imm, sh); } });
        }
        for (int i = 0; i < N; i++) {  // shifted register
            Reg d = rnd(x, AS_ZR), n = rnd(x, AS_ZR), m = rnd(x, AS_ZR);
            Shift s = Shift(ri(0, 2)); int amt = int(re(0, x ? 63 : 31));
            put(f + "_sreg", std::string(ops[op]) + " " + nm(d) + ", " + nm(n) + ", " + nm(m) + ", " + shname[s] + " " + I(amt), [&] {
                switch (op) { case 0: A.add(d, n, m, s, amt); break; case 1: A.adds(d, n, m, s, amt); break;
                              case 2: A.sub(d, n, m, s, amt); break; default: A.subs(d, n, m, s, amt); } });
        }
        for (int i = 0; i < N; i++) {  // extended register
            Reg d = rnd(x, S ? AS_ZR : AS_SP), n = rnd(x, AS_SP);
            Extend e = Extend(ri(0, 7));
            if (!x && (e & 3) == 3) e = Extend(e - 1);
            Reg m = rnd(x && (e & 3) == 3, AS_ZR);
            int amt = int(re(0, 4));
            put(f + "_ereg", std::string(ops[op]) + " " + nm(d) + ", " + nm(n) + ", " + nm(m) + ", " + extname[e] + " " + I(amt), [&] {
                switch (op) { case 0: A.add(d, n, m, e, amt); break; case 1: A.adds(d, n, m, e, amt); break;
                              case 2: A.sub(d, n, m, e, amt); break; default: A.subs(d, n, m, e, amt); } });
        }
    }
    for (int x = 0; x < 2; x++) {
        std::string sx = x ? "_x" : "_w";
        for (int i = 0; i < N; i++) {
            Reg n = rnd(x, AS_SP); uint32_t imm = uint32_t(re(0, 4095)); bool sh = coin();
            put("cmp_imm" + sx, "cmp " + nm(n) + ", " + I(imm) + (sh ? ", lsl #12" : ""), [&] { A.cmp(n, imm, sh); });
            put("cmn_imm" + sx, "cmn " + nm(n) + ", " + I(imm) + (sh ? ", lsl #12" : ""), [&] { A.cmn(n, imm, sh); });
            Reg a = rnd(x, AS_ZR), b = rnd(x, AS_ZR), d = rnd(x, AS_ZR);
            Shift s = Shift(ri(0, 2)); int amt = int(re(0, x ? 63 : 31));
            std::string tail = ", " + std::string(shname[s]) + " " + I(amt);
            put("cmp_sreg" + sx, "cmp " + nm(a) + ", " + nm(b) + tail, [&] { A.cmp(a, b, s, amt); });
            put("cmn_sreg" + sx, "cmn " + nm(a) + ", " + nm(b) + tail, [&] { A.cmn(a, b, s, amt); });
            put("neg" + sx, "neg " + nm(d) + ", " + nm(b) + tail, [&] { A.neg(d, b, s, amt); });
            put("negs" + sx, "negs " + nm(d) + ", " + nm(b) + tail, [&] { A.negs(d, b, s, amt); });
            Extend e = Extend(ri(0, 7));
            if (!x && (e & 3) == 3) e = Extend(e - 1);
            Reg m = rnd(x && (e & 3) == 3, AS_ZR); int ea = int(re(0, 4));
            put("cmp_ereg" + sx, "cmp " + nm(n) + ", " + nm(m) + ", " + extname[e] + " " + I(ea), [&] { A.cmp(n, m, e, ea); });
            // mov register (orr alias, or add #0 when sp is involved)
            Reg md = rnd(x, AS_ZR), ms = rnd(x, AS_ZR);
            int which = int(ri(0, 2));
            if (which == 1) md = x ? sp : wsp;
            if (which == 2 && md.id != 31) ms = x ? sp : wsp;
            if (md.is_sp && ms.is_zr()) ms = x ? X(1) : W(1);
            put("mov_reg" + sx, "mov " + nm(md) + ", " + nm(ms), [&] { A.mov(md, ms); });
        }
    }
}

static void gen_logic() {
    const char* ops[8] = {"and", "orr", "eor", "ands", "bic", "orn", "eon", "bics"};
    for (int op = 0; op < 8; op++) for (int x = 0; x < 2; x++) {
        std::string f = std::string(ops[op]) + (x ? "_x" : "_w");
        for (int i = 0; i < N; i++) {
            Reg d = rnd(x, AS_ZR), n = rnd(x, AS_ZR), m = rnd(x, AS_ZR);
            Shift s = Shift(ri(0, 3)); int amt = int(re(0, x ? 63 : 31));
            put(f + "_sreg", std::string(ops[op]) + " " + nm(d) + ", " + nm(n) + ", " + nm(m) + ", " + shname[s] + " " + I(amt), [&] {
                switch (op) {
                case 0: A.and_(d, n, m, s, amt); break; case 1: A.orr(d, n, m, s, amt); break;
                case 2: A.eor(d, n, m, s, amt); break; case 3: A.ands(d, n, m, s, amt); break;
                case 4: A.bic(d, n, m, s, amt); break; case 5: A.orn(d, n, m, s, amt); break;
                case 6: A.eon(d, n, m, s, amt); break; default: A.bics(d, n, m, s, amt);
                } });
        }
        if (op >= 4) continue;
        for (int i = 0; i < N; i++) {
            uint32_t enc = uint32_t(R() & (x ? 0x1fff : 0xfff));
            uint64_t v;
            if (!decode_logical_imm(enc, x ? 64 : 32, &v)) { i--; continue; }
            Reg d = rnd(x, op == 3 ? AS_ZR : AS_SP), n = rnd(x, AS_ZR);
            put(f + "_imm", std::string(ops[op]) + " " + nm(d) + ", " + nm(n) + ", " + H(v), [&] {
                switch (op) { case 0: A.and_(d, n, v); break; case 1: A.orr(d, n, v); break;
                              case 2: A.eor(d, n, v); break; default: A.ands(d, n, v); } });
        }
    }
    for (int x = 0; x < 2; x++) {
        std::string sx = x ? "_x" : "_w";
        for (int i = 0; i < N; i++) {
            Reg d = rnd(x, AS_ZR), n = rnd(x, AS_ZR), m = rnd(x, AS_ZR);
            Shift s = Shift(ri(0, 3)); int amt = int(re(0, x ? 63 : 31));
            put("mvn" + sx, "mvn " + nm(d) + ", " + nm(m) + ", " + shname[s] + " " + I(amt), [&] { A.mvn(d, m, s, amt); });
            put("tst_sreg" + sx, "tst " + nm(n) + ", " + nm(m) + ", " + shname[s] + " " + I(amt), [&] { A.tst(n, m, s, amt); });
            uint64_t v;
            while (!decode_logical_imm(uint32_t(R() & (x ? 0x1fff : 0xfff)), x ? 64 : 32, &v)) {}
            put("tst_imm" + sx, "tst " + nm(n) + ", " + H(v), [&] { A.tst(n, v); });
        }
    }
}

static void gen_move_bitfield() {
    for (int x = 0; x < 2; x++) {
        std::string sx = x ? "_x" : "_w";
        int w = x ? 64 : 32;
        for (int i = 0; i < N; i++) {
            Reg d = rnd(x, AS_ZR), n = rnd(x, AS_ZR), m = rnd(x, AS_ZR);
            uint32_t imm = uint32_t(re(0, 0xffff)); int sh = 16 * int(ri(0, x ? 3 : 1));
            std::string t = nm(d) + ", " + H(imm) + ", lsl " + I(sh);
            put("movz" + sx, "movz " + t, [&] { A.movz(d, imm, sh); });
            put("movn" + sx, "movn " + t, [&] { A.movn(d, imm, sh); });
            put("movk" + sx, "movk " + t, [&] { A.movk(d, imm, sh); });
            int r = int(re(0, w - 1)), s = int(re(0, w - 1));
            std::string rs = nm(d) + ", " + nm(n) + ", " + I(r) + ", " + I(s);
            put("sbfm" + sx, "sbfm " + rs, [&] { A.sbfm(d, n, r, s); });
            put("bfm" + sx, "bfm " + rs, [&] { A.bfm(d, n, r, s); });
            put("ubfm" + sx, "ubfm " + rs, [&] { A.ubfm(d, n, r, s); });
            int a = int(re(0, w - 1));
            std::string sa = nm(d) + ", " + nm(n) + ", " + I(a);
            put("lsl_imm" + sx, "lsl " + sa, [&] { A.lsl(d, n, a); });
            put("lsr_imm" + sx, "lsr " + sa, [&] { A.lsr(d, n, a); });
            put("asr_imm" + sx, "asr " + sa, [&] { A.asr(d, n, a); });
            put("ror_imm" + sx, "ror " + sa, [&] { A.ror(d, n, a); });
            put("extr" + sx, "extr " + nm(d) + ", " + nm(n) + ", " + nm(m) + ", " + I(a), [&] { A.extr(d, n, m, a); });
            int lsb = int(re(0, w - 1)), wd = int(re(1, w - lsb));
            std::string bf = nm(d) + ", " + nm(n) + ", " + I(lsb) + ", " + I(wd);
            put("sbfx" + sx, "sbfx " + bf, [&] { A.sbfx(d, n, lsb, wd); });
            put("ubfx" + sx, "ubfx " + bf, [&] { A.ubfx(d, n, lsb, wd); });
            put("bfi" + sx, "bfi " + bf, [&] { A.bfi(d, n, lsb, wd); });
            put("bfxil" + sx, "bfxil " + bf, [&] { A.bfxil(d, n, lsb, wd); });
            Reg nw = n.w();
            put("sxtb" + sx, "sxtb " + nm(d) + ", " + nm(nw), [&] { A.sxtb(d, nw); });
            put("sxth" + sx, "sxth " + nm(d) + ", " + nm(nw), [&] { A.sxth(d, nw); });
            if (x) put("sxtw", "sxtw " + nm(d) + ", " + nm(nw), [&] { A.sxtw(d, nw); });
            else {
                put("uxtb", "uxtb " + nm(d) + ", " + nm(nw), [&] { A.uxtb(d, nw); });
                put("uxth", "uxth " + nm(d) + ", " + nm(nw), [&] { A.uxth(d, nw); });
                put("uxtw", "mov " + nm(d) + ", " + nm(nw), [&] { A.uxtw(d, nw); });
            }
            // shifts by register, division
            std::string r3 = nm(d) + ", " + nm(n) + ", " + nm(m);
            put("lslv" + sx, "lsl " + r3, [&] { A.lsl(d, n, m); });
            put("lsrv" + sx, "lsr " + r3, [&] { A.lsr(d, n, m); });
            put("asrv" + sx, "asr " + r3, [&] { A.asr(d, n, m); });
            put("rorv" + sx, "ror " + r3, [&] { A.ror(d, n, m); });
            put("sdiv" + sx, "sdiv " + r3, [&] { A.sdiv(d, n, m); });
            put("udiv" + sx, "udiv " + r3, [&] { A.udiv(d, n, m); });
            std::string r2 = nm(d) + ", " + nm(n);
            put("clz" + sx, "clz " + r2, [&] { A.clz(d, n); });
            put("cls" + sx, "cls " + r2, [&] { A.cls(d, n); });
            put("rbit" + sx, "rbit " + r2, [&] { A.rbit(d, n); });
            put("rev" + sx, "rev " + r2, [&] { A.rev(d, n); });
            put("rev16" + sx, "rev16 " + r2, [&] { A.rev16(d, n); });
            Reg ra = rnd(x, AS_ZR);
            std::string r4 = r3 + ", " + nm(ra);
            put("madd" + sx, "madd " + r4, [&] { A.madd(d, n, m, ra); });
            put("msub" + sx, "msub " + r4, [&] { A.msub(d, n, m, ra); });
            put("mul" + sx, "mul " + r3, [&] { A.mul(d, n, m); });
            put("mneg" + sx, "mneg " + r3, [&] { A.mneg(d, n, m); });
            if (x) {
                Reg nw2 = n.w(), mw2 = m.w();
                std::string l3 = nm(d) + ", " + nm(nw2) + ", " + nm(mw2);
                put("smaddl", "smaddl " + l3 + ", " + nm(ra), [&] { A.smaddl(d, nw2, mw2, ra); });
                put("umaddl", "umaddl " + l3 + ", " + nm(ra), [&] { A.umaddl(d, nw2, mw2, ra); });
                put("smull", "smull " + l3, [&] { A.smull(d, nw2, mw2); });
                put("umull", "umull " + l3, [&] { A.umull(d, nw2, mw2); });
                put("smulh", "smulh " + r3, [&] { A.smulh(d, n, m); });
                put("umulh", "umulh " + r3, [&] { A.umulh(d, n, m); });
            }
            // conditional select / compare
            Cond c = Cond(ri(0, 15)), c14 = Cond(ri(0, 13));
            std::string cs = r3 + ", " + cond_name(c);
            put("csel" + sx, "csel " + cs, [&] { A.csel(d, n, m, c); });
            put("csinc" + sx, "csinc " + cs, [&] { A.csinc(d, n, m, c); });
            put("csinv" + sx, "csinv " + cs, [&] { A.csinv(d, n, m, c); });
            put("csneg" + sx, "csneg " + cs, [&] { A.csneg(d, n, m, c); });
            put("cset" + sx, "cset " + nm(d) + ", " + cond_name(c14), [&] { A.cset(d, c14); });
            put("csetm" + sx, "csetm " + nm(d) + ", " + cond_name(c14), [&] { A.csetm(d, c14); });
            Reg nn = n.is_zr() ? (x ? X(3) : W(3)) : n;  // cinc etc. with zr would print as cset
            std::string ci = nm(d) + ", " + nm(nn) + ", " + cond_name(c14);
            put("cinc" + sx, "cinc " + ci, [&] { A.cinc(d, nn, c14); });
            put("cinv" + sx, "cinv " + ci, [&] { A.cinv(d, nn, c14); });
            put("cneg" + sx, "cneg " + ci, [&] { A.cneg(d, nn, c14); });
            int nzcv = int(ri(0, 15)), imm5 = int(re(0, 31));
            std::string cc = nm(n) + ", " + nm(m) + ", " + I(nzcv) + ", " + cond_name(c);
            put("ccmp_reg" + sx, "ccmp " + cc, [&] { A.ccmp(n, m, nzcv, c); });
            put("ccmn_reg" + sx, "ccmn " + cc, [&] { A.ccmn(n, m, nzcv, c); });
            std::string ci5 = nm(n) + ", " + I(imm5) + ", " + I(nzcv) + ", " + cond_name(c);
            put("ccmp_imm" + sx, "ccmp " + ci5, [&] { A.ccmp(n, imm5, nzcv, c); });
            put("ccmn_imm" + sx, "ccmn " + ci5, [&] { A.ccmn(n, imm5, nzcv, c); });
        }
    }
}

struct LsOp { const char* name; const char* uname; int size; int kind; bool x; };
// kind: 0 ldr, 1 str, 2 ldrb, 3 strb, 4 ldrh, 5 strh, 6 ldrsb, 7 ldrsh, 8 ldrsw
static const LsOp lsops[] = {
    {"ldr", "ldur", 2, 0, false}, {"ldr", "ldur", 3, 0, true}, {"str", "stur", 2, 1, false}, {"str", "stur", 3, 1, true},
    {"ldrb", "ldurb", 0, 2, false}, {"strb", "sturb", 0, 3, false}, {"ldrh", "ldurh", 1, 4, false}, {"strh", "sturh", 1, 5, false},
    {"ldrsb", "ldursb", 0, 6, false}, {"ldrsb", "ldursb", 0, 6, true}, {"ldrsh", "ldursh", 1, 7, false}, {"ldrsh", "ldursh", 1, 7, true},
    {"ldrsw", "ldursw", 2, 8, true},
};

static void do_ls(const LsOp& o, Reg t, const Mem& m) {
    switch (o.kind) {
    case 0: A.ldr(t, m); break; case 1: A.str(t, m); break; case 2: A.ldrb(t, m); break; case 3: A.strb(t, m); break;
    case 4: A.ldrh(t, m); break; case 5: A.strh(t, m); break; case 6: A.ldrsb(t, m); break; case 7: A.ldrsh(t, m); break;
    default: A.ldrsw(t, m);
    }
}
static void do_lsu(const LsOp& o, Reg t, Reg n, int off) {
    switch (o.kind) {
    case 0: A.ldur(t, n, off); break; case 1: A.stur(t, n, off); break; case 2: A.ldurb(t, n, off); break; case 3: A.sturb(t, n, off); break;
    case 4: A.ldurh(t, n, off); break; case 5: A.sturh(t, n, off); break; case 6: A.ldursb(t, n, off); break; case 7: A.ldursh(t, n, off); break;
    default: A.ldursw(t, n, off);
    }
}

static void gen_loadstore() {
    for (const LsOp& o : lsops) {
        std::string f = std::string(o.name) + (o.x ? "_x" : "_w");
        for (int i = 0; i < N; i++) {
            Reg t = rnd(o.x, AS_ZR), b = rnd(true, AS_SP);
            int64_t off = re(0, 4095) << o.size;
            put(f + "_uimm", std::string(o.name) + " " + nm(t) + ", [" + nm(b) + ", " + I(off) + "]", [&] { do_ls(o, t, mem(b, off)); });
            int u = int(re(-256, 255));
            put(f + "_unscaled", std::string(o.uname) + " " + nm(t) + ", [" + nm(b) + ", " + I(u) + "]", [&] { do_lsu(o, t, b, u); });
            // Mem-based auto selection falls back to ldur for negative / unaligned offsets
            int u2 = int(ri(-256, -1));
            put(f + "_auto_unscaled", std::string(o.uname) + " " + nm(t) + ", [" + nm(b) + ", " + I(u2) + "]", [&] { do_ls(o, t, mem(b, u2)); });
            Reg tb = t;
            if (tb.id == b.id) tb = o.x ? X((b.id + 1) % 31) : W((b.id + 1) % 31);
            put(f + "_pre", std::string(o.name) + " " + nm(tb) + ", [" + nm(b) + ", " + I(u) + "]!", [&] { do_ls(o, tb, pre(b, u)); });
            put(f + "_post", std::string(o.name) + " " + nm(tb) + ", [" + nm(b) + "], " + I(u), [&] { do_ls(o, tb, post(b, u)); });
            static const Extend exts[4] = {UXTW, UXTX, SXTW, SXTX};
            Extend e = exts[ri(0, 3)];
            bool sc = coin();
            Reg ix = rnd(e == UXTX || e == SXTX, AS_ZR);
            std::string suffix;
            if (e == UXTX) suffix = sc ? ", lsl " + I(o.size) : "";
            else suffix = std::string(", ") + extname[e] + (sc ? " " + I(o.size) : "");
            put(f + "_reg", std::string(o.name) + " " + nm(t) + ", [" + nm(b) + ", " + nm(ix) + suffix + "]", [&] { do_ls(o, t, memr(b, ix, e, sc)); });
        }
    }
    // pairs
    struct P { const char* name; int kind; bool x; } ps[] = {{"ldp", 0, false}, {"ldp", 0, true}, {"stp", 1, false}, {"stp", 1, true}, {"ldpsw", 2, true}};
    for (const P& p : ps) {
        std::string f = std::string(p.name) + (p.x ? "_x" : "_w");
        int scale = (p.x && p.kind != 2) ? 8 : 4;
        for (int i = 0; i < N; i++) {
            Reg t1 = rnd(p.x, AS_ZR), t2 = rnd(p.x, AS_ZR), b = rnd(true, AS_SP);
            if (p.kind != 1 && t1.id == t2.id) t2 = p.x ? X((t1.id + 1) % 31) : W((t1.id + 1) % 31);
            int64_t off = re(-64, 63) * scale;
            auto emitp = [&](const Mem& m, Reg a, Reg c) {
                if (p.kind == 0) A.ldp(a, c, m); else if (p.kind == 1) A.stp(a, c, m); else A.ldpsw(a, c, m);
            };
            std::string head = std::string(p.name) + " " + nm(t1) + ", " + nm(t2) + ", [" + nm(b);
            put(f + "_off", head + ", " + I(off) + "]", [&] { emitp(mem(b, off), t1, t2); });
            if (t1.id == b.id || t2.id == b.id) continue;
            put(f + "_pre", head + ", " + I(off) + "]!", [&] { emitp(pre(b, off), t1, t2); });
            put(f + "_post", head + "], " + I(off), [&] { emitp(post(b, off), t1, t2); });
        }
    }
}

static void gen_branches() {
    for (int i = 0; i < N; i++) {
        int64_t o26 = re(-(1 << 25), (1 << 25) - 1) * 4, o19 = re(-(1 << 18), (1 << 18) - 1) * 4, o14 = re(-(1 << 13), (1 << 13) - 1) * 4;
        put("b", "b " + I(o26), [&] { A.b_off(o26); });
        put("bl", "bl " + I(o26), [&] { A.bl_off(o26); });
        Cond c = Cond(ri(0, 15));
        put("b_cond", std::string("b.") + cond_name(c) + " " + I(o19), [&] { A.b_off(c, o19); });
        for (int x = 0; x < 2; x++) {
            Reg t = rnd(x, AS_ZR);
            std::string sx = x ? "_x" : "_w";
            put("cbz" + sx, "cbz " + nm(t) + ", " + I(o19), [&] { A.cbz_off(t, o19); });
            put("cbnz" + sx, "cbnz " + nm(t) + ", " + I(o19), [&] { A.cbnz_off(t, o19); });
            int bit = int(re(0, x ? 63 : 31));
            put("tbz" + sx, "tbz " + nm(t) + ", " + I(bit) + ", " + I(o14), [&] { A.tbz_off(t, bit, o14); });
            put("tbnz" + sx, "tbnz " + nm(t) + ", " + I(bit) + ", " + I(o14), [&] { A.tbnz_off(t, bit, o14); });
            put("ldr_lit" + sx, "ldr " + nm(t) + ", " + I(o19), [&] { A.ldr_lit_off(t, o19); });
        }
        Reg xt = rnd(true, AS_ZR);
        put("ldrsw_lit", "ldrsw " + nm(xt) + ", " + I(o19), [&] { A.ldrsw_lit_off(xt, o19); });
        int64_t ao = re(-(1 << 20), (1 << 20) - 1), pg = re(-(1 << 20), (1 << 20) - 1);
        put("adr", "adr " + nm(xt) + ", " + I(ao), [&] { A.adr_off(xt, ao); });
        put("adrp", "adrp " + nm(xt) + ", " + I(pg * 4096), [&] { A.adrp_off(xt, pg); });
        Reg r = X(int(ri(0, 30)));
        put("br", "br " + nm(r), [&] { A.br(r); });
        put("blr", "blr " + nm(r), [&] { A.blr(r); });
        put("mrs_nzcv", "mrs " + nm(xt) + ", nzcv", [&] { A.mrs_nzcv(xt); });
        put("msr_nzcv", "msr nzcv, " + nm(xt), [&] { A.msr_nzcv(xt); });
        put("ret", "ret " + nm(r), [&] { A.ret(r); });
        uint32_t imm16 = uint32_t(re(0, 0xffff));
        put("brk", "brk " + H(imm16), [&] { A.brk(imm16); });
        put("udf", "udf " + I(imm16), [&] { A.udf(imm16); });
    }
    put("ret_default", "ret", [&] { A.ret(); });
    put("nop", "nop", [&] { A.nop(); });
}

// Logical-immediate encoder/decoder: exhaustive round trip over every encoding,
// plus random non-encodable values must be rejected consistently.
static int check_logical_roundtrip() {
    int bad = 0, valid = 0;
    for (unsigned width : {32u, 64u}) {
        for (uint32_t enc = 0; enc < (width == 64 ? 0x2000u : 0x1000u); enc++) {
            uint64_t v;
            if (!decode_logical_imm(enc, width, &v)) continue;
            valid++;
            uint32_t e2;
            uint64_t v2;
            if (!encode_logical_imm(v, width, &e2) || !decode_logical_imm(e2, width, &v2) || v2 != v) {
                if (bad++ < 5) fprintf(stderr, "logical imm roundtrip failed: width %u enc %x v %llx\n", width, enc, (unsigned long long)v);
            }
        }
        for (int i = 0; i < 200000; i++) {
            uint64_t v = R();
            if (width == 32) v &= 0xffffffffu;
            uint32_t e;
            uint64_t v2;
            if (encode_logical_imm(v, width, &e) && (!decode_logical_imm(e, width, &v2) || v2 != v)) {
                if (bad++ < 5) fprintf(stderr, "logical imm encode wrong for %llx\n", (unsigned long long)v);
            }
        }
    }
    fprintf(stderr, "logical-immediate: %d valid encodings round-tripped, %d failures\n", valid, bad);
    return bad;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: encode_gen OUTDIR [N]\n"); return 2; }
    if (argc > 2) N = atoi(argv[2]);
    std::string dir = argv[1];
    A.reserve(size_t(N) * 4 * 600);
    gen_arith();
    gen_logic();
    gen_move_bitfield();
    gen_loadstore();
    gen_branches();
    int bad = check_logical_roundtrip();
    FILE* s = fopen((dir + "/cases.s").c_str(), "w");
    FILE* b = fopen((dir + "/cases.bin").c_str(), "wb");
    FILE* t = fopen((dir + "/cases.tsv").c_str(), "w");
    if (!s || !b || !t) { perror("open"); return 2; }
    fputs(".text\n", s);
    for (const Case& c : cases) {
        fprintf(s, "%s\n", c.text.c_str());
        fwrite(&c.word, 4, 1, b);
        fprintf(t, "%s\t%s\n", c.form.c_str(), c.text.c_str());
    }
    fclose(s); fclose(b); fclose(t);
    fprintf(stderr, "generated %zu cases\n", cases.size());
    return bad ? 1 : 0;
}
