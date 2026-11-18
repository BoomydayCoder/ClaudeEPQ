#!/usr/bin/env python3
"""Register-pressure stress test of the ARM64 JIT (src/jit/codegen.cpp).

usage: tests/jit/stress.py [--seed S] [--count N] [-j N] [--epq build/epq]
                           [--baseline PATH] [--keep DIR] [--gen-only DIR]

Generates programs whose expressions are WIDE (record / list literals, call
arguments, comparison operands: widths 1..40) and DEEP (right-nested arithmetic
with every left operand a computed value that must be held, Select chains from
nested inlined max/min, long And/Or chains, Comma chains from inlining, nested
literals inside literals inside call arguments; depths 1..60), mixed with calls
so held values must survive them, inside append loops (the JIT's append-loop
register cache), as AppendRange / SetIndex / ElemStore / RecSet kids. Every shape
the JIT might evaluate by holding one temporary per element or level.

Expected output comes from the baseline (EPQ_BASELINE, default refbuild/epq_ref);
the program respects the baseline's limits (<= 255 list-literal elements, <= 255
arguments, <= 256 constants per function body: counted below). Each program then
runs on the JIT with default passes, with every pass off, and with
EPQ_NO_RECORDS=1, plus once under the default engine with EPQ_JIT_TEMP_LIMIT=3
(spill paths under a tiny temporary pool, or a silent fallback to the C engine).
Every run must match the baseline's stdout and exit code exactly; "jit internal
error", an abort or a signal is a failure. Exit status 0 iff all passed.
"""
import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ALL_PASSES = ("definite-assignment,constants,demote-globals,fold,simplify,litcmp,inline,recur,pairs,ifconv,"
              "appendopt,listopt,simplify2,reclaim")
CONFIGS = [
    ("jit", {}, ["--engine=jit"]),
    ("jit-nopass", {"EPQ_NO_PASS": ALL_PASSES}, ["--engine=jit"]),
    ("jit-norec", {"EPQ_NO_RECORDS": "1"}, ["--engine=jit"]),
    ("auto-tl3", {"EPQ_JIT_TEMP_LIMIT": "3", "EPQ_NO_TIERUP": "1"}, []),
]

MAX_CONSTS = 200  # per body (baseline: 256)


class Gen:
    def __init__(self, r):
        self.r = r
        self.V = ["v%d" % i for i in range(8)]  # main's variables (globals, demoted to locals)
        self.funcs = []
        self.body = []

    def ri(self, a, b):
        return self.r.randint(a, b)

    def var(self):
        return self.r.choice(self.V)

    # a computed Int: never a bare local, so a held copy needs a temporary
    def comp(self, call_p=0.25):
        k = self.r.random()
        a, b = self.var(), self.var()
        if k < call_p: return "id(%s)" % a
        if k < 0.5: return "(%s * %s)" % (a, b)
        if k < 0.65: return "(%s - %s)" % (a, b)
        if k < 0.75: return "L[%s]" % self.idx8(a)
        if k < 0.85: return "(%s + %s * %s)" % (a, b, self.var())
        return "|%s|" % self.r.choice(["L", "[%s, %s]" % (a, b), "[%s]" % a])

    @staticmethod
    def idx8(a):  # a mod 8, never negative (truncating division), so always in bounds of an 8-list
        m = "(%s - %s / 8 * 8)" % (a, a)
        return "(%s * (%s >= 0) - %s * (%s < 0))" % (m, m, m, m)

    def leaf(self):
        k = self.r.random()
        if k < 0.35: return self.var()
        if k < 0.45: return str(self.ri(0, 9))
        return self.comp()

    def deep(self, d, call_p=0.25):
        """right-nested: every left operand is computed and held while the rest runs"""
        if d <= 0: return self.leaf()
        op = self.r.choice(["+", "-", "*", "+", "-"])
        if op == "*" and self.r.random() < 0.6: op = "+"
        return "(%s %s %s)" % (self.comp(call_p), op, self.deep(d - 1, call_p))

    def elem(self, d):
        k = self.r.random()
        if d > 0 and k < 0.15: return self.deep(self.ri(1, 6))
        if d > 0 and k < 0.25: return "[%s, %s]" % (self.comp(), self.comp())  # pair
        if d > 0 and k < 0.32: return "[%s]" % ", ".join(self.elem(d - 1) for _ in range(self.ri(1, 5)))
        if d > 0 and k < 0.37: return "w%d(%s)" % (self.ri(0, 1), ", ".join(self.comp() for _ in range(3)))
        return self.leaf()

    def rec_lit(self, w, d=1):
        # heterogeneous: Int fields plus one list field -> a record (EPQ_NO_RECORDS: a list of Any)
        f = [self.elem(d) for _ in range(w)]
        f.insert(self.ri(0, len(f)), "[%s]" % self.var())
        return "[%s]" % ", ".join(f)

    def int_lit(self, w):
        return "[%s]" % ", ".join(self.comp() if self.r.random() < 0.8 else self.var() for _ in range(w))

    def andor(self, n, value):
        parts = ["(%s %s %s)" % (self.comp(), self.r.choice(["<", ">", "=", "!="]), self.leaf()) for _ in range(n)]
        out = parts[0]
        for p in parts[1:]:
            out = "(%s %s %s)" % (out, self.r.choice(["&", "|"]), p) if self.r.random() < 0.5 else \
                  "(%s %s %s)" % (p, self.r.choice(["&", "|"]), out)
        if value:
            return out
        return out

    def select_chain(self, n):
        e = self.var()
        for _ in range(n):
            e = "%s(%s, %s)" % (self.r.choice(["mx", "mn"]), self.comp(0.05), e)
        return e

    def gen(self, kinds):
        r = self.r
        pre = ["id := (x) => { => x; };",
               "mx := (a, b) => { (a > b) ? => a; => b; };",
               "mn := (a, b) => { (a < b) ? => a; => b; };",
               "w0 := (a, b, c) => { => [a, b, c]; };",
               "w1 := (a, b, c) => { => a * b - c; };",
               "L := [3, 1, 4, 1, 5, 9, 2, 6];"]
        for i, v in enumerate(self.V):
            pre.append("%s := %d;" % (v, r.randint(-9, 30)))
        pre.append("(i := 0; i < 3; i := i + 1;) @ { %s }" % " ".join("%s := %s + i;" % (v, v) for v in self.V))
        body = []
        for kind in kinds:
            w = r.choice([1, 2, 3, 5, 8, 9, 10, 12, 16, 20, 24, 31, 40])
            d = r.choice([1, 2, 4, 8, 12, 16, 24, 30, 40, 50, 60])
            if kind == "rec":
                lit = self.rec_lit(w)
                form = r.randrange(6)
                if form == 0: body.append(">> %s;" % lit)
                elif form == 1: body.append("R := %s; >> R; >> R[0]; R[%d] := %s; >> R;" % (lit, r.randrange(w + 1), self.comp()))
                elif form == 2: body.append("%s;" % lit)  # statement position (bug01)
                elif form == 3: body.append("([(%s != 0)]) ? >> 1; : >> 0;" % lit)
                elif form == 4: body.append(">> [%s, 1] = [%s, 1];" % (lit, lit))
                else: body.append(">> w0(%s, %s, %s);" % (lit, self.comp(), self.rec_lit(max(1, w // 2))))
            elif kind == "list":
                ww = r.choice([w, w * 3, min(250, w * 6)])
                lit = self.int_lit(ww)
                body.append(">> %s;" % lit if r.random() < 0.6 else "M := %s; >> |M|; >> M;" % lit)
            elif kind == "deep":
                e = self.deep(d, r.choice([0.0, 0.25, 0.6]))
                form = r.randrange(4)
                if form == 0: body.append(">> %s;" % e)
                elif form == 1: body.append("v%d := %s; >> v%d;" % (r.randrange(8), e, r.randrange(8)))
                elif form == 2: body.append("(%s > 0) ? >> 1; : >> 2;" % e)
                else: body.append(">> deep%d(%s, %s);" % (len(self.funcs), self.var(), self.var()))
                if form == 3:
                    # parameters and locals of a function must not shadow main's globals: u0..u7
                    saved, self.V = self.V, ["u%d" % i for i in range(8)]
                    e2 = self.deep(d, 0.2)
                    self.V = saved
                    self.funcs.append("deep%d := (u0, u1) => { u2 := u0 - u1; u3 := u1 * 3; u4 := u0 + 7; u5 := u1; "
                                      "u6 := 2; u7 := u0 * u1; => %s; };" % (len(self.funcs), e2))
            elif kind == "call":
                n = r.choice([1, 3, 7, 8, 9, 12, 16, 24, 33, 40])
                ps = ["p%d" % i for i in range(n)]
                fn = "c%d" % len(self.funcs)
                ret = " + ".join("%s * %d" % (p, i % 5 + 1) for i, p in enumerate(ps))
                self.funcs.append("%s := (%s) => { => %s; };" % (fn, ", ".join(ps), ret))
                args = [self.elem(0) if r.random() < 0.7 else "id(%s)" % self.deep(r.randint(0, 6)) for _ in ps]
                body.append(">> %s(%s);" % (fn, ", ".join(args)))
                if r.random() < 0.5:  # dynamic call (CallDyn, > 8 args through memory)
                    body.append("FS := [%s]; G := FS[0]; >> G(%s);" % (fn, ", ".join(self.comp() for _ in ps)))
            elif kind == "andor":
                n = r.choice([2, 4, 8, 16, 30, 40])
                if r.random() < 0.5: body.append("(%s) ? >> 1; : >> 0;" % self.andor(n, False))
                else: body.append(">> %s;" % self.andor(n, True))
            elif kind == "select":
                n = r.choice([2, 4, 8, 16, 24, 40])
                body.append("v%d := %s; >> v%d;" % (r.randrange(8), self.select_chain(n), r.randrange(8)))
                x = self.var()
                body.append("(%s > %s) ? %s := %s;" % (self.var(), self.var(), x, self.comp(0.0)))
                # nested ifs over scalar assignments (ifconv: Selects whose kids are Selects)
                fn = "s%d" % len(self.funcs)
                s = "x := a * b - %d;" % r.randint(0, 9)
                for k in range(n):
                    s = "(a > %d) ? { %s }" % (r.randint(-3, 30), s) if r.random() < 0.7 else \
                        "(b < %d) ? { %s y := a - b; }" % (r.randint(-3, 30), s)
                self.funcs.append("%s := (a, b) => { x := a; y := b; %s (a > b) ? y := a * 3; => x + y; };" % (fn, s))
                body.append("(i := 0; i < 12; i := i + 1;) @ >> %s(i * %s - 3, %s - i);" % (fn, self.var(), self.var()))
            elif kind == "nested":
                body.append(">> w0(%s, [%s, %s], %s);" % (self.rec_lit(r.randint(1, 12), 3), self.int_lit(r.randint(1, 12)),
                                                          self.rec_lit(r.randint(1, 8), 2), self.deep(r.randint(1, 20))))
            elif kind == "append":
                lit = self.rec_lit(r.choice([2, 5, 10, 16])) if r.random() < 0.5 else self.deep(r.randint(1, 30), 0.0)
                body.append("A := []; (i := 0; i < %d; i := i + 1;) @ A :+ %s; >> |A|; >> A[0]; >> A;" % (r.randint(1, 9), lit))
                body.append("B := [1, 2]; (xx : L) @ B :+ xx; (i := %s * 0 + 1; i < %s * 0 + 6; i := i + 1;) @ B :+ L[i]; >> B;"
                            % (self.deep(r.randint(1, 12), 0.3), self.deep(r.randint(1, 12), 0.3)))
            elif kind == "store":
                body.append("S := [0, 0, 0, 0, 0, 0, 0, 0]; S[%s] := %s; >> S;"
                            % (self.idx8("(%s)" % self.deep(r.randint(1, 30))), self.deep(r.randint(1, 30))))
                body.append("(i := 0; i < 8; i := i + 1;) @ S[i] := %s; >> S;" % self.deep(r.randint(1, 20), 0.0))
                body.append("Q := %s; Q[%d] := %s; >> Q;" % (self.rec_lit(r.choice([3, 8, 16])), 0, self.deep(r.randint(1, 30))))
            elif kind == "cmp":
                a, b = self.int_lit(w), self.int_lit(w)
                body.append(">> %s = %s; >> %s < %s;" % (a, b, self.deep(d // 2 + 1), self.deep(d // 2 + 1)))
        body.append(">> [%s];" % ", ".join(self.V))
        return "\n".join(pre + self.funcs + body) + "\n"


def count_consts(body):
    return len(re.findall(r"(?<![A-Za-z_\d])\d+", body)) + body.count("=>") + 4


def limits_ok(src):
    # the whole program counts as main's body (an over-estimate for main), each definition separately
    if count_consts(src) > MAX_CONSTS:
        return False
    for m in re.finditer(r"\[([^\[\]]*)\]", src):  # innermost literals: elements
        if m.group(1).count(",") >= 250:
            return False
    return True


KINDS = ["rec", "list", "deep", "call", "andor", "select", "nested", "append", "store", "cmp"]


def make(seed):
    r = random.Random(seed)
    for _ in range(50):
        kinds = [r.choice(KINDS) for _ in range(r.randint(1, 4))]
        src = Gen(r).gen(kinds)
        if limits_ok(src):
            return src
    return Gen(r).gen(["deep"])


def run(cmd, path, env=None, timeout=60):
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd + [path], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           env=e, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return "timeout", b"", b""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("--epq", default=os.path.join(ROOT, "build", "epq"))
    ap.add_argument("--baseline", default=os.environ.get("EPQ_BASELINE") or
                    os.path.join(os.path.dirname(ROOT), "refbuild", "epq_ref"))
    ap.add_argument("--keep", help="write every program (and failures) here")
    ap.add_argument("--gen-only", help="only write the programs into this directory")
    a = ap.parse_args()
    out = a.gen_only or a.keep or tempfile.mkdtemp(prefix="epq-stress-")
    os.makedirs(out, exist_ok=True)
    paths = []
    for k in range(a.count):
        p = os.path.join(out, "st_%d.epq" % (a.seed + k))
        with open(p, "w") as f:
            f.write(make(a.seed + k))
        paths.append(p)
    if a.gen_only:
        return 0

    def one(p):
        bc, bo, _ = run([a.baseline], p, timeout=30)
        if bc == "timeout" or (isinstance(bc, int) and bc < 0):
            return p, "skip", None
        fails = []
        for name, env, args in CONFIGS:
            c, o, e = run([a.epq] + args, p, env)
            if c != bc or o != bo or b"internal error" in e:
                fails.append("%s: rc=%s (baseline %s)%s %s" % (name, c, bc, "" if o == bo else " stdout differs",
                                                              e.decode(errors="replace").strip()[:200]))
        return p, "fail" if fails else "ok", fails

    nok = nfail = nskip = 0
    with ThreadPoolExecutor(a.j) as ex:
        for p, st, fails in ex.map(one, paths):
            if st == "ok": nok += 1
            elif st == "skip": nskip += 1
            else:
                nfail += 1
                print("FAIL %s" % p)
                for f in fails:
                    print("     " + f)
    print("stress: %d passed, %d failed, %d skipped (baseline timeout/signal) of %d; programs in %s"
          % (nok, nfail, nskip, len(paths), out))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
