"""Targeted generator for ifconv shapes: loops with small conditional scalar bodies.
usage: python3 tests/passes/ifconv/gen_shapes.py OUTDIR SEED COUNT
then tests/conformance/regen.py BASELINE OUTDIR/*.epq, and tools/fasttest.py OUTDIR with and
without --env EPQ_NO_PASS=ifconv. gen/ holds 24 programs (seed 1) on which the pass fires."""
import os, random, sys

out, seed, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(out, exist_ok=True)
VARS = ["a", "b", "c", "d"]

def atom(r, idx_ok):
    k = r.randrange(6)
    if k == 0: return str(r.randrange(-5, 10))
    if k == 1 and idx_ok: return "l[%s]" % r.choice(idx_ok)
    if k == 2: return "i"
    return r.choice(VARS)

def expr(r, idx_ok, d=0):
    if d > 1 or r.random() < 0.4: return atom(r, idx_ok)
    op = r.choice(["+", "-", "*", "+", "-"])
    return "(%s %s %s)" % (expr(r, idx_ok, d + 1), op, expr(r, idx_ok, d + 1))

def cmpop(r): return r.choice(["<", ">", "=", "<=", ">=", "!="])

def body(r, idx_ok):
    n = r.randint(1, 4)
    ss = []
    for _ in range(n):
        k = r.random()
        if k < 0.08:
            ss.append("%s := %s / (%s);" % (r.choice(VARS), expr(r, idx_ok), atom(r, idx_ok)))
        elif k < 0.14 and idx_ok:
            # unestablished read: may be out of bounds
            ss.append("%s := l[i + %d];" % (r.choice(VARS), r.randint(0, 3)))
        elif k < 0.18:
            ss.append("-m;")
        elif k < 0.22:
            ss.append("l[i] := %s;" % expr(r, idx_ok))
        else:
            ss.append("%s := %s;" % (r.choice(VARS), expr(r, idx_ok)))
    if len(ss) == 1 and r.random() < 0.5: return ss[0]
    return "{ " + " ".join(ss) + " }"

def cond(r, idx_ok):
    k = r.random()
    if k < 0.3:
        return "(%s %s %s)" % (expr(r, idx_ok), cmpop(r), expr(r, idx_ok))
    if k < 0.5:
        j = r.choice(["i + 1", "i + 2", "i"])
        return "(%s < n & l[%s] %s %s)" % (j, j, cmpop(r), expr(r, idx_ok))
    if k < 0.6:
        return "(%s %s %s | %s %s %s)" % (atom(r, idx_ok), cmpop(r), atom(r, idx_ok), atom(r, idx_ok), cmpop(r), atom(r, idx_ok))
    if k < 0.7:
        return "((%s) %s %s)" % ("q()", cmpop(r), atom(r, idx_ok))
    return "(l[i] %s %s)" % (cmpop(r), expr(r, idx_ok))

for t in range(count):
    r = random.Random(seed * 100003 + t)
    n = r.randint(1, 9)
    decl = []
    decl.append("l := [%s];" % ", ".join(str(r.randrange(-9, 20)) for _ in range(n)))
    decl.append("m := [1, 2, 3, 4, 5, 6, 7, 8];")
    decl.append("n := |l|;")
    for v in VARS: decl.append("%s := %d;" % (v, r.randrange(-3, 5)))
    lines = ["cnt := 0;", "q := () => { cnt := cnt + 1; => cnt; };"]
    fn = r.random() < 0.5
    loop = []
    loop.append("(i := 0; i < n; i := i + 1;) @ {")
    for _ in range(r.randint(1, 3)):
        idx_ok = ["i"]
        pre = ""
        if r.random() < 0.3:
            pre = "  t := l[i] + 0;\n"
        loop.append(pre + "  %s ? %s" % (cond(r, idx_ok), body(r, idx_ok)))
    loop.append("  >> [a, b, c, d];")
    loop.append("}")
    tail = [">> [a, b, c, d];", ">> l;"]
    if fn:
        lines.append("f := () => {")
        lines += ["  " + x for x in decl + loop + tail]
        lines.append("  => a + b;")
        lines.append("};")
        lines.append(">> f();")
    else:
        lines += decl + loop + tail
    lines.append(">> cnt;")
    open(os.path.join(out, "g%05d.epq" % t), "w").write("\n".join(lines) + "\n")
