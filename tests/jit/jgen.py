#!/usr/bin/env python3
"""tests/jit/jgen.py OUTDIR SEED COUNT (expected output: tests/conformance/regen.py BASELINE OUTDIR/*.epq): loop-heavy integer programs aimed at the JIT's loop
optimisations (invariant hoisting, inlined-result coalescing, select fusion,
inlined mod, precise loop liveness, list caches)."""
import os, random, sys

out, seed, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(out, exist_ok=True)

def gen(r):
    V = ["a", "b", "c", "d", "e"]
    lines = ["mod := (x, y) => { => x - y * (x / y); };",
             "f := (x, y) => { t := x * 3 - y; (t > 100) ? t := t - 77; => t; };",
             "g := (x) => { => mod(x, 7) + x / 3; };",
             "sw := (l, i, j) => { t := l[i]; l[i] := l[j]; l[j] := t; };"]
    for v in V:
        lines.append("%s := %d;" % (v, r.randint(-20, 50)))
    lines.append("L := [];")
    lines.append("(i := 0; i < 30; i := i + 1;) @ L :+ mod(i * 37 + 11, 23);")
    depth = [0]
    loopvars = []

    def atom():
        k = r.random()
        if k < 0.45: return r.choice(V)
        if k < 0.6 and loopvars: return r.choice(loopvars)
        if k < 0.8: return str(r.choice([0, 1, 2, 3, 5, 7, 8, 16, 100, 4096, 5000, 65537, -3, 1664525]))
        return r.choice(V)

    def idx(e):
        return "mod(%s, 30) * (mod(%s, 30) >= 0)" % (e, e)

    def expr(d=0):
        k = r.random()
        if d > 2 or k < 0.3: return atom()
        if k < 0.45: return "(%s %s %s)" % (expr(d + 1), r.choice(["+", "-", "*"]), expr(d + 1))
        if k < 0.55: return "mod(%s, %s)" % (expr(d + 1), r.choice(["2", "4", "7", "-8", "65537", "3"]))
        if k < 0.62: return "(%s / %s)" % (expr(d + 1), r.choice(["2", "3", "-4", "1000"]))
        if k < 0.72: return "f(%s, %s)" % (expr(d + 1), expr(d + 1))
        if k < 0.78: return "g(%s)" % expr(d + 1)
        if k < 0.88: return "L[%s]" % idx(expr(d + 1))
        return "(%s %s %s)" % (expr(d + 1), r.choice(["<", ">", "=", "<=", ">=", "!="]), expr(d + 1))

    def cond():
        c = "%s %s %s" % (expr(1), r.choice(["<", ">", "=", "<=", ">=", "!="]), expr(1))
        if r.random() < 0.25:
            c = "(%s) %s (%s %s %s)" % (c, r.choice(["&", "|"]), expr(2), r.choice(["<", ">"]), expr(2))
        return c

    def stmt(n):
        k = r.random()
        v = r.choice(V)
        if k < 0.3: return "%s := %s;" % (v, expr())
        if k < 0.45: return "(%s) ? %s := %s;" % (cond(), v, expr())
        if k < 0.52: return "(%s) ? { %s := %s; %s := %s; }" % (cond(), v, atom(), r.choice(V), atom())
        if k < 0.58: return "(%s) ? %s := %s; : %s := %s;" % (cond(), v, expr(), v, expr())
        if k < 0.64: return "L[%s] := %s;" % (idx(expr(1)), expr(1))
        if k < 0.68: return "(%s) ? sw(L, %s, %s);" % (cond(), idx(atom()), idx(atom()))
        if k < 0.85 and depth[0] < 3:
            depth[0] += 1
            lv = "i%d" % len(loopvars)
            lo, hi = r.randint(-3, 3), r.choice(["%d" % r.randint(0, 12), "%s - %s + 4" % (r.choice(V), r.choice(V)), "%d - %s" % (r.randint(5, 15), lv if loopvars else "0")])
            if loopvars and r.random() < 0.5:
                hi = "%s - %s - 1" % (r.choice(V + loopvars), r.choice(loopvars))
            loopvars.append(lv)
            body = " ".join(stmt(n + 1) for _ in range(r.randint(1, 4)))
            loopvars.pop()
            depth[0] -= 1
            return "(%s := %d; %s < %s; %s := %s + 1;) @ { %s }" % (lv, lo, lv, hi, lv, lv, body)
        if k < 0.9: return ">> %s;" % expr()
        return "%s := %s + %s;" % (v, v, expr())

    for _ in range(r.randint(4, 9)):
        lines.append(stmt(0))
    lines.append(">> [a, b, c, d, e];")
    lines.append(">> L;")
    return "\n".join(lines) + "\n"

for k in range(count):
    r = random.Random(seed + k)
    with open(os.path.join(out, "j%d.epq" % (seed + k)), "w") as f:
        f.write(gen(r))
