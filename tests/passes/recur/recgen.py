#!/usr/bin/env python3
"""Targeted generator for the recur pass: self-recursive int functions with
tail calls (swapped / printing / failing arguments), multi self-calls (unroll +
CSE), impure variants (globals, prints), loops around tail calls."""
import random, sys, os

def expr(r, vs, d=0):
    c = r.random()
    if d > 2 or c < 0.3:
        return r.choice(vs + [str(r.randint(-3, 9))])
    op = r.choice(['+', '-', '*', '/', '+', '-'])
    a, b = expr(r, vs, d + 1), expr(r, vs, d + 1)
    if op == '/' and r.random() < 0.7:
        b = '(%s * %s + 1)' % (b, b) if r.random() < 0.5 else str(r.randint(1, 5))
    return '(%s %s %s)' % (a, op, b)

def cond(r, vs):
    return '(%s %s %s)' % (expr(r, vs, 1), r.choice(['<', '>', '=', '<=', '>=', '!=']), expr(r, vs, 2))

def func(r, name, gl):
    np = r.randint(1, 3)
    ps = ['d'] + ['p%d' % i for i in range(np)]
    others = ps[1:]
    lines = ['%s := (%s) => {' % (name, ', '.join(ps))]
    lines.append('    (d <= 0) ? => %s;' % expr(r, others))
    impure = r.random() < 0.3
    if impure:
        k = r.choice(['print', 'gread', 'gwrite'])
        if k == 'print': lines.append('    >> %s;' % expr(r, ps))
        elif k == 'gread': lines.append('    p0 := p0 + %s;' % gl if others else '    >> %s;' % gl)
        else: lines.append('    %s := %s + 1;' % (gl, gl))
    vs = list(ps)
    if r.random() < 0.4:
        lines.append('    t := %s;' % expr(r, vs)); vs.append('t')
    def args(dec):
        a = ['(d - %d)' % dec]
        for p in others:
            c = r.random()
            if c < 0.3: a.append(r.choice(others))          # permutation / parallel-assign trap
            elif c < 0.4: a.append('pr(%s)' % expr(r, vs))  # printing argument
            else: a.append(expr(r, vs))
        return ', '.join(a)
    kind = r.choice(['tail', 'tail', 'multi', 'multi', 'loop'])
    if kind == 'tail':
        for _ in range(r.randint(0, 2)):
            lines.append('    %s ? => %s(%s);' % (cond(r, vs), name, args(r.randint(1, 3))))
        lines.append('    => %s(%s);' % (name, args(1)))
    elif kind == 'multi':
        n = r.randint(2, 3)
        calls = []
        for i in range(n):
            dec = r.randint(1, 2)
            if r.random() < 0.6:
                calls.append('%s(d - %d%s)' % (name, dec, ''.join(', ' + p for p in others)))
            else:
                calls.append('%s(%s)' % (name, args(dec)))
        if r.random() < 0.3:
            lines.append('    %s ? => %s(%s);' % (cond(r, vs), name, args(1)))
        lines.append('    => %s;' % (' %s ' % r.choice(['+', '-', '*'])).join(calls))
    else:
        lines.append('    (i := 0; i < 3; i := i + 1;) @ {')
        lines.append('        %s ? => %s(%s);' % (cond(r, vs + ['i']), name, args(1)))
        lines.append('    }')
        lines.append('    => %s(%s);' % (name, args(2)))
    lines.append('};')
    return lines, np, kind

def prog(seed):
    r = random.Random(seed)
    out = ['pr := (x) => { >> x; => x; };', 'g0 := %d;' % r.randint(0, 5)]
    calls = []
    for k in range(r.randint(1, 3)):
        ls, np, kind = func(r, 'f%d' % k, 'g0')
        out += ls
        depth = r.randint(0, 6 if kind == 'multi' else 25)
        for _ in range(r.randint(1, 3)):
            a = [str(depth)] + [str(r.randint(-5, 20)) for _ in range(np)]
            calls.append('>> f%d(%s);' % (k, ', '.join(a)))
    out += calls
    out.append('>> g0;')
    return '\n'.join(out) + '\n'

d = sys.argv[1]; s0 = int(sys.argv[2]); n = int(sys.argv[3])
os.makedirs(d, exist_ok=True)
for i in range(n):
    open(os.path.join(d, 'rg_%d.epq' % (s0 + i)), 'w').write(prog(s0 + i))
