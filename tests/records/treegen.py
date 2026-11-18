#!/usr/bin/env python3
"""Random tree / cons-list programs that stress pre-shaped empty records.
usage: treegen.py OUTDIR SEED COUNT REF
Writes OUTDIR/tNNNNN.epq + .out + .code (expected from the baseline REF)."""
import os, random, subprocess, sys

HELPERS = {
 'ins0': '''ins0 := (t, v) => {
    (|t| = 0) ? => [v, [], []];
    (v < t[0]) ? t[1] := ins0(t[1], v); : (v > t[0]) ? t[2] := ins0(t[2], v);
    => t;
};''',
 'ins1': '''ins1 := (t, v) => {
    (t = []) ? => [v, MK(), MK()];
    (v < t[0]) ? t[1] := ins1(t[1], v); : t[2] := ins1(t[2], v);
    => t;
};''',
 'ins2': '''ins2 := (t, v) => {
    (t = []) ? => [v, [], []];
    r := t;
    (1) @ {
        (v < t[0]) ? { (t[1] = []) ? { t[1] := [v, [], []]; => r; } t := t[1]; }
        : { (t[2] = []) ? { t[2] := [v, [], []]; => r; } t := t[2]; }
    }
    => r;
};''',
 'size': '''size := (t) => { (t = []) ? => 0; => 1 + size(t[1]) + size(t[2]); };''',
 'has': '''has := (t, v) => { (t != []) @ { (v = t[0]) ? => 1; (v < t[0]) ? t := t[1]; : t := t[2]; } => 0; };''',
 'cons': '''cons := (h, t) => { => [h, t]; };''',
 'clen': '''clen := (c) => { n := 0; (c != []) @ { n := n + 1; c := c[1]; } => n; };''',
 'csum': '''csum := (c) => { (|c| = 0) ? => 0; => c[0] + csum(c[1]); };''',
 'mk': '''MK := () => { => []; };''',
 'leftmost': '''leftmost := (t) => { (t[1] = []) ? => t; => leftmost(t[1]); };''',
 'get': '''get := (l, k) => { => l[k]; };''',
 'app': '''app := (l, v) => { l :+ v; => l; };''',
 'first': '''first := (l) => { => l[0]; };''',
}


def gen(rng):
    out = []
    ins = rng.choice(['ins0', 'ins1', 'ins2'])
    need = {ins, 'size', 'has', 'cons', 'clen', 'csum', 'leftmost', 'get', 'app', 'first', 'mk'}
    for h in ['mk', 'ins0', 'ins1', 'ins2', 'size', 'has', 'cons', 'clen', 'csum', 'leftmost', 'get', 'app', 'first']:
        if h in need:
            out.append(HELPERS[h])
    trees = ['ta', 'tb']
    conses = ['ca', 'cb']
    init = lambda: rng.choice(['[]', 'MK()'])
    for t in trees:
        out.append('%s := %s;' % (t, init()))
    for c in conses:
        out.append('%s := %s;' % (c, init()))
    out.append('zz := 0;')
    n = rng.randint(4, 30)
    def val():
        return rng.choice(['%d' % rng.randint(0, 40), '(i * %d) - (i * %d) / %d * %d' % ((k := rng.randint(3, 17)), k, (m := rng.randint(5, 41)), m)])
    stmts = []
    stmts.append('(i := 0; i < %d; i := i + 1;) @ ta := %s(ta, %s);' % (n, ins, val()))
    if rng.random() < 0.7:
        stmts.append('(i := 0; i < %d; i := i + 1;) @ tb := %s(tb, %s);' % (rng.randint(0, 12), ins, val()))
    stmts.append('(i := 0; i < %d; i := i + 1;) @ ca := cons(%s, ca);' % (rng.randint(0, 12), val()))
    if rng.random() < 0.5:
        stmts.append('(i := 0; i < %d; i := i + 1;) @ cb := [i, cb];' % rng.randint(0, 6))
    ops = [
        lambda: '>> size(%s);' % rng.choice(trees),
        lambda: '>> has(%s, %d);' % (rng.choice(trees), rng.randint(0, 40)),
        lambda: '>> clen(%s);' % rng.choice(conses),
        lambda: '>> csum(%s);' % rng.choice(conses),
        lambda: '>> %s;' % rng.choice(trees + conses),
        lambda: '>> %s = %s;' % (rng.choice(trees + conses + ['[]']), rng.choice(trees + conses + ['[]'])),
        lambda: '>> %s < %s;' % (rng.choice(trees + ['[]']), rng.choice(trees + ['[]', '[5]', '[5, [], [1]]'])),
        lambda: '(|%s| > 0) ? %s[0] := %s[0] + 1;' % ((t := rng.choice(trees)), t, t),
        lambda: '(|%s| > 0) ? { w := leftmost(%s); w[1] := [-1, [], []]; }' % ((t := rng.choice(trees)), t),
        lambda: '(|%s| > 0) ? { w := %s[%d]; >> w; }' % ((t := rng.choice(trees)), t, rng.randint(1, 2)),
        lambda: 'zz := [%s, zz];' % rng.choice(trees),
        lambda: '>> %s(%s);' % ('first', rng.choice(['[7, 8]'] + trees)) if False else '>> first([7, 8]);',
    ]
    bad_ops = [
        # each breaks the record guess in some way
        lambda: '(|%s| > 0) ? { w := %s[%d]; app(w, 3); }' % ((t := rng.choice(trees)), t, rng.randint(1, 2)),
        lambda: '%s :+ 4;' % rng.choice(trees + conses),
        lambda: '(|%s| > 0) ? >> -%s;' % ((t := rng.choice(trees + conses)), t),
        lambda: '(x : %s) @ { zz := zz + 1; }' % rng.choice(trees + conses),
        lambda: '(|%s| > 0) ? >> get(%s, 0);' % ((t := rng.choice(trees + conses)), t),
        lambda: '(|%s| > 0) ? >> first(%s);' % ((t := rng.choice(trees + conses)), t),
        lambda: 'yy := %s; yy := 3;' % rng.choice(trees + conses),
        lambda: '%s := [1, 2, 3, 4];' % rng.choice(trees + conses),
        lambda: '%s := cons(1, %s);' % ((t := rng.choice(trees)), t),
        lambda: '%s := %s;' % (rng.choice(trees), rng.choice(conses)),
        lambda: 'pl := []; pl :+ 1; pl := %s;' % rng.choice(trees + conses),
        lambda: 'ee := MK(); ee :+ 5; >> ee;',
        lambda: 'kk := 1; (|%s| > 1) ? >> %s[kk];' % ((t := rng.choice(trees + conses)), t),
        lambda: 'hh := [%s, 0]; hh[0] := 5;' % rng.choice(trees),
    ]
    body = []
    for _ in range(rng.randint(3, 14)):
        if rng.random() < 0.25:
            body.append(rng.choice(bad_ops)())
        else:
            body.append(rng.choice(ops)())
    # phases: some ops before growth, some after
    cut = rng.randint(0, len(stmts))
    allst = stmts[:cut] + body[:len(body) // 2] + stmts[cut:] + body[len(body) // 2:]
    allst += ['>> ta;', '>> tb;', '>> ca;', '>> cb;', '>> size(ta);']
    return '\n'.join(out + allst) + '\n'


def main():
    d, seed, count, ref = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(d, exist_ok=True)
    kept = 0
    for i in range(count):
        rng = random.Random(seed * 100003 + i)
        src = gen(rng)
        b = os.path.join(d, 't%05d' % i)
        open(b + '.epq', 'w').write(src)
        try:
            r = subprocess.run([ref, b + '.epq'], stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            os.remove(b + '.epq')
            continue
        if r.returncode not in (0, 1) or b'Argument shadows' in r.stderr or len(r.stdout) > 2000000:
            os.remove(b + '.epq')
            continue
        open(b + '.out', 'wb').write(r.stdout)
        open(b + '.code', 'w').write('%d\n' % r.returncode)
        kept += 1
    print('kept', kept)


main()
