#!/usr/bin/env python3
# usage: cmp.py [-v] FILE.epq|FILE.batch ...
# A .batch file holds tests separated by lines "### name" (optionally "### name <<< input text").
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
W = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(W, '..', '..'))
ROOT = os.path.abspath(os.path.join(REPO, '..', '..', '..', '..'))
REF = os.path.join(ROOT, 'refbuild', 'epq_ref')
EPQ = os.path.join(REPO, 'build', 'epq')
env = dict(os.environ, EPQ_CACHE_DIR=os.path.join(W, 'cache'), EPQ_CFLAGS='-O1')
TD = os.path.join(W, 't')
os.makedirs(TD, exist_ok=True)

def run(cmd, f):
    inp = f[:-4] + '.in'
    data = open(inp, 'rb').read() if os.path.exists(inp) else b''
    try:
        p = subprocess.run(cmd + [f], input=data, capture_output=True, timeout=120, env=env)
        return p.stdout.decode(errors='replace'), p.returncode, p.stderr.decode(errors='replace').strip().splitlines()[-1:]
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', -99, []

def expand(a):
    if not a.endswith('.batch'):
        return [a]
    out = []
    name, buf, inp = None, [], None
    def flush():
        if name:
            p = os.path.join(TD, name + '.epq')
            open(p, 'w').write('\n'.join(buf) + '\n')
            ip = os.path.join(TD, name + '.in')
            if inp is not None:
                open(ip, 'w').write(inp + '\n')
            elif os.path.exists(ip):
                os.remove(ip)
            out.append(p)
    for line in open(a):
        line = line.rstrip('\n')
        if line.startswith('### '):
            flush()
            hdr = line[4:]
            if '<<<' in hdr:
                name, inp = hdr.split('<<<', 1)
                name = name.strip(); inp = inp.strip()
            else:
                name, inp = hdr.strip(), None
            buf = []
        else:
            buf.append(line)
    flush()
    return out

verbose = '-v' in sys.argv
files = []
for a in sys.argv[1:]:
    if a != '-v':
        files += expand(a)

def one(f):
    b = run([REF], f)
    e = run([EPQ], f)
    if os.environ.get('LAZY') and b[:2] == e[:2]:
        return f, b, e, b
    p = run(['python3', os.path.join(REPO, 'tools', 'pyref.py')], f)
    return f, b, e, p

with ThreadPoolExecutor(int(os.environ.get("J", "3"))) as ex:
    for f, b, e, p in ex.map(one, files):
        ok = b[:2] == e[:2]
        pok = b[:2] == p[:2]
        tag = ('OK  ' if ok else 'DIFF') + ('' if pok else ' (pyref disagrees)')
        print(tag, os.path.basename(f))
        if not ok or verbose or not pok:
            for n, r in (('base', b), ('epq ', e), ('pyrf', p)):
                print('  ', n, 'code', r[1], 'err', r[2], 'out', repr(r[0][:300]))
