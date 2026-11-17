#!/usr/bin/env python3
"""Targeted tests for the appendopt pass (src/passes/appendopt.cpp).

usage: run.py [--timeout SEC] [--no-fires] -- CMD [ARGS...]
   e.g. run.py -- build/epq --engine=aot --verify-tir

1. Every NAME.epq here is run with CMD (stdin: NAME.in or empty); stdout and
   the exit code must equal NAME.out / NAME.code (generated from the baseline
   with tests/conformance/regen.py).
2. Unless --no-fires: CMD's first word (the epq binary) is run with
   --dump-tir-after=appendopt on the programs in FIRES, and the number of
   AppendRange / Reserve nodes must match (proves the pass fires where intended
   and stays away where it must not).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (AppendRange count, Reserve count); None = don't care
FIRES = {
    'copy_basic': (3, 0),
    'copy_self': (3, 1),  # l :+ l[i] gets only a hint; the alias and both instances of g copy (run-time guard)
    'copy_bound_aliases_dest': (0, None),  # bound reads a list of the destination's region
    'copy_bound_aliases_dest2': (0, None),
    'copy_bound_is_dest': (0, None),
    'copy_nested_lists': (2, None),
    'copy_after_use': (2, None),
    'copy_dynamic': (1, None),
    'hint_counted': (0, 8),  # all counted loops but the one assigning i in its body
    'hint_foreach_merge': (6, 3),  # 4 merge instances; not the for-each over the appended list
}


def main():
    args = sys.argv[1:]
    timeout = 30.0
    fires = True
    while args and args[0] != '--':
        a = args.pop(0)
        if a == '--timeout':
            timeout = float(args.pop(0))
        elif a == '--no-fires':
            fires = False
        else:
            sys.exit(__doc__)
    if not args or args[0] != '--' or len(args) < 2:
        sys.exit(__doc__)
    cmd = args[1:]
    fails = 0
    n = 0
    for f in sorted(os.listdir(HERE)):
        if not f.endswith('.epq'):
            continue
        n += 1
        base = os.path.join(HERE, f[:-4])
        stdin = b''
        if os.path.exists(base + '.in'):
            stdin = open(base + '.in', 'rb').read()
        want = open(base + '.out', 'rb').read()
        code = int(open(base + '.code').read().strip())
        try:
            r = subprocess.run(cmd + [base + '.epq'], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            print('TIMEOUT', f)
            fails += 1
            continue
        if r.stdout != want or r.returncode != code:
            print('FAIL', f, 'exit', r.returncode, 'want', code)
            print('  stderr:', r.stderr.decode(errors='replace')[:400])
            fails += 1
        if fires and f[:-4] in FIRES:
            d = subprocess.run([cmd[0], '--dump-tir-after=appendopt', base + '.epq'], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout).stdout.decode()
            ops = [ln.strip().split(' ')[0] for ln in d.splitlines()]
            got = (ops.count('AppendRange'), ops.count('Reserve'))
            exp = FIRES[f[:-4]]
            if any(e is not None and e != g for e, g in zip(exp, got)):
                print('FIRES', f, '(AppendRange, Reserve) =', got, 'want', exp)
                fails += 1
    print('%d tests, %d failures' % (n, fails))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
