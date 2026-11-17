#!/usr/bin/env python3
"""Regenerate expected NAME.out / NAME.code files from a baseline binary.

usage: regen.py BASELINE [--timeout SEC] [--check] [TEST.epq ...]

With no TEST arguments every *.epq below this directory is processed.
--check   do not write anything; report tests whose stored expectation differs.
A test whose baseline run times out or dies from a signal is reported and its
expectation is NOT written (such a program is outside the well-defined subset).
Exit status: 0 if all runs were clean (and, with --check, all matched), else 1.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find_tests(paths):
    if paths:
        return [os.path.abspath(p) for p in paths]
    out = []
    for root, dirs, files in os.walk(HERE):
        dirs.sort()
        for f in sorted(files):
            if f.endswith('.epq'):
                out.append(os.path.join(root, f))
    return out


def run_one(binary, test, timeout):
    base = test[:-4]
    stdin = b''
    if os.path.exists(base + '.in'):
        with open(base + '.in', 'rb') as f:
            stdin = f.read()
    try:
        r = subprocess.run([binary, test], input=stdin, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    if r.returncode < 0:
        return None, 'signal %d' % -r.returncode
    return (r.stdout, r.returncode), None


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        return 2
    binary = os.path.abspath(args.pop(0))
    timeout = 5.0
    check = False
    paths = []
    while args:
        a = args.pop(0)
        if a == '--timeout':
            timeout = float(args.pop(0))
        elif a == '--check':
            check = True
        else:
            paths.append(a)
    tests = find_tests(paths)
    bad = 0
    changed = 0
    for t in tests:
        rel = os.path.relpath(t, HERE)
        res, err = run_one(binary, t, timeout)
        if err:
            print('BASELINE FAILURE %s: %s' % (rel, err))
            bad += 1
            continue
        out, code = res
        base = t[:-4]
        if check:
            try:
                with open(base + '.out', 'rb') as f:
                    old_out = f.read()
                with open(base + '.code') as f:
                    old_code = int(f.read().strip())
            except (IOError, ValueError):
                old_out, old_code = None, None
            if old_out != out or old_code != code:
                print('DIFFERS %s' % rel)
                changed += 1
            continue
        with open(base + '.out', 'wb') as f:
            f.write(out)
        with open(base + '.code', 'w') as f:
            f.write('%d\n' % code)
    print('%d tests, %d baseline failures%s' % (
        len(tests), bad, (', %d differ' % changed) if check else ''))
    return 1 if (bad or changed) else 0


if __name__ == '__main__':
    sys.exit(main())
