#!/usr/bin/env python3
"""Run an EPQ engine over the conformance corpus and compare stdout + exit code.

usage: run.py [--timeout SEC] [-k SUBSTR] [-j N] [-q] [--max-diff LINES] -- CMD [ARGS...]

The program path is appended to CMD, NAME.in (if present) is fed on stdin.
Example:  run.py -- ../../refbuild/epq_ref
          run.py -k lists/ -- ./build/epq --some-flag
Exit status 0 iff every selected test passes.
"""
import difflib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def find_tests(filt):
    out = []
    for root, dirs, files in os.walk(HERE):
        dirs.sort()
        for f in sorted(files):
            if f.endswith('.epq'):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, HERE)
                if not filt or any(s in rel for s in filt):
                    out.append(p)
    return out


def run_test(cmd, test, timeout):
    base = test[:-4]
    stdin = b''
    if os.path.exists(base + '.in'):
        with open(base + '.in', 'rb') as f:
            stdin = f.read()
    with open(base + '.out', 'rb') as f:
        exp_out = f.read()
    with open(base + '.code') as f:
        exp_code = int(f.read().strip())
    try:
        r = subprocess.run(cmd + [test], input=stdin, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        out, code, err = r.stdout, r.returncode, r.stderr
    except subprocess.TimeoutExpired as e:
        out, code, err = (e.stdout or b''), 'timeout', (e.stderr or b'')
    ok = (out == exp_out and code == exp_code)
    return test, ok, exp_out, exp_code, out, code, err


def main():
    args = sys.argv[1:]
    if '--' not in args:
        print(__doc__)
        return 2
    i = args.index('--')
    opts, cmd = args[:i], args[i + 1:]
    if not cmd:
        print(__doc__)
        return 2
    timeout, filt, jobs, quiet, maxdiff = 10.0, [], os.cpu_count() or 4, False, 40
    while opts:
        a = opts.pop(0)
        if a == '--timeout':
            timeout = float(opts.pop(0))
        elif a == '-k':
            filt.append(opts.pop(0))
        elif a == '-j':
            jobs = int(opts.pop(0))
        elif a == '-q':
            quiet = True
        elif a == '--max-diff':
            maxdiff = int(opts.pop(0))
        else:
            print('unknown option %s' % a)
            return 2
    tests = find_tests(filt)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(lambda t: run_test(cmd, t, timeout), tests))
    failed = []
    for test, ok, exp_out, exp_code, out, code, err in results:
        rel = os.path.relpath(test, HERE)
        if ok:
            continue
        failed.append(rel)
        print('FAIL %s' % rel)
        if quiet:
            continue
        if code != exp_code:
            print('  exit code: expected %s, got %s' % (exp_code, code))
        if out != exp_out:
            d = list(difflib.unified_diff(
                exp_out.decode('utf-8', 'replace').splitlines(),
                out.decode('utf-8', 'replace').splitlines(),
                'expected', 'actual', lineterm='', n=2))
            for line in d[:maxdiff]:
                print('  ' + line[:200])
            if len(d) > maxdiff:
                print('  ... (%d more diff lines)' % (len(d) - maxdiff))
        if err:
            e = err.decode('utf-8', 'replace').strip().splitlines()
            print('  stderr: ' + ' | '.join(e[:3])[:200])
    n = len(results)
    print('%d/%d passed, %d failed' % (n - len(failed), n, len(failed)))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
