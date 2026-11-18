#!/usr/bin/env python3
"""Regenerate NAME.out / NAME.code for tests/integration/{progs,known}/*.epq from the BASELINE.

usage: gen_expected.py [--all] REF_BINARY   (the baseline epq_ref, see tools/build_reference.sh)
Only (re)generates files whose .out is missing unless --all is given.
Run the suite with: python3 tools/fasttest.py --epq-arg=--verify-tir tests/integration/progs
(tests/integration/known holds cases of passes that are OFF by default; run them with
--env EPQ_ENABLE_PASS=<pass> to see the failure.)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def gen(ref, progs, regen):
    if not os.path.isdir(progs):
        return
    for f in sorted(os.listdir(progs)):
        if not f.endswith('.epq'):
            continue
        base = os.path.join(progs, f[:-4])
        if os.path.exists(base + '.out') and not regen:
            continue
        stdin = b''
        if os.path.exists(base + '.in'):
            stdin = open(base + '.in', 'rb').read()
        try:
            r = subprocess.run([ref, base + '.epq'], input=stdin, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=60)
        except subprocess.TimeoutExpired:
            print('TIMEOUT', f)
            continue
        open(base + '.out', 'wb').write(r.stdout)
        open(base + '.code', 'w').write('%d\n' % r.returncode)
        print('%-32s code=%d %s' % (f, r.returncode, r.stderr.decode(errors='replace').strip()[:60]))


def main():
    args = sys.argv[1:]
    regen = '--all' in args
    args = [a for a in args if a != '--all']
    if not args:
        print(__doc__)
        return 2
    for d in ('progs', 'known'):
        gen(args[0], os.path.join(HERE, d), regen)
    return 0


if __name__ == '__main__':
    sys.exit(main())
