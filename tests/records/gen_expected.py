#!/usr/bin/env python3
"""Regenerate NAME.out / NAME.code for tests/records/progs/*.epq from the BASELINE.

usage: gen_expected.py [--all] REF_BINARY   (the baseline epq_ref, see tools/build_reference.sh)
Only (re)generates files whose .out is missing unless --all is given.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    regen = '--all' in args
    args = [a for a in args if a != '--all']
    if not args:
        print(__doc__)
        return 2
    ref = args[0]
    progs = os.path.join(HERE, 'progs')
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
                               stderr=subprocess.PIPE, timeout=20)
            out, code = r.stdout, r.returncode
        except subprocess.TimeoutExpired:
            print('TIMEOUT', f)
            continue
        open(base + '.out', 'wb').write(out)
        open(base + '.code', 'w').write('%d\n' % code)
        print('%-28s code=%d %s' % (f, code, r.stderr.decode(errors='replace').strip()[:60]))


if __name__ == '__main__':
    sys.exit(main())
