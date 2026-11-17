#!/usr/bin/env python3
"""Check that the pairs pass fires (or not) where intended.

    python3 tests/passes/pairs/fires.py -- build/epq [--engine=aot]

Runs `CMD --dump-tir-after=pairs PROG` and checks for PairMake in the dump.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# program -> must the pass fire?
EXPECT = {
    'tests/passes/pairs/progs/fires_basic.epq': True,
    'tests/passes/pairs/progs/dijkstra_like.epq': True,
    'tests/passes/pairs/progs/lists_of_pairs.epq': True,
    'bench/original/dijkstra.epq': True,
    'bench/extra/dijkstra_big.epq': True,
    'tests/passes/pairs/progs/mut_append_far.epq': None,  # other, immutable literals are pairs
    'tests/passes/pairs/progs/mut_pop_far.epq': False,
    'tests/passes/pairs/progs/mut_setindex_global.epq': None,  # other, immutable literals are pairs
    'tests/passes/pairs/progs/mut_foreach_var.epq': False,
    'tests/passes/pairs/progs/mut_outer_elem.epq': None,  # other, immutable literals are pairs
    'tests/passes/pairs/progs/neg_pop_disq.epq': None,  # other, immutable literals are pairs
    'tests/passes/pairs/progs/three_and_two.epq': None,  # informational
}


def main():
    args = sys.argv[1:]
    if '--' not in args:
        print(__doc__)
        return 2
    cmd = args[args.index('--') + 1:]
    bad = 0
    for rel, want in EXPECT.items():
        p = os.path.join(ROOT, rel)
        r = subprocess.run(cmd + ['--dump-tir-after=pairs', p], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        fired = b'PairMake' in r.stdout
        ok = want is None or fired == want
        bad += not ok
        print('%-4s %-55s fired=%s' % ('ok' if ok else 'FAIL', rel, fired))
    print('%d/%d fire checks passed' % (len(EXPECT) - bad, len(EXPECT)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
