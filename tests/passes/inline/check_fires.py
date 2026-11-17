#!/usr/bin/env python3
"""Check that the inline pass fires where intended (and only there).

usage: tests/passes/inline/check_fires.py [EPQ_BINARY]      (default: build/epq)

Runs `EPQ --verify-tir --dump-tir-after=inline` on programs and checks which
CallDirect callees survive, and that InlineBlocks appear. Exit 1 on any failure.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
EPQ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "epq")

# file -> (callee names that must NOT be called directly any more,
#          callee names that MUST still be called directly)
CASES = {
    "tests/passes/inline/nested_args.epq": (["mod", "add", "p"], []),
    "tests/passes/inline/early_returns.epq": (["classify"], []),
    "tests/passes/inline/heapish.epq": (["swap"], []),
    "tests/passes/inline/recursion.epq": (["inc"], ["fib", "even", "odd", "depth", "self"]),
    "tests/passes/inline/short_circuit.epq": (["p", "gt"], []),
    "tests/passes/inline/multi_type.epq": (["id", "fst", "both"], []),
    "bench/original/bubblesort.epq": (["swap", "mod", "bubblesort", "lrandom"], []),
    "bench/original/heapsort.epq": (["swap", "mod"], []),
    "bench/original/mergesort.epq": (["mod", "merge"], ["mergesort"]),
}


def main():
    bad = 0
    for rel, (gone, kept) in CASES.items():
        path = os.path.join(ROOT, rel)
        r = subprocess.run([EPQ, "--verify-tir", "--dump-tir-after=inline", path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        out = r.stdout.decode()
        if r.returncode != 0:
            print("FAIL %s: exit %d %s" % (rel, r.returncode, r.stderr.decode()[:300]))
            bad += 1
            continue
        called = set(re.findall(r"CallDirect f\d+_(\w+)", out))
        errs = []
        for g in gone:
            if g in called:
                errs.append("%s still called directly" % g)
        for k in kept:
            if k not in called:
                errs.append("%s should still be called (recursive)" % k)
        if gone and "InlineBlock" not in out:
            errs.append("no InlineBlock")
        if errs:
            bad += 1
            print("FAIL %s: %s" % (rel, "; ".join(errs)))
        else:
            print("ok   %s" % rel)
    print("%d/%d ok" % (len(CASES) - bad, len(CASES)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
