#!/usr/bin/env python3
"""Times the six original examples: baseline vs build/epq.

usage: tests/core/bench.py [-n RUNS] [--only NAME]
Reports min wall time over RUNS (warm cache) and the `-t` execution time.
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BASE = os.environ.get("EPQ_BASELINE") or os.path.join(os.path.dirname(REPO), "refbuild", "epq_ref")
if not os.path.exists(BASE):
    BASE = os.path.join(REPO, "..", "..", "..", "..", "refbuild", "epq_ref")
EPQ = os.path.join(REPO, "build", "epq")
P = os.path.join(REPO, "tests", "conformance", "programs")
BENCH = os.path.join(REPO, "build", "bench")


def dijkstra_input():
    path = os.path.join(BENCH, "dijkstra_big.in")
    if not os.path.exists(path):
        os.makedirs(BENCH, exist_ok=True)
        import random
        r = random.Random(7)
        n, m = 20000, 100000
        lines = ["%d %d" % (n, m)]
        for _ in range(m):
            lines.append("%d %d %d" % (r.randrange(n), r.randrange(n), r.randrange(1, 1000)))
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    return open(path, "rb").read()


CASES = [
    ("fibonacci", "example_fibonacci.epq", b"40\n"),
    ("eratosthenes", "example_eratosthenes.epq", b"2000000\n"),
    ("dijkstra", "example_dijkstra.epq", None),
    ("bubblesort", "example_bubblesort.epq", b""),
    ("heapsort", "example_heapsort.epq", b""),
    ("mergesort", "example_mergesort.epq", b""),
]


def run(cmd, inp):
    t = time.perf_counter()
    p = subprocess.run(cmd, input=inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    w = time.perf_counter() - t
    m = re.search(rb"Execution time: (\d+) microseconds", p.stderr)
    return w, (int(m.group(1)) if m else None), p.stdout, p.returncode


def main():
    runs = 10
    only = None
    a = sys.argv[1:]
    while a:
        if a[0] == "-n":
            runs = int(a[1]); a = a[2:]
        elif a[0] == "--only":
            only = a[1]; a = a[2:]
        else:
            sys.exit(__doc__)
    print("%-13s %12s %12s %12s %12s %8s" % ("program", "base wall", "base -t", "epq wall", "epq -t", "speedup"))
    for name, f, inp in CASES:
        if only and only not in name:
            continue
        if inp is None:
            inp = dijkstra_input()
        prog = os.path.join(P, f)
        res = {}
        outs = {}
        for eng, cmd in (("base", [BASE, "-t", prog]), ("epq", [EPQ, "-t", prog])):
            run(cmd, inp)  # warm (compile + cache)
            best_w, best_t = 1e9, 1e18
            for _ in range(runs):
                w, t, out, rc = run(cmd, inp)
                best_w = min(best_w, w)
                if t is not None:
                    best_t = min(best_t, t)
                outs[eng] = (out, rc)
            res[eng] = (best_w, best_t)
        ok = "" if outs["base"] == outs["epq"] else "  OUTPUT MISMATCH"
        print("%-13s %10.2fms %10.2fms %10.2fms %10.2fms %7.1fx%s" % (
            name, res["base"][0] * 1e3, res["base"][1] / 1e3, res["epq"][0] * 1e3, res["epq"][1] / 1e3,
            res["base"][0] / res["epq"][0], ok))


if __name__ == "__main__":
    main()
