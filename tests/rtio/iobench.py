#!/usr/bin/env python3
"""I/O micro-benchmarks for the runtime (output formatting, list printing, input).

    python3 tests/rtio/iobench.py [--runs N] [--only NAME,...] [--ref] [-- ENGINE CMD...]

Generates the micro programs under build/rtio/micro/, warms the C backend cache,
then reports min / median of N runs of the `-t` execution time and of wall time.
Default engine: build/epq (EPQ_CACHE_DIR=build/rtio/cache). --ref also times the
baseline binary.
"""
import os
import random
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MICRO = os.path.join(REPO, "build", "rtio", "micro")
REF = os.environ.get("EPQ_BASELINE") or os.path.join(REPO, "..", "refbuild", "epq_ref")
if not os.path.exists(REF):  # inside .claude/worktrees/<name>
    REF = os.path.join(REPO, "..", "..", "..", "..", "refbuild", "epq_ref")

PROGS = {
    "ints": "(i := 0; i < 1000000; i := i + 1;) @ >> i - (i / 100) * 100;\n",
    "bigints": "(i := 0; i < 1000000; i := i + 1;) @ >> i * 2147 - 1000000000;\n",
    "intlist": "l := [];\n(i := 0; i < 1000000; i := i + 1;) @ l :+ i * 7 - (i / 3) * 5;\n>> l;\n",
    "nested1": "l := [];\n(i := 0; i < 100000; i := i + 1;) @ l :+ [i - (i / 10) * 10, [i - (i/7)*7], []];\n>> l;\n",
    "nestedN": "l := [];\n(i := 0; i < 100000; i := i + 1;) @ l :+ [i - (i / 10) * 10, [i - (i/7)*7], []];\n"
               "(i := 0; i < 100000; i := i + 1;) @ >> l[i];\n",
    "readints": "s := 0;\n(i := 0; i < 1000000; i := i + 1;) @ s := s + <<;\n>> s;\n",
    "prompts": "(i := 0; i < 1000000; i := i + 1;) @ <<;\n",
}


def gen():
    os.makedirs(MICRO, exist_ok=True)
    for name, src in PROGS.items():
        p = os.path.join(MICRO, name + ".epq")
        if not os.path.exists(p) or open(p).read() != src:
            with open(p, "w") as f:
                f.write(src)
    rin = os.path.join(MICRO, "readints.in")
    if not os.path.exists(rin):
        rnd = random.Random(1)
        with open(rin, "w") as f:
            f.write(" ".join(str(rnd.randint(-1000000, 1000000)) for _ in range(1000000)) + "\n")


def cases():
    b = os.path.join(REPO, "bench")
    out = [("large_output", os.path.join(b, "extra", "large_output.epq"), None)]
    for n in ("bubblesort", "heapsort", "mergesort"):
        out.append((n, os.path.join(b, "original", n + ".epq"), None))
    out.append(("dijkstra", os.path.join(b, "original", "dijkstra.epq"),
                os.path.join(b, "original", "dijkstra.in")))
    for n in PROGS:
        inp = os.path.join(MICRO, "readints.in") if n in ("readints", "prompts") else None
        out.append((n, os.path.join(MICRO, n + ".epq"), inp))
    return out


def run(cmd, prog, inp, env):
    stdin = open(inp, "rb") if inp else subprocess.DEVNULL
    t0 = time.perf_counter()
    r = subprocess.run(cmd + ["-t", prog], stdin=stdin, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, env=env, timeout=120)
    wall = time.perf_counter() - t0
    if inp:
        stdin.close()
    us = None
    for line in r.stderr.decode(errors="replace").splitlines():
        if line.startswith("Execution time:"):
            us = int(line.split()[2])
    return us, wall * 1e6


def main():
    args = sys.argv[1:]
    runs, only, ref, cmd = 7, None, False, None
    while args:
        a = args.pop(0)
        if a == "--runs":
            runs = int(args.pop(0))
        elif a == "--only":
            only = set(args.pop(0).split(","))
        elif a == "--ref":
            ref = True
        elif a == "--":
            cmd = args
            break
    gen()
    env = dict(os.environ)
    env.setdefault("EPQ_CACHE_DIR", os.path.join(REPO, "build", "rtio", "cache"))
    engines = [("epq", cmd or [os.path.join(REPO, "build", "epq")])]
    if ref:
        engines.append(("ref", [REF]))
    print("%-13s %-5s %10s %10s %10s" % ("case", "eng", "t_min_us", "t_med_us", "wall_min_us"))
    for name, prog, inp in cases():
        if only and name not in only:
            continue
        for en, c in engines:
            run(c, prog, inp, env)  # warm
            ts, ws = [], []
            for _ in range(runs):
                t, w = run(c, prog, inp, env)
                if t is not None:
                    ts.append(t)
                ws.append(w)
            print("%-13s %-5s %10d %10d %10d" % (name, en, min(ts) if ts else -1,
                                                  statistics.median(ts) if ts else -1, min(ws)))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
