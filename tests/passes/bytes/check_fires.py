#!/usr/bin/env python3
"""Fires / no-fires check for pass "bytes" (DESIGN.md §5.12).

Every test here has a `// bytes: yes` or `// bytes: no` header line and marks
the allocation site of the list under test with `// FLAG`. This script runs
`epq --dump-types` (which ends with the TIR region layout after the passes)
and checks that the region allocated on the FLAG line is a byte region
(`List(Ln: Int8)`) exactly when the header says yes. It also checks the
benchmarks the pass is meant for, and that `--no-pass=bytes` turns it off.

    python3 tests/passes/bytes/check_fires.py [--epq build/epq]
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# (program, line of the allocation site, expected)
BENCH = [
    ("bench/original/eratosthenes.epq", 5, True),
    ("bench/extra/sieve_big.epq", 3, True),
    ("bench/extra/nqueens.epq", 15, True),
]


def layout(epq, path, extra=()):
    out = subprocess.run([epq, "--dump-types", *extra, path], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, timeout=60, check=True).stdout.decode()
    res = {}  # line -> set of widths ("Int8", "Int", ...)
    sect = out.split("layout (TIR regions after passes):\n", 1)
    if len(sect) != 2:
        raise RuntimeError("no layout section in --dump-types output of " + path)
    for ln in sect[1].splitlines():
        m = re.match(r"\s*List\(L\d+: (\w+)\)\s+sites:(.*)", ln)
        if not m:
            continue
        for s in re.findall(r"line(\d+)/len\d+", m.group(2)):
            res.setdefault(int(s), set()).add(m.group(1))
    return res


def check(epq, path, line, want, failures):
    for extra, exp in (((), want), (("--no-pass=bytes",), False)):
        lay = layout(epq, path, extra)
        got = "Int8" in lay.get(line, set())
        if got != exp:
            failures.append("%s line %d%s: expected %s, layout %s" %
                            (os.path.relpath(path, ROOT), line, " " + " ".join(extra) if extra else "",
                             "bytes" if exp else "no bytes", sorted(lay.get(line, set()))))
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epq", default=os.path.join(ROOT, "build", "epq"))
    a = ap.parse_args()
    failures = []
    n_yes = n_no = 0
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".epq"):
            continue
        path = os.path.join(HERE, name)
        src = open(path).read().splitlines()
        hdr = [l for l in src[:4] if l.startswith("// bytes:")]
        flag = [i + 1 for i, l in enumerate(src) if "// FLAG" in l]
        if not hdr or len(flag) != 1:
            failures.append(name + ": missing `// bytes:` header or exactly one `// FLAG`")
            continue
        want = hdr[0].split(":", 1)[1].strip().startswith("yes")
        if check(a.epq, path, flag[0], want, failures):
            n_yes += 1
        else:
            n_no += 1
    for rel, line, want in BENCH:
        check(a.epq, os.path.join(ROOT, rel), line, want, failures)
    for f in failures:
        print("FAIL", f)
    print("bytes fires check: %d tests expecting bytes, %d expecting none, %d benchmarks; %d failures" %
          (n_yes, n_no, len(BENCH), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
