#!/usr/bin/env python3
"""`make check`: a quick self-test that needs nothing outside this repository.

usage: tests/check.py [--epq build/epq] [-j N] [--timeout SEC]

Runs, on the default engine (no --engine flag), with every EPQ_* variable cleared,
a private EPQ_CACHE_DIR under build/ and tier-up off (no background compiles):
  - the six example programs of the original in bench/original (X.in on stdin,
    stdout compared with X.out, exit code 0),
  - tests/conformance and tests/core (X.in, X.out, X.code),
  - tests/cache/run.py --quick (unusable cache directory).
Expected outputs are committed (they were produced by the original implementation).
Prints failures and a one-line summary; exit status 0 iff everything passed.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def collect():
    tests = []
    d = os.path.join(REPO, "bench", "original")
    for f in sorted(os.listdir(d)):
        if f.endswith(".epq") and os.path.exists(os.path.join(d, f[:-4] + ".out")):
            tests.append(os.path.join(d, f))
    for suite in ("conformance", "core"):
        for root, dirs, files in os.walk(os.path.join(HERE, suite)):
            dirs.sort()
            for f in sorted(files):
                if f.endswith(".epq") and os.path.exists(os.path.join(root, f[:-4] + ".out")):
                    tests.append(os.path.join(root, f))
    return tests


def run_one(epq, env, timeout, path):
    base = path[:-4]
    stdin = b""
    if os.path.exists(base + ".in"):
        with open(base + ".in", "rb") as f:
            stdin = f.read()
    with open(base + ".out", "rb") as f:
        want = f.read()
    code = 0
    if os.path.exists(base + ".code"):
        with open(base + ".code") as f:
            code = int(f.read().strip())
    try:
        p = subprocess.run([epq, path], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           env=env, timeout=timeout)
        got, rc, err = p.stdout, p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        got, rc, err = b"", "timeout", b""
    if got == want and rc == code:
        return None
    why = ", ".join(([] if got == want else ["stdout differs"]) +
                    ([] if rc == code else ["exit code %s (expected %d)" % (rc, code)]))
    return "%s: %s%s" % (os.path.relpath(path, REPO), why,
                         ("\n    stderr: " + err.decode(errors="replace").strip().splitlines()[0]) if err.strip() else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epq", default=os.path.join(REPO, "build", "epq"))
    ap.add_argument("-j", type=int, default=max(1, min(8, (os.cpu_count() or 2))))
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    if not os.path.exists(a.epq):
        print("check: %s not found (run make first)" % a.epq)
        return 2
    env = {k: v for k, v in os.environ.items() if not k.startswith("EPQ_")}
    env["EPQ_CACHE_DIR"] = os.path.join(REPO, "build", "check-cache")
    env["EPQ_NO_TIERUP"] = "1"
    tests = collect()
    with ThreadPoolExecutor(a.j) as ex:
        fails = [r for r in ex.map(lambda t: run_one(a.epq, env, a.timeout, t), tests) if r]
    for f in fails:
        print("FAIL " + f)
    n = len(tests)
    cache = subprocess.run([sys.executable, os.path.join(HERE, "cache", "run.py"), "--epq", a.epq, "--quick"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cache_out = cache.stdout.decode(errors="replace").strip()
    if cache.returncode != 0:
        print(cache_out)
    ok = not fails and cache.returncode == 0
    n_ex = sum(1 for t in tests if os.path.relpath(t, REPO).startswith("bench"))
    print("make check: %s -- %d/%d programs passed (%d examples, %d conformance+core), %s" % (
        "OK" if ok else "FAILED", n - len(fails), n, n_ex, n - n_ex, cache_out.splitlines()[-1] if cache_out else "cache: ?"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
