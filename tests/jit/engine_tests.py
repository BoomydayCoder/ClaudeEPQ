#!/usr/bin/env python3
"""End-to-end tests of the ARM64 JIT engine (src/jit/codegen.cpp).

usage: python3 tests/jit/engine_tests.py [-j N] [--epq build/epq]

Runs, with `--engine=jit`:
  progs      tests/jit/progs/*.epq (JIT stress programs; expectations captured
             from the baseline with tests/conformance/regen.py)
  conform    tests/conformance (all programs)
  core       tests/core
  tir        --tir-selftest=all (hand-built TIR for every extension op)
  overflow   unbounded recursion -> "Stack overflow" on stderr, exit 1, stdout
             flushed first; also inside a dynamic call
  dumpasm    --dump-asm prints function names; --jit-stats prints compile time
Exit status 0 iff everything passed.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print("%-5s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, **kw)


def suite(name, runner, epq, jobs, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    r = run([sys.executable, runner, "-j", str(jobs), "-q", "--", epq, "--engine=jit"], env=env)
    last = r.stdout.decode(errors="replace").strip().splitlines()[-1:] or [""]
    record(name, r.returncode == 0, last[0])
    if r.returncode != 0:
        sys.stdout.write(r.stdout.decode(errors="replace")[-3000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=3)
    ap.add_argument("--epq", default=os.path.join(ROOT, "build", "epq"))
    a = ap.parse_args()
    epq = a.epq

    # progs: reuse the conformance runner on tests/jit/progs
    sys.path.insert(0, os.path.join(ROOT, "tests", "conformance"))
    progs_runner = os.path.join(HERE, "run_progs.py")
    suite("progs", progs_runner, epq, a.j)
    suite("conform", os.path.join(ROOT, "tests", "conformance", "run.py"), epq, a.j)
    suite("core", os.path.join(ROOT, "tests", "core", "run.py"), epq, a.j)
    for d in ("inference", "runtime", "semantics"):
        if os.path.isdir(os.path.join(ROOT, "tests", "hunt", d)):
            suite("hunt/" + d, progs_runner, epq, a.j, {"EPQ_PROGS_DIR": os.path.join(ROOT, "tests", "hunt", d)})

    r = run([epq, "--engine=jit", "--tir-selftest=all"])
    out = r.stdout.decode(errors="replace")
    record("tir", r.returncode == 0, (out.strip().splitlines() or [""])[-1])

    tmp = os.path.join(ROOT, "build", "jit-tests")
    os.makedirs(tmp, exist_ok=True)
    cases = {
        "overflow_direct": ">> 1;\nf := (n) => { => f(n + 1) + 1; };\n>> f(0);\n",
        "overflow_dynamic": ">> 2;\nfs := [0];\nfs[0] := (n) => { g := fs[0]; => g(n + 1) + 1; };\nh := fs[0];\n>> h(0);\n",
        "overflow_wide": ">> 3;\nf := (a, b, c, d, e, f2, g, h, i, j) => { => f(a, b, c, d, e, f2, g, h, i, j + 1) + [a][0]; };\n>> f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);\n",
    }
    for name, src in cases.items():
        p = os.path.join(tmp, name + ".epq")
        with open(p, "w") as f:
            f.write(src)
        r = run([epq, "--engine=jit", p])
        so, se = r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")
        ok = r.returncode == 1 and se == "Stack overflow\n" and so.startswith(">> ")
        record(name, ok, "" if ok else "rc=%d stdout=%r stderr=%r" % (r.returncode, so[:80], se[:200]))

    # one huge function (> 32 KB: short branches to the cold stubs need islands);
    # the baseline rejects it ("Loop too far"), so compare with the C backend
    import random
    rnd = random.Random(7)
    lines = ["l := [1, 2, 3, 4, 5, 6, 7, 8];", "d := 0;", "d := [1, 2, 3];", "s := 0;",
             "(i := 0; i < 3; i := i + 1;) @ {"]
    for k in range(3000):
        r = rnd.randrange(6)
        if r == 0:
            lines.append("  s := s + l[%d] * l[(|s| + %d) - (|s| + %d) / 8 * 8];" % (k % 8, k, k))
        elif r == 1:
            lines.append("  l[%d] := l[%d] + s / %d;" % (k % 8, (k + 3) % 8, k + 1))
        elif r == 2:
            lines.append("  s := s + |d| + d[%d];" % (k % 3))
        elif r == 3:
            lines.append("  (s > %d) ? s := s - %d; : s := s + 1;" % (k * 7, k))
        elif r == 4:
            lines.append("  s := s - s / 1000003 * 1000003;")
        else:
            lines.append("  (d[%d] = s) ? s := s + 2;" % (k % 3))
    lines += ["}", ">> s; >> l;"]
    p = os.path.join(tmp, "huge.epq")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    env = dict(os.environ)
    env.setdefault("EPQ_CACHE_DIR", os.path.join(tmp, "cache"))
    env.setdefault("EPQ_CFLAGS", "-O1")
    rj = run([epq, "--engine=jit", p])
    ra = run([epq, "--engine=aot", p], env=env)
    ok = rj.returncode == ra.returncode == 0 and rj.stdout == ra.stdout and len(rj.stdout) > 10
    record("huge_function", ok, "" if ok else "jit rc=%d aot rc=%d" % (rj.returncode, ra.returncode))

    p = os.path.join(ROOT, "bench", "original", "bubblesort.epq")
    r = run([epq, "--dump-asm", p])
    s = r.stdout.decode(errors="replace")
    ok = r.returncode == 0 and "<f2_bubblesort>:" in s and "<f3_swap>:" in s
    record("dumpasm", ok)
    r = run([epq, "--jit-stats", p])
    se = r.stderr.decode(errors="replace")
    record("jitstats", r.returncode == 0 and "jit: compile" in se and "code" in se, se.strip())

    bad = [n for n, ok in results if not ok]
    print("%d/%d JIT engine test groups passed" % (len(results) - len(bad), len(results)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
