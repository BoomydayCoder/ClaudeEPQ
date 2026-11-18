#!/usr/bin/env python3
"""Unusable cache directory: the program must still run and print the right output.

usage: tests/cache/run.py [--epq build/epq] [--quick]

For each engine mode (default = auto, --engine=jit, --engine=aot) and each way the
cache can be unusable (EPQ_CACHE_DIR read-only, EPQ_CACHE_DIR an impossible path,
no EPQ_CACHE_DIR and an unwritable $HOME), runs tests/cache/unwritable.epq with
EPQ_TIER_MS=0 (tier-up would trigger at once) and checks stdout, exit code, an
empty stderr and that no tier-up helper was spawned. Forced --engine=aot falls back
to a private directory under $TMPDIR (removed afterwards); with $TMPDIR unusable
too it must fail with exit code 2 and a message naming both directories.
--quick skips the --engine=aot cases (they compile C: a few seconds).
"""
import argparse
import os
import shutil
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PROG = os.path.join(HERE, "unwritable.epq")
WORK = os.path.join(REPO, "build", "cache-test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epq", default=os.path.join(REPO, "build", "epq"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    with open(os.path.join(HERE, "unwritable.out"), "rb") as f:
        expected = f.read()

    if os.path.isdir(WORK):
        for root, dirs, _ in os.walk(WORK):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o755)
        shutil.rmtree(WORK)
    ro = os.path.join(WORK, "readonly")
    home = os.path.join(WORK, "home-ro")
    tmp = os.path.join(WORK, "tmp")
    for d in (ro, home, tmp):
        os.makedirs(d)
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    os.chmod(home, stat.S_IRUSR | stat.S_IXUSR)
    log = os.path.join(WORK, "tier.log")

    caches = [("read-only EPQ_CACHE_DIR", {"EPQ_CACHE_DIR": ro}),
              ("impossible EPQ_CACHE_DIR", {"EPQ_CACHE_DIR": "/dev/null/epq-cache"}),
              ("unwritable HOME", {"HOME": home})]
    modes = [("default", []), ("jit", ["--engine=jit"])]
    if not a.quick:
        modes.append(("aot", ["--engine=aot"]))

    base = {k: v for k, v in os.environ.items() if not k.startswith("EPQ_")}
    base.update({"EPQ_TIER_MS": "0", "EPQ_TIER_LOG": log, "TMPDIR": tmp, "EPQ_CFLAGS": "-O1"})
    fails = n = 0

    def run(label, args, env, want_out, want_code, want_err=None):
        nonlocal fails, n
        n += 1
        if os.path.exists(log):
            os.unlink(log)
        p = subprocess.run([a.epq] + args + [PROG], env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        err = p.stderr.decode(errors="replace")
        spawned = os.path.exists(log) and " spawn " in open(log).read()
        problems = []
        if p.stdout != want_out:
            problems.append("stdout %r" % p.stdout[:200])
        if p.returncode != want_code:
            problems.append("exit code %d" % p.returncode)
        if want_err is None and err:
            problems.append("stderr %r" % err[:300])
        if want_err is not None and not all(w in err for w in want_err):
            problems.append("stderr %r lacks %r" % (err[:300], want_err))
        if spawned:
            problems.append("tier-up helper spawned")
        leftovers = os.listdir(tmp)
        if leftovers:
            problems.append("left in TMPDIR: %s" % leftovers)
            for x in leftovers:
                shutil.rmtree(os.path.join(tmp, x), ignore_errors=True)
        if problems:
            fails += 1
            print("FAIL %s: %s" % (label, "; ".join(problems)))

    for mname, margs in modes:
        for cname, cenv in caches:
            env = dict(base)
            env.update(cenv)
            run("%s, %s" % (mname, cname), margs, env, expected, 0)
    if not a.quick:
        env = dict(base)
        env.update({"EPQ_CACHE_DIR": "/dev/null/epq-cache", "TMPDIR": "/dev/null/epq-tmp"})
        run("aot, impossible cache and TMPDIR", ["--engine=aot"], env, b"", 2,
            want_err=["/dev/null/epq-cache", "/dev/null/epq-tmp"])

    for root, dirs, _ in os.walk(WORK):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)
    shutil.rmtree(WORK, ignore_errors=True)
    print("cache: %d/%d passed%s" % (n - fails, n, "" if not fails else ", %d failed" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
