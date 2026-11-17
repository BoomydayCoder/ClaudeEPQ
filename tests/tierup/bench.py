#!/usr/bin/env python3
"""Tier-up measurements (default code flags; serial; min/median of N runs).

usage: tests/tierup/bench.py [--epq build/epq] [-n 15] [--only probe,trigger,startup,helper,e2e]

  probe    in-process probe cost on a miss (from --verbose)
  trigger  in-process trigger arm cost; wall of a ~1 ms program with and without tier-up
  startup  hit vs today's AOT warm path, wall per run, bench/original
  helper   helper wall time (emit / cc / warm dlopen) for bench/original
  e2e      first vs second run of bench/extra/sieve_big and nqueens: AOT path vs tier-up
"""
import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.path.join(REPO, "build", "tierup", "bench")
ORIG = os.path.join(REPO, "bench", "original")
EPQ = None


def env_for(cache, **extra):
    e = {k: v for k, v in os.environ.items() if not k.startswith("EPQ_")}
    e["EPQ_CACHE_DIR"] = cache
    e["EPQ_TIER_LOG"] = os.path.join(cache, "tier.log")
    e.update({k: str(v) for k, v in extra.items()})
    return e


def fresh(name):
    d = os.path.join(WORK, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def run(args, env, prog, timeout=300):
    inp = os.path.splitext(prog)[0] + ".in"
    with open(inp if os.path.exists(inp) else os.devnull, "rb") as f:
        t = time.perf_counter()
        p = subprocess.run([EPQ] + args + [prog], env=env, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout)
        w = time.perf_counter() - t
    return w, p.returncode, p.stdout, p.stderr.decode(errors="replace")


def busy(cache):
    t = os.path.join(cache, "tier")
    return os.path.isdir(t) and any(n.endswith(".lock") or ".tmp-" in n for n in os.listdir(t))


def wait_idle(cache, limit=600):
    t0 = time.time()
    quiet = 0
    while time.time() - t0 < limit:
        quiet = 0 if busy(cache) else quiet + 1
        if quiet >= 5:
            return
        time.sleep(0.1)


def mm(v):
    v = sorted(v)
    return "min %8.2f  med %8.2f" % (v[0], statistics.median(v))


def progs():
    return sorted(os.path.join(ORIG, n) for n in os.listdir(ORIG) if n.endswith(".epq"))


def b_probe(n):
    c = fresh("probe")
    short = os.path.join(REPO, "tests", "conformance", "expr", "abs_int.epq")
    env = env_for(c, EPQ_TIER_MS=10 ** 6)
    for prog in [short, os.path.join(REPO, "bench", "extra", "dijkstra_big.epq")]:
        P, K, O, A = [], [], [], []
        run(["--engine=aot"], env, prog)
        for _ in range(n * 4):
            e = run(["--verbose"], env, prog)[3]
            m = re.search(r"probe (\d+) us: key (\d+) us, open (\d+)", e)
            a = re.search(r"armed .* in (\d+) us", e)
            P.append(int(m[1]))
            K.append(int(m[2]))
            O.append(int(m[3]))
            A.append(int(a[1]))
        print("%-18s %6d B  probe %s | key %s | open %s | trigger arm %s  (us)" % (
            os.path.basename(prog), os.path.getsize(prog), mm(P), mm(K), mm(O), mm(A)))


def b_trigger(n):
    c = fresh("trigger")
    prog = os.path.join(REPO, "tests", "conformance", "expr", "abs_int.epq")
    on = env_for(c)  # default 20 ms threshold, program ends long before it
    off = env_for(c, EPQ_NO_TIERUP=1)
    run(["--engine=aot"], on, prog)
    W = {"tier-up on": [], "EPQ_NO_TIERUP=1": []}
    for _ in range(n * 10):
        W["tier-up on"].append(run([], on, prog)[0] * 1e3)
        W["EPQ_NO_TIERUP=1"].append(run([], off, prog)[0] * 1e3)
    for k, v in W.items():
        print("%-16s wall ms %s" % (k, mm(v)))
    print("spawns during the short runs: %d" % sum(1 for l in open(os.path.join(c, "tier.log"))
                                                 if " spawn " in l) if os.path.exists(os.path.join(c, "tier.log"))
          else "spawns during the short runs: 0")


def b_startup(n):
    c = fresh("startup")
    print("%-16s %-34s %-34s %s" % ("program", "AOT warm (--engine=aot) wall ms", "tier-up hit wall ms", "exec us (-t)"))
    for prog in progs():
        env = env_for(c)
        run(["--engine=aot"], env, prog)
        run(["--engine=aot"], env, prog)
        run([], env_for(c, EPQ_TIER_MS=0), prog)
        wait_idle(c)
        A, H, X = [], [], []
        for _ in range(n):
            A.append(run(["--engine=aot"], env, prog)[0] * 1e3)
            w, rc, out, err = run(["-t", "--verbose"], env, prog)
            assert "tier-up hit" in err, err
            H.append(w * 1e3)
            X.append(int(re.search(r"Execution time: (\d+)", err)[1]))
        print("%-16s %-34s %-34s %d" % (os.path.basename(prog), mm(A), mm(H), min(X)))


def b_helper(n):
    print("helper (default flags), per program: emit = front end+infer+TIR+passes+emit-c in the helper")
    for prog in progs():
        c = fresh("helper")
        env = env_for(c, EPQ_TIER_MS=0)
        t = time.time()
        run([], env, prog)
        wait_idle(c)
        wall = time.time() - t
        done = [l for l in open(os.path.join(c, "tier.log")) if "compile-done" in l]
        print("%-16s %s  (spawn->idle %.2f s)" % (os.path.basename(prog), done[0].split(" ", 3)[3].strip(), wall))


def b_e2e(n):
    for name in ("sieve_big", "nqueens"):
        prog = os.path.join(REPO, "bench", "extra", name + ".epq")
        rows = []
        for mode in ("before (AOT path, EPQ_NO_TIERUP=1)", "after (auto, tier-up)"):
            F, S = [], []
            for _ in range(max(2, n // 5)):
                c = fresh("e2e")
                env = env_for(c, EPQ_NO_TIERUP=1) if mode.startswith("before") else env_for(c)
                F.append(run([], env, prog)[0] * 1e3)
                wait_idle(c)
                S.append(min(run([], env, prog)[0] * 1e3 for _ in range(3)))
            rows.append((mode, F, S))
        for mode, F, S in rows:
            print("%-10s %-38s first run ms %s | second run ms %s" % (name, mode, mm(F), mm(S)))


def main():
    global EPQ
    ap = argparse.ArgumentParser()
    ap.add_argument("--epq", default=os.path.join(REPO, "build", "epq"))
    ap.add_argument("-n", type=int, default=15)
    ap.add_argument("--only", default="probe,trigger,startup,helper,e2e")
    a = ap.parse_args()
    EPQ = os.path.abspath(a.epq)
    for k in a.only.split(","):
        print("== " + k, flush=True)
        globals()["b_" + k](a.n)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
