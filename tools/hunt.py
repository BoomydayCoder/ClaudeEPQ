#!/usr/bin/env python3
"""hunt -- long-running batch fuzz campaign over several ClaudeEPQ configurations.

  tools/hunt.py --seed 910000 --chunks 40 --chunk 250 [--profiles opt,opt,nasty,mixed]
                [--cflags -O0] [--jit] [--out build/hunt] [--reduce-time 90]

Per chunk (one call of tools/fuzz.py --batch, programs seed..seed+chunk-1, the profile
cycling through --profiles) the engines are:

  baseline   $EPQ_BASELINE (the oracle)
  aot        build/epq --engine=aot --verify-tir, default passes      (EPQ_CFLAGS=--cflags)
  poison     the same with EPQ_POISON_FREED=1 (run-time switch: reuses aot's objects)
  norec      EPQ_NO_RECORDS=1                                   (every --norec-every chunks)
  rnd        EPQ_NO_PASS=<random subset of the passes>, a new subset per chunk (every --rnd-every)
  jit        build/epq --engine=jit per process (only with --jit)

A divergence of `aot` alone is attributed to a default pass by the reducer's case plus
bisection (see tools/README.md, tests/fuzzfound/README.md).  Everything is logged to
OUT/log.txt (one line per chunk: seeds, profile, disabled set, verdict counts).
"""
import argparse
import os
import random
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
PASSES = ["simplify", "litcmp", "inline", "recur", "pairs", "ifconv", "appendopt", "listopt", "simplify2",
          "reclaim"]
INFRA = ["fold", "definite-assignment", "demote-globals", "constants"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--profiles", default="opt,opt,nasty,opt,mixed")
    ap.add_argument("--cflags", default="-O0")
    ap.add_argument("--jit", action="store_true", help="also run build/epq --engine=jit per process")
    ap.add_argument("--jit-only", metavar="BIN", help="only JIT engines of BIN (per process, no C compiles): "
                    "jit, jitnorec, jitpois, jitrnd")
    ap.add_argument("--rnd-every", type=int, default=2, help="rnd engine on every N-th chunk (0: never)")
    ap.add_argument("--norec-every", type=int, default=2, help="norec engine on every N-th chunk (0: never)")
    ap.add_argument("--out", default=os.path.join(REPO, "build", "hunt"))  # absolute paths below
    ap.add_argument("--reduce-time", type=float, default=90.0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    base = os.environ.get("EPQ_BASELINE")
    if not base or not os.path.exists(base):
        sys.exit("set EPQ_BASELINE to the baseline binary")
    epq = "build/epq"  # relative: REPO contains spaces; fuzz.py runs with cwd=REPO
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    profs = a.profiles.split(",")
    log = open(os.path.join(a.out, "log.txt"), "a")
    for k in range(a.chunks):
        seed = a.seed + k * a.chunk
        prof = profs[k % len(profs)]
        cf = "EPQ_CFLAGS=%s" % a.cflags
        if a.jit_only:
            J = os.path.relpath(os.path.abspath(a.jit_only), REPO)
            dis = ",".join(sorted(rng.sample(PASSES + INFRA, rng.randint(1, 5))))
            eng = ["--engine", "baseline=%s" % base,
                   "--engine", "jit=%s --engine=jit --verify-tir" % J,
                   "--engine", "jitnorec=env EPQ_NO_RECORDS=1 %s --engine=jit" % J,
                   "--engine", "jitpois=env EPQ_POISON_FREED=1 %s --engine=jit" % J,
                   "--engine", "jitrnd=env EPQ_NO_PASS=%s %s --engine=jit" % (dis, J)]
            out = os.path.join(a.out, "c%d" % seed)
            run_chunk(a, k, seed, prof, dis, eng, out, log, batch=False)
            continue
        eng = ["--engine", "baseline=%s" % base,
               "--engine", "aot=env %s %s --engine=aot --verify-tir" % (cf, epq),
               "--engine", "poison=env %s EPQ_POISON_FREED=1 %s --engine=aot" % (cf, epq)]
        if a.norec_every and k % a.norec_every == 0:
            eng += ["--engine", "norec=env %s EPQ_NO_RECORDS=1 %s --engine=aot" % (cf, epq)]
        dis = ""
        if a.rnd_every and k % a.rnd_every == (1 % a.rnd_every):
            pool = PASSES + INFRA
            dis = ",".join(sorted(rng.sample(pool, rng.randint(1, 5))))
            eng += ["--engine", "rnd=env %s EPQ_NO_PASS=%s %s --engine=aot --verify-tir" % (cf, dis, epq)]
        if a.jit:
            eng += ["--engine", "jit=%s --engine=jit" % epq]
        out = os.path.join(a.out, "c%d" % seed)
        run_chunk(a, k, seed, prof, dis, eng, out, log, batch=True)


def run_chunk(a, k, seed, prof, dis, eng, out, log, batch):
    if True:
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(TOOLS, "fuzz.py")] + (["--batch"] if batch else []) + [
                            "--seed", str(seed),
                            "--count", str(a.chunk), "--profile", prof, "--out", out, "--jobs", str(a.jobs),
                            "--reduce-time", str(a.reduce_time), "--max-reduce", "3", "--timeout", "10"] + eng,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO)
        txt = p.stdout.decode("utf-8", "replace")
        summ = [l for l in txt.splitlines() if " programs in " in l]
        fails = [l for l in txt.splitlines() if l.startswith(("FAIL", "SUSPECT", "INVALID"))]
        line = "%s seeds %d..%d profile=%s no_pass=%s rc=%d %.0fs | %s" % (
            time.strftime("%H:%M:%S"), seed, seed + a.chunk - 1, prof, dis or "-", p.returncode, time.time() - t0,
            summ[-1] if summ else txt[-300:].replace("\n", " "))
        log.write(line + "\n")
        for l in fails:
            log.write("    " + l + "\n")
        log.flush()
        print(line, flush=True)
        for l in fails:
            print("    " + l, flush=True)
        with open(os.path.join(a.out, "c%d.txt" % seed), "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
