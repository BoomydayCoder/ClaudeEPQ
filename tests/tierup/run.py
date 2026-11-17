#!/usr/bin/env python3
"""Tier-up tests (src/tierup.cpp, DESIGN.md section 7).

usage: tests/tierup/run.py [--epq build/epq] [--suites] [-j 3] [-k NAME]

Scenarios (each in its own EPQ_CACHE_DIR under build/tierup/test/):
  threshold   miss -> helper only after EPQ_TIER_MS -> marker -> hit, same output
  short       programs that finish before the threshold never spawn a helper
  concurrent  20 simultaneous first runs: one compile, all outputs correct
  corrupt     garbage / stale / mismatching / undlopenable artefacts are ignored,
              removed and replaced
  disabled    EPQ_NO_TIERUP=1 is inert (no files, no helper, no log)
  pipe        piped stdout reaches EOF while the helper is still compiling; the
              helper holds no pipe and its stdio is /dev/null
  status      --tier-status / --tier-clear; EPQ_CFLAGS and --no-pass are in the key
  evict       oldest-first eviction beyond EPQ_TIER_CACHE_MB
--suites additionally runs tests/conformance and tests/core in auto mode with a
cold cache, a warm cache and EPQ_TIER_MS=0 (EPQ_CFLAGS=-O1).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.path.join(REPO, "build", "tierup", "test")
LONG = os.path.join(REPO, "bench", "extra", "nqueens.epq")    # ~50-70 ms
SHORT = os.path.join(REPO, "tests", "conformance", "expr", "abs_int.epq")

EPQ = None
FAILS = []


def check(cond, what):
    if not cond:
        FAILS.append(what)
        print("    FAIL: " + what)
    return cond


def env_for(cache, **extra):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith("EPQ_TIER") or k == "EPQ_NO_TIERUP":
            del e[k]
    e.setdefault("EPQ_CFLAGS", "-O1")  # fast compiles; part of the key like any EPQ_* variable
    e["EPQ_CACHE_DIR"] = cache
    e["EPQ_TIER_LOG"] = os.path.join(cache, "tier.log")
    e.update({k: str(v) for k, v in extra.items()})
    return e


def run(args, env, timeout=120, stdin=subprocess.DEVNULL):
    p = subprocess.run([EPQ] + args, env=env, stdin=stdin, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    return p.returncode, p.stdout, p.stderr.decode(errors="replace")


def fresh(name):
    d = os.path.join(WORK, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def log_lines(cache):
    try:
        with open(os.path.join(cache, "tier.log")) as f:
            return [l.split(" ", 2)[1:] for l in f.read().splitlines()]
    except FileNotFoundError:
        return []


def events(cache, kind):
    return [l for l in log_lines(cache) if l and l[0] == kind]


def busy(cache):
    t = os.path.join(cache, "tier")
    if not os.path.isdir(t):
        return False
    return any(n.endswith(".lock") or ".tmp-" in n for n in os.listdir(t))


def wait_idle(cache, limit=300):
    """Waits until no helper holds a lock or has temporaries (stable 0.5 s)."""
    t0 = time.time()
    quiet = 0
    while time.time() - t0 < limit:
        if busy(cache):
            quiet = 0
        else:
            quiet += 1
            if quiet >= 5:
                return True
        time.sleep(0.1)
    return False


def status(cache, prog, env=None):
    rc, out, _ = run(["--tier-status", prog], env or env_for(cache))
    return rc, out.decode().strip()


def expected(cache, prog):
    """Reference output: the C backend forced (no tier-up involvement); also warms the AOT cache."""
    return run(["--engine=aot", prog], env_for(cache))[:2]


# ---------------------------------------------------------------------------

def t_threshold():
    c = fresh("threshold")
    exp = expected(c, LONG)
    env = env_for(c, EPQ_TIER_MS=20)
    t = time.time()
    first = run([LONG], env)
    wall = time.time() - t
    check(first[:2] == exp, "first run output differs from --engine=aot")
    sp = events(c, "spawn")
    check(len(sp) == 1, "expected exactly one spawn, log: %r" % log_lines(c))
    if sp:
        ms = float(sp[0][1].split(" after ")[1].split(" ms")[0])
        print("    first run %.0f ms wall, helper spawned after %.1f ms" % (wall * 1000, ms))
        check(ms >= 20, "helper spawned before the 20 ms threshold (%.1f ms)" % ms)
    check(wait_idle(c), "helper did not finish")
    rc, st = status(c, LONG)
    check(rc == 0 and st.startswith("ready "), "no ready artefact after the helper: %r" % st)
    done = events(c, "compile-done")
    if done:
        print("    helper: " + done[0][1])
    second = run(["--verbose", LONG], env)
    check("tier-up hit" in second[2], "second run did not use the tier-up dylib")
    check(second[:2] == exp, "second run output differs")
    check(len(events(c, "spawn")) == 1, "the hit spawned another helper")
    # -t on a hit: one Execution time line, as on the normal path.
    third = run(["-t", LONG], env)
    check(third[:2] == exp and third[2].count("Execution time:") == 1, "-t on a hit")
    # exit codes / stderr of a failing program go through the same path
    bad = os.path.join(c, "rterr.epq")
    with open(bad, "w") as f:
        f.write(open(LONG).read() + "\nzz := 0;\n>> 7 / zz;\n")
    e0 = run(["--engine=aot", bad], env)
    run([bad], env_for(c, EPQ_TIER_MS=0))
    wait_idle(c)
    e1 = run(["--verbose", bad], env)
    check("tier-up hit" in e1[2], "run-time-error program did not hit")
    check(e1[:2] == e0[:2] and e1[2].splitlines()[-1] == e0[2].splitlines()[-1],
          "run-time error: exit code / stdout / message differ on a hit (%r vs %r)" % (e1[0], e0[0]))


def t_short():
    c = fresh("short")
    exp = expected(c, SHORT)
    r = run([SHORT], env_for(c))
    check(r[:2] == exp, "short program output")
    exp_l = expected(c, LONG)
    r = run([LONG], env_for(c, EPQ_TIER_MS=5000))
    check(r[:2] == exp_l, "long program output (5 s threshold)")
    time.sleep(0.3)
    check(not events(c, "spawn"), "a program that finished before the threshold spawned a helper")
    check(not os.path.exists(os.path.join(c, "tier")), "tier directory created without a helper")


def t_concurrent():
    c = fresh("concurrent")
    exp = expected(c, LONG)
    env = env_for(c, EPQ_TIER_MS=20)
    procs = [subprocess.Popen([EPQ, LONG], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) for _ in range(20)]
    outs = []
    for p in procs:
        o, _ = p.communicate(timeout=120)
        outs.append((p.returncode, o))
    check(all(o == exp for o in outs), "concurrent first runs: wrong output")
    check(wait_idle(c), "helpers did not finish")
    starts = events(c, "compile-start")
    print("    %d spawns, %d trigger-skips, %d helpers bowed out, %d compile(s)" % (
        len(events(c, "spawn")), len(events(c, "trigger-skip-locked")), len(events(c, "helper-locked")), len(starts)))
    check(len(starts) == 1, "expected exactly one compile, got %d" % len(starts))
    check(status(c, LONG)[0] == 0, "no ready artefact")
    r = run(["--verbose", LONG], env)
    check("tier-up hit" in r[2] and r[:2] == exp, "hit after concurrent runs")


def tier_files(c, ext):
    t = os.path.join(c, "tier")
    return [os.path.join(t, n) for n in os.listdir(t) if n.endswith(ext) and ".tmp-" not in n]


def t_corrupt():
    c = fresh("corrupt")
    exp = expected(c, LONG)
    env = env_for(c, EPQ_TIER_MS=0)
    run([LONG], env)
    check(wait_idle(c), "helper did not finish")
    good_marker = open(tier_files(c, ".ready")[0]).read()
    dylib = tier_files(c, ".dylib")[0]

    def case(name, mutate, log_kind):
        n0 = len(events(c, log_kind))
        mutate()
        r = run(["--verbose", LONG], env)
        check(r[:2] == exp, name + ": output differs")
        check("tier-up hit" not in r[2] or "falling back" in r[2], name + ": used the bad artefact")
        check(len(events(c, log_kind)) == n0 + 1, name + ": expected a %s event" % log_kind)
        check(wait_idle(c), name + ": helper did not finish")
        r = run(["--verbose", LONG], env)
        check("tier-up hit" in r[2] and "falling back" not in r[2] and r[:2] == exp,
              name + ": artefact was not replaced")

    marker = lambda: tier_files(c, ".ready")[0]

    def garbage_marker():
        with open(marker(), "w") as f:
            f.write("\x00garbage\n")

    def stale_version():
        with open(marker(), "w") as f:
            f.write(good_marker.replace("epq-tier 1", "epq-tier 0", 1))

    def wrong_key():
        with open(marker(), "w") as f:
            f.write(good_marker.replace(good_marker.split("\n")[1], "0" * 32, 1))

    def dylib_overwritten():
        with open(tier_files(c, ".dylib")[0], "r+b") as f:
            f.seek(4096)
            f.write(b"\xde\xad\xbe\xef" * 64)

    def dylib_missing():
        os.unlink(tier_files(c, ".dylib")[0])

    def undlopenable():
        # A marker whose size/inode/mtime match, naming a file that is no dylib.
        d = tier_files(c, ".dylib")[0]
        with open(d, "wb") as f:
            f.write(b"not a mach-o file\n" * 100)
        st = os.stat(d)
        lines = good_marker.split("\n")
        lines[3] = "%d %d %d %d" % (st.st_size, st.st_ino, int(st.st_mtime), st.st_mtime_ns % 1000000000)
        with open(marker(), "w") as f:
            f.write("\n".join(lines))

    case("garbage marker", garbage_marker, "bad-marker")
    case("stale marker version", stale_version, "bad-marker")
    case("marker for another key", wrong_key, "bad-marker")
    case("dylib overwritten", dylib_overwritten, "bad-marker")
    case("dylib missing", dylib_missing, "bad-marker")
    case("dylib not loadable", undlopenable, "bad-artefact")
    del dylib
    left = [n for n in os.listdir(os.path.join(c, "tier")) if not n.startswith(".")]
    check(sorted(x.rsplit(".", 1)[1] for x in left) == ["dylib", "ready"], "leftover files: %r" % left)


def t_disabled():
    c = fresh("disabled")
    exp = expected(c, LONG)
    env = env_for(c, EPQ_TIER_MS=0, EPQ_NO_TIERUP=1)
    r = run(["--verbose", LONG], env)
    check(r[:2] == exp, "output with EPQ_NO_TIERUP=1")
    check("tier-up" not in r[2], "tier-up diagnostics with EPQ_NO_TIERUP=1")
    time.sleep(0.3)
    check(not os.path.exists(os.path.join(c, "tier")) and not log_lines(c), "EPQ_NO_TIERUP=1 is not inert")
    # forced engines are not auto: no tier-up either
    for flag in ("--engine=jit", "--jit", "--engine=aot", "--aot"):
        r = run(["--verbose", flag, LONG], env_for(c, EPQ_TIER_MS=0))
        check(r[:2] == exp and "tier-up" not in r[2], flag + " engaged tier-up")
    time.sleep(0.3)
    check(not log_lines(c), "forced engine spawned a helper")


def lsof(pid):
    try:
        return subprocess.run(["lsof", "-n", "-P", "-p", str(pid)], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=30).stdout.decode(errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None


def t_pipe():
    c = fresh("pipe")
    exp = expected(c, LONG)
    env = env_for(c, EPQ_TIER_MS=0)  # the helper starts before the program runs
    rfd, wfd = os.pipe()  # stdin pipe we keep open throughout
    t = time.time()
    p = subprocess.Popen([EPQ, LONG], env=env, stdin=rfd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.close(rfd)
    out = p.stdout.read()  # returns at EOF: every holder of the write end has exited
    eof = time.time() - t
    p.wait(timeout=60)
    p.stderr.read()
    still = busy(c)
    print("    EOF after %.0f ms; helper still compiling at EOF: %s" % (eof * 1000, still))
    check(out == exp[1] and p.returncode == exp[0], "piped output")
    check(eof < 5, "EOF took %.1f s" % eof)
    check(still, "helper finished before EOF (test not meaningful; machine too fast?)")
    sp = events(c, "spawn")
    if check(sp, "no helper spawned"):
        pid = int(sp[0][1].rsplit(" ", 1)[1])
        txt = lsof(pid) if still else None
        if txt:
            fds = [l.split() for l in txt.splitlines()[1:]]
            std = [f for f in fds if len(f) > 3 and f[3][:1] in "012" and f[3][1:2] in ("r", "w", "u")]
            check(all(f[-1] == "/dev/null" for f in std), "helper stdio is not /dev/null: %r" % std)
            check(not any("PIPE" in f or "FIFO" in f for f in fds), "helper holds a pipe")
    os.close(wfd)
    check(wait_idle(c), "helper did not finish")


def t_status():
    c = fresh("status")
    env = env_for(c, EPQ_TIER_MS=0)
    rc, st = status(c, LONG)
    check(rc == 1 and st.startswith("absent"), "--tier-status before: %r" % st)
    run([LONG], env)
    wait_idle(c)
    rc, st = status(c, LONG)
    check(rc == 0 and st.startswith("ready ") and os.path.exists(st[6:]), "--tier-status after: %r" % st)
    # code-affecting configuration is part of the key
    rc2, st2 = status(c, LONG, env_for(c, EPQ_CFLAGS="-O2"))
    check(rc2 == 1, "EPQ_CFLAGS not part of the key")
    rc3 = run(["--tier-status", "--no-pass=fold", LONG], env_for(c))[0]
    check(rc3 == 1, "--no-pass not part of the key")
    rc, out, _ = run(["--tier-clear"], env_for(c))
    check(rc == 0 and status(c, LONG)[0] == 1, "--tier-clear")


def t_evict():
    c = fresh("evict")
    env = env_for(c, EPQ_TIER_MS=0, EPQ_TIER_CACHE_MB=0)  # cap 0: keep only the newest
    other = os.path.join(REPO, "bench", "extra", "tak.epq")
    run([LONG], env)
    wait_idle(c)
    check(status(c, LONG)[0] == 0, "first artefact missing")
    run([other], env)
    wait_idle(c)
    check(status(c, other)[0] == 0, "newest artefact was evicted")
    check(status(c, LONG)[0] == 1, "oldest artefact was not evicted")
    check(len(events(c, "evict")) == 1, "expected one evict event")
    r = run([LONG], env_for(c, EPQ_TIER_MS=10 ** 6))
    check(r[:2] == expected(c, LONG), "output after eviction")


def suite(name, env, jobs):
    ok = True
    for s in ("conformance", "core"):
        r = subprocess.run([sys.executable, os.path.join(REPO, "tests", s, "run.py"), "-j", str(jobs), "-q", "--", EPQ],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600)
        last = r.stdout.decode(errors="replace").strip().splitlines()[-1:]
        print("    %-6s %-12s %s" % (name, s, last[0] if last else "?"))
        ok = check(r.returncode == 0, "%s %s failed" % (name, s)) and ok
    return ok


def fill(env, jobs):
    """Runs the helper synchronously for every suite program (the cold run's
    helpers skip programs while both compile slots are busy)."""
    import concurrent.futures as cf
    import glob
    progs = sorted(glob.glob(os.path.join(REPO, "tests", "conformance", "*", "*.epq")) +
                   glob.glob(os.path.join(REPO, "tests", "core", "**", "*.epq"), recursive=True))
    e = dict(env, EPQ_TIER_JOBS=str(jobs + 1))
    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda p: run(["--tier-compile", p], e, timeout=600), progs))
    return len(progs)


def t_suites(jobs):
    c = fresh("suites")
    env = env_for(c, EPQ_CFLAGS="-O1")
    suite("cold", env, jobs)
    check(wait_idle(c, 1800), "helpers did not finish")
    n_cold = len(tier_files(c, ".ready"))
    n_progs = fill(env, jobs)
    n_ready = len(tier_files(c, ".ready"))
    suite("warm", env, jobs)
    print("    %d artefacts after the cold run, %d after filling (%d programs), %d hits in the warm run" % (
        n_cold, n_ready, n_progs, len(events(c, "hit"))))
    check(len(events(c, "hit")) > 0, "warm run had no hits")
    c0 = fresh("suites0")
    env0 = env_for(c0, EPQ_CFLAGS="-O1", EPQ_TIER_MS=0)
    suite("tier0", env0, jobs)
    check(wait_idle(c0, 1800), "helpers did not finish")
    n0 = len(tier_files(c0, ".ready"))
    suite("tier0w", env0, jobs)
    print("    EPQ_TIER_MS=0: %d artefacts, %d hits on the rerun" % (n0, len(events(c0, "hit"))))


def main():
    global EPQ
    ap = argparse.ArgumentParser()
    ap.add_argument("--epq", default=os.path.join(REPO, "build", "epq"))
    ap.add_argument("--suites", action="store_true")
    ap.add_argument("-j", type=int, default=3)
    ap.add_argument("-k", default="")
    a = ap.parse_args()
    EPQ = os.path.abspath(a.epq)
    os.makedirs(WORK, exist_ok=True)
    tests = [("threshold", t_threshold), ("short", t_short), ("concurrent", t_concurrent), ("corrupt", t_corrupt),
             ("disabled", t_disabled), ("pipe", t_pipe), ("status", t_status), ("evict", t_evict)]
    if a.suites:
        tests.append(("suites", lambda: t_suites(a.j)))
    for name, fn in tests:
        if a.k and a.k not in name:
            continue
        n = len(FAILS)
        t = time.time()
        print("%s ..." % name, flush=True)
        fn()
        print("  %s %s (%.1f s)" % (name, "ok" if len(FAILS) == n else "FAILED", time.time() - t), flush=True)
    print("%d failure(s)" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
