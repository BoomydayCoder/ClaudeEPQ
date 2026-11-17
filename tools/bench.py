#!/usr/bin/env python3
"""bench -- benchmark EPQ engines on bench/ (Python 3.9 stdlib only).

For each benchmark and engine:
  1. one verification run (stdout captured through a pipe, drained fully) whose
     stdout + exit code must match the baseline's; mismatching engines are
     reported and their times flagged;
  2. N timed runs, stdout -> /dev/null, stdin from the .in file, wall time measured
     with time.perf_counter() around spawn+wait, and the engine's self-reported
     "Execution time: N microseconds" read from stderr (`-t`).
N is adaptive (more runs for short programs): N = clamp(budget / t_first, min, max),
unless --runs is given.  Runs alternate between engines to spread drift.
Process spawn overhead is measured by timing /usr/bin/true.

  tools/bench.py [--engine NAME=CMD ...] [--bench NAME_OR_GLOB ...] [--runs N]
                 [--budget SECS] [--min-runs N] [--max-runs N] [--out DIR] [--quick]

Writes DIR/bench.md and DIR/bench.json (default build/bench/).  The baseline is
the reference for speedups; geometric means are over benchmarks where both
engines verified.
"""
import argparse
import fnmatch
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epqlib  # noqa: E402

BENCH = os.path.join(epqlib.REPO, "bench")
EXEC_RE = re.compile(rb"Execution time: (\d+) microseconds")


def find_benchmarks(sel):
    out = []
    for sub in ("original", "extra"):
        d = os.path.join(BENCH, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".epq"):
                name = "%s/%s" % (sub, f[:-4])
                if not sel or any(fnmatch.fnmatch(name, s) or fnmatch.fnmatch(f[:-4], s) for s in sel):
                    out.append((name, os.path.join(d, f)))
    return out


def timed_run(argv, stdin_path, timeout):
    """One timed run: stdout to /dev/null, stderr captured (small)."""
    fin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    try:
        t0 = time.perf_counter()
        p = subprocess.Popen(argv, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             start_new_session=True)
        try:
            _, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            p.communicate()
            return None, None, "timeout"
        t1 = time.perf_counter()
    finally:
        if stdin_path:
            fin.close()
    m = EXEC_RE.search(err)
    selft = int(m.group(1)) / 1e6 if m else None
    return t1 - t0, selft, p.returncode


def spawn_overhead(n=30):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        subprocess.run(["/usr/bin/true"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def fmt_t(t):
    if t is None:
        return "-"
    if t < 0.01:
        return "%.2f ms" % (t * 1e3)
    if t < 1:
        return "%.1f ms" % (t * 1e3)
    return "%.3f s" % t


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", action="append", default=[], metavar="NAME=CMD")
    ap.add_argument("--bench", action="append", default=[], help="benchmark name or glob (repeatable)")
    ap.add_argument("--runs", type=int, help="fixed number of timed runs")
    ap.add_argument("--budget", type=float, default=6.0, help="target seconds of timed runs per benchmark/engine")
    ap.add_argument("--min-runs", type=int, default=3)
    ap.add_argument("--max-runs", type=int, default=25)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--quick", action="store_true", help="1 timed run each (smoke test)")
    ap.add_argument("--no-self-time", action="store_true", help="do not pass -t")
    ap.add_argument("--out", default=os.path.join(epqlib.BUILD, "bench"))
    a = ap.parse_args(argv)

    engines = epqlib.engines_from_args(a.engine)
    benches = find_benchmarks(a.bench)
    if not benches:
        print("no benchmarks selected", file=sys.stderr)
        return 2
    ref = engines[0]
    print("engines: " + ", ".join(repr(e) for e in engines))
    sp_min, sp_med = spawn_overhead()
    print("process spawn overhead (/usr/bin/true): min %.2f ms, median %.2f ms" % (sp_min * 1e3, sp_med * 1e3))

    results = []
    any_mismatch = False
    for name, path in benches:
        stdin_path = epqlib.input_for(path)
        row = {"bench": name, "engines": {}}
        # 1. verification
        refout = None
        for e in engines:
            r = epqlib.run_engine(e, path, stdin_path=stdin_path, timeout=a.timeout, max_output=1 << 30)
            st = {"status": r.status(), "verified": None, "first_wall": r.wall}
            if e is ref:
                refout = r
                st["verified"] = (not r.timed_out and r.signal is None)
                st["stdout_bytes"] = len(r.stdout)
            else:
                st["verified"] = (refout is not None and not r.timed_out and r.signal is None
                                  and r.rc == refout.rc and r.stdout == refout.stdout)
                if not st["verified"]:
                    any_mismatch = True
            row["engines"][e.name] = st
        # 2. timing
        flags = [] if a.no_self_time else ["-t"]
        runs = {}
        for e in engines:
            fw = row["engines"][e.name]["first_wall"]
            if a.quick:
                n = 1
            elif a.runs:
                n = a.runs
            else:
                n = int(max(a.min_runs, min(a.max_runs, a.budget / max(fw, 1e-3))))
            runs[e.name] = n
        walls = {e.name: [] for e in engines}
        selfs = {e.name: [] for e in engines}
        for i in range(max(runs.values())):
            for e in engines:
                if i >= runs[e.name] or not row["engines"][e.name]["verified"] and e is not ref and i > 0:
                    continue
                w, s, rc = timed_run(e.argv + flags + [path], stdin_path, a.timeout)
                if w is not None:
                    walls[e.name].append(w)
                if s is not None:
                    selfs[e.name].append(s)
        for e in engines:
            st = row["engines"][e.name]
            ws = walls[e.name]
            st["runs"] = len(ws)
            st["wall_min"] = min(ws) if ws else None
            st["wall_median"] = statistics.median(ws) if ws else None
            ss = selfs[e.name]
            st["self_min"] = min(ss) if ss else None
            st["self_median"] = statistics.median(ss) if ss else None
        rb = row["engines"][ref.name]
        for e in engines[1:]:
            st = row["engines"][e.name]
            if st["verified"] and st["wall_min"] and rb["wall_min"]:
                st["speedup_min"] = rb["wall_min"] / st["wall_min"]
                st["speedup_median"] = rb["wall_median"] / st["wall_median"]
            if st["verified"] and st["self_min"] and rb["self_min"]:
                st["self_speedup"] = rb["self_min"] / st["self_min"]
        results.append(row)
        line = "  %-26s" % name
        for e in engines:
            st = row["engines"][e.name]
            line += "  %s: %s (min of %d)%s" % (e.name, fmt_t(st["wall_min"]), st["runs"],
                                               "" if st["verified"] else " MISMATCH")
        print(line)

    # tables
    md = []
    P = md.append
    P("# EPQ benchmarks")
    P("")
    P("Host: %s %s, Python %s.  Spawn overhead (/usr/bin/true): min %.2f ms, median %.2f ms."
      % (platform.system(), platform.machine(), platform.python_version(), sp_min * 1e3, sp_med * 1e3))
    P("Wall = min / median over N runs, end-to-end (spawn + parse + run), stdout to /dev/null.  "
      "Self = engine-reported `Execution time` (`-t`), min.  Speedup = baseline wall min / engine wall min.")
    P("")
    hdr = "| benchmark | " + " | ".join("%s wall min / med (N) | %s self" % (e.name, e.name) for e in engines)
    hdr += "".join(" | %s speedup (wall) | %s speedup (self)" % (e.name, e.name) for e in engines[1:]) + " |"
    P(hdr)
    P("|" + "---|" * (1 + 2 * len(engines) + 2 * (len(engines) - 1)))
    for row in results:
        cells = [row["bench"]]
        for e in engines:
            st = row["engines"][e.name]
            mark = "" if st["verified"] else " **MISMATCH (%s)**" % st["status"]
            cells.append("%s / %s (%d)%s" % (fmt_t(st["wall_min"]), fmt_t(st["wall_median"]), st["runs"], mark))
            cells.append(fmt_t(st["self_min"]))
        for e in engines[1:]:
            st = row["engines"][e.name]
            cells.append("%.2fx" % st["speedup_min"] if st.get("speedup_min") else "-")
            cells.append("%.2fx" % st["self_speedup"] if st.get("self_speedup") else "-")
        P("| " + " | ".join(cells) + " |")
    geo = {}
    if len(engines) > 1:
        P("")
        P("| engine | geomean speedup (wall min) | geomean speedup (self) | verified |")
        P("|---|---|---|---|")
        for e in engines[1:]:
            g1 = geomean([r["engines"][e.name].get("speedup_min") for r in results])
            g2 = geomean([r["engines"][e.name].get("self_speedup") for r in results])
            nv = sum(1 for r in results if r["engines"][e.name]["verified"])
            geo[e.name] = {"wall": g1, "self": g2, "verified": nv, "total": len(results)}
            P("| %s | %s | %s | %d/%d |" % (e.name, "%.2fx" % g1 if g1 else "-", "%.2fx" % g2 if g2 else "-",
                                            nv, len(results)))
    gb = geomean([r["engines"][ref.name]["wall_min"] for r in results])
    P("")
    P("Geometric mean of %s wall min over %d benchmarks: %s" % (ref.name, len(results), fmt_t(gb)))
    text = "\n".join(md) + "\n"
    print()
    print(text)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "bench.md"), "w") as f:
        f.write(text)
    with open(os.path.join(a.out, "bench.json"), "w") as f:
        json.dump({"engines": [repr(e) for e in engines], "spawn_overhead": {"min": sp_min, "median": sp_med},
                   "results": results, "geomean": geo, "reference_geomean_wall": gb}, f, indent=2)
    print("wrote %s and %s" % (os.path.join(a.out, "bench.md"), os.path.join(a.out, "bench.json")))
    return 1 if any_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
