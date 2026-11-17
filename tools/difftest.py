#!/usr/bin/env python3
"""difftest -- differential runner for EPQ engines.

Runs every program on every engine (first engine = reference, normally the
baseline) and compares stdout bytes + exit code.  stderr is reported but only
compared with --stderr=first-line / --stderr=full.

  tools/difftest.py PROGRAMS_OR_DIRS... [--engine NAME=CMD ...] [--timeout S]
                    [--jobs N] [--out DIR] [--stderr none|first-line|full]

A program's stdin is its sibling FILE.in (foo.epq -> foo.in), else /dev/null.
Reference timeout/crash  -> "skipped (invalid test)", not a failure.
Other engine timeout/signal where the reference finished -> failure.
Exit status: 0 all agree, 1 any mismatch, 2 usage error.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epqlib  # noqa: E402


def compare(ref, other, stderr_mode):
    """Return list of difference descriptions (empty = agree)."""
    diffs = []
    if other.spawn_error:
        return ["could not start: %s" % other.spawn_error]
    if other.timed_out:
        return ["timeout (reference finished in %.3fs)" % ref.wall]
    if other.truncated:
        return ["output limit exceeded"]
    if other.signal is not None:
        return ["killed by signal %d" % other.signal]
    if other.rc != ref.rc:
        diffs.append("exit code %s != reference %s" % (other.rc, ref.rc))
    if other.stdout != ref.stdout:
        a, b = ref.stdout, other.stdout
        i = 0
        n = min(len(a), len(b))
        while i < n and a[i] == b[i]:
            i += 1
        line = a[:i].count(b"\n") + 1
        diffs.append("stdout differs at byte %d (line %d): reference %r vs %r"
                     % (i, line, a[i:i + 40], b[i:i + 40]))
    if stderr_mode == "first-line":
        if epqlib.first_line(other.stderr) != epqlib.first_line(ref.stderr):
            diffs.append("stderr first line %r != reference %r"
                         % (epqlib.first_line(other.stderr), epqlib.first_line(ref.stderr)))
    elif stderr_mode == "full":
        if other.stderr != ref.stderr:
            diffs.append("stderr differs")
    return diffs


def save_case(outdir, prog, stdin_path, engines, results, diffs):
    base = os.path.splitext(os.path.basename(prog))[0]
    d = os.path.join(outdir, base)
    k = 1
    while os.path.exists(d):
        k += 1
        d = os.path.join(outdir, "%s_%d" % (base, k))
    os.makedirs(d)
    shutil.copy(prog, os.path.join(d, base + ".epq"))
    if stdin_path:
        shutil.copy(stdin_path, os.path.join(d, base + ".in"))
    for e, r in zip(engines, results):
        with open(os.path.join(d, e.name + ".stdout"), "wb") as f:
            f.write(r.stdout)
        with open(os.path.join(d, e.name + ".stderr"), "wb") as f:
            f.write(r.stderr)
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump({"program": os.path.abspath(prog), "stdin": stdin_path,
                   "engines": {e.name: {"cmd": e.argv, **r.to_dict()} for e, r in zip(engines, results)},
                   "diffs": diffs}, f, indent=2)
    return d


def run_one(prog, engines, timeout, flags):
    stdin_path = epqlib.input_for(prog)
    results = []
    for i, e in enumerate(engines):
        # the reference gets the full timeout; others get max(timeout, 10x the reference time)
        t = timeout if i == 0 else max(timeout, 10 * results[0].wall + 1)
        r = epqlib.run_engine(e, prog, stdin_path=stdin_path, timeout=t, flags=flags)
        results.append(r)
        if i == 0 and (r.timed_out or r.signal is not None or r.truncated or r.spawn_error):
            # invalid test: do not waste time on the other engines
            for e2 in engines[1:]:
                results.append(epqlib.RunResult())
            break
    return prog, stdin_path, results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--engine", action="append", default=[], metavar="NAME=CMD",
                    help="engine (repeatable); first is the reference. Default: baseline + build/epq jit/aot if present")
    ap.add_argument("--pyref", action="store_true", help="add the Python reference interpreter as an engine")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--jobs", "-j", type=int, default=epqlib.jobs_default())
    ap.add_argument("--out", default=os.path.join(epqlib.BUILD, "difftest", "failures"),
                    help="directory for failing cases (default build/difftest/failures)")
    ap.add_argument("--stderr", choices=["none", "first-line", "full"], default="none")
    ap.add_argument("--flag", action="append", default=[], help="extra flag passed to every engine")
    ap.add_argument("--json", help="write a JSON summary here")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    engines = epqlib.engines_from_args(a.engine)
    if a.pyref:
        engines.append(epqlib.Engine("pyref", [sys.executable, os.path.join(epqlib.TOOLS, "pyref.py")]))
    progs = epqlib.collect_programs(a.paths)
    if not progs:
        print("no programs found", file=sys.stderr)
        return 2
    print("engines: " + ", ".join(repr(e) for e in engines))
    print("programs: %d   timeout: %.1fs   jobs: %d" % (len(progs), a.timeout, a.jobs))

    passed, failed, skipped = [], [], []
    summary = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, p, engines, a.timeout, a.flag) for p in progs]
        for fut in cf.as_completed(futs):
            prog, stdin_path, results = fut.result()
            ref = results[0]
            entry = {"program": prog, "results": {e.name: r.to_dict() for e, r in zip(engines, results)}}
            if ref.spawn_error:
                print("ERROR reference engine could not be started: %s" % ref.spawn_error)
                ex.shutdown(cancel_futures=True)
                return 2
            if ref.timed_out or ref.signal is not None or ref.truncated:
                skipped.append((prog, ref.status()))
                entry["verdict"] = "skipped"
                summary.append(entry)
                print("SKIP  %s  (reference: %s)" % (prog, ref.status()))
                continue
            alld = {}
            for e, r in zip(engines[1:], results[1:]):
                d = compare(ref, r, a.stderr)
                if d:
                    alld[e.name] = d
            if alld:
                saved = save_case(a.out, prog, stdin_path, engines, results, alld) if a.out else None
                failed.append(prog)
                entry["verdict"] = "fail"
                entry["diffs"] = alld
                entry["saved"] = saved
                print("FAIL  %s" % prog)
                for en, ds in alld.items():
                    for d in ds:
                        print("        %-10s %s" % (en, d))
                if saved:
                    print("        saved to %s" % saved)
            else:
                passed.append(prog)
                entry["verdict"] = "pass"
                if a.verbose:
                    times = "  ".join("%s %.3fs" % (e.name, r.wall) for e, r in zip(engines, results))
                    print("ok    %s  [%s]  %s" % (prog, ref.status(), times))
                    if ref.stderr:
                        print("        reference stderr: %s" % epqlib.short(ref.stderr, 120).strip())
            summary.append(entry)

    print()
    print("passed %d   FAILED %d   skipped (reference timeout/crash) %d" % (len(passed), len(failed), len(skipped)))
    for p, st in skipped:
        print("  skipped: %s (%s)" % (p, st))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"engines": [repr(e) for e in engines], "cases": summary,
                       "passed": len(passed), "failed": len(failed), "skipped": len(skipped)}, f, indent=2)
    if len(engines) < 2:
        print("note: only one engine -- nothing was compared")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
