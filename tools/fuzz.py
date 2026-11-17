#!/usr/bin/env python3
"""fuzz -- differential fuzzer for EPQ engines.

Generates random programs inside the well-defined subset (tools/epqgen.py),
runs them on the baseline and every other engine, compares stdout + exit code,
saves and automatically reduces failing cases.

  tools/fuzz.py --seed S --count N [--profile clean|mixed|nasty|any]
                [--engine NAME=CMD ...] [--pyref] [--jobs J] [--timeout T]
                [--out DIR] [--no-reduce] [--reduce-time SECS]
  tools/fuzz.py --self-check --count N [--profile P]     # baseline + pyref only, statistics
  tools/fuzz.py --reduce DIR_OR_FILE [--engine ...]       # reduce an existing failing case
  tools/fuzz.py --emit DIR --seed S --count N             # just write programs (+ .in) to DIR

The first engine is the reference (baseline).  A baseline timeout/crash means the
generated test is invalid (counted separately; should never happen).  `--pyref`
also runs the Python reference interpreter in-process and reports any
pyref-vs-baseline disagreement as a generator/spec/pyref problem ("suspect").
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epqlib  # noqa: E402
import epqgen  # noqa: E402
import pyref  # noqa: E402

TMP = os.path.join(epqlib.BUILD, "fuzz", "tmp")

# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------

_W = {}


def _winit(engines, timeout, use_pyref):
    _W["engines"] = engines
    _W["timeout"] = timeout
    _W["pyref"] = use_pyref
    d = os.path.join(TMP, "w%d" % os.getpid())
    os.makedirs(d, exist_ok=True)
    _W["dir"] = d


def classify(r):
    """Outcome class of a baseline RunResult."""
    if r.timed_out:
        return "timeout"
    if r.truncated:
        return "output-limit"
    if r.signal is not None:
        return "crash(signal %d)" % r.signal
    if r.rc == 0:
        return "ok"
    msg = epqlib.first_line(r.stderr).decode("utf-8", "replace")
    if "syntax error" in msg:
        return "syntax error"
    if msg.startswith("Undefined variable"):
        return "compile: Undefined variable"
    for m in epqgen.COMPILE_ERRORS + ["Too many", "Classes", "Jump too far", "Loop too far"]:
        if msg.startswith(m):
            return "compile: " + m
    return "runtime: " + msg


def run_all(src, inp, engines, timeout, use_pyref, tag="p"):
    d = _W.get("dir") or os.path.join(TMP, "main%d" % os.getpid())
    os.makedirs(d, exist_ok=True)
    pp = os.path.join(d, tag + ".epq")
    ip = os.path.join(d, tag + ".in")
    with open(pp, "w") as f:
        f.write(src)
    with open(ip, "w") as f:
        f.write(inp)
    res = {}
    for i, e in enumerate(engines):
        t = timeout if i == 0 else max(timeout, 10 * res[engines[0].name].wall + 1)
        r = epqlib.run_engine(e, pp, stdin_path=ip, timeout=t)
        res[e.name] = r
        if i == 0 and (r.timed_out or r.signal is not None or r.truncated):
            break
    pr = None
    if use_pyref:
        pr = pyref.run_source_bigstack(src, inp, step_limit=4000000)
    return res, pr


def diff_names(res, pr, engines):
    base = res[engines[0].name]
    bad = []
    for e in engines[1:]:
        r = res.get(e.name)
        if r is None:
            continue
        if r.timed_out or r.signal is not None or r.truncated or r.spawn_error \
                or r.rc != base.rc or r.stdout != base.stdout:
            bad.append(e.name)
    suspect = False
    if pr is not None and pr.status != "limit":
        if pr.rc != base.rc or pr.stdout.encode("latin-1") != base.stdout:
            suspect = True
    return bad, suspect


def wcase(args):
    seed, profile = args
    engines = _W["engines"]
    src, inp, info = epqgen.generate(seed, profile)
    res, pr = run_all(src, inp, engines, _W["timeout"], _W["pyref"])
    base = res[engines[0].name]
    out = {"seed": seed, "info": info, "base_class": classify(base), "base_wall": base.wall,
           "lines": src.count("\n")}
    if base.timed_out or base.signal is not None or base.truncated:
        out["verdict"] = "invalid"
        out["src"], out["inp"] = src, inp
        return out
    bad, suspect = diff_names(res, pr, engines)
    out["pyref"] = None if pr is None else pr.status
    out["suspect"] = suspect
    out["bad"] = bad
    if bad or suspect:
        out["src"], out["inp"] = src, inp
        out["outputs"] = {n: {"stdout": r.stdout.decode("latin-1"), "stderr": r.stderr.decode("latin-1"),
                              **r.to_dict()} for n, r in res.items()}
        if pr is not None:
            out["outputs"]["pyref"] = {"stdout": pr.stdout, "stderr": pr.stderr, "rc": pr.rc, "status": pr.status}
    out["verdict"] = "fail" if bad else ("suspect" if suspect else "pass")
    return out


# ---------------------------------------------------------------------------
# reducer (works on any program text: parsed with pyref's exact parser)
# ---------------------------------------------------------------------------

STMT_CH = {"IF": (1, 2), "WHL": (1,), "FOR": (0, 2, 3), "RNG": (2,)}


def clone(a):
    return pyref.Ast(a.type, [clone(c) for c in a.ch], a.num, a.id)


def node_at(root, path):
    a = root
    for i in path:
        a = a.ch[i]
    return a


def walk_paths(a, path, ctx, out):
    """Collect (path, ctx) with ctx in {"seq", "stmt", "exp"}."""
    out.append((path, ctx))
    t = a.type
    for i, c in enumerate(a.ch):
        p = path + (i,)
        if t == "SEQ":
            walk_paths(c, p, "stmt", out)
        elif t == "BLK":
            walk_paths(c, p, "seq", out)
        elif t in STMT_CH and i in STMT_CH[t]:
            walk_paths(c, p, "stmt", out)
        elif t == "RNG" and i == 0:
            continue
        elif t == "DEF" and i == 0:
            continue
        elif t == "DEF" and i == 1:
            walk_paths(c, p, "stmt", out)
        elif t == "CAL" and i == 1:
            for j, cc in enumerate(c.ch):
                walk_paths(cc, p + (j,), "exp", out)
        elif t == "SET" and i == 0:
            if c.type == "IND":
                for j, cc in enumerate(c.ch):
                    walk_paths(cc, p + (j,), "exp", out)
        else:
            walk_paths(c, p, "exp", out)


class Reducer:
    def __init__(self, inp, engines, timeout, interesting, budget_s=120, max_tests=3000, log=None):
        self.inp = inp
        self.engines = engines
        self.timeout = timeout
        self.interesting = interesting
        self.deadline = time.time() + budget_s
        self.max_tests = max_tests
        self.tests = 0
        self.log = log or (lambda s: None)

    def test(self, root):
        if self.tests >= self.max_tests or time.time() > self.deadline:
            return False
        self.tests += 1
        src = pyref.unparse(root)
        try:
            pyref.parse(pyref.lex(src, "-", []))
        except pyref.SyntaxErr:
            return False
        return self.interesting(src, self.inp)

    def out_of_budget(self):
        return self.tests >= self.max_tests or time.time() > self.deadline

    def reduce(self, root):
        changed = True
        rounds = 0
        while changed and not self.out_of_budget():
            rounds += 1
            changed = False
            for pas in (self.pass_delete, self.pass_unwrap, self.pass_exp, self.pass_list):
                r = pas(root)
                if r is not None:
                    root = r
                    changed = True
            self.log("  reduce round %d: %d lines, %d tests" % (rounds, pyref.unparse(root).count("\n"), self.tests))
        return root

    def seqs(self, root):
        ps = []
        walk_paths(root, (), "seq", ps)
        return [p for p, c in ps if c == "seq" and node_at(root, p).type == "SEQ"]

    def pass_delete(self, root):
        any_change = False
        seq_paths = self.seqs(root)
        for sp in seq_paths:
            try:
                seq = node_at(root, sp)
            except IndexError:
                continue
            if seq.type != "SEQ":
                continue
            n = len(seq.ch)
            chunk = max(1, n // 2)
            while chunk >= 1 and not self.out_of_budget():
                i = 0
                while i < len(node_at(root, sp).ch) and not self.out_of_budget():
                    cand = clone(root)
                    s = node_at(cand, sp)
                    del s.ch[i:i + chunk]
                    if self.test(cand):
                        root = cand
                        any_change = True
                    else:
                        i += chunk
                chunk //= 2
        return root if any_change else None

    def pass_unwrap(self, root):
        any_change = False
        i = 0
        while not self.out_of_budget():
            ps = []
            walk_paths(root, (), "seq", ps)
            stmts = [p for p, c in ps if c == "stmt"]
            if i >= len(stmts):
                break
            p = stmts[i]
            a = node_at(root, p)
            repls = []
            if a.type in STMT_CH:
                repls = [clone(a.ch[j]) for j in STMT_CH[a.type]]
                repls.append(pyref.Ast("SEQ"))
            elif a.type == "BLK":
                repls = [pyref.Ast("SEQ")]
            done = False
            for r in repls:
                cand = clone(root)
                par = node_at(cand, p[:-1])
                par.ch[p[-1]] = r
                if self.test(cand):
                    root = cand
                    any_change = True
                    done = True
                    break
            if not done:
                i += 1
        return root if any_change else None

    def pass_exp(self, root):
        any_change = False
        i = 0
        while not self.out_of_budget():
            ps = []
            walk_paths(root, (), "seq", ps)
            exps = [p for p, c in ps if c == "exp"]
            if i >= len(exps):
                break
            p = exps[i]
            a = node_at(root, p)
            repls = []
            if not (a.type == "INT" and a.num in (0, 1)) and a.type != "ID":
                repls.append(pyref.Ast("INT", num=0))
                repls.append(pyref.Ast("INT", num=1))
            if a.type not in ("SET", "DEF", "LST", "CAL"):
                repls += [clone(c) for c in a.ch if c.type not in ("LST",) or a.type == "IND"]
            elif a.type == "SET":
                repls.append(clone(a.ch[1]))
            elif a.type == "CAL":
                repls += [clone(c) for c in a.ch[1].ch]
            done = False
            for r in repls:
                cand = clone(root)
                par = node_at(cand, p[:-1])
                par.ch[p[-1]] = r
                if self.test(cand):
                    root = cand
                    any_change = True
                    done = True
                    break
            if not done:
                i += 1
        return root if any_change else None

    def pass_list(self, root):
        any_change = False
        i = 0
        while not self.out_of_budget():
            ps = []
            walk_paths(root, (), "seq", ps)
            lsts = [p for p, c in ps if c == "exp" and node_at(root, p).type == "LST" and node_at(root, p).ch]
            if i >= len(lsts):
                break
            p = lsts[i]
            n = len(node_at(root, p).ch)
            done = False
            for j in range(n):
                cand = clone(root)
                del node_at(cand, p).ch[j]
                if self.test(cand):
                    root = cand
                    any_change = True
                    done = True
                    break
            if not done:
                i += 1
        return root if any_change else None


def make_interesting(engines, timeout, want_bad, guard_pyref):
    """Interesting = baseline finishes, the same engines still disagree with it, and
    (if guard_pyref) pyref still agrees with the baseline (keeps the program well-defined)."""
    want = set(want_bad)

    def f(src, inp):
        res, pr = run_all(src, inp, engines, timeout, guard_pyref, tag="red")
        base = res[engines[0].name]
        if base.timed_out or base.signal is not None or base.truncated:
            return False
        bad, suspect = diff_names(res, pr, engines)
        if guard_pyref and suspect:
            return False
        return want <= set(bad) if want else bool(bad)
    return f


def make_interesting_suspect(engines, timeout):
    """For pyref-vs-baseline disagreements: keep the disagreement."""
    def f(src, inp):
        res, pr = run_all(src, inp, engines[:1], timeout, True, tag="red")
        base = res[engines[0].name]
        if base.timed_out or base.signal is not None or base.truncated:
            return False
        if pr is None or pr.status == "limit":
            return False
        return pr.rc != base.rc or pr.stdout.encode("latin-1") != base.stdout
    return f


def reduce_text(src, inp, interesting, budget, log):
    try:
        root = pyref.parse(pyref.lex(src, "-", []))
    except pyref.SyntaxErr:
        log("  (program does not parse; no reduction)")
        return src
    if not interesting(src, inp):
        log("  (original no longer interesting / flaky; no reduction)")
        return src
    red = Reducer(inp, None, None, interesting, budget_s=budget, log=log)
    root = red.reduce(root)
    return pyref.unparse(root)


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------

def save_case(outdir, r, engines):
    d = os.path.join(outdir, "%s_seed%d" % (r["verdict"], r["seed"]))
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    with open(os.path.join(d, "prog.epq"), "w") as f:
        f.write(r["src"])
    with open(os.path.join(d, "prog.in"), "w") as f:
        f.write(r["inp"])
    for name, o in r.get("outputs", {}).items():
        with open(os.path.join(d, name + ".stdout"), "w", encoding="latin-1") as f:
            f.write(o["stdout"])
        with open(os.path.join(d, name + ".stderr"), "w", encoding="latin-1") as f:
            f.write(o["stderr"])
    meta = {k: v for k, v in r.items() if k not in ("src", "inp", "outputs")}
    meta["engines"] = [repr(e) for e in engines]
    meta["outputs"] = {n: {k: v for k, v in o.items() if k not in ("stdout", "stderr")}
                       for n, o in r.get("outputs", {}).items()}
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return d


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def jobs_iter(seed, count, profile):
    for i in range(count):
        yield (seed + i, profile)


def self_check(a, engines):
    engines = engines[:1]
    t0 = time.time()
    classes = {}
    feats = {}
    featprog = {}
    injected = {}
    n = 0
    agree = 0
    pyref_n = 0
    pyref_limit = 0
    suspects = []
    invalid = []
    tot_bytes = 0
    tot_lines = 0
    walls = []
    with cf.ProcessPoolExecutor(max_workers=a.jobs, initializer=_winit,
                                initargs=(engines, a.timeout, True)) as ex:
        for r in ex.map(wcase, jobs_iter(a.seed, a.count, a.profile), chunksize=4):
            n += 1
            c = r["base_class"]
            classes[c] = classes.get(c, 0) + 1
            info = r["info"]
            tot_bytes += info["bytes"]
            tot_lines += r["lines"]
            walls.append(r["base_wall"])
            for k, v in info["features"].items():
                feats[k] = feats.get(k, 0) + v
                featprog[k] = featprog.get(k, 0) + 1
            if info["err_plan"]:
                injected[info["err_plan"]] = injected.get(info["err_plan"], 0) + 1
            if r["verdict"] == "invalid":
                invalid.append(r)
                continue
            if r.get("pyref") == "limit":
                pyref_limit += 1
            else:
                pyref_n += 1
                if not r["suspect"]:
                    agree += 1
                else:
                    suspects.append(r)
    el = time.time() - t0
    walls.sort()
    lines = []
    P = lines.append
    P("self-check: %d programs, profile=%s, seed=%d..%d, %.1fs (%d jobs)"
      % (n, a.profile, a.seed, a.seed + a.count - 1, el, a.jobs))
    P("")
    P("baseline outcomes:")
    tot_rt = sum(v for k, v in classes.items() if k.startswith("runtime"))
    for k, v in sorted(classes.items(), key=lambda kv: (-kv[1], kv[0])):
        P("  %6d  %6.2f%%  %s" % (v, 100.0 * v / n, k))
    P("  -> clean exits %.1f%%, run-time errors %.1f%%, compile errors %.1f%%, timeouts %.2f%%"
      % (100.0 * classes.get("ok", 0) / n, 100.0 * tot_rt / n,
         100.0 * sum(v for k, v in classes.items() if k.startswith("compile")) / n,
         100.0 * classes.get("timeout", 0) / n))
    P("")
    P("pyref vs baseline: %d/%d agree (%.2f%%), %d skipped (pyref step limit)"
      % (agree, pyref_n, 100.0 * agree / max(pyref_n, 1), pyref_limit))
    P("program size: avg %.0f bytes, %.0f lines; baseline wall: median %.1f ms, p99 %.1f ms, max %.1f ms"
      % (tot_bytes / n, tot_lines / n, 1000 * walls[len(walls) // 2], 1000 * walls[int(len(walls) * 0.99)],
         1000 * walls[-1]))
    P("")
    P("injected error plans: " + ", ".join("%s=%d" % kv for kv in sorted(injected.items())))
    P("")
    P("feature coverage (programs using it / total occurrences):")
    for k in sorted(featprog, key=lambda k: -featprog[k]):
        P("  %-34s %6d  %8d" % (k, featprog[k], feats[k]))
    print("\n".join(lines))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for r in suspects[:50]:
            save_case(a.out, r, engines)
        for r in invalid[:50]:
            r.setdefault("outputs", {})
            save_case(a.out, r, engines)
        if suspects or invalid:
            print("\nsaved %d suspect + %d invalid cases to %s" % (min(len(suspects), 50), min(len(invalid), 50), a.out))
        with open(os.path.join(a.out, "selfcheck_%s_%d.txt" % (a.profile, a.seed)), "w") as f:
            f.write("\n".join(lines) + "\n")
    if a.reduce_suspects and suspects:
        for r in suspects[:a.reduce_suspects]:
            print("\nreducing suspect seed %d ..." % r["seed"])
            it = make_interesting_suspect(engines, a.timeout)
            red = reduce_text(r["src"], r["inp"], it, a.reduce_time, print)
            d = os.path.join(a.out, "suspect_seed%d" % r["seed"])
            with open(os.path.join(d, "prog.min.epq"), "w") as f:
                f.write(red)
            print(red)
    return 1 if (suspects or invalid) else 0


def fuzz(a, engines):
    t0 = time.time()
    n = 0
    fails = []
    susp = []
    inval = []
    with cf.ProcessPoolExecutor(max_workers=a.jobs, initializer=_winit,
                                initargs=(engines, a.timeout, a.pyref)) as ex:
        for r in ex.map(wcase, jobs_iter(a.seed, a.count, a.profile), chunksize=2):
            n += 1
            v = r["verdict"]
            if v == "fail":
                fails.append(r)
                print("FAIL    seed %d (%s): %s disagree with %s" % (r["seed"], r["info"]["profile"],
                                                                   ",".join(r["bad"]), engines[0].name))
            elif v == "suspect":
                susp.append(r)
                print("SUSPECT seed %d: pyref disagrees with %s (generator/spec/pyref issue)"
                      % (r["seed"], engines[0].name))
            elif v == "invalid":
                inval.append(r)
                print("INVALID seed %d: baseline %s" % (r["seed"], r["base_class"]))
            if n % 500 == 0:
                print("... %d done, %d fail, %.0f progs/s" % (n, len(fails), n / (time.time() - t0)))
    el = time.time() - t0
    print()
    print("%d programs in %.1fs (%.1f/s): %d FAIL, %d suspect (pyref), %d invalid (baseline timeout/crash)"
          % (n, el, n / max(el, 1e-9), len(fails), len(susp), len(inval)))
    os.makedirs(a.out, exist_ok=True)
    for r in fails + susp + inval:
        r.setdefault("outputs", {})
        d = save_case(a.out, r, engines)
        r["dir"] = d
    if fails or susp or inval:
        print("saved to %s" % a.out)
    if not a.no_reduce:
        for r in fails[:a.max_reduce]:
            print("reducing seed %d ..." % r["seed"])
            it = make_interesting(engines, a.timeout, r["bad"], guard_pyref=not r.get("suspect"))
            red = reduce_text(r["src"], r["inp"], it, a.reduce_time, print)
            with open(os.path.join(r["dir"], "prog.min.epq"), "w") as f:
                f.write(red)
            print("  -> %s (%d lines)" % (os.path.join(r["dir"], "prog.min.epq"), red.count("\n")))
        for r in susp[:a.max_reduce]:
            print("reducing suspect seed %d ..." % r["seed"])
            it = make_interesting_suspect(engines, a.timeout)
            red = reduce_text(r["src"], r["inp"], it, a.reduce_time, print)
            with open(os.path.join(r["dir"], "prog.min.epq"), "w") as f:
                f.write(red)
            print("  -> %s (%d lines)" % (os.path.join(r["dir"], "prog.min.epq"), red.count("\n")))
    return 1 if (fails or susp) else 0


def reduce_existing(a, engines):
    p = a.reduce
    if os.path.isdir(p):
        prog = os.path.join(p, "prog.epq")
        if not os.path.exists(prog):
            cands = [f for f in os.listdir(p) if f.endswith(".epq") and not f.endswith(".min.epq")]
            prog = os.path.join(p, cands[0])
    else:
        prog = p
    inpath = epqlib.input_for(prog)
    inp = open(inpath).read() if inpath else ""
    src = open(prog).read()
    if a.suspect:
        it = make_interesting_suspect(engines, a.timeout)
    else:
        it = make_interesting(engines, a.timeout, [], guard_pyref=a.pyref)
    red = reduce_text(src, inp, it, a.reduce_time, print)
    outp = os.path.splitext(prog)[0] + ".min.epq"
    with open(outp, "w") as f:
        f.write(red)
    print(red)
    print("written %s" % outp)
    return 0


def emit(a):
    os.makedirs(a.emit, exist_ok=True)
    for s, prof in jobs_iter(a.seed, a.count, a.profile):
        src, inp, info = epqgen.generate(s, prof)
        base = os.path.join(a.emit, "fz_%s_%d" % (info["profile"], s))
        with open(base + ".epq", "w") as f:
            f.write(src)
        with open(base + ".in", "w") as f:
            f.write(inp)
    print("wrote %d programs to %s" % (a.count, a.emit))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--profile", default="any", choices=["clean", "mixed", "nasty", "any"])
    ap.add_argument("--engine", action="append", default=[], metavar="NAME=CMD")
    ap.add_argument("--pyref", action="store_true", help="also compare pyref (in-process) against the baseline")
    ap.add_argument("--jobs", "-j", type=int, default=epqlib.jobs_default())
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(epqlib.BUILD, "fuzz", "failures"))
    ap.add_argument("--no-reduce", action="store_true")
    ap.add_argument("--reduce-time", type=float, default=120.0, help="seconds per reduction")
    ap.add_argument("--max-reduce", type=int, default=5, help="reduce at most this many failures")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--reduce-suspects", type=int, default=0, help="self-check: reduce this many suspects")
    ap.add_argument("--reduce", metavar="CASE", help="reduce an existing failing case (dir or .epq)")
    ap.add_argument("--suspect", action="store_true", help="with --reduce: keep a pyref-vs-baseline disagreement")
    ap.add_argument("--emit", metavar="DIR", help="only write generated programs to DIR")
    a = ap.parse_args(argv)
    if a.emit:
        return emit(a)
    engines = epqlib.engines_from_args(a.engine)
    if a.reduce:
        return reduce_existing(a, engines)
    print("engines: " + ", ".join(repr(e) for e in engines) + ("  + pyref" if (a.pyref or a.self_check) else ""))
    if a.self_check:
        return self_check(a, engines)
    if len(engines) < 2 and not a.pyref:
        print("warning: only one engine and no --pyref: nothing to compare (use --self-check?)")
    return fuzz(a, engines)


if __name__ == "__main__":
    sys.exit(main())
