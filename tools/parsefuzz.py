#!/usr/bin/env python3
"""parsefuzz -- parser differential fuzzer (baseline `-e` AST dump vs other engines).

Generates token sequences from the grammar of LANGUAGE.md section 2, biased
toward the tricky spots ('|' abs vs or, '!' vs comparisons, unary minus, ':='
nesting, ':+', '(' at statement start: paren-exp / if / while / for /
for-each / function literal, dangling else, call/index chains, '=>' return vs
function literal, empty statements, comments, weird spacing, invalid
characters), then applies random token insertions/deletions/swaps.

Compares (a) parse success/failure and (b) the AST dump text.

  tools/parsefuzz.py --seed S --count N [--engine NAME=CMD ...] [--mode wrapped|top|both]

Engines: the first (reference, the baseline) is run as `CMD -e FILE` -- it
executes the program after dumping, so each case is either
  wrapped: the fuzzed code is the body of a function literal that is never called
           (`zz := () => { CODE };`), so execution is harmless, or
  top:     the fuzzed code is the top level, made only of terminating constructs,
           run under a short timeout (baseline timeout => skip).
Other engines are run as `CMD -e --parse-only FILE` (ClaudeEPQ, pyref, fake_engine).

Extent of the AST dump in the baseline's stderr: lexer diagnostics
("FILE:L.C: invalid character") come first (they are printed while parsing),
then the dump (one line per top-level statement, each ending in '\\n'), then at
most one compile/run-time error line.  The dump is stderr minus the leading
diagnostics minus a final known error-message line.  `--check-extent` verifies
this rule against a second run without -e.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epqlib  # noqa: E402

KNOWN_MSGS = ["Operands must be integers", "Division by zero", "List is empty",
              "Operand must be an integer or a list", "Invalid comparison", "Operand must be an integer",
              "Invalid operation", "Index out of bounds", "Operand must be a list",
              "Operand must be a function", "Invalid arity", "Undefined global", "Invalid opcode",
              "Invalid argument list", "Argument shadows global variable", "Duplicate argument name",
              "Too many global variables", "Too many elements in list initialiser", "Too many arguments",
              "Jump too far", "Loop too far", "Classes are not implemented"]
DIAG_RE = re.compile(rb"^.*:\d+\.\d+(-\d+)?: invalid character$|^.*invalid character.*$")


def split_stderr(err, reference):
    """Returns (parse_ok, dump_bytes)."""
    lines = err.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    i = 0
    while i < len(lines) and DIAG_RE.match(lines[i]):
        i += 1
    rest = lines[i:]
    if any(b"syntax error" in l for l in rest):
        return False, b""
    if rest:        # (any engine) a trailing compile/run-time error line is not part of the dump
        last = rest[-1].decode("latin-1")
        if last in KNOWN_MSGS or last.startswith("Undefined variable ") or last.startswith("cannot open"):
            rest = rest[:-1]
    if not rest:
        return True, b""
    return True, b"\n".join(rest) + b"\n"


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

IDS = ["a", "b", "x", "f", "l", "i", "n", "foo", "Abc", "x_1", "q9"]


class PGen:
    def __init__(self, rng, top=False):
        self.r = rng
        self.top = top          # only terminating constructs
        self.depth = 0

    def ch(self, p):
        return self.r.random() < p

    def ident(self):
        return self.r.choice(IDS)

    def num(self):
        r = self.r.random()
        if r < 0.7:
            return str(self.r.randint(0, 20))
        return self.r.choice(["2147483647", "2147483648", "4294967296", "99999999999999999999999", "007", "0"])

    def exp(self, d=3):
        """Returns a token list for an expression (not necessarily parenthesised)."""
        if d <= 0:
            return self.atom(0)
        r = self.r.random()
        if r < 0.35:
            op = self.r.choice(["+", "-", "*", "/", "=", ">", "<", ">=", "<=", "!=", "&", "|", "|", ":+", ":+", "&"])
            return self.exp(d - 1) + [op] + self.exp(d - 1)
        if r < 0.42:
            return ["|"] + self.exp(d - 1) + ["|"]
        if r < 0.48:
            return ["!"] + self.exp(d - 1)
        if r < 0.54:
            return ["-"] + self.exp(d - 1)
        if r < 0.62:
            return self.lvalue(d - 1) + [":="] + self.exp(d - 1)
        if r < 0.68:
            return self.exp(d - 1) + ["("] + self.explist(d - 1) + [")"]
        if r < 0.73:
            return self.exp(d - 1) + ["["] + self.exp(d - 1) + ["]"]
        if r < 0.78:
            return ["("] + self.exp(d - 1) + [")"]
        if r < 0.83:
            return self.funclit(d - 1)
        if r < 0.86:
            return self.exp(d - 1) + [".", self.ident()]
        return self.atom(d - 1)

    def lvalue(self, d):
        r = self.r.random()
        if r < 0.6:
            return [self.ident()]
        if r < 0.85:
            return self.atom(d) + ["["] + self.exp(d) + ["]"]
        if r < 0.92:
            return [".", self.ident()]
        return self.exp(d) + [".", self.ident()]

    def atom(self, d):
        r = self.r.random()
        if r < 0.35:
            return [self.ident()]
        if r < 0.6:
            return [self.num()]
        if r < 0.7:
            return ["<<"]
        if r < 0.85 and d > 0:
            return ["["] + self.explist(d) + ["]"]
        if r < 0.9 and d > 0:
            return ["#"] + self.block(d)
        return [self.ident()]

    def explist(self, d):
        n = self.r.choice([0, 1, 1, 2, 3])
        out = []
        for i in range(n):
            if i:
                out.append(",")
            out += self.exp(d)
        return out

    def funclit(self, d):
        r = self.r.random()
        if r < 0.5:
            ps = []
            for i in range(self.r.choice([0, 1, 2, 3])):
                if i:
                    ps.append(",")
                ps += [self.ident()] if self.ch(0.85) else self.exp(1)
            return ["("] + ps + [")", "=>"] + self.block(d)
        return ["("] + self.exp(d) + [")", "=>"] + self.block(d)

    def block(self, d):
        out = ["{"]
        for _ in range(self.r.choice([0, 1, 1, 2, 3])):
            out += self.stmt(d - 1)
        return out + ["}"]

    def stmt(self, d=3):
        if d <= 0:
            return self.exp(1) + [";"]
        r = self.r.random()
        loops = not self.top
        if r < 0.05:
            return [";"]
        if r < 0.30:
            return self.exp(d) + [";"]
        if r < 0.40:
            return [">>"] + self.exp(d) + [";"]
        if r < 0.47:
            return self.block(d)
        if r < 0.60:
            out = ["("] + self.exp(d) + [")", "?"] + self.stmt(d - 1)
            if self.ch(0.5):
                out += [":"] + self.stmt(d - 1)
            return out
        if r < 0.66 and loops:
            return ["("] + self.exp(d) + [")", "@"] + self.stmt(d - 1)
        if r < 0.74 and loops:
            return ["("] + self.stmt(d - 1) + self.exp(d) + [";"] + self.stmt(d - 1) + [")", "@"] + self.stmt(d - 1)
        if r < 0.80:
            body = self.stmt(d - 1)
            return ["(", self.ident(), ":"] + (self.exp(d) if loops else ["["] + self.explist(1) + ["]"]) + [")", "@"] + body
        if r < 0.86:
            return ["=>"] + self.exp(d) + [";"]
        if r < 0.92:
            # function literal at statement start (vs if/while/paren exp)
            return self.funclit(d) + ([";"] if self.ch(0.8) else ["(", ")", ";"])
        return ["("] + self.exp(d) + [")"] + self.r.choice([[";"], ["+", "1", ";"], ["(", ")", ";"], ["[", "0", "]", ";"]])

    def program(self):
        out = []
        for _ in range(self.r.randint(1, 6)):
            out += self.stmt(3)
        return out


TOKSET = ["(", ")", "{", "}", "[", "]", ";", ",", ":", "?", "@", "=>", ":=", ":+", "|", "&", "!", "-", "+",
          "*", "/", "=", "<", ">", "<=", ">=", "!=", ">>", "<<", ".", "#", "x", "1", "f"]


def mutate(rng, toks):
    toks = list(toks)
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        if not toks:
            break
        r = rng.random()
        i = rng.randrange(len(toks))
        if r < 0.3:
            del toks[i]
        elif r < 0.6:
            toks.insert(i, rng.choice(TOKSET))
        elif r < 0.8 and i + 1 < len(toks):
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        else:
            toks.insert(i, toks[i])
    return toks


def render(rng, toks):
    """Join tokens with random spacing / newlines / comments; never glue two tokens
    into a different token."""
    out = []
    prev = ""
    for t in toks:
        r = rng.random()
        if r < 0.03:
            sep = " // comment " + rng.choice(["", "|", "{", ";;"]) + "\n"
        elif r < 0.10:
            sep = "\n"
        elif r < 0.13:
            sep = "\t"
        elif r < 0.45 and safe_glue(prev, t):
            sep = ""
        else:
            sep = " "
        if rng.random() < 0.004:
            sep += rng.choice(["$", "\r", "_", "~", "\\"])
        out.append(sep + t)
        prev = t
    return "".join(out) + "\n"


def safe_glue(a, b):
    if not a:
        return True
    s = a + b
    if (a[-1].isalnum() or a[-1] == "_") and (b[0].isalnum() or b[0] == "_"):
        return False
    for m in (":=", ">>", "<<", ">=", "<=", "!=", ":+", "=>", "//"):
        if m in s and m not in a and m not in b:
            return False
    return True


def make_case(seed, mode):
    rng = random.Random(seed)
    if mode == "both":
        mode = "wrapped" if rng.random() < 0.6 else "top"
    g = PGen(rng, top=(mode == "top"))
    toks = g.program()
    if rng.random() < 0.6:
        toks = mutate(rng, toks)
    code = render(rng, toks)
    if mode == "wrapped":
        src = "zz := () => {\n" + code + "};\n"
    else:
        src = code
    return src, mode


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

_W = {}


def _winit(engines, timeout, parse_only_names):
    _W["engines"] = engines
    _W["timeout"] = timeout
    _W["po"] = parse_only_names
    d = os.path.join(epqlib.BUILD, "parsefuzz", "tmp", "w%d" % os.getpid())
    os.makedirs(d, exist_ok=True)
    _W["dir"] = d


def run_case(src, engines, timeout, po, tag="p"):
    d = _W.get("dir") or os.path.join(epqlib.BUILD, "parsefuzz", "tmp", "main")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, tag + ".epq")
    with open(p, "w", encoding="latin-1") as f:
        f.write(src)
    out = []
    for i, e in enumerate(engines):
        if i == 0 or e.name not in po:
            r = epqlib.run_engine(e, p, timeout=timeout, flags=["-e"])
            ref = (i == 0)
        else:
            r = epqlib.run_engine(e, p, timeout=timeout, flags=["-e", "--parse-only"])
            ref = False
        out.append(r)
        if i == 0 and (r.timed_out or r.signal is not None or r.truncated):
            return out, None
    views = []
    for i, r in enumerate(out):
        if r.timed_out or r.signal is not None or r.spawn_error:
            views.append(("dead", r.status()))
            continue
        ok, dump = split_stderr(r.stderr, reference=(i == 0 or engines[i].name not in po))
        if i > 0 and engines[i].name in po:
            ok = ok and r.rc == 0
        views.append((ok, dump))
    return out, views


def compare(views, engines):
    ref = views[0]
    bad = {}
    for e, v in zip(engines[1:], views[1:]):
        if v[0] == "dead":
            bad[e.name] = "engine died: %s" % v[1]
        elif v[0] != ref[0]:
            bad[e.name] = "parse %s but reference %s" % ("ok" if v[0] else "FAILED", "ok" if ref[0] else "FAILED")
        elif ref[0] and v[1] != ref[1]:
            a, b = ref[1], v[1]
            i = 0
            while i < min(len(a), len(b)) and a[i] == b[i]:
                i += 1
            bad[e.name] = "AST differs at byte %d: ref %r vs %r" % (i, a[i:i + 50], b[i:i + 50])
    return bad


def wcase(args):
    seed, mode = args
    src, m = make_case(seed, mode)
    res, views = run_case(src, _W["engines"], _W["timeout"], _W["po"])
    if views is None:
        return {"seed": seed, "mode": m, "verdict": "skip", "why": res[0].status()}
    bad = compare(views, _W["engines"])
    r = {"seed": seed, "mode": m, "verdict": "fail" if bad else "pass", "parse_ok": bool(views[0][0]),
         "bad": bad}
    if bad:
        r["src"] = src
        r["outputs"] = [(x.stdout.decode("latin-1"), x.stderr.decode("latin-1"), x.status()) for x in res]
    return r


def tok_reduce(src, engines, timeout, po, budget=60):
    """Line- then character-chunk ddmin keeping 'engines still disagree'."""
    deadline = time.time() + budget

    def interesting(s):
        res, views = run_case(s, engines, timeout, po, tag="red")
        return views is not None and bool(compare(views, engines))
    if not interesting(src):
        return src
    for unit in ("line", "word"):
        parts = src.split("\n") if unit == "line" else re.split(r"(\s+)", src)
        n = max(1, len(parts) // 2)
        while n >= 1 and time.time() < deadline:
            i = 0
            while i < len(parts) and time.time() < deadline:
                cand = parts[:i] + parts[i + n:]
                s = ("\n" if unit == "line" else "").join(cand)
                if interesting(s):
                    parts = cand
                else:
                    i += n
            n //= 2
        src = ("\n" if unit == "line" else "").join(parts)
    return src


def check_extent(a, engines):
    """Validate the dump-extent rule on the baseline: dump(-e) == stderr(-e) minus stderr(no -e)."""
    base = engines[0]
    d = os.path.join(epqlib.BUILD, "parsefuzz", "tmp", "extent")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "x.epq")
    n = ok = skipped = 0
    for s in range(a.seed, a.seed + a.count):
        src, m = make_case(s, "top")
        if "<<" in src:
            skipped += 1
            continue
        with open(p, "w", encoding="latin-1") as f:
            f.write(src)
        r1 = epqlib.run_engine(base, p, timeout=a.timeout, flags=["-e"])
        r2 = epqlib.run_engine(base, p, timeout=a.timeout)
        if r1.timed_out or r2.timed_out or r1.signal or r2.signal:
            skipped += 1
            continue
        n += 1
        okp, dump = split_stderr(r1.stderr, True)
        e1, e2 = r1.stderr, r2.stderr
        # e2 = P + S, e1 = P + D + S
        j = 0
        while j < len(e2) and j < len(e1) and e1[j] == e2[j]:
            j += 1
        # P ends at a line boundary within the common prefix
        pre = e2[:j]
        pre = pre[:pre.rfind(b"\n") + 1] if b"\n" in pre else b""
        P_lines = [l for l in pre.split(b"\n") if l and DIAG_RE.match(l)]
        P = b"".join(l + b"\n" for l in P_lines)
        S = e2[len(P):]
        D = e1[len(P):len(e1) - len(S)] if e1.endswith(S) else None
        if not okp:
            D2 = b"" if b"syntax error" in e1 else None
            good = D2 == b""
        else:
            good = (D == dump)
        if good:
            ok += 1
        elif a.verbose:
            print("extent mismatch seed %d: %r vs %r" % (s, D, dump))
    print("dump-extent rule: %d/%d agree with the two-run method (%d skipped: input/timeout)" % (ok, n, skipped))
    return 0 if ok == n else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--mode", choices=["wrapped", "top", "both"], default="both")
    ap.add_argument("--engine", action="append", default=[], metavar="NAME=CMD")
    ap.add_argument("--pyref", action="store_true", help="add pyref (its exact LALR parser) as an engine")
    ap.add_argument("--run-style", action="append", default=[], metavar="NAME",
                    help="run this engine like the baseline (-e, no --parse-only)")
    ap.add_argument("--jobs", "-j", type=int, default=epqlib.jobs_default())
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(epqlib.BUILD, "parsefuzz", "failures"))
    ap.add_argument("--no-reduce", action="store_true")
    ap.add_argument("--max-reduce", type=int, default=5)
    ap.add_argument("--check-extent", action="store_true")
    ap.add_argument("--emit", metavar="DIR", help="only write the generated cases to DIR")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.emit:
        os.makedirs(a.emit, exist_ok=True)
        for s in range(a.seed, a.seed + a.count):
            src, m = make_case(s, a.mode)
            with open(os.path.join(a.emit, "pf_%s_%d.epq" % (m, s)), "w", encoding="latin-1") as f:
                f.write(src)
        return 0
    engines = epqlib.engines_from_args(a.engine)
    if a.pyref:
        engines.append(epqlib.Engine("pyref", [sys.executable, os.path.join(epqlib.TOOLS, "pyref.py")]))
    if a.check_extent:
        return check_extent(a, engines)
    po = set(e.name for e in engines[1:]) - set(a.run_style)
    print("engines: " + ", ".join(repr(e) for e in engines))
    t0 = time.time()
    stats = {"pass": 0, "fail": 0, "skip": 0}
    parse_ok = 0
    fails = []
    with cf.ProcessPoolExecutor(max_workers=a.jobs, initializer=_winit, initargs=(engines, a.timeout, po)) as ex:
        for r in ex.map(wcase, [(s, a.mode) for s in range(a.seed, a.seed + a.count)], chunksize=8):
            stats[r["verdict"]] += 1
            if r["verdict"] != "skip" and r["parse_ok"]:
                parse_ok += 1
            if r["verdict"] == "fail":
                fails.append(r)
                print("FAIL seed %d (%s): %s" % (r["seed"], r["mode"],
                                                 "; ".join("%s: %s" % kv for kv in r["bad"].items())))
    el = time.time() - t0
    nn = stats["pass"] + stats["fail"]
    print("\n%d cases in %.1fs: %d pass, %d FAIL, %d skipped (baseline timeout); reference parse-ok rate %.1f%%"
          % (a.count, el, stats["pass"], stats["fail"], stats["skip"], 100.0 * parse_ok / max(nn, 1)))
    if fails:
        os.makedirs(a.out, exist_ok=True)
        _winit(engines, a.timeout, po)
        for i, r in enumerate(fails):
            d = os.path.join(a.out, "seed%d" % r["seed"])
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
            with open(os.path.join(d, "prog.epq"), "w", encoding="latin-1") as f:
                f.write(r["src"])
            for e, (so, se, st) in zip(engines, r["outputs"]):
                with open(os.path.join(d, e.name + ".stderr"), "w", encoding="latin-1") as f:
                    f.write(se)
            with open(os.path.join(d, "info.json"), "w") as f:
                json.dump({k: v for k, v in r.items() if k not in ("src", "outputs")}, f, indent=2)
            if not a.no_reduce and i < a.max_reduce:
                red = tok_reduce(r["src"], engines, a.timeout, po)
                with open(os.path.join(d, "prog.min.epq"), "w", encoding="latin-1") as f:
                    f.write(red)
                print("reduced seed %d -> %r" % (r["seed"], red[:300]))
        print("saved to %s" % a.out)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
