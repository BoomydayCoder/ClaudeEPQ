"""Shared helpers for the front-end differential tests (Python 3.9, stdlib only).

compare(src) runs the baseline (`epq_ref -e`) and ours (`build/epq -e`) on the
same source and classifies the outcome. Sources are always prefixed by the
caller with "=> 0;" so the baseline returns before executing anything (it
still parses and compiles the whole program).
"""

import os
import sys
import re
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
from epqlib import require_baseline  # noqa: E402
REF = os.environ.get("EPQ_REF") or require_baseline()
EPQ = os.environ.get("EPQ_BIN", os.path.join(REPO, "build", "epq"))
WORK = os.path.join(REPO, "build", "frontend", "work")
TIMEOUT = 5.0

PREFIX = "=> 0; "

SYNTAX_RE = re.compile(r"^(\S+:\d+\.\d+(?:-\d+(?:\.\d+)?)?): syntax error")
BACKEND_MSG = "backend not implemented\n"
CLASSES_MSG = "Classes are not implemented\n"

_local = threading.local()
_counter = [0]
_counter_lock = threading.Lock()


def _workfile():
    p = getattr(_local, "path", None)
    if p is None:
        os.makedirs(WORK, exist_ok=True)
        with _counter_lock:
            _counter[0] += 1
            n = _counter[0]
        p = os.path.join(WORK, "case_%d_%d.epq" % (os.getpid(), n))
        _local.path = p
    return p


def run(binary, path, args=("-e",)):
    """Returns (rc, stdout, stderr) or (None, '', 'TIMEOUT')."""
    try:
        # Run inside the work directory with a bare file name so messages
        # ("FILE:L.C: ...") stay short and free of spaces.
        r = subprocess.run([binary] + list(args) + [os.path.basename(path)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT,
                           cwd=os.path.dirname(path))
    except subprocess.TimeoutExpired:
        return None, "", "TIMEOUT"
    return (r.returncode, r.stdout.decode("utf-8", "replace"),
            r.stderr.decode("utf-8", "replace"))


def split_syntax(stderr):
    """If stderr reports a syntax error: (lines before it, location). Else None."""
    lines = stderr.splitlines(True)
    for i, ln in enumerate(lines):
        m = SYNTAX_RE.match(ln)
        if m:
            return "".join(lines[:i]), m.group(1)
    return None


class Outcome(object):
    __slots__ = ("kind", "detail", "ref", "me")

    def __init__(self, kind, detail, ref, me):
        self.kind = kind      # 'ok', 'ok-syntax', 'errloc', 'DIVERGE', 'TIMEOUT', 'CRASH'
        self.detail = detail
        self.ref = ref
        self.me = me


def compare(src, epq=None, write=True):
    path = _workfile()
    with open(path, "wb") as f:
        f.write(src.encode("utf-8", "surrogateescape") if isinstance(src, str) else src)
    ref = run(REF, path)
    me = run(epq or EPQ, path)
    rrc, rout, rerr = ref
    mrc, mout, merr = me
    if rrc is None:
        return Outcome("TIMEOUT", "baseline timed out", ref, me)
    if mrc is None:
        return Outcome("DIVERGE", "epq timed out", ref, me)
    if mrc not in (0, 1, 2):
        return Outcome("CRASH", "epq exit %r" % mrc, ref, me)
    rs = split_syntax(rerr) if rrc == 1 else None
    ms = split_syntax(merr) if mrc == 1 else None
    if rs is not None or ms is not None:
        if rs is None or ms is None:
            return Outcome("DIVERGE", "parse accept/reject differs (ref %s, epq %s)" % (
                "reject" if rs else "accept", "reject" if ms else "accept"), ref, me)
        if mout:
            return Outcome("DIVERGE", "stdout on syntax error", ref, me)
        if rs != ms:
            return Outcome("errloc", "lex diagnostics or error location differ: ref %r epq %r" % (rs, ms), ref, me)
        return Outcome("ok-syntax", "", ref, me)
    if rrc not in (0, 1):
        return Outcome("CRASH", "baseline exit %r" % rrc, ref, me)
    if mout or rout:
        return Outcome("DIVERGE", "unexpected stdout", ref, me)
    mine = merr
    if mrc == 2:
        if not mine.endswith(BACKEND_MSG):
            return Outcome("DIVERGE", "exit 2 without backend message", ref, me)
        mine = mine[: -len(BACKEND_MSG)]
    if mrc == 1 and mine.endswith(CLASSES_MSG):
        dump = mine[: -len(CLASSES_MSG)]
        if rerr.startswith(dump):
            return Outcome("ok", "classes", ref, me)
        return Outcome("DIVERGE", "AST dump differs (classes case)", ref, me)
    if mine == rerr and (mrc == 1) == (rrc == 1):
        return Outcome("ok", "", ref, me)
    if (mrc == 1) != (rrc == 1):
        return Outcome("DIVERGE", "compile success differs (ref rc %d, epq rc %d)" % (rrc, mrc), ref, me)
    return Outcome("DIVERGE", "stderr differs", ref, me)


def load_cases(path):
    """Parses a '%%'-separated case file into [(name, source)]."""
    cases = []
    name = None
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("%%"):
                if name is not None:
                    cases.append((name, "".join(buf)))
                name = line[2:].strip() or "case%d" % len(cases)
                buf = []
            elif name is not None:
                buf.append(line)
    if name is not None:
        cases.append((name, "".join(buf)))
    return cases


def show(o, src):
    if isinstance(src, bytes):
        src = repr(src)
    out = ["--- source ---", src, "--- baseline rc=%r ---" % (o.ref[0],), o.ref[2],
           "--- epq rc=%r ---" % (o.me[0],), o.me[2]]
    return "\n".join(out)
