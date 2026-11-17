#!/usr/bin/env python3
"""Run an engine over the runtime I/O tests (tests/rtio/progs, made by gen.py).

usage: tests/rtio/run.py [-j N] [-k SUBSTR] [--stdin pipe,file] -- CMD [ARGS...]

Compares stdout bytes (NAME.out, or size + SHA-256 in NAME.sha), exit status
and the first line of stderr. Each test runs once per stdin kind: `pipe`
(input written through a pipe) and `file` (stdin is the NAME.in file itself,
or /dev/null), since the runtime picks its flush-before-read policy from the
kind of stdin.

The buffer-boundary tests are strongest with tiny runtime buffers; the C
backend passes EPQ_CFLAGS to cc, e.g.
  EPQ_CFLAGS='-O1 -DEPQ_OBUF=16 -DEPQ_IBUF=1' tests/rtio/run.py -- build/epq
(EPQ_OBUF must be >= 16; EPQ_IBUF >= 1.)
"""
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROGS = os.path.join(HERE, "progs")


def expected(base):
    with open(base + ".code") as f:
        code = int(f.read().strip())
    with open(base + ".err") as f:
        err = f.read().rstrip("\n")
    if os.path.exists(base + ".out"):
        with open(base + ".out", "rb") as f:
            out = f.read()
        key = (len(out), hashlib.sha256(out).hexdigest())
    else:
        with open(base + ".sha") as f:
            n, h = f.read().split()
        key = (int(n), h)
    return key, code, err


def run_one(cmd, prog, kind, timeout):
    base = prog[:-4]
    inp = base + ".in" if os.path.exists(base + ".in") else None
    try:
        if kind == "file":
            with open(inp or os.devnull, "rb") as f:
                r = subprocess.run(cmd + [prog], stdin=f, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=timeout)
        else:
            data = open(inp, "rb").read() if inp else b""
            r = subprocess.run(cmd + [prog], input=data, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout)
        out, code = r.stdout, r.returncode
        err = r.stderr.decode(errors="replace").splitlines()
        err = err[0] if err else ""
    except subprocess.TimeoutExpired:
        out, code, err = b"", "timeout", ""
    key, ecode, eerr = expected(base)
    got = (len(out), hashlib.sha256(out).hexdigest())
    ok = got == key and code == ecode and err == eerr
    msg = ""
    if not ok:
        msg = "stdout %d bytes (want %d)%s, exit %s (want %s), stderr %r (want %r)" % (
            got[0], key[0], "" if got == key else " DIFFERENT", code, ecode, err, eerr)
        if os.path.exists(base + ".out"):
            want = open(base + ".out", "rb").read()
            i = next((k for k in range(min(len(out), len(want))) if out[k] != want[k]),
                     min(len(out), len(want)))
            msg += "\n    first difference at byte %d: got %r want %r" % (
                i, out[max(0, i - 20):i + 20], want[max(0, i - 20):i + 20])
    return os.path.basename(base), kind, ok, msg


def main():
    args = sys.argv[1:]
    if "--" not in args:
        print(__doc__)
        return 2
    i = args.index("--")
    opts, cmd = args[:i], args[i + 1:]
    jobs, filt, kinds, timeout = 3, None, ["pipe", "file"], 120
    while opts:
        o = opts.pop(0)
        if o == "-j":
            jobs = int(opts.pop(0))
        elif o == "-k":
            filt = opts.pop(0)
        elif o == "--stdin":
            kinds = opts.pop(0).split(",")
        elif o == "--timeout":
            timeout = float(opts.pop(0))
    progs = sorted(os.path.join(PROGS, f) for f in os.listdir(PROGS) if f.endswith(".epq"))
    if filt:
        progs = [p for p in progs if filt in os.path.basename(p)]
    work = [(p, k) for p in progs for k in kinds]
    fails = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for name, kind, ok, msg in ex.map(lambda pk: run_one(cmd, pk[0], pk[1], timeout), work):
            if not ok:
                fails += 1
                print("FAIL %s [%s]: %s" % (name, kind, msg))
    print("%d/%d passed, %d failed" % (len(work) - fails, len(work), fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
