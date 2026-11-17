#!/usr/bin/env python3
"""fake_engine -- a deliberately broken engine used to prove the tools detect
differences.  Wraps a real engine and perturbs its behaviour.

  fake_engine.py MODE REAL_ENGINE_CMD... [flags] FILE

MODE (comma-separated, each applies only to some programs, so a correct tool
must report SOME but not ALL cases):
  digit    stdout: replace the last '7' by '8'
  exit     exit code: 0 <-> 1 when stdout has an odd number of lines
  hang     sleep 1000s when stdout contains '>> 3\\n'
  crash    SIGSEGV itself when stdout contains '>> 5\\n'
  ast      stderr: first '||' -> '&&' (AST dump perturbation)
  parse    exit 1 with a syntax error if the source contains '|' twice on one line
  slow     sleep 0.05s before running (benchmark sanity)
Flags understood by ClaudeEPQ only (--engine=..., --parse-only) are dropped
before calling the wrapped engine; --parse-only is emulated by running the
program with stdin closed and keeping only stderr.
With `--batch MANIFEST` (ClaudeEPQ batch mode) the real batch runs and every
test's output files are perturbed the same way (hang -> exit 124 + TIMEOUT,
crash -> 139 + signal note, as batch mode reports them).
"""
import os
import signal
import subprocess
import sys
import time


def batch(modes, rest):
    """`fake_engine.py MODE REAL_EPQ... --batch MANIFEST ...`: run the real batch,
    then perturb every test's output files exactly as single mode would."""
    man = rest[rest.index("--batch") + 1]
    p = subprocess.run(rest)
    if p.returncode != 0:
        return p.returncode
    for line in open(man).read().splitlines():
        if not line or line.startswith("#"):
            continue
        prog, _, fo, fe, fc = line.split("\t")
        out, err = open(fo, "rb").read(), open(fe, "rb").read()
        rc = int(open(fc).read().strip())
        if "parse" in modes:
            src = open(prog, "rb").read()
            if any(l.count(b"|") >= 2 for l in src.split(b"\n")):
                out, err, rc = b"", ("%s:1.1: syntax error, unexpected '|'\n" % prog).encode(), 1
        if "digit" in modes:
            i = out.rfind(b"7")
            if i >= 0:
                out = out[:i] + b"8" + out[i + 1:]
        if "exit" in modes and out.count(b"\n") % 2 == 1:
            rc = 1 - rc if rc in (0, 1) else rc
        if "ast" in modes:
            err = err.replace(b"||", b"&&", 1)
        if "hang" in modes and b">> 3\n" in out:
            err, rc = err + b"TIMEOUT\n", 124
        elif "crash" in modes and b">> 5\n" in out:
            err, rc = err + b"epq-batch: killed by signal 11 (Segmentation fault: 11)\n", 139
        open(fo, "wb").write(out)
        open(fe, "wb").write(err)
        open(fc, "w").write("%d\n" % rc)
    return 0


def main():
    modes = set(sys.argv[1].split(","))
    rest = sys.argv[2:]
    if "--batch" in rest:
        return batch(modes, rest)
    fname = rest[-1]
    parse_only = "--parse-only" in rest
    rest = [a for a in rest if not a.startswith("--engine=") and a != "--parse-only"]
    if "slow" in modes:
        time.sleep(0.05)
    if "parse" in modes:
        src = open(fname, "rb").read()
        if any(l.count(b"|") >= 2 for l in src.split(b"\n")):
            sys.stderr.write("%s:1.1: syntax error, unexpected '|'\n" % fname)
            return 1
    stdin = subprocess.DEVNULL if parse_only else None
    p = subprocess.run(rest, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err, rc = p.stdout, p.stderr, p.returncode
    if parse_only:
        out = b""
        rc = 0 if rc in (0, 1) and b"syntax error" not in err else 1
    if "digit" in modes:
        i = out.rfind(b"7")
        if i >= 0:
            out = out[:i] + b"8" + out[i + 1:]
    if "exit" in modes and out.count(b"\n") % 2 == 1:
        rc = 1 - rc if rc in (0, 1) else rc
    if "ast" in modes:
        err = err.replace(b"||", b"&&", 1)
    sys.stdout.buffer.write(out)
    sys.stdout.flush()
    sys.stderr.buffer.write(err)
    sys.stderr.flush()
    if "hang" in modes and b">> 3\n" in out:
        time.sleep(1000)
    if "crash" in modes and b">> 5\n" in out:
        os.kill(os.getpid(), signal.SIGSEGV)
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(main())
