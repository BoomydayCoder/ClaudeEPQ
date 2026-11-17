#!/usr/bin/env python3
"""Tests of the `simplify` pass (src/passes/simplify.cpp).

usage: tests/passes/simplify/run.py [-k SUBSTR] [-j N] [-q] -- CMD [ARGS...]
  1. every X.epq here is run with CMD (same format as tests/conformance/run.py;
     expectations generated from the baseline with tests/conformance/regen.py);
  2. for every X.expect: CMD --dump-tir-after=simplify X.epq is checked: the dump
     of the function named on the `func NAME` line must not contain any `absent`
     string and must contain every `present` string (shows the pass fires).
Example: tests/passes/simplify/run.py -j 3 -- build/epq --engine=aot --verify-tir
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "conformance"))
import run as R  # noqa: E402

R.HERE = HERE


def fires(cmd):
    bad = 0
    for f in sorted(os.listdir(HERE)):
        if not f.endswith(".expect"):
            continue
        base = os.path.join(HERE, f[:-7])
        lines = [l.rstrip("\n") for l in open(os.path.join(HERE, f)) if l.strip() and not l.startswith("#")]
        fname = lines[0].split(None, 1)[1]
        dump = subprocess.run(cmd + ["--dump-tir-after=simplify", base + ".epq"], stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60).stdout.decode()
        sect, on = [], False
        for l in dump.split("\n"):
            if l.startswith("func "):
                on = l.startswith("func " + fname + " ")
            if on:
                sect.append(l)
        text = "\n".join(sect)
        for l in lines[1:]:
            kind, s = l.split(None, 1)
            ok = (s not in text) if kind == "absent" else (s in text)
            if not ok or not sect:
                print("FIRES FAIL %s: %s %s" % (f, kind, s))
                bad += 1
    print("fires: %s" % ("ok" if not bad else "%d failed" % bad))
    return bad


if __name__ == "__main__":
    argv = sys.argv[1:]
    cmd = argv[argv.index("--") + 1:] if "--" in argv else []
    rc = R.main()
    if cmd and fires(cmd):
        rc = 1
    sys.exit(rc)
