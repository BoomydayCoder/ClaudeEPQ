#!/usr/bin/env python3
"""TIR extension self-tests (src/tir_selftest.cpp).

Hand-built TIR programs using every extension op (Select, InlineBlock/Break,
Pair*, ListData/ElemLoad/ElemStore/CheckIndex, Free, SameList, Nop) plus
verifier negative tests, each verified, compiled by the C backend, run in a
child process and checked against expected stdout / stderr / exit status.

usage: tests/tir/run.py [EPQ_BINARY] [CASE|all]
Runs once with the default C flags and once with EPQ_CFLAGS=-O1 unless
EPQ_CFLAGS is already set. EPQ_CACHE_DIR defaults to build/tirtest/cache.
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    epq = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "epq")
    case = sys.argv[2] if len(sys.argv) > 2 else "all"
    env = dict(os.environ)
    env.setdefault("EPQ_CACHE_DIR", os.path.join(ROOT, "build", "tirtest", "cache"))
    flag_sets = [env["EPQ_CFLAGS"]] if "EPQ_CFLAGS" in env else [None, "-O1"]
    rc = 0
    for fl in flag_sets:
        e = dict(env)
        if fl is not None:
            e["EPQ_CFLAGS"] = fl
        print("== EPQ_CFLAGS=%s" % (fl if fl is not None else "(default)"))
        sys.stdout.flush()
        r = subprocess.run([epq, "--engine=aot", "--tir-selftest=" + case], env=e, timeout=600)
        rc |= r.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
