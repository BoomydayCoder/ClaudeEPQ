#!/usr/bin/env python3
"""Front-end differential tests against the baseline.

Corpus: the original examples, tests/frontend/snippets.txt (grammar corners)
and tests/frontend/resolver_cases.txt (compile-time errors / scoping). Each
source is prefixed with "=> 0;" and run through `epq_ref -e` and `build/epq -e`;
parse acceptance, the AST dump, lexer diagnostics, the syntax-error location
and the first compile error line must all agree.

usage: run_tests.py [--epq BINARY] [-j N] [-v]
"""

import argparse
import concurrent.futures
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402


def corpus():
    cases = []
    ex = sorted(glob.glob(os.path.join(harness.ROOT, "reference", "EPQ", "examples", "**", "*.txt"), recursive=True))
    for p in ex:
        with open(p, "r", encoding="utf-8") as f:
            cases.append(("example " + os.path.relpath(p, harness.ROOT), f.read()))
    for fn in ("snippets.txt", "resolver_cases.txt"):
        for name, src in harness.load_cases(os.path.join(harness.HERE, fn)):
            cases.append((fn + ": " + name, src))
    # Raw-byte lexer cases that do not fit in a text file. ('\r' is excluded:
    # the original reports it as an invalid character, ClaudeEPQ skips it
    # silently by design.)
    raw = [
        ("NUL byte", b"x := 1\0;\n"),
        ("NUL at end", b"x := 1;\n\0"),
        ("UTF-8 letter", b"x := 1 \xc3\xa9 + 2;\n"),
        ("UTF-8 adjacent", b"x\xe2\x82\xac := 1;"),
        ("form feed", b"x := 1\f;\n"),
        ("vertical tab", b"x := 1\v;\n"),
        ("DEL", b"x := \x7f 1;"),
        ("high bytes then syntax error", b"x := \xff\xfe ) \xff;"),
        ("invalid chars across lines", b"$\n$$\n x $ := 1;"),
        ("invalid char then newline token", b"x := 1;$\n\n$y := 2;"),
        ("no trailing newline", b"x := 1;"),
        ("only invalid", b"$$$"),
    ]
    for name, src in raw:
        cases.append(("raw: " + name, src))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epq", default=harness.EPQ)
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    a.epq = os.path.abspath(a.epq)

    cases = corpus()
    counts = {}
    failures = []

    def one(c):
        name, src = c
        full = (harness.PREFIX.encode() + src) if isinstance(src, bytes) else harness.PREFIX + src
        return name, full, harness.compare(full, epq=a.epq)

    with concurrent.futures.ThreadPoolExecutor(a.j) as ex:
        for name, full, o in ex.map(one, cases):
            counts[o.kind] = counts.get(o.kind, 0) + 1
            if o.kind not in ("ok", "ok-syntax"):
                failures.append((name, full, o))
            elif a.v:
                print("ok   %-10s %s" % (o.kind, name))

    for name, full, o in failures:
        print("FAIL [%s] %s: %s" % (o.kind, name, o.detail))
        print(harness.show(o, full))
    print("%d cases: %s" % (len(cases), ", ".join("%s=%d" % kv for kv in sorted(counts.items()))))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
