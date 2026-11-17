#!/usr/bin/env python3
"""Generate the runtime I/O tests in tests/rtio/progs/.

    python3 tests/rtio/gen.py [--old CMD...]

Each case is NAME.epq (+ NAME.in) with expected NAME.out (stdout bytes; for
outputs over 64 KB, NAME.sha holds the size and SHA-256 instead),
NAME.code (exit status) and NAME.err (first line of stderr, or empty).
Expected results come from the baseline binary (the original implementation)
for every case inside the well-defined subset; cases outside it (EOF or
garbage input: LANGUAGE.md 9.2) are tagged "old" and take their expected
results from `--old CMD` (a ClaudeEPQ build from before the I/O rewrite), so
the rewrite must keep the existing ClaudeEPQ semantics there.
"""
import hashlib
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROGS = os.path.join(HERE, "progs")
REF = os.environ.get("EPQ_BASELINE") or os.path.join(REPO, "..", "refbuild", "epq_ref")
if not os.path.exists(REF):
    REF = os.path.join(REPO, "..", "..", "..", "..", "refbuild", "epq_ref")

BIG = 64 * 1024
INT_MIN = "(-2147483647 - 1)"


def boundary_ints():
    vals = [0, 1, -1, 2147483647, -2147483647]
    for k in range(1, 10):
        p = 10 ** k
        vals += [p - 1, p, p + 1, -(p - 1), -p, -(p + 1)]
    return [v for v in vals if -2147483647 <= v <= 2147483647]


def cases():
    rnd = random.Random(12345)
    out = []

    def add(name, src, stdin=None, ref="baseline"):
        out.append((name, src, stdin, ref))

    ints = boundary_ints()
    # --- integers in every context ------------------------------------------
    src = "".join(">> %d;\n" % v for v in ints)
    src += ">> %s;\n>> -%s;\n>> %s - 1;\n>> %s * -1;\n" % (INT_MIN, INT_MIN, INT_MIN, INT_MIN)
    add("int_static", src)
    lit = ", ".join(str(v) for v in ints)
    add("int_list", "l := [%s];\n>> l;\nl :+ %s;\n>> l;\n>> [%s];\n" % (lit, INT_MIN, INT_MIN))
    # values flowing through Any (a function called with an int and a list)
    add("int_any", "id := (x) => { => x; };\n>> id([]);\n"
        + "".join(">> id(%d);\n" % v for v in ints) + ">> id(%s);\n" % INT_MIN
        + ">> [id(1), id([2, [3]]), id(%s)];\n" % INT_MIN
        + "l := [1, [2]];\nl[0] := [];\n>> l;\nl[1] := %s;\n>> l;\n" % INT_MIN)
    # --- list shapes -------------------------------------------------------------
    add("list_shapes", ">> [];\n>> [[]];\n>> [[], [[]], [1, []]];\n>> [[[[[]]]]];\n"
        ">> [1, [2, [3, [4, []]]], -5];\n>> [[1, 2], [3, 4], []];\n>> [[[1]], [[2, 3]], [[]]];\n"
        "e := [];\n>> e;\ne :+ [];\n>> e;\nx := [[]];\nx[0] :+ 1;\n>> x;\n"
        "p := [];\n(i := 0; i < 20; i := i + 1;) @ p :+ [i, -i];\n>> p;\n"
        "q := [];\n(i := 0; i < 20; i := i + 1;) @ q :+ [i, [i], []];\n>> q;\n")
    add("deep_nest", "l := [];\n(i := 0; i < 20000; i := i + 1;) @ l := [l];\n>> l;\n"
        "m := [7];\n(i := 0; i < 3000; i := i + 1;) @ m := [m, i];\n>> m;\n")
    add("deep_nest_100k", "l := [];\n(i := 0; i < 100000; i := i + 1;) @ l := [l, i];\n>> l;\n")
    # --- big outputs: every value width at many buffer offsets ---------------------
    n = 120000
    add("big_int_lines", "(i := 0; i < %d; i := i + 1;) @ >> i * 7919 - (i / 3) * 104729 * (i - (i / 2) * 2);\n" % n)
    add("big_int_list", "l := [];\n(i := 0; i < %d; i := i + 1;) @ l :+ i * 7919 - (i / 3) * 104729 * (i - (i / 2) * 2);\n>> l;\n"
        "(k := 0; k < 4; k := k + 1;) @ { >> k; >> l; }\n" % n)
    add("big_nested", "l := [];\n(i := 0; i < 60000; i := i + 1;) @ l :+ [i - (i / 10) * 10, [i * 31 - 500000], []];\n"
        ">> l;\n(i := 0; i < 60000; i := i + 7;) @ >> l[i];\n")
    add("big_pairs", "l := [];\n(i := 0; i < 90000; i := i + 1;) @ l :+ [i * 13, i - 45000];\n>> l;\n")
    add("big_any_mixed", "l := [];\n(i := 0; i < 90000; i := i + 1;) @ { (i - (i / 3) * 3 = 0) ? { l :+ [i]; } : { l :+ i * -9; } }\n>> l;\n")
    # straddling lengths: padding prints of every length before a long list
    src = "l := [];\n(i := 0; i < 12000; i := i + 1;) @ l :+ i * 37;\n"
    src += "(pad := 0; pad < 16; pad := pad + 1;) @ {\n  (j := 0; j < pad; j := j + 1;) @ >> 5;\n  >> l;\n  >> [l, pad];\n}\n"
    add("straddle", src)
    # --- prompts and input ----------------------------------------------------------
    add("prompt_interleave", "a := <<;\n>> a;\nb := <<;\nc := <<;\n>> [b, c];\n<<;\n>> a + b + c;\n",
        "10\n20 30\n 40\n")
    add("input_formats", "(i := 0; i < 12; i := i + 1;) @ >> <<;\n",
        "-5 +7\n\n  12 007 3-4\t9\v8\f7\r6\r\n+0 -0 000000000000000000000000000042")
    add("input_limits", "(i := 0; i < 4; i := i + 1;) @ >> <<;\n", "2147483647 -2147483648 +2147483647 -0000002147483648\n")
    nums = [rnd.choice([rnd.randint(-9, 9), rnd.randint(-99999, 99999), rnd.randint(-2147483648, 2147483647),
                        rnd.randint(0, 9999999)]) for _ in range(150000)]
    seps = [" ", "\n", "  ", "\t", " \r\n"]
    text = "".join(("+" if (v >= 0 and rnd.random() < 0.1) else "") + str(v) + rnd.choice(seps) for v in nums)
    add("input_big", "s := 0;\nx := 0;\n(i := 0; i < %d; i := i + 1;) @ { x := <<; s := s + x; (i - (i / 1000) * 1000 = 0) ? >> [i, x, s]; }\n>> s;\n>> x;\n" % len(nums), text)
    ws = " " * 300000 + "\n" * 70000 + "17" + "\t" * 100000 + "-18" + " " * 5
    add("input_long_ws", ">> <<;\n>> <<;\n", ws)
    add("input_long_digits", ">> <<;\n>> <<;\n", "0" * 300000 + "123 " + "-" + "0" * 70000 + "45\n")
    add("input_no_trailing_newline", "a := <<;\nb := <<;\n>> a * 1000 + b;\n", "12 345")
    # --- flushing ---------------------------------------------------------------------
    add("flush_before_error", "(i := 0; i < 30000; i := i + 1;) @ >> i;\n>> [1, 2, 3];\nl := [];\n>> l[0];\n")
    add("flush_before_error_prompt", "a := <<;\n>> a;\n<<;\n>> 1 / 0;\n", "5 6")
    add("flush_toplevel_return", "(i := 0; i < 30000; i := i + 1;) @ { >> i; (i = 29999) ? => 0; }\n>> 99;\n")
    add("flush_return_in_fn_loop", "f := () => { (i := 0; i < 5; i := i + 1;) @ >> i; => 7; };\n>> f();\n(1) @ { >> 8; => 1; }\n")
    # --- outside the subset: EOF / garbage (expected from the old ClaudeEPQ) -----------
    add("input_eof", ">> <<;\n>> <<;\n>> <<;\n", "5", ref="old")
    add("input_empty", ">> <<;\n>> <<;\n", "", ref="old")
    add("input_garbage", ">> <<;\n>> <<;\n>> <<;\n", "7 abc 8", ref="old")
    add("input_sign_only", ">> <<;\n>> <<;\n>> <<;\n", "3 - 4 5", ref="old")
    add("input_sign_eof", ">> <<;\n>> <<;\n", "3 -", ref="old")
    add("input_overflow", ">> <<;\n>> <<;\n>> <<;\n", "1 2147483648 5", ref="old")
    add("input_overflow_neg", ">> <<;\n>> <<;\n", "-2147483649 5", ref="old")
    add("input_huge", ">> <<;\n>> <<;\n", "99999999999999999999999999999999 1", ref="old")
    add("input_hex", ">> <<;\n>> <<;\n", "0x1F 2", ref="old")
    return out


def run(cmd, prog, stdin):
    r = subprocess.run(cmd + [prog], input=stdin or b"", stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    err = r.stderr.decode(errors="replace").splitlines()
    return r.stdout, r.returncode, (err[0] if err else "")


def main():
    args = sys.argv[1:]
    old = None
    if args[:1] == ["--old"]:
        old = args[1:]
    os.makedirs(PROGS, exist_ok=True)
    for name, src, stdin, ref in cases():
        base = os.path.join(PROGS, name)
        with open(base + ".epq", "w") as f:
            f.write(src)
        data = None
        if stdin is not None:
            data = stdin.encode()
            with open(base + ".in", "wb") as f:
                f.write(data)
        if ref == "old":
            if old is None:
                if os.path.exists(base + ".code"):
                    continue
                sys.exit("case %s needs --old CMD" % name)
            cmd = old
        else:
            cmd = [REF]
        out, code, err = run(cmd, base + ".epq", data)
        for ext in (".out", ".sha"):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        if len(out) <= BIG:
            with open(base + ".out", "wb") as f:
                f.write(out)
        else:  # large outputs are stored as size + SHA-256
            with open(base + ".sha", "w") as f:
                f.write("%d %s\n" % (len(out), hashlib.sha256(out).hexdigest()))
        with open(base + ".code", "w") as f:
            f.write("%d\n" % code)
        with open(base + ".err", "w") as f:
            f.write(err + "\n" if err else "")
        print("%-28s %-8s %8d bytes  exit %d  %s" % (name, ref, len(out), code, err))


if __name__ == "__main__":
    main()
