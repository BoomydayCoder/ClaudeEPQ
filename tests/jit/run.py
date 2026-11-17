#!/usr/bin/env python3
"""Build and run every JIT infrastructure test; print a pass/fail summary.

usage: python3 tests/jit/run.py [--n CASES_PER_FORM] [--no-asan] [-k NAME]

Stages:
  encode   encoder cross-check vs clang (byte-for-byte), plus an ASan/UBSan run
           of the generator (exercises the whole assembler front end)
  exec     JIT execution tests (emit -> MAP_JIT -> call), incl. big stack
  stack    big-stack overflow / foreign-fault behaviour (separate processes)
  disasm   disassembly helper smoke test
Outputs go under build/jit/.
"""
import argparse
import os
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "build", "jit")
JIT_SRC = [os.path.join(ROOT, "src", "jit", f) for f in ("a64.cpp", "jitmem.cpp", "bigstack.cpp", "disasm.cpp")]
CXX = ["clang++", "-std=c++20", "-Wall", "-Wextra", "-Wno-unused-parameter", "-I", os.path.join(ROOT, "src")]
SAN = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer"]

results = []


def sh(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, **kw)


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print("%-6s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def build(name, srcs, flags):
    exe = os.path.join(OUT, name)
    r = sh(CXX + flags + srcs + ["-o", exe])
    if r.returncode != 0:
        record("build " + name, False, "\n" + r.stderr[-4000:])
        return None
    return exe


def macho_text(path):
    """Return (__text bytes, nreloc) from a Mach-O 64-bit object."""
    data = open(path, "rb").read()
    magic, _, _, _, ncmds, _, _, _ = struct.unpack_from("<IiiIIIII", data, 0)
    assert magic == 0xFEEDFACF, "not a 64-bit Mach-O"
    off = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd == 0x19:  # LC_SEGMENT_64
            nsects = struct.unpack_from("<I", data, off + 64)[0]
            s = off + 72
            for _ in range(nsects):
                sect = data[s:s + 16].rstrip(b"\0")
                size_, offset = struct.unpack_from("<QI", data, s + 40)
                nreloc = struct.unpack_from("<I", data, s + 60)[0]
                if sect == b"__text":
                    return data[offset:offset + size_], nreloc
                s += 80
        off += size
    raise RuntimeError("no __text section")


def stage_encode(args):
    d = os.path.join(OUT, "enc")
    os.makedirs(d, exist_ok=True)
    exe = build("encode_gen", [os.path.join(ROOT, "tests", "jit", "encode_gen.cpp"), JIT_SRC[0]], ["-O2"])
    if not exe:
        return
    t0 = time.time()
    r = sh([exe, d, str(args.n)])
    if r.returncode != 0:
        record("encode: generator", False, r.stderr[-3000:])
        return
    sys.stderr.write(r.stderr)
    tgen = time.time() - t0
    t0 = time.time()
    obj = os.path.join(d, "cases.o")
    r = sh(["clang", "-c", "-x", "assembler", os.path.join(d, "cases.s"), "-o", obj])
    if r.returncode != 0:
        record("encode: system assembler", False, r.stderr[-4000:])
        return
    tas = time.time() - t0
    ref, nreloc = macho_text(obj)
    ours = open(os.path.join(d, "cases.bin"), "rb").read()
    lines = open(os.path.join(d, "cases.tsv")).read().splitlines()
    if len(ref) != len(ours) or nreloc:
        record("encode: size", False, "clang %d bytes, ours %d bytes, %d relocations" % (len(ref), len(ours), nreloc))
        return
    per_form = Counter()
    bad = defaultdict(list)
    for i, line in enumerate(lines):
        form, text = line.split("\t", 1)
        per_form[form] += 1
        a, b = ours[4 * i:4 * i + 4], ref[4 * i:4 * i + 4]
        if a != b:
            bad[form].append("%s: ours %08x clang %08x" % (text, struct.unpack("<I", a)[0], struct.unpack("<I", b)[0]))
    nbad = sum(len(v) for v in bad.values())
    for form, v in sorted(bad.items()):
        print("  mismatch [%s] %d cases, e.g.\n    %s" % (form, len(v), "\n    ".join(v[:4])))
    record("encode: cross-check vs clang", nbad == 0,
           "%d cases, %d forms, %d mismatches (gen %.2fs, clang %.2fs)" % (len(lines), len(per_form), nbad, tgen, tas))
    if not args.no_asan:
        exe = build("encode_gen_asan", [os.path.join(ROOT, "tests", "jit", "encode_gen.cpp"), JIT_SRC[0]], SAN)
        if exe:
            d2 = os.path.join(OUT, "enc_asan")
            os.makedirs(d2, exist_ok=True)
            r = sh([exe, d2, str(max(100, args.n // 5))])
            same = r.returncode == 0
            record("encode: generator under ASan/UBSan", same, "" if same else r.stderr[-3000:])


def stage_exec(args):
    exe = build("exec_test", [os.path.join(ROOT, "tests", "jit", "exec_test.cpp")] + JIT_SRC, ["-O2"])
    if not exe:
        return
    r = sh([exe] + (["--exhaustive"] if args.exhaustive else []), timeout=1200)
    sys.stdout.write(r.stdout)
    fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
    record("exec: JIT execution tests", r.returncode == 0 and not fails,
           "" if r.returncode == 0 else "exit %d\n%s" % (r.returncode, r.stderr[-3000:]))
    if not args.no_asan:
        exe = build("exec_test_asan", [os.path.join(ROOT, "tests", "jit", "exec_test.cpp")] + JIT_SRC, SAN + ["-DEPQ_ASAN"])
        if exe:
            r = sh([exe, "--asan-safe"], timeout=300)
            record("exec: host-side tests under ASan/UBSan", r.returncode == 0,
                   "" if r.returncode == 0 else r.stdout[-1500:] + r.stderr[-3000:])


def stage_stack(args):
    exe = build("stack_test", [os.path.join(ROOT, "tests", "jit", "stack_test.cpp")] + JIT_SRC, ["-O2"])
    if not exe:
        return
    r = sh([exe, "deep"], timeout=120)
    record("stack: recursion depth 1e7 + libc calls on big stack", r.returncode == 0 and "OK" in r.stdout,
           r.stdout.strip().replace("\n", "; ") + r.stderr[-1000:])
    r = sh([exe, "overflow"], timeout=120)
    record("stack: runaway recursion -> callback, exit 1",
           r.returncode == 1 and r.stderr.strip() == "Stack overflow" and r.stdout.strip() == "",
           "exit %d stderr=%r" % (r.returncode, r.stderr.strip()[-200:]))
    r = sh([exe, "jitoverflow"], timeout=120)
    record("stack: runaway JIT recursion -> callback, exit 1",
           r.returncode == 1 and r.stderr.strip() == "Stack overflow",
           "exit %d stderr=%r" % (r.returncode, r.stderr.strip()[-200:]))
    r = sh([exe, "nullderef"], timeout=120)
    record("stack: unrelated SIGSEGV still crashes normally", r.returncode < 0 and "Stack overflow" not in r.stderr,
           "exit %d" % r.returncode)
    r = sh([exe, "bench"], timeout=120)
    record("stack: startup costs", r.returncode == 0, r.stdout.strip().replace("\n", "; "))


def stage_disasm(args):
    exe = build("disasm_test", [os.path.join(ROOT, "tests", "jit", "disasm_test.cpp")] + JIT_SRC, ["-O2"])
    if not exe:
        return
    r = sh([exe], timeout=60)
    ok = r.returncode == 0 and "<fib>:" in r.stdout and "ret" in r.stdout and "<loop_head>:" in r.stdout
    print(r.stdout)
    record("disasm: objdump listing with symbols", ok, r.stderr[-1000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="random cases per instruction form")
    ap.add_argument("--no-asan", action="store_true")
    ap.add_argument("--exhaustive", action="store_true", help="all 2^32 dividends in the division test (~1 min)")
    ap.add_argument("-k", default="", help="only stages containing this substring")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("encode", stage_encode), ("exec", stage_exec), ("stack", stage_stack), ("disasm", stage_disasm)):
        if args.k in name:
            fn(args)
    npass = sum(1 for r in results if r[1])
    print("\n%d/%d passed" % (npass, len(results)))
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
