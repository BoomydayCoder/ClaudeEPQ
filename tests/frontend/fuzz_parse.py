#!/usr/bin/env python3
"""Quick grammar-based + mutation fuzzer for parse/resolve agreement.

Generates token sequences from the *ambiguous* grammar (no precedence), so the
two parsers must agree on every disambiguation, then optionally mutates them
(insert / delete / replace / duplicate tokens) to probe error paths. Each case
is prefixed with "=> 0;" (and sometimes with global definitions so that the
resolver gets further) and compared with harness.compare().

usage: fuzz_parse.py [-n CASES] [--seed S] [--epq BINARY] [-j N] [--keep-going]
"""

import argparse
import concurrent.futures
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

IDS = ["a", "b", "c", "f", "x"]
NUMS = ["0", "1", "7", "2147483648"]
BINOPS = [":+", "+", "-", "*", "/", "=", ">", "<", ">=", "<=", "!=", "|", "&"]
VOCAB = (IDS + NUMS + BINOPS +
         [":=", ">>", "<<", "=>", "(", ")", ";", "{", "}", "!", "?", ":", "@", ",", ".", "#", "[", "]"])


class Gen(object):
    def __init__(self, rnd, max_depth):
        self.r = rnd
        self.max_depth = max_depth

    def ident(self):
        return self.r.choice(IDS)

    def stmt(self, d):
        r = self.r
        if d >= self.max_depth:
            return r.choice([[";"], [self.ident(), ";"], [self.ident(), ":=", "1", ";"]])
        k = r.randrange(100)
        if k < 6:
            return [";"]
        if k < 34:
            return self.exp(d + 1) + [";"]
        if k < 40:
            return [">>"] + self.exp(d + 1) + [";"]
        if k < 45:
            return ["=>"] + self.exp(d + 1) + [";"]
        if k < 52:
            return self.block(d + 1)
        if k < 64:
            s = ["("] + self.exp(d + 1) + [")", "?"] + self.stmt(d + 1)
            if r.random() < 0.5:
                s += [":"] + self.stmt(d + 1)
            return s
        if k < 72:
            return ["("] + self.exp(d + 1) + [")", "@"] + self.stmt(d + 1)
        if k < 88:
            return (["("] + self.stmt(d + 1) + self.exp(d + 1) + [";"] + self.stmt(d + 1) +
                    [")", "@"] + self.stmt(d + 1))
        return ["(", self.ident(), ":"] + self.exp(d + 1) + [")", "@"] + self.stmt(d + 1)

    def block(self, d):
        out = ["{"]
        for _ in range(self.r.randrange(3)):
            out += self.stmt(d + 1)
        return out + ["}"]

    def explist(self, d):
        n = self.r.choice([0, 1, 1, 2, 3])
        out = []
        for i in range(n):
            if i:
                out.append(",")
            out += self.exp(d + 1)
        return out

    def idtype(self, d):
        k = self.r.randrange(10)
        if k < 6 or d >= self.max_depth:
            return [self.ident()]
        if k < 8:
            return self.exp(d + 1) + ["["] + self.exp(d + 1) + ["]"]
        if k < 9:
            return [".", self.ident()]
        return self.exp(d + 1) + [".", self.ident()]

    def bar_atom(self, d):
        """An operand/lookahead for '|' chains: covers every token class that
        can follow `'|' exp '|'` (shift set NUMBER ID INPUT ! #, reduce set
        '|' - . [ ( and others)."""
        r = self.r
        k = r.randrange(14)
        if k == 0:
            return [r.choice(NUMS)]
        if k == 1:
            return [self.ident()]
        if k == 2:
            return ["<<"]
        if k == 3:
            return ["!"] + self.bar_atom(d + 1)
        if k == 4:
            return ["#", "{", "}"]
        if k == 5:
            return ["-"] + self.bar_atom(d + 1)
        if k == 6:
            return ["(", self.ident(), ")"]
        if k == 7:
            return ["[", "]"]
        if k == 8:
            return [".", self.ident()]
        if k == 9:
            return [self.ident(), ":=", self.ident()]
        if k == 10:
            return [self.ident(), ":+", self.ident()]
        if k == 11:
            return [self.ident(), r.choice(BINOPS), self.ident()]
        if k == 12 and d < self.max_depth:
            return self.exp(d + 1)
        return []

    def bar_chain(self, d):
        out = ["|"]
        for _ in range(self.r.randrange(1, 5)):
            out += self.bar_atom(d) + ["|"]
        if self.r.random() < 0.3:
            out += self.bar_atom(d)
        return out

    def exp(self, d):
        r = self.r
        if d >= self.max_depth:
            return [r.choice(IDS + NUMS + ["<<"])]
        if r.random() < 0.06:
            return self.bar_chain(d)
        k = r.randrange(100)
        if k < 12:
            return self.idtype(d + 1) + [":="] + self.exp(d + 1)
        if k < 32:
            return self.exp(d + 1) + [r.choice(BINOPS)] + self.exp(d + 1)
        if k < 42:
            return ["|"] + self.exp(d + 1) + ["|"]
        if k < 47:
            return ["!"] + self.exp(d + 1)
        if k < 52:
            return ["-"] + self.exp(d + 1)
        if k < 58:
            return ["("] + self.exp(d + 1) + [")"]
        if k < 64:
            return [r.choice(NUMS)]
        if k < 72:
            return [self.ident()]
        if k < 74:
            return ["<<"]
        if k < 78:
            return ["["] + self.explist(d + 1) + ["]"]
        if k < 82:
            return ["("] + self.explist(d + 1) + [")", "=>"] + self.block(d + 1)
        if k < 84:
            return ["("] + self.exp(d + 1) + [")", "=>"] + self.block(d + 1)
        if k < 91:
            return self.exp(d + 1) + ["("] + self.explist(d + 1) + [")"]
        if k < 95:
            return ["#"] + self.block(d + 1)
        return self.idtype(d + 1)


def mutate(r, toks):
    toks = list(toks)
    for _ in range(r.choice([1, 1, 2, 3])):
        k = r.randrange(5)
        pos = r.randrange(len(toks) + 1)
        if k == 0 and toks:
            del toks[min(pos, len(toks) - 1)]
        elif k == 1:
            toks.insert(pos, r.choice(VOCAB))
        elif k == 2 and toks:
            toks[min(pos, len(toks) - 1)] = r.choice(VOCAB)
        elif k == 3 and toks:
            a = r.randrange(len(toks))
            b = min(len(toks), a + r.randrange(1, 5))
            toks[pos:pos] = toks[a:b]
        else:
            toks.insert(pos, r.choice(["$", "@", "?", ":", "|", "("]))
    return toks


def render(r, toks):
    out = []
    for t in toks:
        out.append(t)
        x = r.random()
        out.append(" " if x < 0.85 else ("\n" if x < 0.93 else ""))
    return "".join(out)


def make_case(seed):
    r = random.Random(seed)
    g = Gen(r, r.choice([3, 4, 5, 6, 7]))
    mode = r.randrange(10)
    if mode == 0:
        toks = [r.choice(VOCAB) for _ in range(r.randrange(1, 14))]
    else:
        toks = []
        for _ in range(r.randrange(1, 4)):
            toks += g.stmt(0)
        if mode < 4:
            toks = mutate(r, toks)
    pre = harness.PREFIX
    if r.random() < 0.5:
        pre += "a := 0; b := 0; c := 0; x := 0; f := 0;\n"
    return pre + render(r, toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epq", default=harness.EPQ)
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--show", type=int, default=5, help="divergences to print")
    a = ap.parse_args()
    epq = os.path.abspath(a.epq)

    faildir = os.path.join(harness.REPO, "build", "frontend", "fuzz_failures")
    counts = {}
    shown = 0
    nfail = 0

    def one(i):
        src = make_case(a.seed * 1000003 + i)
        return i, src, harness.compare(src, epq=epq)

    with concurrent.futures.ThreadPoolExecutor(a.j) as ex:
        for i, src, o in ex.map(one, range(a.n)):
            key = o.kind + ("/" + o.detail if o.kind == "ok" and o.detail else "")
            counts[key] = counts.get(key, 0) + 1
            if o.kind in ("DIVERGE", "CRASH", "errloc"):
                nfail += 1
                os.makedirs(faildir, exist_ok=True)
                with open(os.path.join(faildir, "fail_%d_%d.epq" % (a.seed, i)), "w") as f:
                    f.write(src)
                if shown < a.show:
                    shown += 1
                    print("FAIL [%s] case %d: %s" % (o.kind, i, o.detail))
                    print(harness.show(o, src))
    print("fuzz_parse: %d cases (seed %d): %s" % (a.n, a.seed, ", ".join(
        "%s=%d" % kv for kv in sorted(counts.items()))))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
