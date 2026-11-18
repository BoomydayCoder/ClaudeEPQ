"""optgen -- optimisation-targeted program snippets for the fuzzer (profile `opt`).

tools/epqgen.py builds programs from a small kind system; it almost never produces
the shapes that ClaudeEPQ's optimisation passes look for (fixed-shape records
mutated through aliases, pairs, tail / doubly-recursive pure functions, copy loops,
guarded conditional updates, short-lived lists, ...).  This module generates
self-contained *snippets* of such code, with deliberate variety, and then
*perturbs* them so that a pass precondition is sometimes violated far away (a
"tail" statement placed at the end of the program, or in a function defined early
and only called late).

Guarantees (same as epqgen, by construction):
* Every name a snippet uses starts with `O` followed by a family tag and a
  program-unique number (epqgen only produces lower-case names), every parameter
  name is unique in the program (so no `Argument shadows global variable`), and
  every function only reads globals and its own parameters / locals.
* Locals are only defined by statement-level `x := e;` in a statement list, block
  or loop body, or by a for-init; every top-level variable is first assigned in
  straight-line top-level code (so it is a global that is defined before use).
* Termination: every loop is counted (literal or clamped bounds), walks a finite
  acyclic structure (cons lists / trees, built from fresh nodes only, never
  linked into themselves), or makes strict progress on a bounded index (heap
  sift).  Recursion depth is bounded by a literal or clamped argument.
* Cost is small (sizes <= ~40, fib/binomial arguments <= 14).
* `<<` is only read in straight-line top-level code; each read is immediately
  clamped into a small range, so any input (also epqgen's huge values) is fine.
  `reads` counts them.
* Functions are never printed.  Lists stay acyclic.  Comparisons with `<` only
  meet int/int or list/list at every position actually reached, unless the
  snippet is marked `risky`.
* `risky` snippets may stop the program with a run-time error that depends on
  the input (division by zero in a pure function, out-of-bounds copy ranges,
  error-order checks); a program has at most one, placed last.

Constants / globals are checked by the caller (epqgen.generate) on the AST.
"""


class Snip:
    __slots__ = ("head", "tail", "reads", "risky", "fam", "nglob")

    def __init__(self, fam):
        self.fam = fam
        self.head = []
        self.tail = []
        self.reads = 0
        self.risky = False
        self.nglob = 0


FAMILIES = ["rec", "tree", "cons", "pairs", "helpers", "recur", "append", "copy", "idxloop",
            "ifconv", "litcmp", "reclaim", "glob", "recloop", "inl2", "alias", "grow", "rec2", "glue"]


class OptGen:
    def __init__(self, rng, start=0, glue=True):
        self.r = rng
        self.k = start
        self.glue = glue
        self.exports = []   # (name, kind) of values later snippets may use (kinds below, f_glue)

    def export(self, name, kind):
        if self.glue:
            self.exports.append((name, kind))

    # ------------------------------------------------------------------ utils
    def nm(self, t):
        self.k += 1
        return "O%s%d" % (t, self.k)

    def ch(self, p):
        return self.r.random() < p

    def ri(self, a, b):
        return self.r.randint(a, b)

    def pick(self, xs):
        return self.r.choice(xs)

    def read(self, s, lo, hi, tag="n"):
        """Top-level clamped input read; returns the global's name."""
        v = self.nm(tag)
        s.reads += 1
        s.nglob += 1
        s.head.append("%s := <<;" % v)
        s.head.append("(%s > %d) ? %s := %d;" % (v, hi, v, hi))
        s.head.append("(%s < %d) ? %s := %d;" % (v, lo, v, lo))
        return v

    def size(self, s, lo, hi):
        """A size: literal or clamped input."""
        if self.ch(0.35):
            return self.read(s, lo, hi)
        return str(self.ri(lo, hi))

    def iexp(self, vs, d=2):
        r = self.r
        if d <= 0 or r.random() < 0.35:
            if vs and r.random() < 0.75:
                return r.choice(vs)
            return str(r.randint(0, 9))
        op = r.choice(["+", "-", "*", "+", "-"])
        return "(%s %s %s)" % (self.iexp(vs, d - 1), op, self.iexp(vs, d - 1))

    def mod(self, a, m):
        """a mod m (m a positive literal), as a - a / m * m (sign follows a)."""
        return "(%s - %s / %d * %d)" % (a, a, m, m)

    def glob(self, s, t, init):
        v = self.nm(t)
        s.nglob += 1
        s.head.append("%s := %s;" % (v, init))
        return v

    def fn(self, s, t, params, body):
        """Defines a global function; params are fresh names given as count or list."""
        f = self.nm(t)
        s.nglob += 1
        s.head.append("%s := (%s) => { %s };" % (f, ", ".join(params), " ".join(body)))
        return f

    def params(self, n, t="p"):
        return [self.nm(t) for _ in range(n)]

    # =============================================================== records
    def f_rec(self):
        """Fixed-shape heterogeneous records, mutated through aliases."""
        s = Snip("rec")
        shape = self.pick(["il", "ip", "ili", "iii"])
        a, b = self.params(2)
        if shape == "il":
            lit = "[%s, [%s]]" % (a, b)
            l0 = "[%d, [%d]]" % (self.ri(0, 9), self.ri(0, 9))
            inner = "[%d]" % self.ri(0, 9)
        elif shape == "ip":
            lit = "[%s, [%s, %s + 1]]" % (a, b, a)
            l0 = "[%d, [%d, %d]]" % (self.ri(0, 9), self.ri(0, 9), self.ri(0, 9))
            inner = "[%d, %d]" % (self.ri(0, 9), self.ri(0, 9))
        elif shape == "ili":
            lit = "[%s, [%s, %s], %s * 2]" % (a, b, b, b)
            l0 = "[%d, [%d, %d], %d]" % (self.ri(0, 9), self.ri(0, 9), self.ri(0, 9), self.ri(0, 9))
            inner = "[%d, %d]" % (self.ri(0, 9), self.ri(0, 9))
        else:
            lit = "[%s, %s, %s + %s]" % (a, b, a, b)
            l0 = "[%d, %d, %d]" % (self.ri(0, 9), self.ri(0, 9), self.ri(0, 9))
            inner = None
        mk = self.fn(s, "mk", [a, b], ["=> %s;" % lit])
        # a pure helper: either inlinable or recursive (not inlined -> recur CSE)
        q = self.nm("p")
        if self.ch(0.5):
            sq = self.fn(s, "sq", [q], ["=> %s * %s + 1;" % (q, q)])
        else:
            sq = self.fn(s, "sq", [q], ["(%s < 1) ? => 0;" % q, "=> %s + %s(%s - 1) + 1;" % (q, "SELF", q)])
            s.head[-1] = s.head[-1].replace("SELF", sq)
        r1 = self.glob(s, "r", "%s(%d, %d)" % (mk, self.ri(0, 9), self.ri(0, 9)) if self.ch(0.6) else l0)
        r2 = self.glob(s, "r", r1)
        hold = self.glob(s, "hd", "[%s, %s(%d, %d)]" % (r1, mk, self.ri(0, 9), self.ri(0, 9)))
        rp, vp = self.params(2)
        setf = self.fn(s, "set", [rp, vp], ["%s[0] := %s;" % (rp, vp), "=> %s;" % vp])
        bump = self.fn(s, "bmp", [], ["%s[0] := %s[0] + 1;" % (r1, r1), "=> 0;"])
        fp = self.nm("p")
        tail1 = "%s[1][0]" % fp if inner else "%s[1]" % fp
        fs = self.fn(s, "fs", [fp], ["=> %s[0] * 2 + %s;" % (fp, tail1)])
        K = lambda: str(self.ri(0, 12))
        stm = [
            ">> %s(%s[0]) + (%s[0] := %s) + %s(%s[0]);" % (sq, r1, r2, K(), sq, r1),
            ">> %s[0] + %s(%s, %s) + %s[0];" % (r1, setf, r2, K(), r1),
            ">> %s[0] + (%s[0][0] := %s) + %s[0];" % (r1, hold, K(), r2),
            ">> %s(%s[0]) + %s(%s[0], %s) + %s(%s[0]);" % (sq, r2, setf, hold, K(), sq, r2),
            ">> %s[0] + %s() + %s[0];" % (r2, bump, r2),
            ">> %s(%s) + (%s[0] := %s) + %s(%s);" % (fs, r1, r2, K(), fs, r2),
            ">> %s; >> %s;" % (r1, hold),
            ">> %s = %s; >> %s = %s(%d, %d); >> %s[1] = %s;" % (r1, r2, r1, mk, self.ri(0, 9), self.ri(0, 9), hold, r1),
            ">> %s < %s(%d, %d); >> %s(%d, %d) < %s;" % (r1, mk, self.ri(0, 9), self.ri(0, 9), mk, self.ri(0, 9),
                                                    self.ri(0, 9), r2),
        ]
        if inner:
            stm += [
                "%s[1][0] := %s; >> %s;" % (r2, K(), r1),
                "%s[1] := %s; >> %s; >> %s[1][0] + %s[1][0];" % (r2, inner, r1, r1, r2),
                "%s[1] :+ %s; >> %s[1];" % (r1, K(), r2),
                ">> %s[1][0] + (%s[1][0] := %s) + %s[1][0];" % (r1, r2, K(), r1),
            ]
        self.r.shuffle(stm)
        s.head += stm[:self.ri(3, len(stm))]
        # list of records, loops, a function over records
        n = self.size(s, 0, 12)
        rs = self.glob(s, "rs", "[]")
        i = self.nm("i")
        s.head.append("(%s := 0; %s < %s; %s := %s + 1;) @ %s :+ %s(%s, %s * 2);" % (i, i, n, i, i, rs, mk, i, i))
        acc = self.glob(s, "ac", "0")
        x = self.nm("x")
        s.head.append("(%s : %s) @ %s := %s + %s(%s);" % (x, rs, acc, acc, fs, x))
        if self.ch(0.5):
            j = self.nm("i")
            s.head.append("(%s := 0; %s < |%s|; %s := %s + 1;) @ { (%s[%s][0] > 3) ? %s[%s][0] := %s[%s][0] - 1; }"
                          % (j, j, rs, j, j, rs, j, rs, j, rs, j))
        s.head.append(">> %s; >> %s;" % (acc, rs))
        self.rec_violation(s, [r1], inner is not None, mk)
        self.export(r1, "rec")  # exports
        return s

    def rec_violation(self, s, names, has_inner, mk):
        """Sometimes break the record shape far away (in the tail)."""
        if not self.ch(0.45):
            return
        r = names[0]
        kv = self.read(s, 0, 1, "kv")
        idf = self.nm("id")
        pp = self.nm("p")
        s.nglob += 1
        s.head.append("%s := (%s) => { => %s; };" % (idf, pp, pp))
        v = self.pick(["idx", "app", "pop", "each", "dyn", "store", "storeT", "fnapp", "len"])
        if v == "idx":
            s.tail.append(">> %s[%s];" % (r, kv))
        elif v == "app":
            s.tail.append("%s :+ %d; >> %s;" % (r, self.ri(0, 9), r))
        elif v == "pop":
            s.tail.append(">> -%s; >> %s;" % (r, r))
        elif v == "each":
            x = self.nm("x")
            s.tail.append("(%s : %s) @ >> %s;" % (x, r, x))
        elif v == "dyn":
            fl = self.glob(s, "fl", "[%s, %s]" % (idf, idf))
            s.tail.append(">> %s[%s](%s);" % (fl, kv, r))
        elif v == "store":
            s.tail.append("%s[%s] := %d; >> %s;" % (r, kv, self.ri(0, 9), r))
        elif v == "storeT":
            s.tail.append("%s[0] := [%d]; >> %s;" % (r, self.ri(0, 9), r))
        elif v == "fnapp":
            pp2 = self.nm("p")
            vf = self.fn(s, "vf", [pp2], ["%s :+ %d;" % (pp2, self.ri(0, 9)), "=> |%s|;" % pp2])
            s.tail.append(">> %s(%s(%d, %d)); >> %s(%s);" % (vf, mk, self.ri(0, 9), self.ri(0, 9), vf, r))
        else:
            s.tail.append(">> |%s|; >> %s(%s) = %s;" % (r, idf, r, r))

    # ================================================================== trees
    def f_tree(self):
        s = Snip("tree")
        d, v, t, t2, w = self.params(5)
        mkT = self.nm("mkT")
        s.nglob += 1
        s.head.append("%s := (%s, %s) => { (%s < 1) ? => []; => [%s, %s(%s - 1, %s * 2), %s(%s - 1, %s * 2 + 1)]; };"
                      % (mkT, d, v, d, v, mkT, d, v, mkT, d, v))
        sumT = self.nm("sumT")
        s.nglob += 1
        s.head.append("%s := (%s) => { (%s = []) ? => 0; => %s[0] + %s(%s[1]) + %s(%s[2]); };"
                      % (sumT, t, t, t, sumT, t, sumT, t))
        ins = self.nm("ins")
        s.nglob += 1
        s.head.append("%s := (%s, %s) => { (%s = []) ? => [%s, [], []]; (%s < %s[0]) ? %s[1] := %s(%s[1], %s); : "
                      "%s[2] := %s(%s[2], %s); => %s; };"
                      % (ins, t2, w, t2, w, w, t2, t2, ins, t2, w, t2, ins, t2, w, t2))
        T = self.glob(s, "T", "%s(%s, 1)" % (mkT, self.size(s, 0, 4)))
        n = self.size(s, 0, 14)
        U = self.glob(s, "U", "[]")
        i = self.nm("i")
        m = self.pick([7, 11, 13, 17])
        s.head.append("(%s := 0; %s < %s; %s := %s + 1;) @ %s := %s(%s, %s);"
                      % (i, i, n, i, i, U, ins, U, self.mod("(%s * %d + %d)" % (i, self.ri(2, 9), self.ri(0, 9)), m)))
        s.head.append(">> %s(%s); >> %s(%s); >> %s;" % (sumT, T, sumT, U, U))
        # iterative walk down a path (nullable record loop)
        p = self.glob(s, "pw", self.pick([T, U]))
        c = self.glob(s, "cw", "0")
        s.head.append("(%s != []) @ { %s := %s + %s[0]; %s := %s[%d]; }" % (p, c, c, p, p, p, self.ri(1, 2)))
        s.head.append(">> %s;" % c)
        if self.ch(0.6):
            dp = self.nm("p")
            a1, b1 = self.nm("a"), self.nm("b")
            dep = self.nm("dep")
            s.nglob += 1
            s.head.append("%s := (%s) => { (%s = []) ? => 0; %s := %s(%s[1]); %s := %s(%s[2]); (%s > %s) ? => %s + 1; "
                          "=> %s + 1; };" % (dep, dp, dp, a1, dep, dp, b1, dep, dp, a1, b1, a1, b1))
            s.head.append(">> %s(%s) + %s(%s);" % (dep, T, dep, U))
        # mutate through an alias, read back through the other
        q = self.glob(s, "q", T)
        s.head.append("(%s != []) ? >> %s[0] + (%s[0] := %d) + %s(%s);" % (T, T, q, self.ri(0, 9), sumT, T))
        if self.ch(0.5):
            s.head.append("(%s != []) ? %s[1] := %s(%s[1], %d);" % (U, U, ins, U, self.ri(0, 20)))
            s.head.append(">> %s = %s; >> %s < %s; >> %s;" % (U, T, T, U, U))
        if self.ch(0.4):
            # an outer list of trees
            fr = self.glob(s, "fr", "[%s, %s, []]" % (T, U))
            x = self.nm("x")
            tot = self.glob(s, "tt", "0")
            s.head.append("(%s : %s) @ %s := %s + %s(%s);" % (x, fr, tot, tot, sumT, x))
            s.head.append(">> %s;" % tot)
        if self.ch(0.4):
            kv = self.read(s, 0, 2, "kv")
            v2 = self.pick(["idx", "app", "pop", "each", "len"])
            if v2 == "idx":
                s.tail.append("(%s != []) ? >> %s[%s];" % (U, U, kv))
            elif v2 == "app":
                s.tail.append("%s :+ %d; >> %s;" % (T, self.ri(0, 9), T))
            elif v2 == "pop":
                s.tail.append("(%s != []) ? >> -%s;" % (U, U))
                s.tail.append(">> %s;" % U)
            elif v2 == "each":
                x = self.nm("x")
                s.tail.append("(%s : %s) @ >> %s = [];" % (x, T, x))
            else:
                s.tail.append(">> |%s| + |%s|;" % (T, U))
        self.export(U, "tree"); self.export(T, "tree")  # exports
        return s

    # ============================================================== cons lists
    def f_cons(self):
        s = Snip("cons")
        n = self.size(s, 0, 15)
        C = self.glob(s, "C", "[]")
        i = self.nm("i")
        s.head.append("(%s := 0; %s < %s; %s := %s + 1;) @ %s := [%s, %s];"
                      % (i, i, n, i, i, C, self.iexp([i], 2), C))
        c1, c2, acc, c3, r = self.params(5)
        cs = self.nm("cs")
        s.nglob += 1
        s.head.append("%s := (%s) => { (%s = []) ? => 0; => %s[0] + %s(%s[1]); };" % (cs, c1, c1, c1, cs, c1))
        ca = self.nm("ca")
        s.nglob += 1
        s.head.append("%s := (%s, %s) => { (%s = []) ? => %s; => %s(%s[1], %s + %s[0]); };"
                      % (ca, c2, acc, c2, acc, ca, c2, acc, c2))
        rev = self.fn(s, "rev", [c3], ["%s := [];" % r, "(%s != []) @ { %s := [%s[0], %s]; %s := %s[1]; }"
                                       % (c3, r, c3, r, c3, c3), "=> %s;" % r])
        s.head.append(">> %s; >> %s(%s); >> %s(%s) = %s(%s, 0);" % (C, rev, C, cs, C, ca, C))
        bn, bi, bc = self.nm("p"), self.nm("i"), self.nm("c")
        bld = self.fn(s, "bld", [bn], ["%s := [];" % bc, "(%s := 0; %s < %s; %s := %s + 1;) @ %s := [%s * %d, %s];"
                                      % (bi, bi, bn, bi, bi, bc, bi, self.ri(1, 5), bc), "=> %s;" % bc])
        D = self.glob(s, "D", "%s(%d)" % (bld, self.ri(0, 8)))
        E = self.glob(s, "E", D)
        s.head.append("(%s != []) ? >> %s[0] + (%s[0] := %d) + %s[0];" % (D, D, E, self.ri(0, 9), D))
        s.head.append(">> %s < %s; >> %s = %s(%s(%s)); >> %s = [];" % (C, D, D, rev, rev, D, C))
        if self.ch(0.5):
            # destructive append at the end of a cons list
            p = self.glob(s, "pw", D)
            s.head.append("(%s != []) ? { (%s[1] != []) @ %s := %s[1]; %s[1] := [%d, []]; }"
                          % (p, p, p, p, p, self.ri(0, 9)))
            s.head.append(">> %s; >> %s(%s, 0);" % (E, ca, E))
        if self.ch(0.4):
            v = self.pick(["app", "idx", "each", "pop", "store"])
            kv = self.read(s, 0, 1, "kv")
            if v == "app":
                s.tail.append("%s :+ 1; >> %s;" % (D, D))
            elif v == "idx":
                s.tail.append("(%s != []) ? >> %s[%s];" % (C, C, kv))
            elif v == "each":
                x = self.nm("x")
                s.tail.append("(%s : %s) @ >> %s;" % (x, D, x))
            elif v == "pop":
                s.tail.append("(%s != []) ? { >> -%s; >> %s; }" % (C, C, C))
            else:
                s.tail.append("(%s != []) ? %s[%s] := [%d]; >> %s;" % (C, C, kv, self.ri(0, 9), C))
        self.export(C, "cons"); self.export(D, "cons")  # exports
        return s

    # =================================================================== pairs
    def f_pairs(self):
        s = Snip("pairs")
        n = self.size(s, 0, 16)
        P = self.glob(s, "P", "[]")
        i = self.nm("i")
        m = self.pick([5, 7, 9, 11])
        s.head.append("(%s := 0; %s < %s; %s := %s + 1;) @ %s :+ [%s, %s];"
                      % (i, i, n, i, i, P, self.mod("(%s * %d + %d)" % (i, self.ri(2, 9), self.ri(0, 9)), m),
                         self.iexp([i], 1)))
        kind = self.pick(["isort", "heap", "adj", "isort", "heap"])
        if kind == "isort":
            a, j, t = self.nm("i"), self.nm("j"), self.nm("t")
            s.head.append("(%s := 1; %s < |%s|; %s := %s + 1;) @ { %s := %s; (%s > 0 & %s[%s] < %s[%s - 1]) @ "
                          "{ %s := %s[%s]; %s[%s] := %s[%s - 1]; %s[%s - 1] := %s; %s := %s - 1; } }"
                          % (a, a, P, a, a, j, a, j, P, j, P, j, t, P, j, P, j, P, j, P, j, t, j, j))
            s.head.append(">> %s;" % P)
        elif kind == "heap":
            h, x, k, p, t = self.params(5)
            hp = self.fn(s, "hp", [h, x], [
                "%s :+ %s;" % (h, x), "%s := |%s| - 1;" % (k, h),
                "(%s > 0 & %s[%s] < %s[(%s - 1) / 2]) @ { %s := (%s - 1) / 2; %s := %s[%s]; %s[%s] := %s[%s]; "
                "%s[%s] := %s; %s := %s; }" % (k, h, k, h, k, p, k, t, h, k, h, k, h, p, h, p, t, k, p),
                "=> 0;"])
            h2, top, last, k2, go, c, t2 = self.params(7)
            hpop = self.fn(s, "hpop", [h2], [
                "%s := %s[0];" % (top, h2), "%s := -%s;" % (last, h2),
                "(|%s| > 0) ? { %s[0] := %s; %s := 0; %s := 1; (%s) @ { %s := 2 * %s + 1; "
                "(%s + 1 < |%s| & %s[%s + 1] < %s[%s]) ? %s := %s + 1; "
                "(%s < |%s| & %s[%s] < %s[%s]) ? { %s := %s[%s]; %s[%s] := %s[%s]; %s[%s] := %s; %s := %s; } : %s := 0; } }"
                % (h2, h2, last, k2, go, go, c, k2, c, h2, h2, c, h2, c, c, c,
                   c, h2, h2, c, h2, k2, t2, h2, c, h2, c, h2, k2, h2, k2, t2, k2, c, go),
                "=> %s;" % top])
            H = self.glob(s, "H", "[]")
            x2 = self.nm("x")
            s.head.append("(%s : %s) @ %s(%s, %s);" % (x2, P, hp, H, x2))
            s.head.append("%s(%s, [%d, %d]);" % (hp, H, self.ri(0, 9), self.ri(0, 9)))
            s.head.append("(|%s| > 0) @ >> %s(%s);" % (H, hpop, H))
        else:
            V = self.ri(2, 6)
            A = self.glob(s, "A", "[%s]" % ", ".join(["[]"] * V))
            e, u = self.nm("e"), self.nm("u")
            s.head.append("(%s : %s) @ %s[%s[0] - %s[0] / %d * %d] :+ [%s[1], %s[0]];" % (e, P, A, e, e, V, V, e, e))
            tot = self.glob(s, "tt", "0")
            e2 = self.nm("e")
            s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ (%s : %s[%s]) @ %s := %s + %s[0] * %s[1];"
                          % (u, u, V, u, u, e2, A, u, tot, tot, e2, e2))
            s.head.append(">> %s; >> %s;" % (tot, A))
        best = self.glob(s, "bp", "[%d, %d]" % (self.ri(0, 9), self.ri(0, 9)))
        x3 = self.nm("x")
        s.head.append("(%s : %s) @ (%s < %s) ? %s := %s;" % (x3, P, x3, best, best, x3))
        s.head.append(">> %s; >> %s = [%d, %d]; >> %s != [];" % (best, best, self.ri(0, 9), self.ri(0, 9), P))
        if self.ch(0.6):
            # pairs meeting lists of other lengths / literals / records / functions
            pp, pk = self.params(2)
            mkp = self.fn(s, "mkp", [pp, pk], ["=> [%s + %s, %s - %s];" % (pp, pk, pp, pk)])
            Q = self.glob(s, "Q", "%s(%d, %d)" % (mkp, self.ri(0, 9), self.ri(0, 9)))
            other = self.pick(["[%d]" % self.ri(0, 9), "[%d, %d, %d]" % (self.ri(0, 9), self.ri(0, 9), self.ri(0, 9)),
                               "[]", "[[%d], %d]" % (self.ri(0, 9), self.ri(0, 9))])
            ops = ["%s = %s" % (Q, other), "%s != %s" % (other, Q), "|%s|" % Q, "%s = %s" % (Q, best),
                   "[%s, %d] = [%s(%d, %d), %d]" % (Q, self.ri(0, 3), mkp, self.ri(0, 9), self.ri(0, 9), self.ri(0, 3)),
                   "%s[0] * 10 + %s[1]" % (Q, Q)]
            if not other.startswith("[["):
                ops.append(self.pick(["%s < %s" % (Q, other), "%s < %s" % (other, Q)]))
            self.r.shuffle(ops)
            s.head.append(">> [%s];" % ", ".join(ops[:self.ri(2, len(ops))]))
            rk = self.glob(s, "rk", "[%d, %s]" % (self.ri(0, 9), Q))      # a record holding a pair
            s.head.append(">> %s; >> %s[1][0] + %s[1][1];" % (rk, rk, rk))
            if self.ch(0.4):
                kv = self.read(s, 0, 1, "kv")
                s.head.append(">> %s[%s] + %s[1 - %s];" % (Q, kv, Q, kv))
            if self.ch(0.3):
                ppl = self.glob(s, "pl", "[%s, %s(%d, %d)]" % (Q, mkp, self.ri(0, 9), self.ri(0, 9)))
                x4, i4, t4 = self.nm("x"), self.nm("i"), self.nm("t")
                s.head.append("(%s := 0; %s < |%s| - 1; %s := %s + 1;) @ (%s[%s + 1] < %s[%s]) ? "
                              "{ %s := %s[%s]; %s[%s] := %s[%s + 1]; %s[%s + 1] := %s; }"
                              % (i4, i4, ppl, i4, i4, ppl, i4, ppl, i4, t4, ppl, i4, ppl, i4, ppl, i4, ppl, i4, t4))
                s.head.append("(%s : %s) @ >> %s;" % (x4, ppl, x4))
        if self.ch(0.45):
            v = self.pick(["setel", "app", "pop", "each", "dyn"])
            if v == "setel":
                s.tail.append("(|%s| > 0) ? %s[0][1] := %d; >> %s;" % (P, P, self.ri(0, 9), P))
            elif v == "app":
                s.tail.append("(|%s| > 0) ? %s[0] :+ %d; >> %s;" % (P, P, self.ri(0, 9), P))
            elif v == "pop":
                s.tail.append("(|%s| > 0) ? >> -%s[0]; >> %s;" % (P, P, P))
            elif v == "each":
                x = self.nm("x")
                s.tail.append("(%s : %s) @ >> %s;" % (x, best, x))
            else:
                pp = self.nm("p")
                g = self.fn(s, "dg", [pp], ["=> %s[1];" % pp])
                gl = self.glob(s, "fl", "[%s]" % g)
                s.tail.append(">> %s[0](%s);" % (gl, best))
        self.export(P, "pairs")  # exports
        return s

    # ================================================================ helpers
    def f_helpers(self):
        """Small helper functions called in loops (inlining)."""
        s = Snip("helpers")
        l, a, b, t = self.params(4)
        swap = self.fn(s, "sw", [l, a, b], ["%s := %s[%s];" % (t, l, a), "%s[%s] := %s[%s];" % (l, a, l, b),
                                           "%s[%s] := %s;" % (l, b, t), "=> 0;"])
        x, y = self.params(2)
        mn = self.fn(s, "mn", [x, y], ["(%s < %s) ? => %s;" % (x, y, x), "=> %s;" % y])
        x2, y2 = self.params(2)
        mx = self.fn(s, "mx", [x2, y2], ["(%s > %s) ? => %s;" % (x2, y2, x2), "=> %s;" % y2])
        x3, m3 = self.params(2)
        md = self.fn(s, "md", [x3, m3], ["=> %s - %s / %s * %s;" % (x3, x3, m3, m3)])
        x4 = self.nm("p")
        ev = self.fn(s, "ev", [x4], ["=> %s - %s / 2 * 2 = 0;" % (x4, x4)])
        n = self.size(s, 0, 14)
        L = self.glob(s, "L", "[]")
        i = self.nm("i")
        mm = self.pick([7, 10, 13])
        s.head.append("(%s := 0; %s < %s; %s := %s + 1;) @ %s :+ %s(%s * %d + %d, %d);"
                      % (i, i, n, i, i, L, md, i, self.ri(3, 11), self.ri(0, 9), mm))
        # bubble sort with swap
        i2, j2 = self.nm("i"), self.nm("j")
        s.head.append("(%s := 0; %s < |%s|; %s := %s + 1;) @ (%s := 0; %s + 1 < |%s| - %s; %s := %s + 1;) @ "
                      "(%s[%s] > %s[%s + 1]) ? %s(%s, %s, %s + 1);"
                      % (i2, i2, L, i2, i2, j2, j2, L, i2, j2, j2, L, j2, L, j2, swap, L, j2, j2))
        s.head.append(">> %s;" % L)
        lo = self.glob(s, "lo", "%d" % self.ri(50, 99))
        hi = self.glob(s, "hi", "-1")
        ce = self.glob(s, "ce", "0")
        e = self.nm("e")
        body = ["%s := %s(%s, %s);" % (lo, mn, lo, e), "%s := %s(%s, %s);" % (hi, mx, hi, e),
                "(%s(%s)) ? %s := %s + 1;" % (ev, e, ce, ce)]
        self.r.shuffle(body)
        s.head.append("(%s : %s) @ { %s }" % (e, L, " ".join(body)))
        s.head.append(">> [%s, %s, %s];" % (lo, hi, ce))
        if self.ch(0.5):
            # helper called in a counted loop with a guarded index
            i3 = self.nm("i")
            acc = self.glob(s, "ac", "0")
            s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ { (%s < |%s|) ? %s := %s + %s(%s[%s], %s); }"
                          % (i3, i3, self.ri(3, 20), i3, i3, i3, L, acc, acc, mx, L, i3, i3))
            s.head.append(">> %s;" % acc)
        if self.ch(0.3):
            # impure look-alike helper: prints / reads a changing global
            gv = self.glob(s, "gv", "%d" % self.ri(0, 5))
            pp = self.nm("p")
            imp = self.fn(s, "im", [pp], ["=> %s + %s;" % (pp, gv)])
            pp2 = self.nm("p")
            prf = self.fn(s, "pr", [pp2], [">> %s;" % pp2, "=> %s * 2;" % pp2])
            s.head.append(">> %s(1) + (%s := %s + 1) + %s(1);" % (imp, gv, gv, imp))
            s.head.append(">> %s(%d) + %s(%d);" % (prf, self.ri(0, 9), prf, self.ri(0, 9)))
        self.export(L, "ilist")  # exports
        return s

    # ================================================================== recur
    def f_recur(self):
        s = Snip("recur")
        kinds = self.r.sample(["fib", "binom", "gcd", "acc", "swapt", "impure", "fail", "cnt", "fibl", "tlist"],
                              self.ri(2, 4))
        for kd in kinds:
            if kd == "fib":
                n = self.nm("p")
                f = self.nm("fib")
                s.nglob += 1
                s.head.append("%s := (%s) => { (%s < 2) ? => %s; => %s(%s - 1) + %s(%s - 2); };"
                              % (f, n, n, n, f, n, f, n))
                arg = self.size(s, -2, 14)
                if self.ch(0.6):
                    # equal pure calls on both branches / under a condition and after it (CSE must
                    # not treat a conditional first call as unconditional)
                    cv = self.glob(s, "cv", "0")
                    a2 = self.ri(0, 12)
                    s.head.append("(%s > %d) ? %s := %s(%d) + 1; : %s := %s(%d) * 2;"
                                  % (arg, self.ri(0, 12), cv, f, a2, cv, f, a2))
                    s.head.append(">> %s; >> (%s > %d & %s(%d) > 3) + %s(%d);"
                                  % (cv, arg, self.ri(0, 12), f, a2, f, a2))
                s.head.append(">> %s(%s); >> %s(%d) * 2 + %s(%d);" % (f, arg, f, self.ri(0, 12), f, self.ri(0, 12)))
            elif kd == "fibl":
                # fib in a loop over a list (pure-call CSE across iterations)
                n = self.nm("p")
                f = self.nm("fib")
                s.nglob += 1
                s.head.append("%s := (%s) => { (%s < 2) ? => 1; => %s(%s - 1) + %s(%s - 2) + 1; };"
                              % (f, n, n, f, n, f, n))
                i = self.nm("i")
                ac = self.glob(s, "ac", "[]")
                s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ %s :+ %s(%s) - %s(%s - 1);"
                              % (i, i, self.ri(1, 12), i, i, ac, f, i, f, i))
                s.head.append(">> %s;" % ac)
            elif kd == "binom":
                n, k = self.params(2)
                f = self.nm("bin")
                s.nglob += 1
                s.head.append("%s := (%s, %s) => { (%s = 0 | %s = %s) ? => 1; => %s(%s - 1, %s - 1) + %s(%s - 1, %s); };"
                              % (f, n, k, k, k, n, f, n, k, f, n, k))
                nn = self.ri(0, 12)
                s.head.append(">> %s(%d, %d);" % (f, nn, self.ri(0, nn)))
            elif kd == "gcd":
                a, b = self.params(2)
                f = self.nm("gcd")
                s.nglob += 1
                s.head.append("%s := (%s, %s) => { (%s = 0) ? => %s; => %s(%s, %s - %s / %s * %s); };"
                              % (f, a, b, b, a, f, b, a, a, b, b))
                s.head.append(">> %s(%d, %d) + %s(%s, %d);" % (f, self.ri(0, 999), self.ri(0, 999), f,
                                                              self.size(s, 0, 99), self.ri(1, 99)))
            elif kd == "acc":
                n, ac = self.params(2)
                f = self.nm("sum")
                s.nglob += 1
                s.head.append("%s := (%s, %s) => { (%s < 1) ? => %s; => %s(%s - 1, %s + %s * %s); };"
                              % (f, n, ac, n, ac, f, n, ac, n, n))
                s.head.append(">> %s(%s, 0);" % (f, self.size(s, -3, 3000) if self.ch(0.3) else str(self.ri(0, 5000))))
            elif kd == "swapt":
                a, b, n = self.params(3)
                f = self.nm("swt")
                s.nglob += 1
                s.head.append("%s := (%s, %s, %s) => { (%s < 1) ? => %s * 100 + %s; => %s(%s, %s %s %s, %s - 1); };"
                              % (f, a, b, n, n, a, b, f, b, a, self.pick(["+", "-", "*"]), b, n)
                              if self.ch(0.6) else
                              "%s := (%s, %s, %s) => { (%s < 1) ? => %s * 100 + %s; => %s(%s, %s, %s - 1); };"
                              % (f, a, b, n, n, a, b, f, b, a, n))
                s.head.append(">> %s(%d, %d, %s);" % (f, self.ri(0, 9), self.ri(0, 9), self.size(s, -1, 30)))
            elif kd == "impure":
                gv = self.glob(s, "gv", "%d" % self.ri(0, 3))
                n = self.nm("p")
                f = self.nm("imp")
                s.nglob += 1
                s.head.append("%s := (%s) => { (%s < 2) ? => %s + %s; => %s(%s - 1) + %s(%s - 2); };"
                              % (f, n, n, n, gv, f, n, f, n))
                st, gf = self.params(2)
                setg = self.fn(s, "sg", [st], ["%s := %s;" % (gv, st), "=> 0;"])
                a = self.ri(0, 10)
                s.head.append(">> %s(%d) + %s(%d) + %s(%d);" % (f, a, setg, self.ri(0, 9), f, a))
                s.head.append(">> %s(%d) + (%s := %s + 1) + %s(%d);" % (f, a, gv, gv, f, a))
                if self.ch(0.5):
                    pr = self.nm("pr")
                    q = self.nm("p")
                    s.nglob += 1
                    s.head.append("%s := (%s) => { (%s < 1) ? { >> %s; => 0; } => %s(%s - 1) + %s(%s - 1) + 1; };"
                                  % (pr, q, q, q, pr, q, pr, q))
                    s.head.append(">> %s(%d);" % (pr, self.ri(0, 4)))
            elif kd == "fail":
                n = self.nm("p")
                f = self.nm("fl")
                s.nglob += 1
                off = self.ri(2, 4)
                s.head.append("%s := (%s) => { (%s < 1) ? => %d / (%s + %d); => %s(%s - 1) + %s(%s - 2); };"
                              % (f, n, n, self.ri(10, 99), n, off, f, n, f, n))
                arg = self.size(s, -off - 1, 10) if self.ch(0.5) else str(self.ri(-off + 1, 10))
                # arg = -off -> division by zero; base cases reach n-1 >= -1
                s.head.append(">> %s(%s) + %s(%s);" % (f, arg, f, arg))
            elif kd == "tlist":
                n, acc, t = self.params(3)
                f = self.nm("tl")
                s.nglob += 1
                if self.ch(0.5):
                    s.head.append("%s := (%s, %s) => { (%s < 1) ? => %s; %s := [%s, %s * 2]; %s :+ %s; => %s(%s - 1, %s); };"
                                  % (f, n, acc, n, acc, t, n, n, acc, t, f, n, acc))
                    tr = self.nm("tr")
                    s.head.append("%s := %s(%s, []); %s := [5, 5, 5]; >> %s; >> %s(%d, [[0]]);"
                                  % (tr, f, self.size(s, -1, 12), self.nm("jk"), tr, f, self.ri(0, 5)))
                    s.nglob += 2
                else:
                    s.head.append("%s := (%s, %s) => { (%s < 1) ? => %s; %s := [%s, %s]; => %s(%s - 1, %s + %s[0] + |%s|); };"
                                  % (f, n, acc, n, acc, t, n, n, f, n, acc, t, t))
                    s.head.append(">> %s(%s, 0);" % (f, self.size(s, -1, 40)))
            else:
                # tail-recursive counting loop through two parameters, and a list accumulator
                l, n = self.params(2)
                f = self.nm("cnt")
                s.nglob += 1
                s.head.append("%s := (%s, %s) => { (%s < 1) ? => %s; %s :+ %s; => %s(%s, %s - 1); };"
                              % (f, l, n, n, l, l, n, f, l, n))
                s.head.append(">> %s([], %s);" % (f, self.size(s, -1, 20)))
        if "fail" in kinds:
            s.risky = True
        return s

    # ================================================================= append
    def f_append(self):
        s = Snip("append")
        a = self.size(s, -2, 6)
        b = self.size(s, 0, 25)
        st = self.ri(1, 3)
        L = self.glob(s, "L", self.pick(["[]", "[%d]" % self.ri(0, 9)]))
        M = self.glob(s, "M", "[]")
        i = self.nm("i")
        cmp_ = self.pick(["<", "<=", "<"])
        body = ["%s :+ %s;" % (L, self.iexp([i], 1))]
        if self.ch(0.5):
            body.append("%s :+ %s;" % (M, self.iexp([i], 1)))
        if self.ch(0.3):
            body.append("%s :+ -%s;" % (L, i))
        if self.ch(0.2):
            body.append("(%s = %s) ? %s :+ %d;" % (i, a, M, self.ri(0, 9)))
        s.head.append("(%s := %s; %s %s %s; %s := %s + %d;) @ { %s }" % (i, a, i, cmp_, b, i, i, st, " ".join(body)))
        s.head.append(">> %s; >> |%s|;" % (L, M))
        # in a function, bounds from parameters
        pn, pl, pi = self.nm("p"), self.nm("l"), self.nm("i")
        f = self.fn(s, "ga", [pn], ["%s := [];" % pl, "(%s := 0; %s < %s; %s := %s + 1;) @ { %s :+ %s * %s; %s }"
                                     % (pi, pi, pn, pi, pi, pl, pi, pi,
                                        "(%s = %d) ? => %s;" % (pi, self.ri(0, 20), pl) if self.ch(0.3) else ""),
                                     "=> %s;" % pl])
        s.head.append(">> %s(%s); >> |%s(%d)|;" % (f, self.size(s, -3, 20), f, self.ri(0, 40)))
        if self.ch(0.5):
            # merge-shaped while loops
            pa, pb, mi, mj, mo = self.params(5)
            mg = self.fn(s, "mg", [pa, pb], [
                "%s := 0; %s := 0; %s := [];" % (mi, mj, mo),
                "(%s < |%s| & %s < |%s|) @ { (%s[%s] < %s[%s]) ? { %s :+ %s[%s]; %s := %s + 1; } : "
                "{ %s :+ %s[%s]; %s := %s + 1; } }" % (mi, pa, mj, pb, pa, mi, pb, mj, mo, pa, mi, mi, mi,
                                                      mo, pb, mj, mj, mj),
                "(%s < |%s|) @ { %s :+ %s[%s]; %s := %s + 1; }" % (mi, pa, mo, pa, mi, mi, mi),
                "(%s < |%s|) @ { %s :+ %s[%s]; %s := %s + 1; }" % (mj, pb, mo, pb, mj, mj, mj),
                "=> %s;" % mo])
            s.head.append(">> %s(%s, %s);" % (mg, "[1, 4, 9]" if self.ch(0.5) else L, self.pick(["[]", M, L, "[2, 3]"])))
            s.head.append(">> %s(%s, %s);" % (mg, M, M))
        if self.ch(0.4):
            # for-each that appends to another list, several times per iteration
            x = self.nm("x")
            N = self.glob(s, "N", "[]")
            s.head.append("(%s : %s) @ { %s :+ %s; %s :+ [%s]; }" % (x, L, N, x, N, x))
            s.head.append(">> %s;" % N)
        if self.ch(0.3):
            # while-loop with counter used afterwards
            j = self.glob(s, "j", "0")
            s.head.append("(%s < %d) @ { %s :+ %s; %s := %s + 1; }" % (j, self.ri(0, 12), M, j, j, j))
            s.head.append(">> %s; >> %s;" % (j, M))
        self.export(L, "ilist"); self.export(M, "ilist")  # exports
        return s

    def f_copy(self):
        """Copy loops L :+ M[i] with various ranges, aliasing and loop-variable use.
        Lengths are tracked statically so that a bound re-evaluated from a list that the
        loop appends to (L aliasing M) is never used: such loops use literal bounds."""
        s = Snip("copy")
        mlen = self.ri(0, 10)
        M = self.glob(s, "M", "[%s]" % ", ".join(str(self.ri(0, 99)) for _ in range(mlen)))
        alias = self.ch(0.3)
        if alias:
            L = self.glob(s, "L", M)
            llen = None
        else:
            llen = self.ri(0, 1)
            L = self.glob(s, "L", "[]" if llen == 0 else "[%d]" % self.ri(0, 9))
        lens = {M: mlen}

        def ln(x):
            return lens[M] if (x == M or alias) else llen

        for _ in range(self.ri(1, 3)):
            mode = self.pick(["in", "in", "empty", "rev", "oob", "while", "after", "fn", "inp"])
            src = self.pick([M, M, L])
            n = ln(src)
            same = alias or src == L
            lo = self.ri(0, n)
            hi = self.ri(lo, n)
            los, his = str(lo), str(hi)
            if not same and self.ch(0.4):
                k = n - hi
                his = "|%s|" % src if k == 0 else "|%s| - %d" % (src, k)
            cnt = hi - lo
            if mode == "empty":
                los, his = str(self.ri(3, 9)), str(self.ri(-2, 3))
                cnt = max(0, int(his) - int(los))
            elif mode == "oob":
                s.risky = True
                if self.ch(0.5):
                    los = self.size(s, -2, 2)
                    cnt = 0
                else:
                    his = str(n + self.ri(1, 3))
                    cnt = 0
            elif mode == "inp":
                his = self.size(s, 0, n)
                if los != "0" and int(los) > 0:
                    los = "0"
                cnt = 0
            if mode == "rev":
                i = self.nm("i")
                s.head.append("(%s := %d; %s >= %d; %s := %s - 1;) @ %s :+ %s[%s];" % (i, n - 1, i, lo, i, i, L, src, i))
                cnt = max(0, n - lo)
            elif mode == "while":
                j = self.glob(s, "j", los)
                s.head.append("(%s < %s) @ { %s :+ %s[%s]; %s := %s + 1; }" % (j, his, L, src, j, j, j))
                s.head.append(">> %s;" % j)
            elif mode == "after":
                q = self.glob(s, "q", "0")
                s.head.append("(%s := %s; %s < %s; %s := %s + 1;) @ { %s :+ %s[%s]; }" % (q, los, q, his, q, q, L, src, q))
                s.head.append(">> %s;" % q)
            elif mode == "fn":
                pd, ps, pa, pb, pi = self.params(5)
                f = self.fn(s, "cp", [pd, ps, pa, pb], ["(%s := %s; %s < %s; %s := %s + 1;) @ %s :+ %s[%s];"
                                                        % (pi, pa, pi, pb, pi, pi, pd, ps, pi), "=> %s;" % pd])
                d = self.pick(["[]", L])
                s.head.append(">> %s(%s, %s, %s, %s);" % (f, d, src, los, his))
                if d == "[]":
                    cnt = 0
            else:
                i = self.nm("i")
                s.head.append("(%s := %s; %s < %s; %s := %s + 1;) @ %s :+ %s[%s];" % (i, los, i, his, i, i, L, src, i))
            # length bookkeeping (unknown counts make later bounds use the smaller known value)
            if alias:
                lens[M] += cnt
            else:
                llen += cnt
            s.head.append(">> %s;" % L)
        s.head.append(">> %s; >> |%s| + |%s|;" % (M, L, M))
        self.export(M, "ilist")  # exports
        return s

    # =========================================================== indexed loops
    def f_idxloop(self):
        s = Snip("idxloop")
        n = self.ri(1, 12)
        L = self.glob(s, "L", "[%s]" % ", ".join(str(self.ri(0, 30)) for _ in range(n)))
        A = self.glob(s, "A", L)
        sm = self.glob(s, "sm", "0")
        pl = self.nm("p")
        if self.ch(0.5):
            grow = self.fn(s, "gr", [pl], ["(|%s| < %d) ? %s :+ %d;" % (pl, self.ri(n, n + 10), pl, self.ri(0, 9)),
                                          "=> 0;"])
        else:
            # recursive, so never inlined: the loop sees a CallDirect that mutates the list
            grow = self.nm("gr")
            gd = self.nm("p")
            s.nglob += 1
            s.head.append("%s := (%s) => { (|%s| < %d) ? { %s :+ %d; %s(%s); } => 0; };"
                          % (grow, pl, pl, self.ri(n, n + 6), pl, self.ri(0, 9), grow, pl))
        pl2 = self.nm("p")
        if self.ch(0.5):
            shr = self.fn(s, "sh", [pl2], ["(|%s| > %d) ? => -%s;" % (pl2, self.ri(0, 3), pl2), "=> 0;"])
        else:
            shr = self.nm("sh")
            sd = self.nm("p")
            s.nglob += 1
            s.head.append("%s := (%s, %s) => { (%s > 0) ? => %s(%s, %s - 1); (|%s| > %d) ? => -%s; => 0; };"
                          % (shr, pl2, sd, sd, shr, pl2, sd, pl2, self.ri(0, 3), pl2))
            shr = "%s_D_" % shr
        i = self.nm("i")
        muts = [
            "(%s = %d) ? %s :+ %d;" % (i, self.ri(0, 5), A, self.ri(0, 9)),
            "(%s = %d) ? %s(%s);" % (i, self.ri(0, 5), grow, A),
            "(%s = %d) ? %s := %s + %s;" % (i, self.ri(0, 5), sm, sm,
                                            "%s(%s, %d)" % (shr[:-3], A, self.ri(0, 2)) if shr.endswith("_D_")
                                            else "%s(%s)" % (shr, A)),
            "(%s = %d & |%s| > 1) ? -%s;" % (i, self.ri(0, 5), A, A),
            "(%s = %d) ? %s := [%d, %d, %d];" % (i, self.ri(0, 5), L, self.ri(0, 9), self.ri(0, 9), self.ri(0, 9)),
            "%s(%s);" % (grow, L),
            "(%s + 1 < |%s| & %s < |%s|) ? %s[%s + 1] := %s[%s] + 1;" % (i, A, i, L, A, i, L, i),
        ]
        body = self.r.sample(muts, self.ri(0, 2))
        self.r.shuffle(body)
        body.insert(0, "%s := %s + %s[%s];" % (sm, sm, L, i))
        bound = self.pick(["|%s|" % L, "|%s|" % A])
        s.head.append("(%s := 0; %s < %s & %s < 40; %s := %s + 1;) @ { (%s < |%s|) ? { %s } }"
                      % (i, i, bound, i, i, i, i, L, " ".join(body)))
        s.head.append(">> %s; >> %s; >> %s;" % (sm, L, A))
        # a second indexed loop with cached header candidates (reads only + stores in place)
        j = self.nm("i")
        s.head.append("(%s := 0; %s < |%s|; %s := %s + 1;) @ { %s[%s] := %s[%s] * 2 - %s; }"
                      % (j, j, A, j, j, A, j, A, j, j))
        s.head.append(">> %s;" % A)
        if self.ch(0.35):
            # error-order: index / stored value with side effects
            s.risky = True
            cnt = self.glob(s, "cn", "0")
            inc = self.fn(s, "inc", [], ["%s := %s + 1;" % (cnt, cnt), "=> %s;" % cnt])
            pp = self.nm("p")
            pop = self.fn(s, "pp", [pp], ["=> -%s;" % pp])
            k = self.nm("i")
            form = self.pick([
                "%s := %s + %s[%s()];" % (sm, sm, A, inc),
                "%s[%s] := %s(%s);" % (A, k, pop, A),
                "%s[%s()] := %s[%s];" % (A, inc, A, k),
                "%s := %s + %s[%s] / (%s - %s);" % (sm, sm, A, k, k, self.ri(0, 4)),
            ])
            s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ { %s }" % (k, k, self.ri(1, 14), k, k, form))
            s.head.append(">> %s; >> %s;" % (A, sm))
            if self.ch(0.5):
                # a guarded index incremented twice per iteration: the second read is not
                # covered by the loop guard (fails at the last iteration on an odd count)
                w = self.glob(s, "w", "0")
                s.head.append("(%s < |%s|) @ { %s := %s + %s[%s]; %s := %s + 1; %s := %s + %s[%s]; %s := %s + 1; }"
                              % (w, A, sm, sm, A, w, w, w, sm, sm, A, w, w, w))
                s.head.append(">> %s;" % sm)
        self.export(A, "ilist"); self.export(L, "ilist")  # exports
        return s

    # ================================================================= ifconv
    def f_ifconv(self):
        s = Snip("ifconv")
        n = self.ri(0, 14)
        L = self.glob(s, "L", "[%s]" % ", ".join(str(self.ri(-20, 40)) for _ in range(n)))
        mn = self.glob(s, "mn", "%d" % self.ri(0, 50))
        mx = self.glob(s, "mx", "-1")
        mi = self.glob(s, "mi", "-1")
        c = self.glob(s, "c", "0")
        q = self.glob(s, "q", "0")
        i = self.nm("i")
        e = "%s[%s]" % (L, i)
        opts = [
            "(%s < %s) ? %s := %s;" % (e, mn, mn, e),
            "(%s > %s) ? { %s := %s; %s := %s; }" % (e, mx, mx, e, mi, i),
            "(%s - %s / 2 * 2 = 0) ? %s := %s + 1;" % (e, e, c, c),
            "(%s != 0) ? %s := %s + 100 / %s;" % (e, q, q, e),
            "(%s + 1 < |%s|) ? %s := %s + %s[%s + 1];" % (i, L, q, q, L, i),
            "(%s > %s) ? %s := %s - 1; : %s := %s + 2;" % (e, self.ri(0, 20), c, c, c, c),
            "(%s & %s > 3) ? %s := %s * 2;" % (i, e, q, q),
            "(%s | %s = 7) ? %s := %s + %s;" % (e, i, c, c, i),
            "(%s < %s - 1 & %s[%s + 1] < %s) ? %s := %s + 1;" % (i, "|%s|" % L, L, i, e, c, c),
        ]
        body = self.r.sample(opts, self.ri(2, 5))
        s.head.append("(%s := 0; %s < |%s|; %s := %s + 1;) @ { %s }" % (i, i, L, i, i, " ".join(body)))
        s.head.append(">> [%s, %s, %s, %s, %s];" % (mn, mx, mi, c, q))
        if self.ch(0.7):
            # established-bounds near misses: the guarded body indexes something the condition
            # did (or did not quite) establish; inlined swaps; a second list / alias; while
            # loops whose index changes after the guarded update
            M = self.glob(s, "M", self.pick([L, "[%s]" % ", ".join(str(self.ri(0, 30)) for _ in range(self.ri(0, 6)))]))
            sw = None
            if self.ch(0.5):
                pl, pa, pb, pt = self.params(4)
                sw = self.fn(s, "sw", [pl, pa, pb], ["%s := %s[%s];" % (pt, pl, pa), "%s[%s] := %s[%s];" % (pl, pa, pl, pb),
                                                    "%s[%s] := %s;" % (pl, pb, pt), "=> 0;"])
            j = self.nm("i")
            ej, ej1 = "%s[%s]" % (L, j), "%s[%s + 1]" % (L, j)
            t = self.nm("t")
            p = self.nm("p")
            opts = [
                "(%s > %s) ? { %s := %s; %s[%s] := %s; %s[%s + 1] := %s; }" % (ej, ej1, t, ej, L, j, ej1, L, j, t),
                "(%s > %s) ? %s := %s;" % (ej, ej1, mn, ej1),
                "(%s > %d) ? %s := %s + %s;" % (ej, self.ri(0, 20), q, q, "%s[1 + %s]" % (L, j)),
                "(%s < |%s| & %s < %d) ? %s := %s + %s[%s];" % (j, M, ej, self.ri(0, 30), c, c, M, j),
                "(%s < |%s|) ? { (%s[%s] > %s) ? %s := %s[%s]; }" % (j, M, M, j, ej, mx, M, j),
                "{ %s := %s + 1; (%s < %s[%s]) ? %s := %s[%s] - %s; }" % (p, j, ej, L, p, q, L, p, ej),
                "(%s > 0 & %s[%s - 1] < %s) ? %s := %s[%s - 1];" % (j, L, j, ej, mn, L, j),
            ]
            if sw:
                opts.append("(%s > %s) ? %s(%s, %s, %s + 1);" % (ej, ej1, sw, L, j, j))
            body = self.r.sample(opts, self.ri(1, 4))
            s.head.append("(%s := 0; %s + 1 < |%s|; %s := %s + 1;) @ { %s }" % (j, j, L, j, j, " ".join(body)))
            s.head.append(">> %s; >> [%s, %s, %s, %s];" % (L, mn, mx, c, q))
            if self.ch(0.5):
                w = self.glob(s, "w", "0")
                s.head.append("(%s + 1 < |%s|) @ { (%s[%s] < %s[%s + 1]) ? %s := %s + %s[%s + 1]; %s := %s + %d; }"
                              % (w, L, L, w, L, w, c, c, L, w, w, w, self.ri(1, 2)))
                s.head.append(">> %s;" % c)
        # heap child selection with guarded index inside a function
        h, k, ch, go, t = self.params(5)
        sift = self.fn(s, "sd", [h, k], [
            "%s := 1;" % go,
            "(%s) @ { %s := 2 * %s + 1; (%s + 1 < |%s| & %s[%s + 1] %s %s[%s]) ? %s := %s + 1; "
            "(%s < |%s| & %s[%s] %s %s[%s]) ? { %s := %s[%s]; %s[%s] := %s[%s]; %s[%s] := %s; %s := %s; } : %s := 0; }"
            % (go, ch, k, ch, h, h, ch, "<", h, ch, ch, ch, ch, h, h, ch, "<", h, k, t, h, ch, h, ch, h, k,
               h, k, t, k, ch, go),
            "=> %s;" % h])
        H = self.glob(s, "H", "[%s]" % ", ".join(str(self.ri(0, 30)) for _ in range(self.ri(0, 9))))
        j = self.nm("i")
        s.head.append("(%s := |%s| / 2; %s >= 0; %s := %s - 1;) @ %s(%s, %s);" % (j, H, j, j, j, sift, H, j))
        s.head.append(">> %s;" % H)
        if self.ch(0.5):
            # select in a function returning min/abs-like values
            a, b, r = self.params(3)
            f = self.fn(s, "sel", [a, b], ["%s := %s;" % (r, a), "(%s < %s) ? %s := %s;" % (b, a, r, b),
                                          "(%s < 0) ? %s := -%s;" % (r, r, r), "=> %s;" % r])
            k2 = self.nm("i")
            ac = self.glob(s, "ac", "0")
            s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ %s := %s + %s(%s - %d, %d - %s);"
                          % (k2, k2, self.ri(1, 15), k2, k2, ac, ac, f, k2, self.ri(0, 9), self.ri(0, 9), k2))
            s.head.append(">> %s;" % ac)
        self.export(L, "ilist"); self.export(H, "ilist")  # exports
        return s

    # ================================================================= litcmp
    def f_litcmp(self):
        s = Snip("litcmp")
        kv = self.read(s, 0, 6, "kv")
        idf = self.nm("id")
        pp = self.nm("p")
        s.nglob += 1
        s.head.append("%s := (%s) => { => %s; };" % (idf, pp, pp))
        vals = [
            ("int", "%d" % self.ri(0, 3)),
            ("list", "[]"),
            ("list", "[%d]" % self.ri(0, 2)),
            ("list", "[%d, %d]" % (self.ri(0, 2), self.ri(0, 2))),
            ("nlist", "[[], %d]" % self.ri(0, 2)),
            ("nlist", "[[%d]]" % self.ri(0, 2)),
        ]
        names = []
        for kind, init in self.r.sample(vals, self.ri(2, 5)):
            v = self.glob(s, "v", init)
            names.append((v, kind))
        # a dynamic value: int or list depending on the input
        dv = self.glob(s, "dv", "[%d]" % self.ri(0, 2))
        s.head.append("(%s > %d) ? %s := %d;" % (kv, self.ri(0, 5), dv, self.ri(0, 2)))
        s.head.append("(%s = %d) ? %s := [];" % (kv, self.ri(0, 6), dv))
        names.append((dv, "dyn"))
        fv = self.glob(s, "fv", idf)
        names.append((fv, "fn"))
        lits_eq = ["[]", "[0]", "[1]", "[0, 1]", "[1, 0]", "0", "1", "[[]]", "[[], 1]"]
        lits_lt = ["[]", "[0]", "[1]", "[0, 1]", "[2]"]
        out = []
        for v, kind in names:
            for _ in range(self.ri(1, 3)):
                op = self.pick(["=", "!=", "=", "!="])
                l = self.pick(lits_eq)
                out.append("%s %s %s" % (v, op, l) if self.ch(0.6) else "%s %s %s" % (l, op, v))
            if kind in ("list", "nlist") and self.ch(0.7):
                l = self.pick(lits_lt) if kind == "list" else "[]"
                out.append(self.pick(["%s < %s" % (v, l), "%s < %s" % (l, v), "%s >= %s" % (v, l), "%s > %s" % (l, v)]))
        self.r.shuffle(out)
        s.head.append(">> [%s];" % ", ".join(out[:12]))
        # in a loop over a heterogeneous list and through a function parameter
        hl = self.glob(s, "hl", "[%s]" % ", ".join(n for n, k in names if k != "fn"))
        x = self.nm("x")
        c = self.glob(s, "c", "0")
        s.head.append("(%s : %s) @ { (%s = []) ? %s := %s + 1; (%s != [%d]) ? %s := %s + 10; }"
                      % (x, hl, x, c, c, x, self.ri(0, 2), c, c))
        s.head.append(">> %s;" % c)
        pq = self.nm("p")
        isn = self.fn(s, "isn", [pq], ["=> %s = [];" % pq])
        if self.ch(0.5):
            # a literal on the left with non-constant elements, compared with an impure right side
            # that changes those elements' variables
            cv = self.glob(s, "cv", str(self.ri(0, 2)))
            stp = self.fn(s, "st", [], ["%s := %s + 1;" % (cv, cv), "=> [%s, %s];" % (cv, cv)])
            s.head.append(">> [%s, %s] = %s(); >> [%s] != %s(); >> [%d, %s] = %s();"
                          % (cv, cv, stp, cv, stp, self.ri(1, 4), cv, stp))
        s.head.append(">> [%s];" % ", ".join("%s(%s)" % (isn, n) for n, k in names))
        if self.ch(0.3):
            s.risky = True
            v, k = self.pick(names)
            s.head.append(">> %s < [%d];" % (v, self.ri(0, 2)))
        elif self.ch(0.25):
            # literal elements that may fail (only reached when the lengths match, or always?)
            s.risky = True
            kv2 = self.read(s, -1, 3, "kv")
            v, k = self.pick([nk for nk in names if nk[1] != "fn"] or names)
            late = self.nm("Lg")
            pq2 = self.nm("p")
            lf = self.fn(s, "lf", [pq2], ["=> %s = [%s, %s];" % (pq2, late, pq2)])
            el = self.pick(["%d / %s" % (self.ri(1, 9), kv2), "[1, 2][%s]" % kv2, "%s[%s]" % (v, kv2)
                            if k == "list" else "%d / %s" % (self.ri(1, 9), kv2)])
            s.head.append(">> %s = [%s]; >> [%s, %d] != %s;" % (v, el, el, self.ri(0, 2), v))
            s.head.append(">> %s(%s);" % (lf, v))
            s.head.append("%s := %d;" % (late, self.ri(0, 3)))
            s.nglob += 1
        return s

    # ================================================================ reclaim
    def f_reclaim(self):
        s = Snip("reclaim")
        keep = self.glob(s, "kp", "[]")
        idf = self.nm("id")
        pp = self.nm("p")
        s.nglob += 1
        s.head.append("%s := (%s) => { => %s; };" % (idf, pp, pp))
        pa, pb, pc = self.params(3)
        pick = self.fn(s, "pk", [pa, pb, pc], ["(%s) ? => %s;" % (pc, pa), "=> %s;" % pb])
        fns = []
        for _ in range(self.ri(2, 4)):
            n, t, u, i, w, dp, w2 = self.params(7)
            f = self.nm("rf")
            body = ["%s := [%s, %s + 1, %s * 2];" % (t, n, n, n), "%s := [];" % u,
                    "(%s := 0; %s < %d; %s := %s + 1;) @ %s :+ %s[%s] * 2;" % (i, i, 3, i, i, u, t, i)]
            if self.ch(0.7):
                # make the function big enough not to be inlined (> 40 TIR nodes)
                v, j, sm = self.params(3)
                body += ["%s := [];" % v, "%s := 0;" % sm,
                         "(%s := 0; %s < %s + %d; %s := %s + 1;) @ { %s :+ [%s, %s * %s]; %s := %s + %s[%s - %s / 3 * 3]; }"
                         % (j, j, "|%s|" % u, self.ri(0, 4), j, j, v, j, j, j, sm, sm, t, j, j),
                         "(%s > %d) ? %s[0] := %s + %s[|%s| - 1][1];" % (sm, self.ri(0, 30), t, sm, v, v)
                         if self.ch(0.5) else "%s := %s + |%s|;" % (sm, sm, v)]
            ret = self.pick(["int", "int", "t", "u", "alias", "store", "storeel", "idret", "pick", "nested",
                             "aliasx", "aliasx", "elemret", "setelret"])
            # self-recursion through a depth parameter: never inlined, so reclaim sees it
            if ret in ("int", "store", "storeel"):
                body.append("(%s > 0) ? %s[0] := %s[0] + %s(%s + 1, %s - 1);" % (dp, u, u, f, n, dp))
            else:
                body.append("(%s > 0) ? { %s := %s(%s + 1, %s - 1); %s := %s + |%s|; }" % (dp, w2, f, n, dp, n, n, w2))
            if ret == "int":
                body.append("=> %s[0] + %s[2] + |%s|;" % (u, u, t))
            elif ret == "t":
                body.append("=> %s;" % t)
            elif ret == "u":
                body.append("=> %s;" % u)
            elif ret == "alias":
                body += ["%s := %s;" % (w, t), "=> %s;" % w]
            elif ret == "store":
                body += ["%s :+ %s;" % (keep, u), "=> |%s|;" % u]
            elif ret == "storeel":
                body += ["%s := [%s, 1];" % (w, t), "%s :+ %s;" % (keep, w), "=> 0;"]
            elif ret == "idret":
                body.append("=> %s(%s);" % (idf, self.pick([t, u])))
            elif ret == "elemret":
                body += ["%s := [%s];" % (w, n), "%s :+ %s;" % (w, t), "=> %s;" % w]
            elif ret == "setelret":
                if self.ch(0.5):
                    body += ["%s := [0, 0];" % w, "%s[%d] := %s;" % (w, self.ri(0, 1), u), "=> %s;" % w]
                else:
                    # indexed by a variable: a plain list, not a record (SetIndex, not RecSet)
                    body += ["%s := [[], [], []];" % w, "%s[%s - %s / 3 * 3] := %s;" % (w, dp, dp, self.pick([t, u])),
                             "=> %s;" % w]
            elif ret == "aliasx":
                body.append("=> %s;" % self.alias_expr(self.pick([t, u]), (idf, "[%s]" % t)))
            elif ret == "pick":
                body.append("=> %s(%s, %s, %s - %s / 2 * 2);" % (pick, t, u, n, n))
            else:
                body += ["%s := [%s, [%s]];" % (w, t, n), "=> %s[%d];" % (w, self.ri(0, 1))]
            s.nglob += 1
            s.head.append("%s := (%s, %s) => { %s };" % (f, n, dp, " ".join(body)))
            fns.append((f, ret))
        if self.ch(0.5):
            pl, pn, pt, pd, pw = self.params(5)
            ps = self.nm("pst")
            how = self.pick(["%s :+ %s;" % (pl, pt), "(|%s| > 0) ? %s[0] := %s;" % (pl, pl, pt),
                             "%s :+ [%s];" % (pl, pt)])
            s.nglob += 1
            s.head.append("%s := (%s, %s, %s) => { %s := [%s, %s * 3]; %s (%s > 0) ? %s := %s(%s, %s + 1, %s - 1); "
                          "=> |%s|; };" % (ps, pl, pn, pd, pt, pn, pn, how, pd, pw, ps, pl, pn, pd, pt))
            G = self.glob(s, "pg", "[[%d]]" % self.ri(0, 9))
            s.head.append(">> %s(%s, %d, %d); %s := [77, 77]; >> %s;" % (ps, G, self.ri(0, 9), self.ri(0, 3),
                                                                         self.nm("jk"), G))
            s.nglob += 1
        # a driver function calling the helpers (frees happen in non-main functions)
        dn, di, ds, dt = self.params(4)
        dbody = ["%s := 0;" % ds, "%s := [%s, %s];" % (dt, dn, dn)]
        lp = []
        for f, ret in fns:
            if ret in ("int", "store", "storeel"):
                lp.append("%s := %s + %s(%s, %d);" % (ds, ds, f, di, self.ri(0, 2)))
            else:
                lp.append("%s := %s + |%s(%s, %d)|;" % (ds, ds, f, di, self.ri(0, 2)))
        dbody.append("(%s := 0; %s < %s; %s := %s + 1;) @ { %s }" % (di, di, dn, di, di, " ".join(lp)))
        dbody.append("(%s > 0) ? %s := %s + DRV(%s - 1);" % (dn, ds, ds, dn))
        dbody.append("=> %s + %s[1];" % (ds, dt))
        drv = self.fn(s, "drv", [dn], dbody)
        s.head[-1] = s.head[-1].replace("DRV", drv)
        s.head.append(">> %s(%d); >> %s(%s);" % (drv, self.ri(0, 5), drv, self.size(s, 0, 6)))
        res = []
        for f, ret in fns:
            r = self.glob(s, "rr", "%s(%d, %d)" % (f, self.ri(0, 9), self.ri(0, 3)))
            junk = self.glob(s, "jk", "[%d, %d]" % (self.ri(0, 99), self.ri(0, 99)))
            res.append(r)
            i = self.nm("i")
            acc = self.glob(s, "ac", "0")
            s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ { %s := %s + |[%s]|; %s := [%s, %s]; }"
                          % (i, i, self.ri(1, 8), i, i, acc, acc, i, junk, i, i) if ret == "int" else
                          "(%s := 0; %s < %d; %s := %s + 1;) @ %s := [%s(%s, %d)];" % (i, i, self.ri(1, 6), i, i, junk, f, i,
                                                                                self.ri(0, 2)))
            s.head.append(">> %s; >> %s;" % (r, junk))
        s.head.append(">> [%s]; >> %s;" % (", ".join(res), keep))
        if self.ch(0.5):
            # a list that is returned through two levels and then mutated
            g = self.fn(s, "rg", [], ["=> %s(%d, %d);" % (fns[0][0], self.ri(0, 9), self.ri(0, 3))])
            r2 = self.glob(s, "rr", "%s()" % g)
            self.glob(s, "z", "[9, 9, 9]")
            s.head.append(">> %s;" % r2)
            s.tail.append(">> %s; >> %s;" % (r2, keep))
        return s

    # ================================================================== glob
    def f_glob(self):
        """Constant / late-defined / main-only globals, perturbed far away."""
        s = Snip("glob")
        K = self.glob(s, "K", str(self.ri(1, 9)))
        x = self.nm("p")
        late = self.nm("Lg")
        f = self.fn(s, "gf", [x], ["=> %s * %s + %s;" % (x, K, late)])
        if self.ch(0.15):
            # read before definition: Undefined global
            s.risky = True
            s.head.append(">> %s(%d);" % (f, self.ri(0, 9)))
        s.nglob += 1
        s.head.append("%s := %d;" % (late, self.ri(0, 9)))
        s.head.append(">> %s(%d) + %s(%s);" % (f, self.ri(0, 9), f, K))
        i = self.nm("i")
        acc = self.glob(s, "ac", "0")
        s.head.append("(%s := 0; %s < %s; %s := %s + 1;) @ %s := %s + %s(%s);" % (i, i, K, i, i, acc, acc, f, i))
        s.head.append(">> %s;" % acc)
        # main-only global (demotable), list-valued and int-valued
        mv = self.glob(s, "mv", "[%s]" % ", ".join(str(self.ri(0, 9)) for _ in range(self.ri(0, 6))))
        mc = self.glob(s, "mc", "0")
        j = self.nm("i")
        s.head.append("(%s := 0; %s < |%s|; %s := %s + 1;) @ { %s := %s + %s[%s] * %s; %s[%s] := %s; }"
                      % (j, j, mv, j, j, mc, mc, mv, j, K, mv, j, mc))
        s.head.append(">> %s; >> %s;" % (mv, mc))
        if self.ch(0.5):
            v = self.pick(["setK", "typeK", "readmv", "setlate", "kself"])
            if v == "setK":
                g = self.fn(s, "sk", [], ["%s := %s + 1;" % (K, K), "=> 0;"])
                s.tail.append("%s(); >> %s(%d);" % (g, f, self.ri(0, 9)))
            elif v == "typeK":
                s.tail.append("%s := [%s]; >> %s;" % (K, K, K))
            elif v == "readmv":
                g = self.fn(s, "rm", [], ["=> |%s| + %s;" % (mv, mc)])
                s.tail.append(">> %s();" % g)
            elif v == "setlate":
                s.tail.append("%s := %s + 1; >> %s(%d);" % (late, late, f, self.ri(0, 9)))
            else:
                s.tail.append("%s := %s; >> %s(1);" % (K, K, f))
        self.export(mv, "ilist")  # exports
        return s

    # =============================================================== recloop
    def f_recloop(self):
        """Lists stored in record fields: indexed loops, appends and copies through the field."""
        s = Snip("recloop")
        n = self.ri(0, 8)
        M = self.glob(s, "M", "[%s]" % ", ".join(str(self.ri(0, 50)) for _ in range(n)))
        R = self.glob(s, "R", "[%d, %s]" % (self.ri(0, 9), M if self.ch(0.6) else "[%d]" % self.ri(0, 9)))
        R2 = self.glob(s, "R", "[%s, []]" % R if self.ch(0.5) else "[%d, []]" % self.ri(0, 9))
        i = self.nm("i")
        sm = self.glob(s, "sm", "0")
        body = ["%s := %s + %s[1][%s];" % (sm, sm, R, i)]
        extra = self.pick(["", "app", "pop", "field", "callee"])
        if extra == "app":
            body.append("(%s = %d & |%s| < 20) ? %s[1] :+ %s;" % (i, self.ri(0, 3), M, R, i))
        elif extra == "pop":
            body.append("(%s = %d & |%s[1]| > 1) ? %s := %s + -%s[1];" % (i, self.ri(0, 3), R, sm, sm, R))
        elif extra == "field":
            body.append("(%s = %d) ? %s[1] := [%d, %d, %d];" % (i, self.ri(0, 3), R, self.ri(0, 9), self.ri(0, 9),
                                                               self.ri(0, 9)))
        elif extra == "callee":
            pr = self.nm("p")
            g = self.fn(s, "ga", [pr], ["(|%s[1]| < 12) ? %s[1] :+ %s[0];" % (pr, pr, pr), "=> 0;"])
            body.append("(%s = %d) ? %s(%s);" % (i, self.ri(0, 3), g, R))
        s.head.append("(%s := 0; %s < |%s[1]|; %s := %s + 1;) @ { (%s < |%s[1]|) ? { %s } }"
                      % (i, i, R, i, i, i, R, " ".join(body)))
        s.head.append(">> %s; >> %s; >> %s;" % (sm, R, M))
        # copy loops from / into a field
        j = self.nm("i")
        D = self.glob(s, "D", "[]")
        s.head.append("(%s := 0; %s < |%s[1]|; %s := %s + 1;) @ %s :+ %s[1][%s];" % (j, j, R, j, j, D, R, j))
        if self.ch(0.5):
            k = self.nm("i")
            lim = self.ri(0, 6)
            s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ (%s < |%s|) ? %s[1] :+ %s[%s];"
                          % (k, k, lim, k, k, k, D, R2, D, k))
            s.head.append(">> %s;" % R2)
        s.head.append(">> %s;" % D)
        # record holding a list, returned by a function, then mutated through both
        pa, pl = self.params(2)
        mk = self.fn(s, "mkr", [pa, pl], ["=> [%s, %s];" % (pa, pl)])
        Q = self.glob(s, "Q", "%s(%d, %s)" % (mk, self.ri(0, 9), D))
        s.head.append("%s[1] :+ %d; >> %s; >> %s[1] = %s; >> %s[0] + (%s[0] := %d) + %s[0];"
                      % (Q, self.ri(0, 9), D, Q, D, Q, Q, self.ri(0, 9), Q))
        if self.ch(0.4):
            v = self.pick(["idx", "app", "each"])
            kv = self.read(s, 0, 1, "kv")
            if v == "idx":
                s.tail.append(">> %s[%s];" % (Q, kv))
            elif v == "app":
                s.tail.append("%s :+ %s; >> |%s|;" % (Q, kv, Q))
            else:
                x = self.nm("x")
                s.tail.append("(%s : %s) @ >> %s;" % (x, R, x))
        self.export(R, "rec"); self.export(D, "ilist")  # exports
        return s

    # ================================================================== inl2
    def f_inl2(self):
        """Inlining: callees that assign their parameters, argument evaluation order,
        calls with side effects in arguments, callees mutating lists passed in."""
        s = Snip("inl2")
        a, b = self.params(2)
        f1 = self.fn(s, "pa", [a, b], ["%s := %s + %s;" % (a, a, b), "%s := %s * 2;" % (b, a), "=> %s - %s;" % (a, b)])
        G = self.glob(s, "G", str(self.ri(0, 9)))
        H = self.glob(s, "H", str(self.ri(0, 9)))
        s.head.append(">> %s(%s, %s); >> [%s, %s];" % (f1, G, H, G, H))
        s.head.append(">> %s(%s, %s := %s + 1) + %s;" % (f1, G, G, G, G))
        s.head.append(">> %s(%s := %d, %s) + %s(%s, %s := %d);" % (f1, H, self.ri(0, 9), H, f1, H, H, self.ri(0, 9)))
        c = self.glob(s, "cn", "0")
        tick = self.fn(s, "tk", [], ["%s := %s + 1;" % (c, c), "=> %s;" % c])
        s.head.append(">> %s(%s(), %s()) * 10 + %s;" % (f1, tick, tick, c))
        l, v = self.params(2)
        push = self.fn(s, "ps", [l, v], ["%s :+ %s;" % (l, v), "%s := [];" % l, "=> |%s|;" % l])
        L = self.glob(s, "L", "[]")
        i = self.nm("i")
        s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ %s := %s + %s(%s, %s);"
                      % (i, i, self.ri(1, 8), i, i, G, G, push, L, i))
        s.head.append(">> %s; >> %s;" % (L, G))
        # nested calls of inlinable helpers inside conditions and indexes
        x, y = self.params(2)
        mn = self.fn(s, "mn", [x, y], ["(%s < %s) ? => %s;" % (x, y, x), "=> %s;" % y])
        j = self.nm("i")
        s.head.append("(%s := 0; %s < |%s|; %s := %s + 1;) @ (%s[%s(%s, |%s| - 1)] > %s(%s, 3)) ? %s := %s + 1;"
                      % (j, j, L, j, j, L, mn, j, L, mn, j, c, c))
        s.head.append(">> %s;" % c)
        if self.ch(0.5):
            # tail-recursive function swapping two list arguments
            p1, p2, pn = self.params(3)
            sw = self.nm("swl")
            s.nglob += 1
            s.head.append("%s := (%s, %s, %s) => { (%s < 1) ? => [%s, %s]; %s :+ %s; => %s(%s, %s, %s - 1); };"
                          % (sw, p1, p2, pn, pn, p1, p2, p1, pn, sw, p2, p1, pn))
            A = self.glob(s, "A", "[]")
            s.head.append(">> %s(%s, %s, %d); >> %s(%s, [], %d);" % (sw, A, self.pick(["[]", A]), self.ri(0, 7),
                                                                     sw, A, self.ri(0, 5)))
        return s

    # ================================================================= alias
    def alias_expr(self, L, names):
        """An expression whose value is the list L itself, reached by an unusual route.
        `names` provides fresh helper names (identity function, holder list)."""
        idf, hold = names
        return self.pick([
            "%s :+ %d" % (L, self.ri(0, 9)),        # append's value is the list
            "%s | 0" % L,                           # a list is true: | returns it
            "0 | %s" % L,
            "1 & %s" % L,                           # & returns its right operand
            "%s(%s)" % (idf, L),
            "[%s][0]" % L,
            "[%d, %s][1]" % (self.ri(0, 9), L),     # through a record field
            "-[%s]" % L,                            # pop returns the element itself
            "(%s)" % L,
            "%s[0]" % hold,                         # through an element stored earlier
        ])

    def f_alias(self):
        """A list aliased by an unusual route, then mutated through the alias while an
        indexed loop / cached header reads the original (listopt uniqueness, appendopt,
        reclaim escape analysis)."""
        s = Snip("alias")
        n = self.ri(1, 9)
        L = self.glob(s, "L", "[%s]" % ", ".join(str(self.ri(0, 30)) for _ in range(n)))
        idf = self.nm("id")
        pp = self.nm("p")
        s.nglob += 1
        s.head.append("%s := (%s) => { => %s; };" % (idf, pp, pp))
        hold = self.glob(s, "hd", "[%s]" % L)
        A = self.glob(s, "A", self.alias_expr(L, (idf, hold)))
        if self.ch(0.3):
            # chained / conditional aliasing
            B = self.glob(s, "B", "[]")
            s.head.append("(|%s| > %d) ? %s := %s; : %s := %s;" % (L, self.ri(0, 5), B, A, B, L))
            A = B
        if self.ch(0.25):
            x = self.nm("x")
            E = self.glob(s, "E", "[]")
            s.head.append("(%s : %s) @ %s := %s;" % (x, hold, E, x))   # for-each variable aliases the element
            A = E
        sm = self.glob(s, "sm", "0")
        i = self.nm("i")
        mut = self.pick([
            "(%s = %d & |%s| < %d) ? %s :+ %s;" % (i, self.ri(0, 4), A, n + 6, A, i),
            "(%s = %d & |%s| > 1) ? %s := %s + -%s;" % (i, self.ri(0, 4), A, sm, sm, A),
            "(%s + 1 < |%s|) ? %s[%s + 1] := %s[%s + 1] + %s;" % (i, A, A, i, A, i, i),
            "(|%s| < %d) ? %s :+ %s[%s];" % (A, n + 4, A, L, i),
        ])
        s.head.append("(%s := 0; %s < |%s| & %s < 30; %s := %s + 1;) @ { (%s < |%s|) ? { %s := %s + %s[%s]; %s } }"
                      % (i, i, L, i, i, i, i, L, sm, sm, L, i, mut))
        s.head.append(">> %s; >> %s; >> %s = %s;" % (sm, L, A, L))
        # a copy loop whose destination is the alias
        if self.ch(0.5):
            j = self.nm("i")
            lim = self.ri(0, n)
            s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ %s :+ %s[%s];" % (j, j, lim, j, j, A, L, j))
            s.head.append(">> %s;" % L)
        # functions: alias of a fresh list escapes through an unusual route
        for _ in range(self.ri(1, 3)):
            pn, t, u, w, dp = self.params(5)
            f = self.nm("af")
            ex = self.alias_expr(t, (idf, "[%s]" % t)) if self.ch(0.8) else u
            body = ["%s := [%s, %s + 1];" % (t, pn, pn), "%s := [%s];" % (u, pn),
                    "(%s > 0) ? %s := %s(%s + 1, %s - 1);" % (dp, w, f, pn, dp) if self.ch(0.6) else "%s := 0;" % w,
                    "=> %s;" % ex]
            s.nglob += 1
            s.head.append("%s := (%s, %s) => { %s };" % (f, pn, dp, " ".join(body)))
            r = self.glob(s, "rr", "%s(%d, %d)" % (f, self.ri(0, 9), self.ri(0, 3)))
            self.glob(s, "jk", "[%d, %d, %d]" % (self.ri(0, 99), self.ri(0, 99), self.ri(0, 99)))
            s.head.append(">> %s; %s :+ %d; >> %s;" % (r, r, self.ri(0, 9), r))
        self.export(L, "ilist")
        return s

    # ================================================================== grow
    def f_grow(self):
        """Bigger lists (hundreds to thousands of elements): interleaved growth of several
        lists (grow-in-place vs move), Reserve hints, bulk self-copies whose source buffer
        moves while copying, temp lists freed in between; observed through checksums."""
        s = Snip("grow")
        n = self.pick([self.ri(50, 300), self.ri(300, 1500), self.ri(1, 40)])
        A = self.glob(s, "A", "[]")
        B = self.glob(s, "B", "[]")
        i = self.nm("i")
        body = ["%s :+ %s * %d;" % (A, i, self.ri(1, 7))]
        if self.ch(0.6):
            body.append("%s :+ %s;" % (B, self.pick([i, "-%s" % i])))
        if self.ch(0.4):
            body.append("(%s - %s / %d * %d = 0) ? %s := [%s, %s];" % (i, i, 3, 3, self.nm("tmp"), i, i))
        self.r.shuffle(body)
        s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ { %s }" % (i, i, n, i, i, " ".join(body)))
        ck, ck2, cx = self.params(3)
        sumf = self.fn(s, "ck", [ck], ["%s := 0;" % ck2, "(%s : %s) @ %s := %s * 31 + %s;" % (cx, ck, ck2, ck2, cx),
                                     "=> %s;" % ck2])
        s.head.append(">> |%s|; >> %s(%s); >> |%s|;" % (A, sumf, A, B))
        # bulk self-copy: source = destination, the buffer moves while it grows
        lo = self.ri(0, max(0, n - 1))
        hi = self.ri(lo, n)
        j = self.nm("i")
        s.head.append("(%s := %d; %s < %d; %s := %s + 1;) @ %s :+ %s[%s];" % (j, lo, j, hi, j, j, A, A, j))
        s.head.append(">> |%s|; >> %s(%s);" % (A, sumf, A))
        # copy into another list with an allocation in between every few elements
        C = self.glob(s, "C", "[]")
        k = self.nm("i")
        junk = self.glob(s, "jk", "[]")
        md = self.ri(3, 17)
        s.head.append("(%s := 0; %s < |%s|; %s := %s + %d;) @ { %s :+ %s[%s]; (%s - %s / %d * %d = 0) ? %s :+ [%s]; }"
                      % (k, k, A, k, k, self.ri(1, 3), C, A, k, k, k, md, md, junk, k))
        s.head.append(">> |%s|; >> %s(%s); >> |%s|;" % (C, sumf, C, junk))
        # recursive function that builds and drops temps (reclaim frees them at the bump end)
        pn, pr, pt, pi = self.params(4)
        tf = self.nm("tf")
        s.nglob += 1
        s.head.append("%s := (%s, %s) => { %s := [%s]; (%s := 1; %s < %s; %s := %s + 1;) @ %s :+ %s; "
                      "(%s > 0) ? %s[0] := %s(%s / 2 + 1, %s - 1); => |%s| + %s[0]; };"
                      % (tf, pn, pr, pt, pn, pi, pi, pn, pi, pi, pt, pi, pr, pt, tf, pn, pr, pt, pt))
        D = self.glob(s, "D", "[]")
        m = self.nm("i")
        s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ { %s :+ %s(%s * %d + 1, %d); %s :+ %s; }"
                      % (m, m, self.ri(1, 12), m, m, D, tf, m, self.ri(5, 60), self.ri(0, 3), A, m))
        s.head.append(">> %s; >> |%s|; >> %s(%s);" % (D, A, sumf, A))
        if self.ch(0.4):
            # pop most of it again, then grow again
            q = self.nm("i")
            s.head.append("(%s := 0; %s < %d & |%s| > 0; %s := %s + 1;) @ -%s;" % (q, q, self.ri(1, n + 5), A, q, q, A))
            q2 = self.nm("i")
            s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ %s :+ %s;" % (q2, q2, self.ri(1, 200), q2, q2, A, q2))
            s.head.append(">> |%s|; >> %s(%s);" % (A, sumf, A))
        self.export(B, "ilist")
        return s

    # ================================================================== rec2
    def f_rec2(self):
        """Record-typing stress: function fields, two shapes through one generic helper,
        nullable flips, input-dependent shapes, mutually nested shapes, records popped
        out of lists, int-terminated chains."""
        s = Snip("rec2")
        parts = self.r.sample(["fnfield", "twoshape", "nullflip", "union", "mutual", "popped", "intchain",
                               "grown", "grown", "mkempty", "wide", "wide"], self.ri(2, 4))
        for part in parts:
            if part == "fnfield":
                a, b = self.params(2)
                f1 = self.fn(s, "fa", [a], ["=> %s * 2;" % a])
                f2 = self.fn(s, "fb", [b], ["=> %s + 100;" % b])
                R = self.glob(s, "R", "[%s, %d]" % (f1, self.ri(0, 9)))
                s.head.append(">> %s[0](%s[1]); %s[0] := %s; >> %s[0](%s[1]); >> %s[0] = %s; >> %s[1];"
                              % (R, R, R, f2, R, R, R, f1, R))
                pr = self.nm("p")
                ap = self.fn(s, "ap", [pr], ["%s[1] := %s[0](%s[1]);" % (pr, pr, pr), "=> %s[1];" % pr])
                s.head.append(">> %s(%s) + %s(%s);" % (ap, R, ap, R))
            elif part == "twoshape":
                a, b = self.params(2)
                m1 = self.fn(s, "m1", [a], ["=> [%s, [%s]];" % (a, a)])
                m2 = self.fn(s, "m2", [b], ["=> [%s, %s, %s];" % (b, b, b)])
                r = self.nm("p")
                use = self.fn(s, "us", [r], ["%s[0] := %s[0] + 1;" % (r, r), "=> %s[0] * 10 + |%s|;" % (r, r)])
                X = self.glob(s, "X", "%s(%d)" % (m1, self.ri(0, 9)))
                Y = self.glob(s, "Y", "%s(%d)" % (m2, self.ri(0, 9)))
                s.head.append(">> %s(%s) + %s(%s); >> %s; >> %s; >> %s = %s;" % (use, X, use, Y, X, Y, X, Y))
                if self.ch(0.5):
                    Z = self.glob(s, "Z", "[%s, %s]" % (X, Y))
                    x = self.nm("x")
                    s.head.append("(%s : %s) @ >> %s(%s);" % (x, Z, use, x))
            elif part == "nullflip":
                a = self.nm("p")
                mk = self.fn(s, "mk", [a], ["=> [%s, [%s, %s]];" % (a, a, a)])
                N = self.glob(s, "N", "%s(%d)" % (mk, self.ri(0, 9)))
                s.head.append(">> %s[1][0]; %s := []; >> %s = []; >> |%s|; %s := %s(%d); >> %s[0] + %s[1][1];"
                              % (N, N, N, N, N, mk, self.ri(0, 9), N, N))
                q = self.nm("p")
                g = self.fn(s, "nf", [q], ["(%s = []) ? => -1;" % q, "=> %s[0];" % q])
                s.head.append(">> %s(%s) + %s([]) + %s(%s(%d));" % (g, N, g, g, mk, self.ri(0, 9)))
            elif part == "union":
                kv = self.read(s, 0, 1, "kv")
                R = self.glob(s, "R", "[0, 0, 0]")
                s.head.append("(%s) ? %s := [%d, %d, %d]; : %s := [%d, [%d]];"
                              % (kv, R, self.ri(0, 9), self.ri(0, 9), self.ri(0, 9), R, self.ri(0, 9), self.ri(0, 9)))
                s.head.append(">> %s[0]; >> %s; >> |%s|; %s[0] := %s[0] + 1; >> %s;" % (R, R, R, R, R, R))
            elif part == "mutual":
                X = self.glob(s, "X", "[%d, []]" % self.ri(0, 9))
                Y = self.glob(s, "Y", "[[], %d]" % self.ri(0, 9))
                s.head.append("%s[1] := %s; %s[0] := [%d, []]; >> %s; >> %s[1][0][0] + %s[1][1];"
                              % (X, Y, Y, self.ri(0, 9), X, X, X))
                s.head.append("%s[0][1] := [[], %d]; >> %s; >> %s = %s;" % (Y, self.ri(0, 9), X, X, X))
            elif part == "popped":
                a = self.nm("p")
                mk = self.fn(s, "mk", [a], ["=> [%s, [%s]];" % (a, a)])
                RL = self.glob(s, "RL", "[%s(%d), %s(%d), %s(%d)]" % (mk, self.ri(0, 9), mk, self.ri(0, 9),
                                                                     mk, self.ri(0, 9)))
                R = self.glob(s, "R", "-%s" % RL)
                s.head.append("%s[0] := %d; %s[1] :+ %d; >> %s; >> %s; >> %s[0] = %s;"
                              % (R, self.ri(0, 9), R, self.ri(0, 9), RL, R, RL, R))
                x = self.nm("x")
                s.head.append("(%s : %s) @ %s[0] := %s[0] + %s[1][0];" % (x, RL, x, x, x))
                s.head.append(">> %s;" % RL)
            elif part == "grown":
                # an empty literal filled by appends to a fixed shape (pre-shaped record), then
                # read with literal indexes; sometimes a distant extra append / variable index
                G = self.glob(s, "G", "[]")
                s.head.append("%s :+ %d; %s :+ [%d]; %s :+ %d;" % (G, self.ri(0, 9), G, self.ri(0, 9), G, self.ri(0, 9)))
                s.head.append(">> %s[0] + %s[1][0] + %s[2]; %s[1] := [%d, %d]; >> %s;"
                              % (G, G, G, G, self.ri(0, 9), self.ri(0, 9), G))
                q = self.nm("p")
                hd = self.fn(s, "hd", [q], ["=> %s[0];" % q])
                s.head.append(">> %s(%s) + %s([%d, %d]);" % (hd, G, hd, self.ri(0, 9), self.ri(0, 9)))
                v = self.pick(["", "", "app", "idx", "pop"])
                if v == "app":
                    s.tail.append("%s :+ %d; >> %s;" % (G, self.ri(0, 9), G))
                elif v == "idx":
                    kv = self.read(s, 0, 2, "kv")
                    s.tail.append(">> %s[%s];" % (G, kv))
                elif v == "pop":
                    s.tail.append(">> -%s; >> %s;" % (G, G))
            elif part == "wide":
                # wide heterogeneous literals (records with many fields) built from main's
                # locals / globals, also in statement and condition position, compared with
                # ints and lists (the shape of tests/fuzzfound/bug01: JIT out of temporaries)
                ks = [self.glob(s, "v", str(self.ri(0, 9))) for _ in range(self.ri(3, 6))]
                dv = self.glob(s, "dv", "0")
                s.head.append("(%s) ? %s := [%d];" % (ks[0], dv, self.ri(0, 9)))
                blk = self.nm("i")
                s.head.append("{ %s := %s + 1; (%s := 0; %s < %d; %s := %s + 1;) @ { %s := %s + %s; } }"
                              % (ks[1], ks[1], blk, blk, self.ri(0, 3), blk, blk, ks[2], ks[2], blk))
                n = self.ri(8, 16)
                fields = [self.pick(ks + [str(self.ri(0, 9)), "%s + 1" % self.pick(ks)]) for _ in range(n)] + [dv]
                self.r.shuffle(fields)
                lit = "[%s]" % ", ".join(fields)
                form = self.pick(["([(%s != 0)]) ? ;" % lit, "([%s = []]) ? >> 1;" % lit, "%s;" % lit,
                                  ">> [%s, %d] = [%s, %d];" % (lit, self.ri(0, 1), lit, self.ri(0, 1)),
                                  "{ (%s : [1]) @ { %s := 1; } ([(%s != 0)]) ? ; }" % (self.nm("x"), ks[0], lit)])
                s.head.append(form)
                W = self.glob(s, "W", lit)
                s.head.append(">> %s[0]; >> %s[%d]; >> %s;" % (W, W, n, W))
            elif part == "mkempty":
                mk = self.fn(s, "MK", [], ["=> [];"])
                t, v = self.params(2)
                ins = self.nm("ins")
                s.nglob += 1
                s.head.append("%s := (%s, %s) => { (|%s| = 0) ? => [%s, %s(), %s()]; (%s < %s[0]) ? %s[1] := %s(%s[1], %s); "
                              ": (%s > %s[0]) ? %s[2] := %s(%s[2], %s); => %s; };"
                              % (ins, t, v, t, v, mk, mk, v, t, t, ins, t, v, v, t, t, ins, t, v, t))
                T = self.glob(s, "T", "%s()" % mk)
                i = self.nm("i")
                s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ %s := %s(%s, %s);"
                              % (i, i, self.ri(0, 9), i, i, T, ins, T, self.mod("(%s * 7 + %d)" % (i, self.ri(0, 9)), 11)))
                s.head.append(">> %s; >> |%s|;" % (T, T))
            else:
                C = self.glob(s, "C", "0")
                i = self.nm("i")
                s.head.append("(%s := 0; %s < %d; %s := %s + 1;) @ %s := [%s, %s];" % (i, i, self.ri(0, 8), i, i, C, i, C))
                p = self.glob(s, "pw", C)
                c = self.glob(s, "cw", "0")
                s.head.append("(%s != 0) @ { %s := %s + %s[0]; %s := %s[1]; }" % (p, c, c, p, p, p))
                s.head.append(">> %s; >> %s;" % (C, c))
        return s

    # ================================================================== glue
    def f_glue(self):
        """Values exported by earlier snippets passed through shared polymorphic helpers
        (generic instances; for-each / variable index / generic store on records and
        pairs from other snippets = distant violations), boxed together and compared."""
        s = Snip("glue")
        if len(self.exports) < 2:
            return None
        ex = self.r.sample(self.exports, min(len(self.exports), self.ri(2, 4)))
        x1 = self.nm("p")
        fst = self.fn(s, "fst", [x1], ["(%s = []) ? => -1;" % x1, "=> %s[0];" % x1])
        x2 = self.nm("p")
        ln = self.fn(s, "len", [x2], ["=> |%s|;" % x2])
        x3, r3, e3 = self.params(3)
        cp = self.fn(s, "cp", [x3], ["%s := [];" % r3, "(%s : %s) @ %s :+ %s;" % (e3, x3, r3, e3), "=> %s;" % r3])
        x4, r4, i4 = self.params(3)
        rv = self.fn(s, "rv", [x4], ["%s := [];" % r4, "(%s := |%s| - 1; %s >= 0; %s := %s - 1;) @ %s :+ %s[%s];"
                                     % (i4, x4, i4, i4, i4, r4, x4, i4), "=> %s;" % r4])
        x5, v5 = self.params(2)
        st = self.fn(s, "s0", [x5, v5], ["(%s != []) ? %s[0] := %s;" % (x5, x5, v5), "=> %s;" % v5])
        for n, k in ex:
            ops = [">> %s(%s);" % (fst, n), ">> %s(%s);" % (ln, n), ">> %s(%s) = %s;" % (cp, n, n),
                   ">> %s(%s(%s)) = %s;" % (rv, rv, n, n), ">> %s(%s);" % (rv, n)]
            if k in ("ilist", "rec", "tree", "cons"):
                ops.append(">> %s[0] + %s(%s, %d) + %s(%s);" % (n, st, n, self.ri(0, 9), fst, n)
                           if k == "rec" else ">> %s(%s, %d) + %s(%s);" % (st, n, self.ri(0, 9), fst, n))
            if k in ("ilist", "pairs"):
                ops.append("(%s : %s(%s)) @ >> %s;" % (self.nm("x"), cp, n, "%s(%s)" % (ln, n)))
            self.r.shuffle(ops)
            s.head += ops[:self.ri(1, 4)]
        box = self.glob(s, "bx", "[%s]" % ", ".join(n for n, k in ex))
        x = self.nm("x")
        s.head.append("(%s : %s) @ >> %s(%s) + %s(%s);" % (x, box, ln, x, ln, "%s(%s)" % (cp, x)))
        s.head.append(">> %s(%s) = %s;" % (fst, box, ex[0][0]))
        same = [(a, b) for a in ex for b in ex if a[0] < b[0] and a[1] == b[1] and a[1] in ("ilist", "pairs")]
        for (a, _), (b, _) in same[:2]:
            s.head.append(">> %s < %s; >> %s = %s;" % (a, b, a, b))
        return s

    # ================================================================ program
    def snippet(self, fam=None):
        fam = fam or self.pick(FAMILIES)
        if fam == "glue" and not self.glue:
            fam = "rec"
        s = getattr(self, "f_" + fam)()
        if s is None:
            s = self.f_rec()
        return s


def program(rng, n_lo=4, n_hi=7, risky_p=0.25):
    """A whole `opt` program: (list of snippets in order).  At most one risky snippet, last."""
    g = OptGen(rng)
    snips = []
    risky = None
    fams = [f for f in FAMILIES if f != "glue"]
    rng.shuffle(fams)
    # glue (uses exports of earlier snippets) goes somewhere after the second snippet
    fams.insert(rng.randint(2, len(fams)), "glue")
    want = rng.randint(n_lo, n_hi)
    allow_risky = rng.random() < risky_p
    tries = 0
    while len(snips) < want and tries < 40:
        n0 = len(g.exports)
        s = g.snippet(fams[tries % len(fams)] if rng.random() < 0.7 else None)
        tries += 1
        if s.risky:
            del g.exports[n0:]  # risky snippets run last (or not at all): nothing may use their values
            if allow_risky and risky is None:
                risky = s
            continue
        snips.append(s)
    return snips, risky, g


def assemble(snips, risky, rng):
    """Head of every snippet in order, then the tails (shuffled), then the risky snippet."""
    lines = []
    for s in snips:
        lines += s.head
    tails = [t for s in snips for t in s.tail]
    rng.shuffle(tails)
    lines += tails
    if risky is not None:
        lines += risky.head + risky.tail
    return "\n".join(lines) + "\n"
