"""epqgen -- random EPQ program generator for the differential fuzzer.

Generates programs strictly inside the well-defined subset (docs/LANGUAGE.md
section 9).  Programs are built as pyref.Ast trees and printed with
pyref.unparse().

Invariants enforced by construction
-----------------------------------
* Locals are only introduced at clean positions: statement-level `x := e;` in a
  statement list / branch body / loop body, a for-init, or the root of an if
  condition.  Nested assignments only ever target existing variables.
* Scopes are tracked exactly like section 4 (blocks, list literals, if, while,
  for, for-each); every read is of a variable in scope or of a global that is
  already assigned when the read executes (except injected errors).
* Global/local status follows the section 4.1 walk: new names at walk-reachable
  top-level positions are globals.
* Value "kinds" (a small static type system) make most operations valid:
    I              integer
    L(e)           list whose elements all have kind e (invariant)
    D(k, f)        "dynamic" of level k: an integer, or a list of D(k-1, f) values
                   (or, if f, a function).  D(0, False) == I.
    F(sig, b)      function with signature sig, of rank < b
  Every list object has one element kind for its whole life, so aliasing can
  never break the generator's assumptions.  Lists only contain values of a
  strictly lower level -> no cycles.  Values whose kind may contain a function
  are never printed.
* Termination: loops are counted (`for` with a counter the body never assigns,
  `while` with a counter incremented as the last body statement), for-each
  bodies only grow lists through appends guarded by the global append budget
  `zb`; every append inside a loop or a function is guarded.  Functions have
  ranks and only call functions of lower rank, plus self-recursion through an
  explicit depth parameter `dp` (never assigned).  Dynamic calls of values of
  unknown kind happen only in straight-line top-level code.
* A cost model (statements x loop trip counts + callee costs) keeps the work per
  program small; `<<` is only read in top-level code and the generated .in has
  enough numbers for every read.
* Baseline limits: <= ~170 constants (integer literals + function literals +
  1 per for-each) per function and at top level, <= ~150 globals, list literals
  <= 50 elements, few locals.
"""
import random

import pyref
from pyref import Ast

I = ("I",)


def L(e):
    return ("L", e)


def dk(k, f):
    if k == 0 and not f:
        return I
    return ("D", k, f)


def F(sig, b):
    return ("F", sig, b)


def level(k):
    if k[0] == "I" or k[0] == "F":
        return 0
    if k[0] == "L":
        return 1 + level(k[1])
    return k[1]


def has_func(k):
    t = k[0]
    if t == "F":
        return True
    if t == "D":
        return k[2]
    if t == "L":
        return has_func(k[1])
    return False


def fits(s, t):
    if s == t:
        return True
    if t[0] == "D":
        if s == I:
            return True
        if s[0] == "L" and t[1] >= 1 and s[1] == dk(t[1] - 1, t[2]):
            return True
        if s[0] == "F" and t[2]:
            return True
        return False
    if t[0] == "F" and s[0] == "F":
        return s[1] == t[1] and s[2] <= t[2]
    return False


def is_listish(k):
    return k[0] == "L" or (k[0] == "D" and k[1] >= 1)


def elem_of(k):
    if k[0] == "L":
        return k[1]
    if k[0] == "D":
        return dk(k[1] - 1, k[2])
    return None


def kstr(k):
    t = k[0]
    if t == "I":
        return "I"
    if t == "L":
        return "L(%s)" % kstr(k[1])
    if t == "D":
        return "D%d%s" % (k[1], "f" if k[2] else "")
    return "F%d" % len(k[1][0])


# --------------------------------------------------------------------------
# AST builders
# --------------------------------------------------------------------------

def N(t, *ch):
    return Ast(t, list(ch))


def INT(v, spelling=""):
    return Ast("INT", num=pyref.to_i32(v), id=spelling)


def ID(name):
    return Ast("ID", id=name)


def SEQ(stmts=None):
    return Ast("SEQ", list(stmts or []))


def BLK(stmts):
    return Ast("BLK", [SEQ(stmts)])


def EXP(e):
    return N("EXP", e)


def PRINT(e):
    return N("PRINT", e)


def SET(name, e):
    return N("SET", ID(name), e)


def IF(c, a, b=None):
    return N("IF", c, a, b if b is not None else SEQ())


def CALL(f, args):
    return N("CAL", f, Ast("LST", list(args)))


def LIST(items):
    return Ast("LST", list(items))


def count_appends(a):
    n = 1 if a.type == "APP" else 0
    if a.type == "DEF":
        return 0
    for c in a.ch:
        n += count_appends(c)
    return n


def count_consts(body):
    """Constants the original compiler puts into the Program for this body:
    integer literals, function literals (their bodies are separate programs) and
    one `1` per for-each."""
    n = 0
    stack = [body]
    while stack:
        a = stack.pop()
        if a.type == "INT":
            n += 1
        elif a.type == "DEF":
            n += 1
            continue
        elif a.type == "RNG":
            n += 1
        stack.extend(a.ch)
    return n


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

PROFILES = {
    # weights for kinds of new variables, plus knobs
    "clean": dict(kinds={"I": 60, "LI": 28, "LLI": 10, "D1": 1, "F": 1},
                  p_err=0.20, p_loose=0.01, p_loose_prog=0.04, p_dyncall=0.0, nfun=(1, 4), nmain=(12, 28),
                  p_funclit=0.0, p_slot=0.0, p_compile_err=0.015),
    "mixed": dict(kinds={"I": 40, "LI": 20, "LLI": 8, "D1": 8, "D2": 5, "LD1": 4, "F": 6,
                         "D1f": 3, "LF": 3, "D0f": 2},
                  p_err=0.18, p_loose=0.02, p_loose_prog=0.08, p_dyncall=0.02, nfun=(2, 6), nmain=(14, 32),
                  p_funclit=0.03, p_slot=0.5, p_compile_err=0.015),
    "nasty": dict(kinds={"I": 26, "LI": 12, "LLI": 6, "D1": 12, "D2": 10, "LD1": 6, "LD2": 3,
                         "F": 10, "D1f": 6, "D2f": 4, "LF": 5, "D0f": 4},
                  p_err=0.15, p_loose=0.03, p_loose_prog=0.12, p_dyncall=0.05, nfun=(3, 7), nmain=(16, 34),
                  p_funclit=0.06, p_slot=0.8, p_compile_err=0.015),
}

ERROR_KINDS = ["Operands must be integers", "Division by zero", "List is empty",
               "Operand must be an integer or a list", "Invalid comparison",
               "Operand must be an integer", "Invalid operation", "Index out of bounds",
               "Operand must be a list", "Operand must be a function", "Invalid arity",
               "Undefined global"]
COMPILE_ERRORS = ["Undefined variable", "Invalid argument list", "Argument shadows global variable",
                  "Duplicate argument name"]

APPEND_BUDGET = 60
CONST_CAP = 165
COST_MAIN = 6000
COST_FN = 400


class Var:
    __slots__ = ("name", "kind", "known", "ro", "glob")

    def __init__(self, name, kind, ro=False, glob=False):
        self.name = name
        self.kind = kind
        self.known = kind
        self.ro = ro
        self.glob = glob


class FnInfo:
    __slots__ = ("name", "rank", "params", "ret", "rec", "maxdepth", "cost", "branch", "body_cost")

    def __init__(self, name, rank, params, ret, rec):
        self.name = name
        self.rank = rank
        self.params = params        # list of kinds (excluding dp)
        self.ret = ret
        self.rec = rec
        self.maxdepth = 0
        self.branch = 1
        self.cost = 1
        self.body_cost = 1

    @property
    def sig(self):
        return (tuple(self.params), self.ret)

    def kind(self):
        return F(self.sig, self.rank + 1) if not self.rec else None


class Ctx:
    def __init__(self, fn, rank):
        self.fn = fn                # FnInfo or None (top level)
        self.rank = rank            # may call functions of rank < self.rank (top level: huge)
        self.scopes = [[]]
        self.dead = []              # names of locals whose scope closed (for reuse)
        self.mult = 1
        self.loop = 0
        self.guarded = False
        self.reach = fn is None     # top-level walk-reachable position
        self.straight = fn is None  # straight-line top-level (executes at most once)
        self.depth = 0
        self.cost = 0
        self.nlocals = 0
        self.rec_calls = 0
        self.body = None

    @property
    def guard_needed(self):
        return (self.fn is not None or self.loop > 0) and not self.guarded

    def vars(self):
        for s in self.scopes:
            for v in s:
                yield v


class Gen:
    def __init__(self, seed, profile="mixed"):
        self.rng = random.Random(seed)
        if profile == "any":
            profile = self.rng.choice(["clean", "mixed", "mixed", "nasty"])
        self.profile = profile
        self.P = dict(PROFILES[profile])
        self.loose = self.rng.random() < self.P["p_loose_prog"]
        if not self.loose:
            self.P["p_loose"] = 0.0
            self.P["p_dyncall"] = 0.0
        self.allow_empty = False
        self.globals = {}           # name -> Var
        self.gdefined = set()       # globals assigned by now (top-level straight line)
        self.fns = []
        self.slots = []             # (Var, sig, bound)
        self.ncounter = 0
        self.nreads = 0
        self.features = {}
        self.top_cost = 0
        self.err_plan = None
        self.err_done = False
        self.ngl = 0
        self.no_calls = False
        self.fn_globals = set()
        self.lit_depth = 0

    # ---------------------------------------------------------------- utils
    def feat(self, k, n=1):
        self.features[k] = self.features.get(k, 0) + n

    def chance(self, p):
        return self.rng.random() < p

    def wpick(self, d):
        tot = sum(d.values())
        r = self.rng.random() * tot
        for k, w in d.items():
            r -= w
            if r <= 0:
                return k
        return k

    def fresh(self, prefix):
        self.ncounter += 1
        return "%s%d" % (prefix, self.ncounter)

    def kind_from_tag(self, tag, ctx_rank=None):
        if tag == "I":
            return I
        if tag == "LI":
            return L(I)
        if tag == "LLI":
            return L(L(I))
        if tag == "D1":
            return dk(1, False)
        if tag == "D2":
            return dk(2, False)
        if tag == "LD1":
            return L(dk(1, False))
        if tag == "LD2":
            return L(dk(2, False))
        if tag == "D1f":
            return dk(1, True)
        if tag == "D2f":
            return dk(2, True)
        if tag == "D0f":
            return dk(0, True)
        if tag in ("F", "LF"):
            cands = [f for f in self.fns if not f.rec and (ctx_rank is None or f.rank < ctx_rank)]
            if not cands:
                return I
            f = self.rng.choice(cands)
            b = ctx_rank if ctx_rank is not None else len(self.fns) + 1
            k = F(f.sig, max(f.rank + 1, min(b, 10 ** 6)))
            return k if tag == "F" else L(k)
        return I

    def choose_kind(self, ctx_rank=None, allow_func=True):
        for _ in range(10):
            tag = self.wpick(self.P["kinds"])
            if not allow_func and tag in ("F", "LF", "D1f", "D2f", "D0f"):
                continue
            return self.kind_from_tag(tag, ctx_rank)
        return I

    def lit(self, ctx, small=False):
        r = self.rng.random()
        if small or r < 0.55:
            v = self.rng.randint(0, 9)
            return INT(v)
        if r < 0.75:
            return INT(self.rng.randint(10, 1000))
        if r < 0.85:
            v = self.rng.choice([2147483647, 2147483646, 65536, 46341, 1000000007, 123456789])
            self.feat("big_literal")
            return INT(v)
        if r < 0.92:
            self.feat("int_min")
            return INT(2147483648, "2147483648")
        if r < 0.97:
            v = self.rng.choice([4294967295, 4294967296, 4294967297, 99999999999, 3000000000])
            self.feat("wrapping_literal")
            return INT(v, str(v))
        self.feat("huge_literal")
        return INT(-1, "99999999999999999999999")

    # ------------------------------------------------------------ variables
    def readable(self, ctx, pred):
        """Variables (locals in scope + defined globals) whose known kind satisfies pred."""
        out = [v for v in ctx.vars() if pred(v.known)]
        for g in self.globals.values():
            if (ctx.fn is not None or g.name in self.gdefined) and g.name in self.fn_visible(ctx) and pred(g.kind):
                out.append(g)
        return out

    def fn_visible(self, ctx):
        if ctx.fn is None:
            return self.gdefined
        return self.fn_globals

    def assignable(self, ctx, pred):
        out = [v for v in ctx.vars() if not v.ro and pred(v.kind)]
        for g in self.globals.values():
            if not g.ro and g.name in self.fn_visible(ctx) and pred(g.kind) and g.name != "zb":
                out.append(g)
        return out

    def define_local(self, ctx, kind, prefix="v", ro=False):
        name = None
        if ctx.dead and self.chance(0.25):
            cand = [n for n in ctx.dead if n not in self.globals and not any(v.name == n for v in ctx.vars())]
            if cand:
                name = self.rng.choice(cand)
                self.feat("name_reuse")
        if name is None:
            name = self.fresh(prefix)
        v = Var(name, kind, ro=ro)
        ctx.scopes[-1].append(v)
        ctx.nlocals += 1
        return v

    def define_global(self, kind, prefix="g", ro=False):
        name = self.fresh(prefix)
        v = Var(name, kind, ro=ro, glob=True)
        self.globals[name] = v
        return v

    def push(self, ctx):
        ctx.scopes.append([])

    def pop(self, ctx):
        s = ctx.scopes.pop()
        for v in s:
            ctx.dead.append(v.name)
        ctx.nlocals -= len(s)

    def reset_known(self, ctx):
        for v in ctx.vars():
            v.known = v.kind

    def snapshot(self, ctx):
        return {id(v): v.known for v in ctx.vars()}

    def restore(self, ctx, snap):
        for v in ctx.vars():
            if id(v) in snap:
                v.known = snap[id(v)]

    def merge(self, ctx, s1, s2):
        for v in ctx.vars():
            a = s1.get(id(v), v.kind)
            b = s2.get(id(v), v.kind)
            v.known = a if a == b else v.kind

    def note_assign(self, v, kind, stmt_level):
        if v.glob:
            return
        v.known = kind if stmt_level else v.kind

    # ---------------------------------------------------------- expressions
    def gen(self, ctx, kind, d):
        """Expression whose value fits `kind`.  Returns (ast, actual_kind)."""
        t = kind[0]
        if t == "I":
            return self.gen_int(ctx, d)
        if t == "L":
            return self.gen_list(ctx, kind, d)
        if t == "F":
            return self.gen_fn(ctx, kind, d)
        # D(k, f)
        k, f = kind[1], kind[2]
        r = self.rng.random()
        if r < 0.3:
            return self.gen_int(ctx, d)
        if r < 0.6 and k >= 1:
            return self.gen_list(ctx, L(dk(k - 1, f)), d)
        if r < 0.7 and f:
            fk = self.any_fn_kind(ctx)
            if fk is not None:
                return self.gen_fn(ctx, fk, d)
        src = self.source(ctx, kind, d)
        if src is not None:
            return src
        return self.gen_int(ctx, d)

    def any_fn_kind(self, ctx):
        cands = [f for f in self.fns if not f.rec]
        if not cands:
            return None
        f = self.rng.choice(cands)
        return F(f.sig, f.rank + 1)

    def source(self, ctx, kind, d):
        """A variable read / element read / call / nested assignment producing kind."""
        opts = []
        vs = self.readable(ctx, lambda k: fits(k, kind))
        if vs:
            opts += ["var"] * 6
        if d > 0:
            opts += ["elem", "call", "assign"]
            if self.loose and self.chance(0.3):
                opts.append("pop")
        self.rng.shuffle(opts)
        for o in opts:
            if o == "var":
                v = self.rng.choice(vs)
                return ID(v.name), v.known
            if o == "elem":
                r = self.gen_elem(ctx, kind, d)
                if r:
                    return r
            if o == "call":
                r = self.gen_call(ctx, kind, d)
                if r:
                    return r
            if o == "assign":
                r = self.gen_nested_assign(ctx, kind, d)
                if r:
                    return r
            if o == "pop":
                r = self.gen_pop(ctx, kind, d)
                if r:
                    return r
        return None

    def list_vars(self, ctx, pred):
        return self.readable(ctx, lambda k: is_listish(k) and pred(k))

    def safe_index(self, ctx, lv, d):
        """Index expression for list variable lv (mostly in range)."""
        p_bad = self.P["p_loose"] * 0.5
        r = self.rng.random()
        if r < p_bad:
            self.feat("raw_index")
            return self.gen_int(ctx, min(d, 1))[0]
        if r < 0.35:
            return INT(0) if self.rng.random() < 0.5 else N("SUB", N("ABS", ID(lv.name)), INT(1))
        # a variable here could be INT_MIN (|INT_MIN| < 0) -> only in loose programs
        kk = self.int_atom(ctx) if (self.loose and self.chance(0.4)) else INT(self.rng.randint(0, 12))
        # |k| - |k| / |v| * |v|  ==  |k| mod |v|
        ak = N("ABS", kk)
        return N("SUB", ak, N("MUL", N("DIV", N("ABS", self.clone(kk)), N("ABS", ID(lv.name))), N("ABS", ID(lv.name))))

    def clone(self, a):
        return Ast(a.type, [self.clone(c) for c in a.ch], a.num, a.id)

    def int_atom(self, ctx):
        vs = self.readable(ctx, lambda k: k == I)
        if vs and self.chance(0.7):
            return ID(self.rng.choice(vs).name)
        return self.lit(ctx, small=True)

    def gen_elem(self, ctx, kind, d):
        lvs = self.list_vars(ctx, lambda k: fits(elem_of(k), kind) and (k[0] == "L" or self.chance(self.P["p_loose"] * 3)))
        if not lvs:
            return None
        lv = self.rng.choice(lvs)
        lk = lv.known
        self.feat("index")
        return N("IND", ID(lv.name), self.safe_index(ctx, lv, d - 1)), elem_of(lk)

    def gen_pop(self, ctx, kind, d):
        lvs = self.list_vars(ctx, lambda k: k[0] == "L" and fits(k[1], kind))
        if not lvs:
            return None
        lv = self.rng.choice(lvs)
        self.feat("pop")
        return N("NEG", ID(lv.name)), lv.known[1]

    def gen_nested_assign(self, ctx, kind, d):
        vs = self.assignable(ctx, lambda k: fits(k, kind))
        if not vs:
            return None
        v = self.rng.choice(vs)
        sv = self.allow_empty
        self.allow_empty = False
        e, ek = self.gen(ctx, v.kind, d - 1)
        self.allow_empty = sv
        if not (fits(ek, v.kind) and fits(ek, kind)):
            return None
        self.note_assign(v, ek, False)
        self.feat("nested_assign")
        return SET(v.name, e), ek

    def gen_int(self, ctx, d):
        if d <= 0 or self.chance(0.25):
            r = self.rng.random()
            vs = self.readable(ctx, lambda k: k == I)
            if vs and r < 0.6:
                return ID(self.rng.choice(vs).name), I
            return self.lit(ctx), I
        choices = {"arith": 30, "cmp": 10, "eq": 6, "not": 3, "neg": 3, "abs": 6, "andor": 5,
                   "src": 25, "len": 6, "listcmp": 3, "loose": 20 * self.P["p_loose"],
                   "input": 2 if ctx.fn is None and ctx.mult <= 4 and self.nreads < 40 else 0}
        c = self.wpick(choices)
        if c == "arith":
            op = self.rng.choice(["ADD", "SUB", "MUL", "MUL", "DIV", "ADD", "SUB"])
            a, _ = self.gen_int(ctx, d - 1)
            b, _ = self.gen_int(ctx, d - 1)
            if op == "DIV":
                self.feat("div")
                if self.chance(0.75) or (not self.loose and self.chance(0.6)):
                    b = N("OR", b, INT(self.rng.choice([1, 2, 3, 7])))
                elif not self.loose:
                    b = INT(self.rng.choice([1, 2, 3, 7, 10, 100, 2147483647]))
                    if self.chance(0.4):
                        b = N("NEG", b)
                elif self.chance(0.3):
                    self.feat("int_min_div")
                    a = INT(2147483648, "2147483648")
                    b = N("NEG", INT(1))
            elif op == "MUL":
                self.feat("mul")
            return N(op, a, b), I
        if c == "cmp":
            op = self.rng.choice(["LT", "GT", "LE", "GE", "EQ", "NE"])
            a, _ = self.gen_int(ctx, d - 1)
            b, _ = self.gen_int(ctx, d - 1)
            return N(op, a, b), I
        if c == "eq":
            k = self.choose_kind(ctx.rank)
            sv = self.allow_empty
            self.allow_empty = True
            a, _ = self.gen(ctx, k, d - 1)
            b, _ = self.gen(ctx, k if self.chance(0.7) else self.choose_kind(ctx.rank), d - 1)
            self.allow_empty = sv
            if has_func(k):
                self.feat("func_eq")
            elif is_listish(k):
                self.feat("list_eq")
            return N(self.rng.choice(["EQ", "NE"]), a, b), I
        if c == "not":
            a, _ = self.gen_int(ctx, d - 1)
            return N("NOT", a), I
        if c == "neg":
            a, _ = self.gen_int(ctx, d - 1)
            return N("NEG", a), I
        if c == "abs":
            a, _ = self.gen_int(ctx, d - 1)
            return N("ABS", a), I
        if c == "andor":
            a, _ = self.gen_int(ctx, d - 1)
            b, _ = self.gen_int(ctx, d - 1)
            self.feat("andor_value")
            return N(self.rng.choice(["AND", "OR"]), a, b), I
        if c == "len":
            lvs = self.list_vars(ctx, lambda k: not has_func(k) or self.chance(self.P["p_loose"]))
            if lvs:
                self.feat("len")
                return N("ABS", ID(self.rng.choice(lvs).name)), I
            k = self.rng.choice([L(I), L(L(I))])
            a, _ = self.gen_list(ctx, k, d - 1)
            return N("ABS", a), I
        if c == "listcmp":
            ks = [L(I), L(I), L(L(I))]
            if self.loose:
                ks += [dk(1, False), dk(2, False)]
            k = self.rng.choice(ks)
            sv = self.allow_empty
            self.allow_empty = True
            a, _ = self.gen(ctx, k, d - 1)
            b, _ = self.gen(ctx, k, d - 1)
            self.allow_empty = sv
            self.feat("list_lt")
            return N(self.rng.choice(["LT", "GT", "LE", "GE"]), a, b), I
        if c == "loose":
            # operand of a possibly-wrong kind in an absorbing position (result is int or an error)
            k = self.choose_kind(ctx.rank)
            a, _ = self.gen(ctx, k, d - 1)
            b, _ = self.gen_int(ctx, d - 1)
            op = self.rng.choice(["ADD", "SUB", "MUL", "LT", "GT", "NOT", "ABS", "DIV"])
            self.feat("loose_op")
            if op == "NOT":
                return N("NOT", a), I
            if op == "ABS":
                return N("ABS", a), I
            if op == "DIV":
                return N("DIV", b, N("OR", a, INT(1))), I
            return (N(op, a, b) if self.chance(0.5) else N(op, b, a)), I
        if c == "input":
            self.nreads += ctx.mult
            self.feat("input")
            return Ast("INP"), I
        r = self.source(ctx, I, d)
        if r is not None:
            return r
        return self.lit(ctx), I

    def gen_list(self, ctx, kind, d):
        e = kind[1]
        r = self.rng.random()
        if d > 0 and r < 0.45:
            s = self.source(ctx, kind, d)
            if s is not None:
                return s
        if d > 0 and r < 0.55 and not ctx.guard_needed:
            lvs = self.readable(ctx, lambda k: k == kind)
            if lvs:
                lv = self.rng.choice(lvs)
                a = ID(lv.name)
                n = self.rng.choice([1, 1, 2, 3])
                sv = self.allow_empty
                self.allow_empty = False
                for _ in range(n):
                    a = N("APP", a, self.gen(ctx, e, d - 1)[0])
                self.allow_empty = sv
                self.feat("append_expr")
                if n > 1:
                    self.feat("append_chain")
                return a, kind
        # literal
        if e == I and self.chance(0.05) and ctx.fn is None and ctx.mult == 1:
            n = self.rng.randint(10, 40)
            d = 0
        elif self.allow_empty and self.chance(0.25):
            n = 0
            self.feat("empty_list")
        else:
            n = self.rng.choice([1, 1, 2, 2, 3, 3, 4, 5])
        items = []
        self.push(ctx)          # a list literal opens a scope
        for _ in range(n):
            items.append(self.gen(ctx, e, d - 1 if d > 0 else 0)[0])
        self.pop(ctx)
        if e[0] == "D" or (e[0] == "L" and e[1][0] == "D"):
            self.feat("hetero_list")
        if level(kind) >= 2:
            self.feat("nested_list")
        return LIST(items), kind

    def gen_fn(self, ctx, kind, d):
        sig, b = kind[1], kind[2]
        cands = [f for f in self.fns if not f.rec and f.sig == sig and f.rank < b]
        opts = []
        if cands:
            opts.append(("name", None))
        vs = self.readable(ctx, lambda k: fits(k, kind))
        if vs:
            opts.append(("var", None))
        if d > 0:
            r = self.gen_elem(ctx, kind, d) if self.chance(0.3) else None
            if r:
                return r
        if d > 0 and self.chance(self.P["p_funclit"] * 3) and b > 0:
            lit = self.gen_funclit(ctx, sig, min(b - 1, ctx.rank - 1 if ctx.fn else b - 1))
            if lit is not None:
                return lit, kind
        if not opts:
            # always exists: the function the kind was derived from
            f = [f for f in self.fns if not f.rec and f.sig == sig]
            if f:
                g = min(f, key=lambda x: x.rank)
                return ID(g.name), F(sig, g.rank + 1)
            return INT(0), I
        o = self.rng.choice(opts)[0]
        if o == "name":
            f = self.rng.choice(cands)
            self.feat("func_value")
            return ID(f.name), F(sig, f.rank + 1)
        v = self.rng.choice(vs)
        return ID(v.name), v.known

    def gen_call(self, ctx, want, d):
        """A call whose result fits `want` (any kind if want is None)."""
        sv = self.allow_empty
        self.allow_empty = False
        try:
            return self._gen_call(ctx, want, d)
        finally:
            self.allow_empty = sv

    def _gen_call(self, ctx, want, d):
        if ctx.cost > self.cost_limit(ctx) or self.no_calls:
            return None
        cands = []
        for f in self.fns:
            if f.rank < ctx.rank and (want is None or fits(f.ret, want)):
                if f.cost * ctx.mult + ctx.cost <= self.cost_limit(ctx):
                    cands.append(("fn", f))
        for v in self.readable(ctx, lambda k: k[0] == "F" and k[2] <= ctx.rank):
            sig = v.known[1]
            if want is None or fits(sig[1], want):
                fs = [f for f in self.fns if f.sig == sig and not f.rec]
                c = max([f.cost for f in fs] + [COST_FN])
                if c * ctx.mult + ctx.cost <= self.cost_limit(ctx):
                    cands.append(("var", v))
        if ctx.fn is not None and ctx.fn.rec and ctx.loop == 0 and ctx.rec_calls < ctx.fn.branch \
                and (want is None or fits(ctx.fn.ret, want)) and self.chance(0.5):
            cands.append(("self", ctx.fn))
        if not cands:
            return None
        kind, x = self.rng.choice(cands)
        if kind == "self":
            ctx.rec_calls += 1
            args = [N("SUB", ID("dp"), INT(1))]
            for pk in x.params:
                args.append(self.gen(ctx, pk, max(0, d - 2))[0])
            self.feat("recursion")
            return CALL(ID(x.name), args), x.ret
        if kind == "fn":
            f = x
            args = []
            if f.rec:
                dep = self.rng.randint(0, f.maxdepth)
                args.append(INT(dep))
                if dep >= 50:
                    self.feat("deep_recursion")
            for pk in f.params:
                a, ak = self.gen(ctx, pk, max(0, d - 1))
                args.append(a)
                if pk[0] == "D":
                    self.feat("poly_arg_" + kstr(ak)[:1])
                if is_listish(ak) and a.type == "ID":
                    self.feat("alias_arg")
            ctx.cost += f.cost * ctx.mult
            self.feat("call")
            return CALL(ID(f.name), args), f.ret
        v = x
        sig = v.known[1]
        args = [self.gen(ctx, pk, max(0, d - 1))[0] for pk in sig[0]]
        ctx.cost += COST_FN * ctx.mult
        self.feat("call_through_var")
        return CALL(ID(v.name), args), sig[1]

    def cost_limit(self, ctx):
        return COST_FN if ctx.fn is not None else COST_MAIN

    def gen_printable(self, ctx, d):
        k = self.choose_kind(ctx.rank, allow_func=False)
        if self.chance(0.35):
            vs = self.readable(ctx, lambda kk: not has_func(kk))
            if vs:
                v = self.rng.choice(vs)
                return ID(v.name)
        sv = self.allow_empty
        self.allow_empty = True
        e = self.gen(ctx, k, d)[0]
        self.allow_empty = sv
        return e

    # ----------------------------------------------------------- statements
    def gen_stmts(self, ctx, n):
        out = []
        for _ in range(n):
            if count_consts(SEQ(out)) + self.const_base(ctx) > CONST_CAP:
                break
            s = self.gen_stmt(ctx)
            if s is not None:
                out.extend(s if isinstance(s, list) else [s])
        return out

    def const_base(self, ctx):
        return getattr(ctx, "const_base", 0)

    def wrap_guard(self, ctx, stmts):
        """Guard statements containing appends with the global append budget."""
        n = sum(count_appends(s) for s in stmts)
        if n == 0 or not ctx.guard_needed:
            return stmts
        self.feat("guarded_append")
        body = list(stmts) + [EXP(SET("zb", N("ADD", ID("zb"), INT(n))))]
        return [IF(N("LT", ID("zb"), INT(APPEND_BUDGET)), BLK(body))]

    def gen_stmt(self, ctx):
        P = self.P
        d = 3 if ctx.depth < 2 else 2
        w = {"print": 26, "def": 14, "assign": 10, "store": 6, "append": 7, "pop": 3,
             "if": 8, "for": 6, "while": 3, "foreach": 6, "block": 3, "callstmt": 5,
             "ret": 3 if ctx.fn is not None else 0, "sc": 3, "alias": 3,
             "funclit": 100 * P["p_funclit"], "dyncall": 100 * P["p_dyncall"] if ctx.straight else 0,
             "slotredef": 4 if (ctx.straight and self.slots) else 0,
             "topret": 0.4 if ctx.fn is None else 0,
             "ifdef": 3}
        if ctx.depth >= 3 or ctx.cost > self.cost_limit(ctx) * 0.8:
            for k in ("if", "for", "while", "foreach", "block", "ifdef"):
                w[k] = 0
        if ctx.mult > 60:
            for k in ("for", "while", "foreach"):
                w[k] = 0
        if ctx.nlocals > 60:
            w["def"] = 0
            w["ifdef"] = 0
        c = self.wpick(w)
        ctx.cost += ctx.mult
        if c == "print":
            e = self.gen_printable(ctx, d)
            self.feat("print")
            return self.wrap_guard(ctx, [PRINT(e)])
        if c == "def":
            return self.gen_def(ctx, d)
        if c == "assign":
            vs = self.assignable(ctx, lambda k: True)
            if not vs:
                return self.gen_def(ctx, d)
            v = self.rng.choice(vs)
            e, ek = self.gen(ctx, v.kind, d)
            self.note_assign(v, ek, True)
            st = [EXP(SET(v.name, e))]
            if self.chance(0.4) and not has_func(v.kind):
                st.append(PRINT(ID(v.name)))
            return self.wrap_guard(ctx, st)
        if c == "store":
            lvs = self.list_vars(ctx, lambda k: k[0] == "L")
            if not lvs:
                return None
            lv = self.rng.choice(lvs)
            lk = lv.known
            idx = self.safe_index(ctx, lv, d - 1)
            e, _ = self.gen(ctx, lk[1], d - 1)
            self.feat("store")
            st = [EXP(N("SET", N("IND", ID(lv.name), idx), e))]
            if not has_func(lk) and self.chance(0.4):
                st.append(PRINT(ID(lv.name)))
            return self.wrap_guard(ctx, st)
        if c == "append":
            lvs = self.list_vars(ctx, lambda k: k[0] == "L")
            if not lvs:
                return None
            lv = self.rng.choice(lvs)
            lk = lv.known
            a = ID(lv.name)
            n = self.rng.choice([1, 1, 1, 2, 3])
            for _ in range(n):
                a = N("APP", a, self.gen(ctx, lk[1], d - 1)[0])
            self.feat("append")
            if n > 1:
                self.feat("append_chain")
            st = [EXP(a)]
            if not has_func(lk) and self.chance(0.5):
                st.append(PRINT(ID(lv.name)))
            return self.wrap_guard(ctx, st)
        if c == "pop":
            lvs = self.list_vars(ctx, lambda k: k[0] == "L")
            if not lvs:
                return None
            lv = self.rng.choice(lvs)
            self.feat("pop")
            pe = N("NEG", ID(lv.name))
            body = PRINT(pe) if not has_func(lv.known) else EXP(pe)
            if not self.loose or self.chance(0.8):
                return IF(N("GT", N("ABS", ID(lv.name)), INT(1)), body)
            return body
        if c == "if":
            return self.gen_if(ctx, d)
        if c == "ifdef":
            return self.gen_ifdef(ctx, d)
        if c == "for":
            return self.gen_for(ctx, d)
        if c == "while":
            return self.gen_while(ctx, d)
        if c == "foreach":
            return self.gen_foreach(ctx, d)
        if c == "block":
            self.push(ctx)
            ctx.reach_save = ctx.reach
            old = (ctx.reach, ctx.depth)
            ctx.reach = False
            ctx.depth += 1
            body = self.gen_stmts(ctx, self.rng.randint(1, 4))
            ctx.reach, ctx.depth = old
            self.pop(ctx)
            self.feat("block")
            return BLK(body)
        if c == "callstmt":
            r = self.gen_call(ctx, None, d)
            if r is None:
                return None
            e, k = r
            if not has_func(k) and self.chance(0.6):
                return self.wrap_guard(ctx, [PRINT(e)])
            return self.wrap_guard(ctx, [EXP(e)])
        if c == "ret":
            e, _ = self.gen(ctx, ctx.fn.ret, d)
            self.feat("early_return")
            return IF(self.gen_int(ctx, 1)[0], N("RET", e))
        if c == "topret":
            self.feat("toplevel_return")
            r = N("RET", self.gen_int(ctx, 1)[0])
            if ctx.straight and self.chance(0.5):
                return IF(N("GT", self.gen_int(ctx, 1)[0], INT(5)), r)
            return IF(N("EQ", INT(1), INT(self.rng.choice([0, 0, 1]))), r)
        if c == "sc":
            vs = self.assignable(ctx, lambda k: k == I)
            if not vs:
                return None
            v = self.rng.choice(vs)
            side = SET(v.name, N("ADD", ID(v.name), INT(1)))
            self.note_assign(v, I, False)
            cnd = self.gen_int(ctx, 1)[0]
            e = N(self.rng.choice(["AND", "OR"]), cnd, side)
            self.feat("shortcircuit_side")
            return [EXP(e), PRINT(ID(v.name))]
        if c == "alias":
            return self.gen_alias(ctx, d)
        if c == "funclit":
            fs = [f for f in self.fns if not f.rec and f.rank < ctx.rank]
            if not fs or ctx.reach:
                return None
            f = self.rng.choice(fs)
            lit = self.gen_funclit(ctx, f.sig, f.rank)
            if lit is None:
                return None
            v = self.define_local(ctx, F(f.sig, f.rank + 1))
            st = [EXP(SET(v.name, lit))]
            args = [self.gen(ctx, pk, 1)[0] for pk in f.sig[0]]
            call = CALL(ID(v.name), args)
            st.append(PRINT(call) if not has_func(f.sig[1]) else EXP(call))
            ctx.cost += COST_FN * ctx.mult
            return st
        if c == "dyncall":
            vs = self.readable(ctx, lambda k: k[0] == "D" and k[2] or k[0] == "F")
            if not vs:
                return None
            v = self.rng.choice(vs)
            nargs = self.rng.randint(0, 3)
            args = [self.gen_int(ctx, 1)[0] for _ in range(nargs)]
            self.feat("dyncall")
            ctx.cost += COST_FN
            call = CALL(ID(v.name), args)
            return EXP(call) if self.chance(0.5) else PRINT(N("EQ", call, self.gen_int(ctx, 1)[0]))
        if c == "slotredef":
            sv, sig, b = self.rng.choice(self.slots)
            fs = [f for f in self.fns if not f.rec and f.sig == sig and f.rank < b]
            if self.chance(0.3):
                lit = self.gen_funclit(ctx, sig, b - 1)
                if lit is not None:
                    self.feat("slot_redef_literal")
                    return EXP(SET(sv.name, lit))
            if not fs:
                return None
            self.feat("slot_redef")
            return EXP(SET(sv.name, ID(self.rng.choice(fs).name)))
        return None

    def gen_def(self, ctx, d):
        k = self.choose_kind(ctx.rank)
        e, ek = self.gen(ctx, k, d)
        if ctx.reach:
            v = self.define_global(k)
            self.gdefined.add(v.name)
            self.feat("main_global")
        else:
            v = self.define_local(ctx, k)
        v.known = ek if not v.glob else v.kind
        st = [EXP(SET(v.name, e))]
        if not has_func(k) and self.chance(0.5):
            st.append(PRINT(ID(v.name)))
        if k[0] == "D":
            self.feat("dyn_var")
        return self.wrap_guard(ctx, st)

    def body_ctx(self, ctx, loop_mult=None):
        """Enter a nested statement body."""
        old = (ctx.reach, ctx.straight, ctx.depth, ctx.mult, ctx.loop)
        ctx.depth += 1
        if loop_mult is not None:
            ctx.mult *= loop_mult
            ctx.loop += 1
            ctx.straight = False
        return old

    def leave(self, ctx, old):
        ctx.reach, ctx.straight, ctx.depth, ctx.mult, ctx.loop = old

    def gen_body(self, ctx, n, single_ok=True):
        self.push(ctx)
        if single_ok and self.chance(0.25):
            body = self.gen_stmts(ctx, 1)
            if len(body) == 1 and not (ctx.reach and body[0].type == "EXP"):
                self.pop(ctx)
                return body[0]
        else:
            body = self.gen_stmts(ctx, n)
        self.pop(ctx)
        return BLK(body)

    def gen_if(self, ctx, d):
        self.push(ctx)          # if-statement scope
        c, _ = self.gen_int(ctx, d) if self.chance(0.8) else self.gen(ctx, self.choose_kind(ctx.rank), d)
        old = self.body_ctx(ctx)
        ctx.reach = False
        ctx.straight = ctx.straight  # executes at most once if parent does
        snap = self.snapshot(ctx)
        a = self.gen_body(ctx, self.rng.randint(1, 4))
        s1 = self.snapshot(ctx)
        self.restore(ctx, snap)
        b = None
        if self.chance(0.5):
            b = self.gen_body(ctx, self.rng.randint(1, 3))
        s2 = self.snapshot(ctx)
        self.merge(ctx, s1, s2)
        self.leave(ctx, old)
        self.pop(ctx)
        self.feat("if")
        return IF(c, a, b)

    def gen_ifdef(self, ctx, d):
        """(v := e) ? ... : ...   -- a definition at the root of an if condition."""
        self.push(ctx)
        k = self.choose_kind(ctx.rank)
        e, ek = self.gen(ctx, k, d)
        if ctx.reach:
            v = self.define_global(k)
            self.gdefined.add(v.name)
        else:
            v = self.define_local(ctx, k)
            v.known = ek
        cond = SET(v.name, e)
        old = self.body_ctx(ctx)
        ctx.reach = False
        snap = self.snapshot(ctx)
        a = self.gen_body(ctx, self.rng.randint(1, 3))
        s1 = self.snapshot(ctx)
        self.restore(ctx, snap)
        b = self.gen_body(ctx, self.rng.randint(1, 3)) if self.chance(0.5) else None
        s2 = self.snapshot(ctx)
        self.merge(ctx, s1, s2)
        self.leave(ctx, old)
        self.pop(ctx)
        self.feat("if_cond_def")
        return IF(cond, a, b)

    def gen_for(self, ctx, d):
        n = self.rng.choice([0, 1, 2, 3, 3, 4, 5, 6, 8])
        if ctx.mult * max(n, 1) > 80:
            n = 2
        self.push(ctx)          # for scope
        cv = self.define_local(ctx, I, prefix="c", ro=True)
        down = self.chance(0.25)
        if down:
            init = EXP(SET(cv.name, INT(n)))
            cond = N("GT", ID(cv.name), INT(0))
            step = EXP(SET(cv.name, N("SUB", ID(cv.name), INT(1))))
        else:
            init = EXP(SET(cv.name, INT(0)))
            cond = N("LT", ID(cv.name), INT(n))
            step = EXP(SET(cv.name, N("ADD", ID(cv.name), INT(1))))
        old = self.body_ctx(ctx, max(n, 1))
        ctx.reach = False
        self.reset_known(ctx)
        body = self.gen_body(ctx, self.rng.randint(1, 4))
        self.leave(ctx, old)
        self.reset_known(ctx)
        self.pop(ctx)
        self.feat("for")
        return N("FOR", init, cond, step, body)

    def gen_while(self, ctx, d):
        n = self.rng.choice([0, 1, 2, 3, 4, 5])
        if ctx.mult * max(n, 1) > 80:
            n = 2
        if ctx.reach:
            wv = self.define_global(I, prefix="w", ro=True)
            self.gdefined.add(wv.name)
        else:
            wv = self.define_local(ctx, I, prefix="w", ro=True)
        pre = EXP(SET(wv.name, INT(0)))
        self.reset_known(ctx)
        self.push(ctx)          # while scope
        cond = N("LT", ID(wv.name), INT(n))
        if self.chance(0.3):
            cond = N("AND", cond, self.gen_int(ctx, 1)[0])
            self.feat("while_and")
        old = self.body_ctx(ctx, max(n, 1))
        ctx.reach = False
        self.push(ctx)
        body = self.gen_stmts(ctx, self.rng.randint(1, 4))
        body.append(EXP(SET(wv.name, N("ADD", ID(wv.name), INT(1)))))
        self.pop(ctx)
        self.leave(ctx, old)
        self.reset_known(ctx)
        self.pop(ctx)
        self.feat("while")
        return [pre, N("WHL", cond, BLK(body))]

    def gen_foreach(self, ctx, d):
        lvs = self.list_vars(ctx, lambda k: True)
        loose = False
        if not (self.loose and self.chance(0.3)):
            lvs = [v for v in lvs if v.known[0] == "L"]
        if lvs and self.chance(0.75):
            lv = self.rng.choice(lvs)
            lk = lv.known
            loose = lk[0] == "D"
            lexp = ID(lv.name)
            target = lv
        else:
            lk = self.rng.choice([L(I), L(I), L(L(I)), L(dk(1, False))])
            lexp, lk = self.gen_list(ctx, lk, 1)
            target = None
        ek = elem_of(lk)
        self.push(ctx)          # for-each scope
        ev = self.define_local(ctx, ek, prefix="e", ro=True)
        old = self.body_ctx(ctx, 8)
        ctx.reach = False
        self.reset_known(ctx)
        self.push(ctx)
        body = self.gen_stmts(ctx, self.rng.randint(1, 3))
        if target is not None and lk[0] == "L" and self.chance(0.4):
            # mutate the iterated list inside the body (bounded by the append budget)
            if self.chance(0.6):
                body.extend(self.wrap_guard(ctx, [EXP(N("APP", ID(target.name), self.gen(ctx, ek, 1)[0]))]))
                self.feat("foreach_grow")
            else:
                body.append(IF(N("GT", N("ABS", ID(target.name)), INT(2)), EXP(N("NEG", ID(target.name)))))
                self.feat("foreach_shrink")
        if not has_func(ek) and self.chance(0.6):
            body.insert(0, PRINT(ID(ev.name)))
        self.pop(ctx)
        self.leave(ctx, old)
        self.reset_known(ctx)
        self.pop(ctx)
        self.feat("foreach")
        if loose:
            self.feat("foreach_dyn")
        return N("RNG", ID(ev.name), lexp, BLK(body))

    def gen_alias(self, ctx, d):
        """alias a list, mutate through one name, observe through the other."""
        lvs = self.list_vars(ctx, lambda k: k[0] == "L" and not has_func(k))
        if not lvs or ctx.reach:
            return None
        lv = self.rng.choice(lvs)
        k = lv.known
        st = []
        av = self.define_local(ctx, k)
        st.append(EXP(SET(av.name, ID(lv.name))))
        if self.chance(0.5) and not ctx.guard_needed:
            st.append(EXP(N("APP", ID(av.name), self.gen(ctx, k[1], 1)[0])))
        else:
            st.append(IF(N("GT", N("ABS", ID(av.name)), INT(0)),
                         EXP(N("SET", N("IND", ID(av.name), INT(0)), self.gen(ctx, k[1], 1)[0]))))
        st.append(PRINT(ID(lv.name)))
        st.append(PRINT(N("EQ", ID(lv.name), ID(av.name))))
        self.feat("alias")
        return st

    # ------------------------------------------------------------ functions
    def gen_funclit(self, ctx, sig, rank):
        """A nested function literal with the given signature and rank (calls only < rank)."""
        if rank < 0 or self.lit_depth >= 1:
            return None
        self.lit_depth += 1
        try:
            return self._gen_funclit(ctx, sig, rank)
        finally:
            self.lit_depth -= 1

    def _gen_funclit(self, ctx, sig, rank):
        fake = FnInfo("<lit>", rank, list(sig[0]), sig[1], False)
        sub = Ctx(fake, rank)
        params = []
        for pk in sig[0]:
            params.append(self.define_local(sub, pk, prefix="p").name)
        old_reads = self.nreads
        body = self.gen_fn_body(sub, fake, self.rng.randint(1, 3))
        self.nreads = old_reads
        self.feat("nested_funclit")
        return N("DEF", Ast("LST", [ID(p) for p in params]), body)

    def gen_fn_body(self, sub, fi, n):
        sub.const_base = 0
        self.push(sub)          # the body block
        stmts = []
        if fi.rec:
            sub.rec_calls = 10 ** 6     # no self-call in the base case
            base, _ = self.gen(sub, fi.ret, 1)
            sub.rec_calls = 0
            stmts.append(IF(N("LT", ID("dp"), INT(1)), N("RET", base)))
        stmts += self.gen_stmts(sub, n)
        if fi.rec and sub.rec_calls == 0:
            # force one self call
            sub.rec_calls = 1
            args = [N("SUB", ID("dp"), INT(1))] + [self.gen(sub, pk, 1)[0] for pk in fi.params]
            e = CALL(ID(fi.name), args)
            stmts.append(PRINT(e) if not has_func(fi.ret) else EXP(e))
            self.feat("recursion")
        if fi.ret == I and self.chance(0.2):
            pass                # fall off the end: returns 0
        else:
            stmts.append(N("RET", self.gen(sub, fi.ret, 2)[0]))
        self.pop(sub)
        return BLK(stmts)

    def gen_function(self, fi):
        sub = Ctx(fi, fi.rank)
        params = []
        if fi.rec:
            v = Var("dp", I, ro=True)
            sub.scopes[0].append(v)
            params.append("dp")
        for pk in fi.params:
            params.append(self.define_local(sub, pk, prefix="p").name)
        body = self.gen_fn_body(sub, fi, self.rng.randint(2, 8))
        fi.body_cost = max(sub.cost, 1)
        if fi.rec:
            fi.branch = max(1, sub.rec_calls)
            if fi.branch == 1:
                fi.maxdepth = self.rng.choice([3, 10, 30, 100, 300]) if fi.body_cost < 40 else 10
                calls = fi.maxdepth + 1
            else:
                fi.maxdepth = self.rng.choice([2, 3, 4, 5])
                calls = 2 ** (fi.maxdepth + 1)
            fi.cost = fi.body_cost * calls
            while fi.cost > 4000 and fi.maxdepth > 1:
                fi.maxdepth //= 2
                calls = fi.maxdepth + 1 if fi.branch == 1 else 2 ** (fi.maxdepth + 1)
                fi.cost = fi.body_cost * calls
        else:
            fi.cost = fi.body_cost
        lit = N("DEF", Ast("LST", [ID(p) for p in params]), body)
        return EXP(SET(fi.name, lit))

    # ------------------------------------------------------------- errors
    def error_stmt(self, ctx, kind):
        g = self.rng
        lvs = self.list_vars(ctx, lambda k: k[0] == "L")
        ivs = self.readable(ctx, lambda k: k == I)
        fs = [f for f in self.fns if not f.rec]
        lv = ID(g.choice(lvs).name) if lvs and g.random() < 0.6 else LIST([INT(1), INT(2)])
        iv = ID(g.choice(ivs).name) if ivs and g.random() < 0.6 else INT(g.randint(1, 9))
        fv = ID(g.choice(fs).name) if fs else N("DEF", Ast("LST"), BLK([]))
        if kind == "Operands must be integers":
            return PRINT(N(g.choice(["ADD", "SUB", "MUL"]), iv, self.clone(lv)))
        if kind == "Division by zero":
            return PRINT(N("DIV", iv if g.random() < 0.7 else self.clone(lv), N("SUB", self.clone(iv), self.clone(iv))))
        if kind == "List is empty":
            return PRINT(N("NEG", LIST([]))) if g.random() < 0.5 else \
                EXP(SET(self.fresh("x") if False else "zb", N("ADD", ID("zb"), N("NEG", LIST([])))))
        if kind == "Operand must be an integer or a list":
            return PRINT(N(g.choice(["NEG", "ABS"]), fv))
        if kind == "Invalid comparison":
            return PRINT(N(g.choice(["LT", "GT", "LE", "GE"]), self.clone(lv), iv)) if g.random() < 0.5 else \
                PRINT(N("LT", LIST([fv]), LIST([self.clone(fv)])))
        if kind == "Operand must be an integer":
            return PRINT(N("NOT", self.clone(lv)))
        if kind == "Invalid operation":
            return PRINT(N("IND", iv, INT(0))) if g.random() < 0.5 else PRINT(N("IND", self.clone(lv), LIST([])))
        if kind == "Index out of bounds":
            return PRINT(N("IND", self.clone(lv), N("ABS", self.clone(lv)))) if g.random() < 0.5 else \
                PRINT(N("IND", self.clone(lv), N("NEG", INT(1))))
        if kind == "Operand must be a list":
            return EXP(N("APP", iv, INT(3)))
        if kind == "Operand must be a function":
            return PRINT(CALL(self.clone(lv) if g.random() < 0.5 else iv, [INT(1)]))
        if kind == "Invalid arity":
            if fs:
                f = g.choice(fs)
                n = len(f.params) + g.choice([1, 2])
                return EXP(CALL(ID(f.name), [INT(0)] * n))
            return EXP(CALL(N("DEF", Ast("LST", [ID("q")]), BLK([])), []))
        if kind == "Undefined global":
            nm = self.fresh("u")
            self.late_undefined = nm
            return PRINT(ID(nm))
        # compile errors
        if kind == "Undefined variable":
            return PRINT(ID(self.fresh("nv")))
        if kind == "Invalid argument list":
            return EXP(N("DEF", Ast("LST", [INT(1)]), BLK([])))
        if kind == "Argument shadows global variable":
            return EXP(N("DEF", Ast("LST", [ID("zb")]), BLK([])))
        if kind == "Duplicate argument name":
            return EXP(N("DEF", Ast("LST", [ID("q"), ID("q")]), BLK([])))
        raise AssertionError(kind)

    # ------------------------------------------------------------- program
    def program(self):
        P = self.P
        g = self.rng
        self.late_undefined = None
        top = Ctx(None, 10 ** 6)
        top.const_base = 0
        stmts = [EXP(SET("zb", INT(0)))]
        self.globals["zb"] = Var("zb", I, ro=True, glob=True)
        self.gdefined.add("zb")
        # prologue: data globals (no functions yet)
        for _ in range(g.randint(1, 4)):
            k = self.choose_kind(allow_func=False)
            e, ek = self.gen(top, k, 2)
            v = self.define_global(k)
            self.gdefined.add(v.name)
            stmts.append(EXP(SET(v.name, e)))
        # functions: decide signatures first
        nf = g.randint(*P["nfun"])
        infos = []
        for i in range(nf):
            np_ = g.choice([0, 1, 1, 2, 2, 3])
            rec = g.random() < 0.3
            fi = FnInfo(self.fresh("f"), i, [], I, rec)
            infos.append(fi)
        self.fns = []
        late = []
        # late globals: referenced by functions, assigned after the function section
        nlate = g.randint(0, 2)
        late_vars = []
        for _ in range(nlate):
            k = self.choose_kind(allow_func=False)
            v = self.define_global(k)
            late_vars.append(v)
        self.fn_globals = set(self.gdefined) | {v.name for v in late_vars}
        for fi in infos:
            # signature kinds may refer to already generated functions
            np_ = g.choice([0, 1, 1, 2, 2, 3])
            fi.params = [self.choose_kind(fi.rank) for _ in range(np_)]
            fi.ret = self.choose_kind(fi.rank)
            # slots usable by this function: created when it is a good time
            if self.fns and g.random() < P["p_slot"] * 0.4 and len(self.slots) < 3:
                low = g.choice([f for f in self.fns if not f.rec] or [None])
                if low is not None:
                    sv = self.define_global(F(low.sig, fi.rank), prefix="h")
                    self.slots.append((sv, low.sig, fi.rank))
                    self.fn_globals.add(sv.name)
                    late.append(EXP(SET(sv.name, ID(low.name))))
            stmts.append(self.gen_function(fi))
            self.fns.append(fi)
            if late_vars:
                self.feat("late_global")
        self.no_calls = True
        for v in late_vars:
            e, ek = self.gen(top, v.kind, 2)
            late.append(EXP(SET(v.name, e)))
        self.no_calls = False
        stmts += late
        for v in late_vars:
            self.gdefined.add(v.name)
        for sv, _, _ in self.slots:
            self.gdefined.add(sv.name)
        # error plan
        r = g.random()
        if r < P["p_compile_err"]:
            self.err_plan = g.choice(COMPILE_ERRORS)
        elif r < P["p_compile_err"] + P["p_err"]:
            self.err_plan = g.choice(ERROR_KINDS)
        top.const_base = count_consts(SEQ(stmts))
        nmain = g.randint(*P["nmain"])
        main = []
        err_at = g.randint(0, nmain) if self.err_plan else -1
        for i in range(nmain):
            if i == err_at:
                main.append(self.error_stmt(top, self.err_plan))
                self.feat("inject:" + self.err_plan)
                if self.late_undefined:
                    main.append(EXP(SET(self.late_undefined, INT(1))))
            if count_consts(SEQ(main)) + top.const_base > CONST_CAP:
                break
            s = self.gen_stmt(top)
            if s is not None:
                main.extend(s if isinstance(s, list) else [s])
        if err_at == nmain:
            main.append(self.error_stmt(top, self.err_plan))
            self.feat("inject:" + self.err_plan)
            if self.late_undefined:
                main.append(EXP(SET(self.late_undefined, INT(1))))
        # final observation of every printable global
        for v in self.globals.values():
            if v.name in self.gdefined and not has_func(v.kind) and v.name != "zb":
                main.append(PRINT(ID(v.name)))
        stmts += main
        self.top_cost = top.cost
        return SEQ(stmts)

    def inputs(self):
        n = self.nreads + 4
        g = self.rng
        vals = []
        for _ in range(n):
            r = g.random()
            if r < 0.6:
                vals.append(g.randint(0, 20))
            elif r < 0.8:
                vals.append(g.randint(-100, 100))
            elif r < 0.95:
                vals.append(g.randint(-2 ** 31, 2 ** 31 - 1))
            else:
                vals.append(g.choice([0, -1, 2147483647, -2147483648]))
        return " ".join(map(str, vals)) + "\n"


def generate(seed, profile="mixed"):
    """Returns (source_text, stdin_text, info dict)."""
    for attempt in range(20):
        gen = Gen(seed * 1000003 + attempt if attempt else seed, profile)
        root = gen.program()
        # baseline limits
        tops = count_consts(root)
        ok = tops <= 200
        nglob = len(gen.globals) + len(gen.fns) + 2
        ok = ok and nglob < 200

        def walk(a):
            nonlocal ok
            if a.type == "DEF":
                if count_consts(a.ch[1]) > 200:
                    ok = False
            if a.type == "LST" and len(a.ch) > 50:
                ok = False
            for c in a.ch:
                walk(c)
        walk(root)
        if not ok:
            continue
        src = pyref.unparse(root)
        info = {"profile": gen.profile, "features": gen.features, "err_plan": gen.err_plan,
                "consts_top": tops, "globals": nglob, "cost": gen.top_cost, "reads": gen.nreads,
                "bytes": len(src), "attempt": attempt}
        return src, gen.inputs(), info
    raise RuntimeError("could not generate a program within limits for seed %d" % seed)
