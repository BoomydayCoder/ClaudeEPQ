#!/usr/bin/env python3
"""pyref -- a small, obviously-correct reference interpreter for EPQ, written
directly from docs/LANGUAGE.md.  Python 3.9 standard library only.

It exists to cross-check the fuzzer and the spec: generated programs are run on
both pyref and the baseline; any disagreement means the generator left the
well-defined subset, the spec is wrong, or pyref is wrong.

Parsing: instead of a hand-written parser, pyref builds an LALR(1) automaton
from the grammar of LANGUAGE.md section 2 at start-up (DeRemer/dragon-book
lookahead propagation) and resolves conflicts exactly as bison does (rule
precedence = precedence of the rule's LAST terminal, or %prec; S/R without
precedence -> shift; R/R -> earlier rule).  The resulting parser therefore
accepts exactly the language of the original parser.yy, and `-e` reproduces
the original's AST dump (Ast::print_self) byte for byte.

Semantics follow LANGUAGE.md including the ClaudeEPQ choices of section 9 for
programs outside the well-defined subset (input at EOF -> 0, functions print as
<function>, classes rejected at compile time, no size limits).

CLI:  pyref.py [-e] [-t] [--parse-only] FILE     (program input on stdin)
API:  run_source(text, stdin_text, step_limit) -> Result
"""
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Lexer (mirrors scanner.ll: flex longest match, first rule wins on ties)
# ---------------------------------------------------------------------------

SINGLE = set("-+*/();{}=!?:&|@,.#[]")
MULTI = {":=": "ASSIGN", ">>": "PRINT", "<<": "INPUT", ">=": "GE", "<=": "LE",
         "!=": "NE", ":+": "APPEND", "=>": "DEF"}
LONG_MAX = (1 << 63) - 1


def to_i32(x):
    return ((x + 0x80000000) & 0xFFFFFFFF) - 0x80000000


class Tok:
    __slots__ = ("kind", "val", "line", "col")

    def __init__(self, kind, val, line, col):
        self.kind = kind
        self.val = val
        self.line = line
        self.col = col


def lex(text, fname, diag):
    """Returns list of tokens ending with END.  `diag` collects stderr lines."""
    toks = []
    i = 0
    n = len(text)
    line = 1
    col = 1
    while i < n:
        c = text[i]
        if c == " " or c == "\t":
            i += 1
            col += 1
            continue
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            if j < 0:
                j = n
            col += j - i
            i = j
            continue
        two = text[i:i + 2]
        if two in MULTI:
            toks.append(Tok(MULTI[two], None, line, col))
            i += 2
            col += 2
            continue
        if c in SINGLE:
            toks.append(Tok(c, None, line, col))
            i += 1
            col += 1
            continue
        if c == ">":
            toks.append(Tok("GT", None, line, col))
            i += 1
            col += 1
            continue
        if c == "<":
            toks.append(Tok("LT", None, line, col))
            i += 1
            col += 1
            continue
        if "0" <= c <= "9":
            j = i
            while j < n and "0" <= text[j] <= "9":
                j += 1
            v = int(text[i:j])
            if v > LONG_MAX:
                v = LONG_MAX          # strtol saturates
            toks.append(Tok("NUMBER", to_i32(v), line, col))
            col += j - i
            i = j
            continue
        if ("a" <= c <= "z") or ("A" <= c <= "Z"):
            j = i + 1
            while j < n and (text[j].isascii() and (text[j].isalnum() or text[j] == "_")):
                j += 1
            toks.append(Tok("ID", text[i:j], line, col))
            col += j - i
            i = j
            continue
        diag.append("%s:%d.%d: invalid character" % (fname, line, col))
        i += 1
        col += 1
    toks.append(Tok("END", None, line, col))
    return toks


# ---------------------------------------------------------------------------
# AST (mirrors exptree.hh / exptree.cc)
# ---------------------------------------------------------------------------

class Ast:
    __slots__ = ("type", "num", "id", "ch")

    def __init__(self, type, ch=None, num=0, id=""):
        self.type = type
        self.ch = ch if ch is not None else []
        self.num = num
        self.id = id


BINOPS = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "SET": ":=", "EQ": "==",
          "AND": "&&", "OR": "||", "GT": ">", "LT": "<", "GE": ">=", "LE": "<=",
          "NE": "!=", "APP": ":+"}


def ast_dump(a, out):
    t = a.type
    if t == "SEQ":
        for c in a.ch:
            ast_dump(c, out)
            out.append("\n")
    elif t == "EXP":
        ast_dump(a.ch[0], out)
    elif t in BINOPS:
        out.append("(")
        ast_dump(a.ch[0], out)
        out.append(" " + BINOPS[t] + " ")
        ast_dump(a.ch[1], out)
        out.append(")")
    elif t == "NEG":
        out.append("-")
        ast_dump(a.ch[0], out)
    elif t == "INT":
        out.append(str(a.num))
    elif t == "LST":
        out.append("[")
        for i, c in enumerate(a.ch):
            ast_dump(c, out)
            if i != len(a.ch) - 1:
                out.append(", ")
        out.append("]")
    elif t == "ID":
        out.append(a.id)
    elif t == "IND":
        ast_dump(a.ch[0], out)
        out.append("[")
        ast_dump(a.ch[1], out)
        out.append("]")
    elif t == "PRINT":
        out.append("print ")
        ast_dump(a.ch[0], out)
    elif t == "BLK":
        out.append("{\n")
        ast_dump(a.ch[0], out)
        out.append("}")
    elif t == "INP":
        out.append("input")
    elif t == "NOT":
        out.append("!")
        ast_dump(a.ch[0], out)
    elif t == "IF":
        out.append("if ")
        ast_dump(a.ch[0], out)
        out.append(" then ")
        ast_dump(a.ch[1], out)
        out.append(" else ")
        ast_dump(a.ch[2], out)
    elif t == "WHL":
        out.append("while ")
        ast_dump(a.ch[0], out)
        out.append(" do ")
        ast_dump(a.ch[1], out)
    elif t == "FOR":
        out.append("for [")
        ast_dump(a.ch[0], out)
        out.append(", ")
        ast_dump(a.ch[1], out)
        out.append(", ")
        ast_dump(a.ch[2], out)
        out.append("] do ")
        ast_dump(a.ch[3], out)
    elif t == "RNG":
        out.append("for [")
        ast_dump(a.ch[0], out)
        out.append(" in ")
        ast_dump(a.ch[1], out)
        out.append("] do ")
        ast_dump(a.ch[2], out)
    elif t == "CAL":
        ast_dump(a.ch[0], out)
        out.append("(")
        args = a.ch[1].ch
        for i, c in enumerate(args):
            ast_dump(c, out)
            if i != len(args) - 1:
                out.append(", ")
        out.append(")")
    elif t == "DEF":
        out.append("(")
        ps = a.ch[0].ch
        for i, c in enumerate(ps):
            ast_dump(c, out)
            if i != len(ps) - 1:
                out.append(", ")
        out.append(")")
        out.append(" => ")
        ast_dump(a.ch[1], out)
    elif t == "RET":
        out.append("return ")
        ast_dump(a.ch[0], out)
    elif t == "ABS":
        out.append("|")
        ast_dump(a.ch[0], out)
        out.append("|")
    elif t == "CLS":
        out.append("class ")
        ast_dump(a.ch[0], out)
    elif t == "GET":
        ast_dump(a.ch[0], out)
        out.append(".")
        ast_dump(a.ch[1], out)
    elif t == "SGET":
        out.append(".")
        ast_dump(a.ch[0], out)
    else:
        raise AssertionError(t)


def ast_to_text(a):
    out = []
    ast_dump(a, out)
    return "".join(out)


# ---------------------------------------------------------------------------
# Unparser: Ast -> source text that parses back to the same Ast (used by the
# fuzzer's printer and by the reducer).  Operands are parenthesised generously.
# An INT node may carry a spelling override in .id (e.g. "2147483648").
# ---------------------------------------------------------------------------

UNOPS_SRC = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "SET": ":=", "EQ": "=",
             "AND": "&", "OR": "|", "GT": ">", "LT": "<", "GE": ">=", "LE": "<=",
             "NE": "!=", "APP": ":+"}
_ATOMIC = ("INT", "ID", "INP", "LST", "CAL", "IND")


def _u_int(a):
    if a.id:
        return a.id
    return str(a.num if a.num >= 0 else a.num + (1 << 32))


def _u_atom(a):
    if a.type in _ATOMIC:
        return _u_exp(a)
    if a.type in UNOPS_SRC:
        return "(" + _u_bin(a) + ")"
    return "(" + _u_exp(a) + ")"


def _u_lhs(a):
    if a.type == "ID":
        return a.id
    if a.type == "IND":
        return _u_atom(a.ch[0]) + "[" + _u_exp(a.ch[1]) + "]"
    if a.type == "GET":
        return _u_atom(a.ch[0]) + "." + a.ch[1].id
    if a.type == "SGET":
        return "." + a.ch[0].id
    return _u_atom(a)


def _u_bin(a):
    if a.type == "SET":
        return _u_lhs(a.ch[0]) + " := " + _u_top(a.ch[1])
    return _u_operand(a.ch[0], False) + " " + UNOPS_SRC[a.type] + " " + _u_operand(a.ch[1], True)


def _u_operand(a, right):
    if a.type in UNOPS_SRC:
        return "(" + _u_bin(a) + ")"
    return _u_atom(a)


def _u_exp(a):
    t = a.type
    if t == "INT":
        return _u_int(a)
    if t == "ID":
        return a.id
    if t == "INP":
        return "<<"
    if t == "LST":
        return "[" + ", ".join(_u_exp(c) for c in a.ch) + "]"
    if t in UNOPS_SRC:
        return "(" + _u_bin(a) + ")"
    if t == "NEG":
        return "-" + _u_atom(a.ch[0])
    if t == "NOT":
        return "!" + _u_atom(a.ch[0])
    if t == "ABS":
        return "|" + _u_atom(a.ch[0]) + "|"
    if t == "IND":
        return _u_atom(a.ch[0]) + "[" + _u_exp(a.ch[1]) + "]"
    if t == "CAL":
        return _u_atom(a.ch[0]) + "(" + ", ".join(_u_exp(c) for c in a.ch[1].ch) + ")"
    if t == "DEF":
        return "(" + ", ".join(_u_exp(c) for c in a.ch[0].ch) + ") => " + _u_stmt(a.ch[1], 0, False)
    if t == "CLS":
        return "#" + _u_stmt(a.ch[0], 0, False)
    if t in ("GET", "SGET"):
        return _u_lhs(a)
    raise AssertionError(t)


def _u_top(a):
    """Expression in a position delimited by ; ) etc.: no outer parentheses needed."""
    if a.type in UNOPS_SRC:
        return _u_bin(a)
    return _u_exp(a)


def _ind(n):
    return "    " * n


def _u_stmt(a, ind, dangle):
    t = a.type
    if t == "SEQ":
        if not a.ch:
            return ";"
        return "{\n" + "".join(_ind(ind + 1) + _u_stmt(c, ind + 1, False) + "\n" for c in a.ch) + _ind(ind) + "}"
    if t == "EXP":
        return _u_top(a.ch[0]) + ";"
    if t == "PRINT":
        return ">> " + _u_top(a.ch[0]) + ";"
    if t == "RET":
        return "=> " + _u_top(a.ch[0]) + ";"
    if t == "BLK":
        body = a.ch[0].ch
        if not body:
            return "{ }"
        return "{\n" + "".join(_ind(ind + 1) + _u_stmt(c, ind + 1, False) + "\n" for c in body) + _ind(ind) + "}"
    if t == "IF":
        s = "(" + _u_top(a.ch[0]) + ") ? " + _u_stmt(a.ch[1], ind, True)
        e = a.ch[2]
        if e.type == "SEQ" and not e.ch:
            if dangle:
                s += " : ;"
        else:
            s += " : " + _u_stmt(e, ind, dangle)
        return s
    if t == "WHL":
        return "(" + _u_top(a.ch[0]) + ") @ " + _u_stmt(a.ch[1], ind, dangle)
    if t == "FOR":
        return ("(" + _u_stmt(a.ch[0], ind, False) + " " + _u_top(a.ch[1]) + "; " +
                _u_stmt(a.ch[2], ind, False) + ") @ " + _u_stmt(a.ch[3], ind, dangle))
    if t == "RNG":
        return "(" + a.ch[0].id + " : " + _u_top(a.ch[1]) + ") @ " + _u_stmt(a.ch[2], ind, dangle)
    return _u_top(a) + ";"


def unparse(root):
    """root: the program SEQ."""
    return "".join(_u_stmt(c, 0, False) + "\n" for c in root.ch)


# ---------------------------------------------------------------------------
# Grammar (parser.yy, same rule order) and LALR(1) construction
# ---------------------------------------------------------------------------

PREC_LEVELS = [
    ("nonassoc", ["ID"]),
    ("right", ["ASSIGN"]),
    ("nonassoc", ["EXPLIST"]),
    ("left", ["APPEND"]),
    ("nonassoc", ["@"]),
    ("right", ["?", ":"]),
    ("left", ["|"]),
    ("left", ["&"]),
    ("right", ["!"]),
    ("left", ["=", "GT", "LT", "GE", "LE", "NE"]),
    ("left", ["+", "-"]),
    ("left", ["*", "/"]),
    ("right", ["UMINUS"]),
    ("left", ["."]),
    ("left", ["[", "]"]),
    ("left", ["(", ")"]),
    ("left", ["ABS"]),
]
TOKPREC = {}
for _lvl, (_assoc, _toks) in enumerate(PREC_LEVELS):
    for _t in _toks:
        TOKPREC[_t] = (_lvl + 1, _assoc)

TERMINALS = ["END", "ASSIGN", "PRINT", "DEF", "NUMBER", "ID", "INPUT", "GE", "LE", "NE",
             "APPEND", "GT", "LT"] + list("-+*/();{}=!?:&|@,.#[]")
NONTERMS = ["$accept", "program", "sequence", "block", "stmt", "explistf", "explist",
            "idtype", "exp"]


def _bin(t):
    return lambda v: Ast(t, [v[0], v[2]])


def _lst1(v):
    return Ast("LST", [v[0]])


def _lstadd(v):
    v[0].ch.append(v[2])
    return v[0]


def _seqadd(v):
    v[0].ch.append(v[1])
    return v[0]


# (lhs, rhs, action(values)->Ast, %prec or None)
RULES = [
    ("$accept", ["program"], lambda v: v[0], None),
    ("program", ["sequence", "END"], lambda v: v[0], None),
    ("sequence", [], lambda v: Ast("SEQ"), None),
    ("sequence", ["sequence", "stmt"], _seqadd, None),
    ("block", ["{", "sequence", "}"], lambda v: Ast("BLK", [v[1]]), None),
    ("stmt", [";"], lambda v: Ast("SEQ"), None),
    ("stmt", ["exp", ";"], lambda v: Ast("EXP", [v[0]]), None),
    ("stmt", ["PRINT", "exp", ";"], lambda v: Ast("PRINT", [v[1]]), None),
    ("stmt", ["block"], lambda v: v[0], None),
    ("stmt", ["(", "exp", ")", "?", "stmt", ":", "stmt"], lambda v: Ast("IF", [v[1], v[4], v[6]]), None),
    ("stmt", ["(", "exp", ")", "?", "stmt"], lambda v: Ast("IF", [v[1], v[4], Ast("SEQ")]), None),
    ("stmt", ["(", "exp", ")", "@", "stmt"], lambda v: Ast("WHL", [v[1], v[4]]), None),
    ("stmt", ["(", "stmt", "exp", ";", "stmt", ")", "@", "stmt"],
     lambda v: Ast("FOR", [v[1], v[2], v[4], v[7]]), None),
    ("stmt", ["(", "ID", ":", "exp", ")", "@", "stmt"],
     lambda v: Ast("RNG", [Ast("ID", id=v[1]), v[3], v[6]]), None),
    ("stmt", ["DEF", "exp", ";"], lambda v: Ast("RET", [v[1]]), None),
    ("explistf", ["exp"], _lst1, "EXPLIST"),
    ("explistf", ["explistf", ",", "exp"], _lstadd, "EXPLIST"),
    ("explist", [], lambda v: Ast("LST"), "EXPLIST"),
    ("explist", ["explistf"], lambda v: v[0], "EXPLIST"),
    ("idtype", ["ID"], lambda v: Ast("ID", id=v[0]), None),
    ("idtype", ["exp", "[", "exp", "]"], lambda v: Ast("IND", [v[0], v[2]]), None),
    ("idtype", [".", "ID"], lambda v: Ast("SGET", [Ast("ID", id=v[1])]), None),
    ("idtype", ["exp", ".", "ID"], lambda v: Ast("GET", [v[0], Ast("ID", id=v[2])]), None),
    ("exp", ["idtype", "ASSIGN", "exp"], _bin("SET"), None),
    ("exp", ["exp", "APPEND", "exp"], _bin("APP"), None),
    ("exp", ["exp", "+", "exp"], _bin("ADD"), None),
    ("exp", ["exp", "-", "exp"], _bin("SUB"), None),
    ("exp", ["exp", "*", "exp"], _bin("MUL"), None),
    ("exp", ["exp", "/", "exp"], _bin("DIV"), None),
    ("exp", ["exp", "=", "exp"], _bin("EQ"), None),
    ("exp", ["exp", "GT", "exp"], _bin("GT"), None),
    ("exp", ["exp", "LT", "exp"], _bin("LT"), None),
    ("exp", ["exp", "GE", "exp"], _bin("GE"), None),
    ("exp", ["exp", "LE", "exp"], _bin("LE"), None),
    ("exp", ["exp", "NE", "exp"], _bin("NE"), None),
    ("exp", ["|", "exp", "|"], lambda v: Ast("ABS", [v[1]]), "ABS"),
    ("exp", ["exp", "|", "exp"], _bin("OR"), None),
    ("exp", ["exp", "&", "exp"], _bin("AND"), None),
    ("exp", ["!", "exp"], lambda v: Ast("NOT", [v[1]]), None),
    ("exp", ["-", "exp"], lambda v: Ast("NEG", [v[1]]), "UMINUS"),
    ("exp", ["(", "exp", ")"], lambda v: v[1], "("),
    ("exp", ["NUMBER"], lambda v: Ast("INT", num=v[0]), None),
    ("exp", ["[", "explist", "]"], lambda v: v[1], None),
    ("exp", ["idtype"], lambda v: v[0], None),
    ("exp", ["INPUT"], lambda v: Ast("INP"), None),
    ("exp", ["(", "explist", ")", "DEF", "block"], lambda v: Ast("DEF", [v[1], v[4]]), None),
    ("exp", ["(", "exp", ")", "DEF", "block"], lambda v: Ast("DEF", [Ast("LST", [v[1]]), v[4]]), None),
    ("exp", ["exp", "(", "explist", ")"], lambda v: Ast("CAL", [v[0], v[2]]), None),
    ("exp", ["#", "block"], lambda v: Ast("CLS", [v[1]]), None),
]


def rule_prec(r):
    lhs, rhs, _, pr = r
    if pr is not None:
        return TOKPREC.get(pr)
    last = None
    for s in rhs:
        if s in TERMINALS:
            last = s
    # bison: precedence of the LAST terminal of the rule (none if it has none)
    return TOKPREC.get(last) if last is not None else None


def build_lalr():
    nt = set(NONTERMS)
    prods_of = {}
    for i, r in enumerate(RULES):
        prods_of.setdefault(r[0], []).append(i)
    # FIRST / nullable
    nullable = set()
    first = {x: set() for x in NONTERMS}
    changed = True
    while changed:
        changed = False
        for lhs, rhs, _, _ in RULES:
            allnull = True
            for s in rhs:
                if s in nt:
                    before = len(first[lhs])
                    first[lhs] |= first[s]
                    if len(first[lhs]) != before:
                        changed = True
                    if s not in nullable:
                        allnull = False
                        break
                else:
                    if s not in first[lhs]:
                        first[lhs].add(s)
                        changed = True
                    allnull = False
                    break
            if allnull and lhs not in nullable:
                nullable.add(lhs)
                changed = True

    def first_seq(seq, la):
        out = set()
        for s in seq:
            if s in nt:
                out |= first[s]
                if s not in nullable:
                    return out
            else:
                out.add(s)
                return out
        out.add(la)
        return out

    def closure0(items):
        res = set(items)
        work = list(items)
        while work:
            p, d = work.pop()
            rhs = RULES[p][1]
            if d < len(rhs) and rhs[d] in nt:
                for q in prods_of[rhs[d]]:
                    it = (q, 0)
                    if it not in res:
                        res.add(it)
                        work.append(it)
        return res

    def closure1(pairs):
        # pairs: dict item -> set(la)
        res = {k: set(v) for k, v in pairs.items()}
        work = [(k, a) for k, v in pairs.items() for a in v]
        while work:
            (p, d), a = work.pop()
            rhs = RULES[p][1]
            if d < len(rhs) and rhs[d] in nt:
                f = first_seq(rhs[d + 1:], a)
                for q in prods_of[rhs[d]]:
                    it = (q, 0)
                    s = res.setdefault(it, set())
                    for b in f:
                        if b not in s:
                            s.add(b)
                            work.append((it, b))
        return res

    # LR(0) collection
    start = frozenset([(0, 0)])
    states = [start]
    index = {start: 0}
    gotos = {}
    k = 0
    while k < len(states):
        cl = closure0(states[k])
        nxt = {}
        for p, d in cl:
            rhs = RULES[p][1]
            if d < len(rhs):
                nxt.setdefault(rhs[d], set()).add((p, d + 1))
        for sym in sorted(nxt):
            ks = frozenset(nxt[sym])
            if ks not in index:
                index[ks] = len(states)
                states.append(ks)
            gotos[(k, sym)] = index[ks]
        k += 1
    # lookahead propagation
    la = {(s, it): set() for s, ks in enumerate(states) for it in ks}
    la[(0, (0, 0))].add("END")
    prop = {}
    for s, ks in enumerate(states):
        for kit in ks:
            cl = closure1({kit: {"#"}})
            for (p, d), las in cl.items():
                rhs = RULES[p][1]
                if d < len(rhs):
                    t = gotos[(s, rhs[d])]
                    tgt = (t, (p, d + 1))
                    for a in las:
                        if a == "#":
                            prop.setdefault((s, kit), []).append(tgt)
                        else:
                            la[tgt].add(a)
    changed = True
    while changed:
        changed = False
        for src, tgts in prop.items():
            sl = la[src]
            for t in tgts:
                tl = la[t]
                if not sl <= tl:
                    tl |= sl
                    changed = True
    # action table
    action = []
    gototab = []
    for s, ks in enumerate(states):
        cl = closure1({it: la[(s, it)] for it in ks})
        acts = {}
        reds = {}
        for (p, d), las in cl.items():
            rhs = RULES[p][1]
            if d == len(rhs):
                for a in las:
                    reds.setdefault(a, []).append(p)
        shifts = {}
        gt = {}
        for (ss, sym), t in gotos.items():
            if ss == s:
                if sym in nt:
                    gt[sym] = t
                else:
                    shifts[sym] = t
        for a in set(shifts) | set(reds):
            rs = sorted(reds.get(a, []))
            sh = shifts.get(a)
            err = False
            if sh is not None:
                tp = TOKPREC.get(a)
                keep = []
                for r in rs:
                    rp = rule_prec(RULES[r])
                    if sh is None or err:
                        keep.append(r)
                        continue
                    if rp is None or tp is None:
                        keep.append(r)        # unresolved: shift wins below
                        continue
                    if tp[0] > rp[0]:
                        continue              # shift
                    if tp[0] < rp[0]:
                        sh = None
                        keep.append(r)
                        continue
                    assoc = tp[1]
                    if assoc == "left":
                        sh = None
                        keep.append(r)
                    elif assoc == "right":
                        continue
                    else:
                        sh = None
                        err = True
                rs = keep
            if err and sh is None and not rs:
                acts[a] = ("e",)
                continue
            if err:
                acts[a] = ("e",)
                continue
            if sh is not None:
                acts[a] = ("s", sh)
            elif rs:
                if rs[0] == 0:
                    acts[a] = ("a",)
                else:
                    acts[a] = ("r", rs[0])
        action.append(acts)
        gototab.append(gt)
    return action, gototab


_TABLES = None


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = build_lalr()
    return _TABLES


class SyntaxErr(Exception):
    pass


def parse(toks, fname="-"):
    action, gototab = tables()
    stack = [0]
    vals = []
    pos = 0
    ntok = len(toks)
    while True:
        tok = toks[pos] if pos < ntok else toks[-1]
        st = stack[-1]
        act = action[st].get(tok.kind)
        if act is None or act[0] == "e":
            raise SyntaxErr("%s:%d.%d: syntax error, unexpected %s" % (fname, tok.line, tok.col, tok.kind))
        if act[0] == "s":
            stack.append(act[1])
            vals.append(tok.val)
            pos += 1
        elif act[0] == "r":
            lhs, rhs, fn, _ = RULES[act[1]]
            n = len(rhs)
            if n:
                args = vals[-n:]
                del vals[-n:]
                del stack[-n:]
            else:
                args = []
            vals.append(fn(args))
            stack.append(gototab[stack[-1]][lhs])
        else:  # accept
            return vals[-1]


# ---------------------------------------------------------------------------
# Resolver + closure compiler
# ---------------------------------------------------------------------------

class CompileError(Exception):
    pass


class EPQError(Exception):
    pass


class StepLimit(Exception):
    pass


class TopReturn(Exception):
    pass


class Ret(Exception):
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


DEBUG = bool(os.environ.get("PYREF_DEBUG"))


class Func:
    """A function literal site.  Identity of the object == identity of the site."""
    __slots__ = ("arity", "nslots", "body", "site")

    def __init__(self, arity, site):
        self.arity = arity
        self.site = site
        self.nslots = 0
        self.body = None


UNDEF = object()


def find_globals(a, out):
    t = a.type
    if t in ("BLK", "FOR", "DEF"):
        return
    if t == "IF":
        find_globals(a.ch[0], out)
        return
    if t == "SET":
        nm = a.ch[0].id
        if nm not in out:
            out[nm] = len(out)
    for c in a.ch:
        find_globals(c, out)


def lt(a, b):
    ta = type(a)
    tb = type(b)
    if ta is int and tb is int:
        return a < b
    if ta is list and tb is list:
        n = min(len(a), len(b))
        for i in range(n):
            if lt(a[i], b[i]):
                return True
            if lt(b[i], a[i]):
                return False
        return len(a) < len(b)
    raise EPQError("Invalid comparison")


def truthy(v):
    if type(v) is int:
        return v != 0
    return True


def fmt(v, out):
    if type(v) is int:
        out.append(str(v))
    elif type(v) is list:
        out.append("[")
        first = True
        for x in v:
            if not first:
                out.append(", ")
            first = False
            fmt(x, out)
        out.append("]")
    else:
        out.append("<function>")


class Machine:
    def __init__(self, stdin_text, step_limit, out_limit):
        self.out = []
        self.outlen = 0
        self.out_limit = out_limit
        self.inp = stdin_text
        self.ipos = 0
        self.steps = 0
        self.step_limit = step_limit
        self.globals = []

    def tick(self):
        self.steps += 1
        if self.steps > self.step_limit:
            raise StepLimit()

    def write(self, s):
        self.out.append(s)
        self.outlen += len(s)
        if self.outlen > self.out_limit:
            raise StepLimit()

    _num = re.compile(r"\s*([+-]?\d+)")

    def read_int(self):
        m = self._num.match(self.inp, self.ipos)
        if not m:
            self.ipos = len(self.inp)
            return 0
        self.ipos = m.end()
        v = int(m.group(1))
        if v > 2147483647:
            v = 2147483647
        if v < -2147483648:
            v = -2147483648
        return v


class FnComp:
    """Per-function compile state (mirrors the original Compiler)."""

    def __init__(self, gindex, func):
        self.gindex = gindex
        self.func = func
        self.local_index = {}
        self.scopes = [[]]
        self.nslots = 0

    def begin(self):
        self.scopes.append([])

    def end(self):
        for nm in self.scopes.pop():
            del self.local_index[nm]

    def define(self, name):
        s = self.nslots
        self.nslots += 1
        self.local_index[name] = s
        self.scopes[-1].append(name)
        return s


def compile_program(root):
    gindex = {}
    find_globals(root, gindex)
    main = Func(0, root)
    fc = FnComp(gindex, main)
    body = C(fc).stmt(root)
    main.nslots = fc.nslots
    main.body = body
    return main, gindex


class C:
    def __init__(self, fc):
        self.fc = fc
        self.hidden = 0

    # -- variables
    def setter(self, name):
        fc = self.fc
        if name in fc.gindex:
            g = fc.gindex[name]

            def sg(m, fr, v):
                m.globals[g] = v
            return sg
        if name in fc.local_index:
            s = fc.local_index[name]
        else:
            s = fc.define(name)

        def sl(m, fr, v):
            fr[s] = v
        return sl

    def getter(self, name):
        fc = self.fc
        if name in fc.gindex:
            g = fc.gindex[name]

            def gg(m, fr):
                v = m.globals[g]
                if v is UNDEF:
                    raise EPQError("Undefined global")
                return v
            return gg
        if name not in fc.local_index:
            raise CompileError("Undefined variable " + name)
        s = fc.local_index[name]
        return lambda m, fr: fr[s]

    # -- statements: closures f(m, fr) -> None
    def stmt(self, a):
        t = a.type
        fc = self.fc
        if t == "SEQ":
            parts = [self.stmt(c) for c in a.ch]
            if not parts:
                return lambda m, fr: None
            if len(parts) == 1:
                return parts[0]

            def seq(m, fr):
                for p in parts:
                    p(m, fr)
            return seq
        if t == "BLK":
            fc.begin()
            s = self.stmt(a.ch[0])
            fc.end()
            return s
        if t == "EXP":
            e = self.exp(a.ch[0])
            return lambda m, fr: (e(m, fr), None)[1]
        if t == "PRINT":
            e = self.exp(a.ch[0])

            def pr(m, fr):
                v = e(m, fr)
                o = [">> "]
                fmt(v, o)
                o.append("\n")
                m.write("".join(o))
            return pr
        if t == "IF":
            fc.begin()
            c = self.exp(a.ch[0])
            fc.begin()
            s1 = self.stmt(a.ch[1])
            fc.end()
            fc.begin()
            s2 = self.stmt(a.ch[2])
            fc.end()
            fc.end()

            def if_(m, fr):
                v = c(m, fr)
                if (v != 0) if type(v) is int else True:
                    s1(m, fr)
                else:
                    s2(m, fr)
            return if_
        if t == "WHL":
            fc.begin()
            c = self.exp(a.ch[0])
            fc.begin()
            b = self.stmt(a.ch[1])
            fc.end()
            fc.end()

            def wh(m, fr):
                while True:
                    v = c(m, fr)
                    if not ((v != 0) if type(v) is int else True):
                        break
                    m.tick()
                    b(m, fr)
            return wh
        if t == "FOR":
            fc.begin()
            init = self.stmt(a.ch[0])
            c = self.exp(a.ch[1])
            fc.begin()
            b = self.stmt(a.ch[3])
            fc.end()
            step = self.stmt(a.ch[2])
            fc.end()

            def fo(m, fr):
                init(m, fr)
                while True:
                    v = c(m, fr)
                    if not ((v != 0) if type(v) is int else True):
                        break
                    m.tick()
                    b(m, fr)
                    step(m, fr)
            return fo
        if t == "RNG":
            fc.begin()
            hi = fc.define("%di" % self.hidden)
            hl = fc.define("%dl" % self.hidden)
            self.hidden += 1
            le = self.exp(a.ch[1])
            name = a.ch[0].id
            setx = self.setter(name)
            fc.begin()
            setx2 = self.setter(name)
            b = self.stmt(a.ch[2])
            fc.end()
            fc.end()

            def rng(m, fr):
                fr[hi] = 0
                L = le(m, fr)
                fr[hl] = L
                setx(m, fr, 0)
                while True:
                    i = fr[hi]
                    if type(L) is int:
                        n = to_i32(abs(L))
                    elif type(L) is list:
                        n = len(L)
                    else:
                        raise EPQError("Operand must be an integer or a list")
                    if not (i < n):
                        break
                    m.tick()
                    if type(L) is not list:
                        raise EPQError("Invalid operation")
                    setx2(m, fr, L[i])
                    b(m, fr)
                    fr[hi] = i + 1
            return rng
        if t == "RET":
            e = self.exp(a.ch[0])
            top = fc.func.site.type != "DEF"     # the program itself

            if top:
                def rt(m, fr):
                    e(m, fr)
                    raise TopReturn()
            else:
                def rt(m, fr):
                    raise Ret(e(m, fr))
            return rt
        # anything else in statement position (cannot happen from the grammar)
        e = self.exp(a)
        return lambda m, fr: (e(m, fr), None)[1]

    # -- expressions: closures f(m, fr) -> value
    def exp(self, a):
        f = self._exp(a)
        if not DEBUG:
            return f

        def dbg(m, fr):
            try:
                return f(m, fr)
            except EPQError as e:
                if not hasattr(e, "node"):
                    e.node = unparse(Ast("SEQ", [Ast("EXP", [a])])).strip()
                raise
        return dbg

    def _exp(self, a):
        t = a.type
        fc = self.fc
        if t == "INT":
            v = a.num
            return lambda m, fr: v
        if t == "ID":
            return self.getter(a.id)
        if t in ("ADD", "SUB", "MUL", "DIV"):
            x = self.exp(a.ch[0])
            y = self.exp(a.ch[1])
            if t == "ADD":
                def f(m, fr):
                    p = x(m, fr)
                    q = y(m, fr)
                    if type(p) is not int or type(q) is not int:
                        raise EPQError("Operands must be integers")
                    return ((p + q + 0x80000000) & 0xFFFFFFFF) - 0x80000000
            elif t == "SUB":
                def f(m, fr):
                    p = x(m, fr)
                    q = y(m, fr)
                    if type(p) is not int or type(q) is not int:
                        raise EPQError("Operands must be integers")
                    return ((p - q + 0x80000000) & 0xFFFFFFFF) - 0x80000000
            elif t == "MUL":
                def f(m, fr):
                    p = x(m, fr)
                    q = y(m, fr)
                    if type(p) is not int or type(q) is not int:
                        raise EPQError("Operands must be integers")
                    return ((p * q + 0x80000000) & 0xFFFFFFFF) - 0x80000000
            else:
                def f(m, fr):
                    p = x(m, fr)
                    q = y(m, fr)
                    if type(q) is int and q == 0:
                        raise EPQError("Division by zero")
                    if type(p) is not int or type(q) is not int:
                        raise EPQError("Operands must be integers")
                    r = abs(p) // abs(q)
                    if (p < 0) != (q < 0):
                        r = -r
                    return to_i32(r)
            return f
        if t == "NEG":
            x = self.exp(a.ch[0])

            def neg(m, fr):
                v = x(m, fr)
                if type(v) is int:
                    return to_i32(-v)
                if type(v) is list:
                    if not v:
                        raise EPQError("List is empty")
                    return v.pop()
                raise EPQError("Operand must be an integer or a list")
            return neg
        if t == "ABS":
            x = self.exp(a.ch[0])

            def ab(m, fr):
                v = x(m, fr)
                if type(v) is int:
                    return to_i32(abs(v))
                if type(v) is list:
                    return len(v)
                raise EPQError("Operand must be an integer or a list")
            return ab
        if t in ("EQ", "NE", "LT", "GT", "LE", "GE"):
            x = self.exp(a.ch[0])
            y = self.exp(a.ch[1])
            if t == "EQ":
                return lambda m, fr: 1 if x(m, fr) == y(m, fr) else 0
            if t == "NE":
                return lambda m, fr: 0 if x(m, fr) == y(m, fr) else 1
            if t == "LT":
                def f(m, fr):
                    p = x(m, fr)
                    q = y(m, fr)
                    return 1 if lt(p, q) else 0
            elif t == "GT":
                def f(m, fr):
                    p = x(m, fr)
                    q = y(m, fr)
                    return 1 if lt(q, p) else 0
            elif t == "GE":
                def f(m, fr):
                    p = x(m, fr)
                    q = y(m, fr)
                    return 0 if lt(p, q) else 1
            else:
                def f(m, fr):
                    p = x(m, fr)
                    q = y(m, fr)
                    return 0 if lt(q, p) else 1
            return f
        if t == "NOT":
            x = self.exp(a.ch[0])

            def nt(m, fr):
                v = x(m, fr)
                if type(v) is not int:
                    raise EPQError("Operand must be an integer")
                return 1 if v == 0 else 0
            return nt
        if t == "AND":
            x = self.exp(a.ch[0])
            y = self.exp(a.ch[1])

            def an(m, fr):
                v = x(m, fr)
                if not truthy(v):
                    return v
                return y(m, fr)
            return an
        if t == "OR":
            x = self.exp(a.ch[0])
            y = self.exp(a.ch[1])

            def orr(m, fr):
                v = x(m, fr)
                if truthy(v):
                    return v
                return y(m, fr)
            return orr
        if t == "SET":
            tgt = a.ch[0]
            if tgt.type == "ID":
                e = self.exp(a.ch[1])
                s = self.setter(tgt.id)

                def st(m, fr):
                    v = e(m, fr)
                    s(m, fr, v)
                    return v
                return st
            if tgt.type == "IND":
                l = self.exp(tgt.ch[0])
                i = self.exp(tgt.ch[1])
                e = self.exp(a.ch[1])

                def si(m, fr):
                    L = l(m, fr)
                    I = i(m, fr)
                    v = e(m, fr)
                    if type(L) is not list or type(I) is not int:
                        raise EPQError("Invalid operation")
                    if I < 0 or I >= len(L):
                        raise EPQError("Index out of bounds")
                    L[I] = v
                    return v
                return si
            raise CompileError("Classes are not implemented")
        if t == "IND":
            l = self.exp(a.ch[0])
            i = self.exp(a.ch[1])

            def gi(m, fr):
                L = l(m, fr)
                I = i(m, fr)
                if type(L) is not list or type(I) is not int:
                    raise EPQError("Invalid operation")
                if I < 0 or I >= len(L):
                    raise EPQError("Index out of bounds")
                return L[I]
            return gi
        if t == "APP":
            l = self.exp(a.ch[0])
            e = self.exp(a.ch[1])

            def ap(m, fr):
                L = l(m, fr)
                v = e(m, fr)
                if type(L) is not list:
                    raise EPQError("Operand must be a list")
                L.append(v)
                return L
            return ap
        if t == "LST":
            fc.begin()
            es = [self.exp(c) for c in a.ch]
            fc.end()
            return lambda m, fr: [e(m, fr) for e in es]
        if t == "INP":
            def inp(m, fr):
                m.write("<< ")
                return m.read_int()
            return inp
        if t == "DEF":
            ps = a.ch[0].ch
            f = Func(len(ps), a)
            sub = FnComp(fc.gindex, f)
            for p in ps:
                if p.type != "ID":
                    raise CompileError("Invalid argument list")
                if p.id in fc.gindex:
                    raise CompileError("Argument shadows global variable")
                if p.id in sub.local_index:
                    raise CompileError("Duplicate argument name")
                sub.define(p.id)
            f.body = C(sub).stmt(a.ch[1])
            f.nslots = sub.nslots
            return lambda m, fr: f
        if t == "CAL":
            args = [self.exp(c) for c in a.ch[1].ch]
            callee = self.exp(a.ch[0])
            n = len(args)

            def call(m, fr):
                vals = [x(m, fr) for x in args]
                f = callee(m, fr)
                if type(f) is not Func:
                    raise EPQError("Operand must be a function")
                if f.arity != n:
                    raise EPQError("Invalid arity")
                m.tick()
                nf = vals + [0] * (f.nslots - n)
                try:
                    f.body(m, nf)
                except Ret as r:
                    return r.v
                return 0
            return call
        if t in ("CLS", "GET", "SGET"):
            raise CompileError("Classes are not implemented")
        raise CompileError("cannot compile " + t)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Result:
    __slots__ = ("stdout", "stderr", "rc", "status", "ast")

    def __init__(self, stdout, stderr, rc, status, ast=None):
        self.stdout = stdout
        self.stderr = stderr
        self.rc = rc
        self.status = status      # ok | error | compile | syntax | limit
        self.ast = ast

    def __repr__(self):
        return "Result(rc=%d, status=%s)" % (self.rc, self.status)


def run_source(text, stdin_text="", step_limit=20000000, out_limit=64 << 20, fname="-",
               dump_ast=False, parse_only=False):
    diag = []
    toks = lex(text, fname, diag)
    try:
        root = parse(toks, fname)
    except SyntaxErr as e:
        diag.append(str(e))
        return Result("", "\n".join(diag) + "\n", 1, "syntax")
    dump = ast_to_text(root) if dump_ast else None
    pre = "".join(d + "\n" for d in diag)
    if dump_ast:
        pre += dump
    if parse_only:
        return Result("", pre, 0, "ok", dump)
    try:
        main, gindex = compile_program(root)
    except CompileError as e:
        return Result("", pre + str(e) + "\n", 1, "compile", dump)
    m = Machine(stdin_text, step_limit, out_limit)
    m.globals = [UNDEF] * len(gindex)
    fr = [0] * main.nslots
    try:
        main.body(m, fr)
        rc, status, err = 0, "ok", ""
    except TopReturn:
        rc, status, err = 0, "ok", ""
    except EPQError as e:
        rc, status, err = 1, "error", str(e) + "\n"
        if DEBUG and hasattr(e, "node"):
            err += "pyref: failing expression: %s\n" % e.node
    except StepLimit:
        rc, status, err = 1, "limit", "pyref: step/output limit exceeded\n"
    except RecursionError:
        rc, status, err = 1, "limit", "pyref: recursion limit exceeded\n"
    return Result("".join(m.out), pre + err, rc, status, dump)


def _with_big_stack(fn):
    import threading
    box = []
    old_size = threading.stack_size(1 << 30)
    old_lim = sys.getrecursionlimit()
    sys.setrecursionlimit(400000)
    try:
        th = threading.Thread(target=lambda: box.append(fn()))
        th.start()
        th.join()
    finally:
        sys.setrecursionlimit(old_lim)
        threading.stack_size(old_size)
    return box[0]


def run_source_bigstack(*a, **kw):
    return _with_big_stack(lambda: run_source(*a, **kw))


def main(argv):
    dump = False
    timing = False
    parse_only = False
    fname = None
    for a in argv[1:]:
        if a == "-e":
            dump = True
        elif a == "-t":
            timing = True
        elif a == "--parse-only":
            parse_only = True
        elif a.startswith("--engine"):
            pass
        else:
            fname = a
    if fname is None:
        sys.stderr.write("No file passed\n")
        return 1
    try:
        with open(fname, "rb") as f:
            text = f.read().decode("latin-1")
    except OSError:
        sys.stderr.write("cannot open %s\n" % fname)
        return 1
    stdin_text = "" if parse_only else sys.stdin.buffer.read().decode("latin-1")
    t0 = time.perf_counter()
    r = run_source_bigstack(text, stdin_text, step_limit=1 << 62, fname=fname,
                            dump_ast=dump, parse_only=parse_only)
    t1 = time.perf_counter()
    sys.stdout.buffer.write(r.stdout.encode("latin-1"))
    sys.stdout.flush()
    sys.stderr.write(r.stderr)
    if timing and r.status not in ("syntax",):
        sys.stderr.write("Execution time: %d microseconds\n" % int((t1 - t0) * 1e6))
    return r.rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
