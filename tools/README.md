# ClaudeEPQ test & benchmark tools

Everything here is Python 3.9 standard library (plus one POSIX shell script). No installs.
All tools run engines in their own process group with a timeout (the whole group is
SIGKILLed on timeout), cap captured output, and run in parallel across all cores
where that does not distort results (`bench.py` is deliberately serial).

Scratch output goes to `build/` (git-ignored): `build/difftest/`, `build/fuzz/`,
`build/parsefuzz/`, `build/bench/`, `build/fasttest/`.

## fasttest.py — run test suites on the C backend fast (use this first)

    python3 tools/fasttest.py tests/conformance tests/core tests/hunt
    python3 tools/fasttest.py -k lists/ --epq-arg=--verify-tir --env EPQ_NO_PASS=listopt tests/conformance
    python3 tools/fasttest.py [--epq PATH] [--epq-arg ARG ...] [--env K=V ...] [-k SUBSTR ...]
                              [-j N] [--timeout SEC] [--max-diff LINES] [-q] [--keep] [--also CMD] DIR...

**Why.** macOS security-scans every newly created binary on its first exec/dlopen
(~200 ms or more each, strictly one at a time for the whole machine, however many cores).
The per-process runners (`tests/conformance/run.py`, `difftest.py`) make the C backend
create one new dylib per program: ~560 scans for conformance+core+hunt, and every agent
on the machine waits in the same queue. `fasttest.py` creates **one** dylib per run, and
**none** when the same set of programs is run again.

**What it does.** Collects `X.epq` (recursively; `X.in` = stdin, expected `X.out` bytes and
`X.code`, default 0; tests without `X.out` are skipped), writes a manifest to
`build/fasttest/<run-id>/manifest.tsv`, runs `build/epq --engine=aot [ARGS] --batch MANIFEST`
once, compares stdout + exit code, prints `FAIL path` with a short unified diff and the
first stderr lines, then `N/M passed, K failed`. Exit 0 iff all pass. Output files of
failures stay in `build/fasttest/<run-id>/` (`--keep`: all of them). `--epq-arg` options and
`--env` variables apply to every program exactly as on a normal command line and are part
of the cache key. `--also CMD` additionally runs every test through CMD per process (e.g.
`--also "build/epq --engine=jit"`, which creates no new binaries) and checks it too.

Measured (564 programs of conformance+core+hunt, 10 cores): cold (nothing cached) 21.5 s;
objects cached, new link 2.0 s; fully warm 1.6 s. The old per-process runner on the same
set pays one scan per uncached program (minutes, more when other agents are scanning too).

**`epq --batch MANIFEST [-j N] [--timeout SEC]`** (src/batch.cpp) is the engine underneath and
can be used directly: one test per line, five TAB-separated fields
`PROGRAM  STDIN-or-'-'  STDOUT-FILE  STDERR-FILE  EXITCODE-FILE`. Per program it runs the
normal pipeline in a child of `epq` itself (compile-time errors give exactly the normal
stderr and exit code 1), compiles the C with `cc -c` into `$EPQ_CACHE_DIR/<key>.o` (same key
as the dylib cache; parallel; reused), links all objects into one
`$EPQ_CACHE_DIR/batch-<set-hash>.dylib` (reused for an identical object set), dlopens it once
and runs every program in a forked child (no exec) with the normal run path (big stack,
stack-overflow message, `-t`). Timeout ⇒ exit code 124 and `TIMEOUT` on stderr; killed by a
signal ⇒ 128+signal and a note on stderr. A one-line summary goes to stderr. Verified
byte-identical (stdout, stderr, exit code) to single-program runs on all 564 programs.

**When to use the old runners instead:** for engines that are not the C backend (the
baseline, the JIT — they create no new binaries, so per-process is fine), to test the
single-program path itself (cache handling, `--verbose`, `dlopen` errors, unwritable cache
directory), or for programs that depend on being the only program in their process image.
`--verbose` output differs in batch mode (no per-program cc / cache lines).

## Engines

Every tool takes engines as repeatable `--engine NAME=COMMAND` options. `COMMAND`
is shell-split (quote paths with spaces: `--engine "b='/path with space/epq_ref'"`);
a bare existing path is taken as one argument. Flags and the program file are
appended. **The first engine is the reference**; an engine named `baseline` is
always moved to the front.

Default (no `--engine`): `baseline` = `$EPQ_BASELINE`, else `../refbuild/epq_ref`
next to the repo, else `build/tools/ref/epq_ref`; plus `jit` = `build/epq --engine=jit`
and `aot` = `build/epq --engine=aot` if `build/epq` exists.

Helpers:

* `tools/pyref.py` — reference interpreter (below); usable as an engine:
  `--engine "pyref=python3 tools/pyref.py"` (or `--pyref` on difftest/parsefuzz/fuzz).
* `tools/fake_engine.py MODE REAL_CMD...` — deliberately broken wrapper used to prove
  the tools detect differences. Modes (comma-separated): `digit` (last `7`→`8` in
  stdout), `exit` (flip exit code when stdout has an odd number of lines), `hang`,
  `crash` (SIGSEGV), `ast` (first `||`→`&&` in the AST dump), `parse` (spurious
  syntax error when a line has two `|`), `slow`, `none`.

## build_reference.sh — the baseline binary

    tools/build_reference.sh ../reference/EPQ build/tools/ref      # -> build/tools/ref/epq_ref

Copies the original sources into `OUT/src` (refuses to write inside the original
repo), applies the single documented patch to the copy of `vm.cc` (the two
`pop()>pop()` / `pop()<pop()` lines made evaluation-order-explicit; each must match
exactly once), and builds with the original's flags
`-std=c++17 -O3 -flto -march=native -DNDEBUG -w -Wno-register`. The pre-generated
`lex.yy.c`/`parser.tab.cc` are used as-is. Verified: the produced sources are
identical to `refbuild/src` and the binary is byte-identical to `refbuild/epq_ref`.

## difftest.py — differential runner

    tools/difftest.py tests/ bench/ [--engine ...] [--timeout 10] [--jobs 10]
                      [--stderr none|first-line|full] [--out build/difftest/failures] [--json FILE] [-v]

Runs every program (`*.epq`, `*.txt`; stdin = sibling `foo.in`, else /dev/null) on
every engine, compares stdout bytes + exit code (stderr reported; compared only with
`--stderr`). Reference timeout/crash ⇒ "skipped (invalid test)", other engines are
not run. Another engine timing out (limit: max(timeout, 10×reference+1 s)), crashing
or exceeding the output cap where the reference finished ⇒ failure. Failing cases
are saved with the program, input, every engine's stdout/stderr and a `report.json`.
Exit 1 on any mismatch.

`--batch` (also on `fuzz.py`): every C-backend engine (an engine whose command runs
`build/epq` without `--engine=jit`, also behind `env K=V …` or `fake_engine.py`) gets all
programs through `epq --batch` (`--batch-chunk N` programs per invocation, default 300;
one new dylib per chunk and engine configuration, none on a rerun) instead of one process
and one new dylib per program. The reference, the JIT and pyref still run per process.
Verdicts, saved cases and the reducer are unchanged (the reducer runs its candidates per
process). Per-program wall times of batch engines are reported as 0. Use it whenever an
`aot` engine is in the list:

    tools/fuzz.py --batch --seed 1 --count 5000 --engine baseline=../refbuild/epq_ref \
        --engine "aot=build/epq --engine=aot" --engine "aot-nolist=env EPQ_NO_PASS=listopt build/epq --engine=aot"

## fuzz.py — random-program differential fuzzer (+ reducer)

    tools/fuzz.py --seed 1 --count 20000 --profile any [--engine ...] [--pyref] [--jobs 10]
    tools/fuzz.py --self-check --count 4000 --profile nasty          # baseline + pyref statistics
    tools/fuzz.py --reduce build/fuzz/failures/fail_seed123 [--engine ...] [--pyref]
    tools/fuzz.py --reduce CASE --suspect                             # keep a pyref/baseline disagreement
    tools/fuzz.py --emit DIR --seed 1 --count 100                     # just write programs + .in files

Generator: `tools/epqgen.py` (see its docstring for the invariants). Programs stay in
the well-defined subset (LANGUAGE.md §9): locals are introduced only at clean
positions, scopes are tracked exactly as §4 (including the global-discovery walk),
a small kind system (int / list-of-kind / dynamic level-k / function-of-rank) keeps
operations mostly valid, lists are acyclic by construction (elements always have a
lower nesting level), functions are never printed, every loop is counted (counters
the body never assigns), every append inside a loop or function is guarded by a
global append budget (`zb`) so for-each over a growing list terminates, functions
only call lower-ranked functions plus self-recursion through a never-assigned depth
parameter `dp`, `<<` reads are counted and the `.in` file provides twice as many
numbers, and a cost model bounds work. Limits: ≤165 constants per function/top
level (checked after generation, regenerated otherwise), list literals ≤ 40, ≤ ~150
globals.

Profiles: `clean` (ints, int lists, lists of int lists), `mixed`, `nasty`
(variables of dynamic kind that change type, heterogeneous lists, polymorphic
functions, functions in lists / as arguments / returned / compared / stored in
redefinable global slots `h*` and redefined, aliasing through params/globals/nested
lists then observed through another alias, for-each over lists grown/shrunk in the
body, `:+` chains, short-circuit side effects, top-level return, globals read
before their definition, deep-ish recursion, 32-bit overflow, `INT_MIN` literals,
`INT_MIN / -1`, wrapping and saturating literals); `any` picks per program. A
per-program "loose" flag (4–12%) enables error-prone operations (indexing values of
unknown kind, unguarded pops, raw indices, dynamic calls); in addition each
program has ~15–20% chance of one injected error of a uniformly chosen kind (all 12
run-time errors and 4 compile errors). Every program prints a lot (`>>` sprinkled
everywhere plus a final dump of all printable globals).

Verdicts: `FAIL` (an engine disagrees with the baseline), `SUSPECT` (only with
`--pyref`: pyref disagrees with the baseline ⇒ the generator left the subset, the
spec is wrong, or pyref is wrong), `INVALID` (baseline timeout/crash ⇒ generator
bug). Failing cases go to `build/fuzz/failures/<verdict>_seed<N>/` (`prog.epq`,
`prog.in`, `<engine>.stdout/.stderr`, `info.json`) and are automatically reduced to
`prog.min.epq` (`--no-reduce`, `--max-reduce`, `--reduce-time`).

Reducer: parses the program with pyref's exact parser and works on the AST:
ddmin-style deletion of statement chunks in every statement list, replacing compound
statements by their bodies, replacing expressions by `0`/`1`/a subexpression,
deleting list-literal elements; repeated to a fixpoint or the time budget. A
candidate is kept iff the baseline still finishes within the timeout AND the same
engines still disagree with it AND (when the original failure had pyref agreeing
with the baseline) pyref still agrees with the baseline — the last condition keeps
the reduced program inside the well-defined subset.

`--self-check` runs N programs on the baseline and pyref only and prints the outcome
distribution (clean exit / each run-time error / compile errors / timeouts), the
pyref-vs-baseline agreement rate, sizes, baseline time distribution, injected-error
counts and feature coverage; suspects/invalid cases are saved (`--reduce-suspects K`
reduces them).

## pyref.py — reference interpreter

    python3 tools/pyref.py [-e] [-t] [--parse-only] FILE < input
    PYREF_DEBUG=1 python3 tools/pyref.py FILE      # also print the failing expression on a run-time error

Implements LANGUAGE.md directly as a closure-compiling tree walker. The parser is an
LALR(1) automaton built at start-up (0.2 s) from the grammar of §2 with bison's
conflict resolution (rule precedence = precedence of the rule's last terminal or
`%prec`; unresolved S/R → shift; R/R → earlier rule), so it accepts exactly the
original's language and `-e` reproduces the original AST dump byte for byte
(6000/6000 parsefuzz cases agree). Semantics outside the subset follow ClaudeEPQ's
§9 choices (EOF input → 0, functions print as `<function>`, classes rejected).
API: `pyref.run_source(text, stdin_text, step_limit)`, `pyref.unparse(ast)`.

## parsefuzz.py — parser differential fuzzer

    tools/parsefuzz.py --seed 1 --count 5000 [--mode wrapped|top|both] [--engine ...] [--pyref]
    tools/parsefuzz.py --check-extent --count 300       # validate the dump-extraction rule

Grammar-based generation biased to the tricky spots (abs bars vs or, `!` vs
comparisons, unary minus, nested `:=`, `:+`, `(` at statement start, for headers,
dangling else, call/index chains, `=>` return vs function literal, `.`/`#` class
syntax, empty statements, comments, random spacing, rare invalid characters),
then 0–3 random token insertions/deletions/swaps/duplications (~57% of cases still
parse). Compares parse success and the AST dump. The reference (baseline) runs
`-e FILE` — it executes the program after dumping, so cases are either *wrapped*
(code inside a never-called function literal: `zz := () => { … };`) or *top* (top
level, terminating constructs only, short timeout; reference timeout ⇒ skip). Other
engines run `-e --parse-only FILE`. Dump extent in the baseline's stderr: leading
`FILE:L.C: invalid character` lines are lexer diagnostics (printed during parsing,
before the dump), a final known compile/run-time message line is not part of the
dump, everything in between is the dump; `--check-extent` checks this against a
second run without `-e` (93/93 agree). Failures saved to `build/parsefuzz/failures/`
and reduced by line/word ddmin.

## bench/ and bench.py

    bench/original/   the six original examples, unchanged (fibonacci.in = 40,
                      eratosthenes.in = 2000000, dijkstra.in = 20000 vertices /
                      100000 edges from bench/gen_dijkstra_input.py)
    bench/extra/      18 additional programs, each tuned to 0.4–2 s on the baseline
    python3 bench/gen_dijkstra_input.py --vertices 20000 --edges 100000 --seed 12345 > bench/original/dijkstra.in

    tools/bench.py [--engine ...] [--bench 'extra/*'] [--runs N | --budget 6 --min-runs 3 --max-runs 25]
                   [--quick] [--out build/bench]

Per benchmark: one verification run per engine (stdout through a pipe, fully
drained, + exit code compared with the baseline), then N timed runs (stdout to
/dev/null, stdin from the `.in`), alternating engines; N adaptive = budget / first
run time clamped to [min, max]. Reports wall min/median, the engine's self-reported
`Execution time` (`-t`), speedups vs the baseline and geometric means; spawn overhead
from `/usr/bin/true`. Writes `bench.md` + `bench.json`. Exit 1 if any engine's output
mismatched.

Every extra benchmark starts with `N := …;` so it can be scaled down; all were
checked against pyref at small N (identical output) and checked for accidental
global/local name collisions.
