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

Default (no `--engine`): `baseline` = `$EPQ_BASELINE`, else `build/tools/ref/epq_ref`
(the default output of `build_reference.sh`; a warning says how to get it when neither
exists); plus `jit` = `build/epq --engine=jit`
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

    git clone https://github.com/BoomydayCoder/EPQ ../EPQ      # the original (read-only use)
    tools/build_reference.sh ../EPQ build/tools/ref              # -> build/tools/ref/epq_ref

Copies the original sources into `OUT/src` (refuses to write inside the original
repo), applies the single documented patch to the copy of `vm.cc` (the two
`pop()>pop()` / `pop()<pop()` lines made evaluation-order-explicit; each must match
exactly once), and builds with the original's flags
`-std=c++17 -O3 -flto -march=native -DNDEBUG -w -Wno-register`. The pre-generated
`lex.yy.c`/`parser.tab.cc` are used as-is. Verified: the binary is byte-identical to
the baseline the test expectations were recorded with.

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

    tools/fuzz.py --batch --seed 1 --count 5000 --engine "baseline=$EPQ_BASELINE" \
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
`INT_MIN / -1`, wrapping and saturating literals); `any` picks per program (clean/mixed/nasty; `opt` only when asked for, see below). A
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

**Pitfall in a git worktree:** `../refbuild/epq_ref` "next to the repo" does not exist
there, and the default engine list then silently starts with `jit` (so `--self-check`
compares pyref with the JIT, and every program costs a C compile when the JIT falls
back). Always `export EPQ_BASELINE=$ROOT/refbuild/epq_ref` (the first line of output
names the engines; check it says `baseline=`).

`--known SUBSTR` (repeatable; env `EPQ_FUZZ_KNOWN`, `|`-separated): a divergence whose
stderr contains SUBSTR is printed as `KNOWN …` and not counted as a FAIL (keeps one
open, already reported bug from flooding a campaign). In `--batch` mode a batch child
that times out is re-run once per process (`RECHECK seed …` line) before it counts:
on a heavily loaded machine (load 100+) forked batch children are sometimes starved.

### Profile `opt` — optimisation-targeted programs (tools/optgen.py)

The kind-based generator almost never produces the shapes the optimisation passes look
for (measured with optcov.py below: recur, ifconv, appendopt and nullable records fired
in < 1 % of clean/mixed/nasty programs, and a re-introduced copy of the recur × records
CSE bug found by the integrator was hit by 0 of 600 nasty programs). `--profile opt`
builds every program from 4–7 *snippets* of tools/optgen.py, each a self-contained piece
of top-level code (names `O<tag><n>`, unique per program, so they never collide with each
other or with epqgen's lower-case names), one family each:

| family | shapes |
|---|---|
| rec | fixed-shape records `[int, list]`, `[int, [int,int]]`, `[int, list, int]`, `[int,int,int]` from literals and maker functions; mutated through aliases (other variable, parameter, element of an outer list, global in a function) and read back in the same expression (`sq(r[0]) + (r2[0] := 4) + sq(r[0])` with `sq` inlinable or recursive); compared, printed, in lists, for-each over a list of records |
| tree / cons | `[v, l, r]` trees and `[v, next]` cons lists with `[]` as nil, built recursively (BST insert) and iteratively, walked by while loops, tail-recursive / doubly recursive sums, reverse, destructive append at the end, mutation through aliases |
| rec2 | record-typing stress: function fields, two shapes through one helper, nullable flips (`N := []` then a record again), input-dependent shape, mutually nested shapes, records popped out of lists, int-terminated chains, empty literals grown by appends (pre-shaped records), `MK()` empties, wide (8–16 field) heterogeneous literals in statement / condition position |
| recloop | lists in record fields: indexed loops, appends, pops, field replacement and copy loops through the field |
| pairs | lists of `[int, int]`: insertion sort by pair, binary heap of pairs (push / pop with sift), adjacency lists, min pair; pairs vs lists of other lengths / literals, a record holding a pair, variable index into a pair |
| helpers / inl2 | swap / min / max / mod / predicate helpers called in loops (bubble sort, reductions); helpers assigning their parameters, argument evaluation order with side effects, callees mutating list arguments, tail-recursive functions swapping two list arguments |
| recur | fib (two self calls), binomial, gcd and accumulator tail recursion, argument swaps `=> f(b, a, n - 1)`, impure look-alikes (global read that changes in the same expression, printing), pure functions that fail for some inputs, tail recursion with local lists (escaping into the accumulator or dropped) |
| append / copy / grow | counted loops with 1–3 appends per iteration (`<`, `<=`, step 1–3, bounds from input), early return; copy loops `L :+ M[i]` with in-bounds / empty / reversed / partly out-of-bounds ranges, `L` aliasing `M` (lengths tracked statically so a re-evaluated bound never follows a growing list), loop variable used afterwards, in a function; merge loops; hundreds to thousands of elements with interleaved growth, self-copies whose buffer moves, temps freed in between, pop then regrow |
| idxloop / alias | indexed loops over a list mutated in the loop through an alias or a callee (inlinable or recursive, so the loop sees a real call), reassigned, popped; lists aliased through `:+` results, `|`, `&`, `[L][0]`, record fields, `-[L]`, for-each variables, chained / conditional assignment; error-order forms (index / stored value with side effects or failures) |
| ifconv | conditional min / max / counters / guarded index / division in loops, `&` chains whose first operand guards the second, heap sift-down child selection, established-bounds near misses (index the condition did / did not evaluate, a second shorter list, inlined swaps, while loops) |
| litcmp | `x = []`, `x != [0, 1]`, `[] < x`, `x >= [1]`, … on ints, int lists, nested lists, a dynamic value, a function, in loops over heterogeneous lists and through a function parameter; literal elements that may fail |
| reclaim | short-lived lists in self-recursive (never inlined) functions: dropped, returned, aliased, returned through identity / pick / `\|` / `&` / `:+` / record field / pop, stored in a global, into a list parameter, into an element; a recursive driver; allocations after each call so that a premature free is observed (run also with `EPQ_POISON_FREED=1`) |
| glob | single-literal globals (constants) used in functions and loops, globals defined after the function that reads them, main-only globals (demotable); far-away reassignment / type change / read before definition |
| glue | values exported by earlier snippets (int lists, records, pairs, trees, cons lists) passed through shared polymorphic helpers (first / len / copy by for-each / reverse by variable index / store at 0), boxed together, compared |

Perturbation: about 40 % of the rec / tree / cons / pairs / recloop / glob / rec2 snippets
add a *tail* statement that breaks the optimisation's precondition — variable index, append,
pop, for-each, dynamic call, store of another type, reassignment — and all tails run after
every snippet's main part, i.e. far away from the code that was optimised; glue does the
same through generic functions. Risky snippets (that may stop the program with a run-time
error depending on the input) are at most one per program and placed last. Inputs are
small numbers; every read is clamped right away, so epqgen's huge inputs are fine too.

`mixed` / `nasty` now also get 1–2 snippets (35 % / 45 % of programs) inserted at random
top-level positions (own RNG stream, so the epqgen part of a seed's program is unchanged;
`clean` is unchanged). Self-check (1000 `opt` programs): ~95 % clean exits, ~5 % run-time
errors (all from risky snippets), 0 timeouts, pyref agrees 100 %.

## optcov.py — which optimisations fire on generated programs

    tools/optcov.py --profile opt --profile nasty --count 1000 --seed 910000 [--jobs J]
    tools/optcov.py --dir tests/passes          # any directory of .epq files
    tools/optcov.py --profile opt --show recur_unroll   # seeds where it fires / does not

Front end only (no C compiler): for each program `--dump-types` (records: fixed /
nullable regions) and `--dump-tir-after=P` for the whole pipeline; a pass fires when its
dump differs from the previous pass's; refined by op counts (Reserve / AppendRange,
Select added by ifconv, Free added by reclaim, Pair ops, While added by recur = tail-call
loop, InlineBlock under a Comma added by recur = self-unrolling; recur_cse = recur changed
something else). `demoted` = some global is demoted; `constants` = `--no-pass=constants`
changes the TIR. Prints profile × feature = % of compiling programs where it fires.

## hunt.py — campaign driver

    tools/hunt.py --seed 940000 --chunks 200 --chunk 150 [--cflags -O0] [--profiles opt,opt,nasty,opt,mixed]
    tools/hunt.py --seed 1300000 --chunks 200 --chunk 1000 --jit-only build/epq --known "out of temporaries"

Runs fuzz.py chunk after chunk (logging one line per chunk to OUT/log.txt). C mode
(`--batch`): `aot` (default passes, `--verify-tir`), `poison` (EPQ_POISON_FREED=1: a
run-time switch, so it reuses aot's objects for free), and on alternating chunks `norec`
(EPQ_NO_RECORDS=1) and `rnd` (EPQ_NO_PASS = a random 1–5 of the 10 passes + 4 build
steps, logged). JIT mode (`--jit-only BIN`, per process, no C compiles, ~20–40 programs/s):
`jit` (`--verify-tir`), `jitnorec`, `jitpois`, `jitrnd`, `jitx` (EPQ_IFCONV_STORES=1 plus
`--mono` or `--no-dedup` at random).

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
