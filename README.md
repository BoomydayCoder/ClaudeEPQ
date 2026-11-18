# ClaudeEPQ

A fast implementation of the EPQ programming language. The original implementation, a stack-bytecode
virtual machine, is [BoomydayCoder/EPQ](https://github.com/BoomydayCoder/EPQ). ClaudeEPQ runs the same
programs and produces byte-identical output and exit codes. Instead of interpreting bytecode, it infers
static types for the whole program and compiles it to native ARM64 code.

"Fast" here means less wall-clock time from starting `epq` to the program finishing, compared with
the original on the same machine and inputs.

## Benchmarks

Measured on an Apple M5 (macOS 26) with `tools/bench.py`, 17 Nov 2026: the original (built as its README
says, with the one evaluation-order fix that `tools/build_reference.sh` applies) against
ClaudeEPQ's default engine (the JIT) and its C backend. Every output was checked against the original's.
Best of 7–15 runs. The machine was in ordinary use but quiet (1-minute load about 3).

Two numbers per engine: **run** is execution time only (`-t`), **total** is wall-clock from starting the
process to its exit, as timed by the benchmark tool; about 2 ms of that is process spawn that no
implementation can avoid.

The six example programs from the original repository:

| Program | Original run | JIT run | Speedup | Original total | JIT total | Speedup |
|---|---|---|---|---|---|---|
| bubblesort (5,000) | 1,395 ms | 18.1 ms | 77× | 1,400 ms | 22.5 ms | 62× |
| dijkstra (20,000 vertices, 100,000 edges) | 619 ms | 8.1 ms | 77× | 630 ms | 12.6 ms | 50× |
| eratosthenes (n = 2,000,000) | 388 ms | 5.5 ms | 71× | 394 ms | 9.5 ms | 42× |
| heapsort (5,000) | 13.6 ms | 0.26 ms | 52× | 16.7 ms | 4.4 ms | 3.8× |
| mergesort (5,000) | 10.6 ms | 0.41 ms | 26× | 14.2 ms | 4.4 ms | 3.3× |
| fibonacci (n = 40, iterative) | 0.02 ms | 0.01 ms | – | 2.8 ms | 2.9 ms | 1.0× |

The last three finish in under a millisecond once compiled, so their totals are almost entirely process
start-up; iterative fibonacci is too short to speed up, and end to end it is level with the original (a
run of ClaudeEPQ costs roughly 0.3 ms more to start than the original does).

Across all 24 benchmarks (the six above plus 18 larger programs in `bench/extra`):

| | JIT (default) | C backend (`--engine=aot`) |
|---|---|---|
| Execution time, geometric mean over 22 programs | 48× (range 19×–120×) | 53× |
| Total time, geometric mean over 23 programs | 25× | 26× |

Those means leave out two programs so they are not flattering: iterative fibonacci (too short to
measure) and recursive fibonacci, where merging duplicate calls of a pure function changes the amount of
work and gives about 15,000× (427 ms → 0.03 ms). The C backend figures are with its compiled code already
cached; its first run of a program costs a C compile on top. The slowest relative results are the
recursion-heavy `tak` (19×) and the pair-record sort (24×).

The full per-program table is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Reproduce with
`tools/build_reference.sh` and `python3 tools/bench.py` (see "How it is tested").

## Requirements

- An Apple Silicon Mac (arm64) running macOS 26.
- Xcode Command Line Tools (`xcode-select --install`): clang, make, python3.

Nothing else: no packages, no downloads.

## Build and run

    make                         # builds build/epq
    make check                   # quick self-test (a few seconds)
    build/epq program.epq        # run a program; it reads its input from stdin
    build/epq -t program.epq     # also print "Execution time: N microseconds" to stderr, like the original
    build/epq -h                 # all options

`-e` prints the parsed program in the original's format. `--parse-only` only checks syntax.

## Engines

By default (`--engine=auto`), `epq` compiles the program in-process with its own ARM64 JIT and runs it
immediately. The JIT handles the whole language. If a program ever used a construct it did not support,
`epq` would fall back to the C engine without saying so (`--verbose` says why). The C engine
(`--engine=aot`) emits C, compiles it with the system `cc -O3` and caches the result. A program that is
still running after 20 ms starts a one-off background compile of its C version. The result is cached
under `~/.cache/claude-epq` (or `$EPQ_CACHE_DIR`). Later runs of the same source with the same settings
use the cached version, but only while it measures faster than the JIT. Set `EPQ_NO_TIERUP=1` to turn
the background compile off; `build/epq --tier-clear` empties that cache. `--engine=jit` and
`--engine=aot` force one engine.

## How it works

Front end: a hand-written lexer and parser that accept exactly the original's grammar, then a resolver
for its scoping rules. Next comes whole-program **type inference with specialisation**. Each function
is cloned for the argument types it is called with, so integers become bare 32-bit values and lists of
integers become plain arrays. Only values that really mix types carry run-time tags. Fixed-shape lists
such as `[key, [a, b]]` get per-position types (records), and trees built from them become nullable
records. The program becomes a typed IR (TIR), optimised by these passes, all on by default:

- **simplify**: constant folding and propagation, strength reduction, dead-code removal (run twice; the second run is **simplify2**).
- **litcmp**: comparisons with list literals (`t != []`) without allocating the literal.
- **inline**: inlines small non-recursive functions.
- **recur**: self tail calls become loops; duplicate calls of pure recursive functions are merged.
- **pairs**: two-integer lists that are never mutated become one 64-bit value.
- **ifconv**: small conditional assignments become branch-free selects.
- **appendopt**: append loops get capacity reservations; copy loops become bulk copies.
- **listopt**: keeps list lengths and data pointers in registers; removes redundant bounds checks.
- **reclaim**: frees lists at points where the compiler proves they are dead (static reclamation).
- **bytes**: lists whose stored values are all provably 0–255 (flags, marks) use one byte per element.

The JIT then emits ARM64 code directly. The C engine compiles the same optimised TIR. Details:
[docs/DESIGN.md](docs/DESIGN.md) (architecture) and [docs/LANGUAGE.md](docs/LANGUAGE.md) (the language as
the original implements it, established by experiment).

## How it is tested

The original implementation is the oracle. Every expected output in the repository was produced by
the original (built with `tools/build_reference.sh`).

- `make check` needs nothing outside the repository. It runs the original's six example programs, the
  conformance suite (389 small programs, one feature or error path each) and the core suite on the
  default engine, and compares them with the committed expected outputs.
- `tests/` also has suites for each optimisation pass, records, run-time I/O, inference corner cases,
  the JIT and tier-up. `python3 tools/fasttest.py SUITE_DIR...` runs any of them on the C engine in a
  single batch (`--also "build/epq --engine=jit"` checks the JIT too).
- Differential testing against the original needs its binary. Build it from a clone of the original:

      git clone https://github.com/BoomydayCoder/EPQ ../EPQ
      tools/build_reference.sh ../EPQ build/tools/ref        # -> build/tools/ref/epq_ref (or set EPQ_BASELINE)

  Then `tools/difftest.py DIR...` compares any programs across the original, the JIT and the C engine.
  `tools/fuzz.py --batch --seed 1 --count 2000` generates random programs inside the well-defined subset and
  compares them the same way (with a reducer for failures). `tools/parsefuzz.py` fuzzes the parser
  against the original's AST dump. `tools/bench.py` produces the benchmark table. See
  [tools/README.md](tools/README.md).

## Intentional differences from the original

ClaudeEPQ matches the original on every program in the well-defined subset
([LANGUAGE.md §9](docs/LANGUAGE.md#9-the-well-defined-subset)). Outside it, the original's behaviour
is an accident of its implementation, and ClaudeEPQ does the obvious thing instead:

- Defining a variable inside a larger expression (`a + (x := 1)`, inside a list literal, in a loop
  condition) is an ordinary definition. The original reads the wrong stack slot.
- A failed `<<` read (EOF, non-numeric text, out of 32-bit range) yields 0.
- Printing a function prints `<function>` instead of the original's internal bytecode.
- None of the original's size limits apply (256 globals, 255 list elements or arguments, 256 constants
  per function, jump distances).
- Lists and functions are always true and never a zero divisor. The original decides this from the low
  32 bits of a pointer.
- Class syntax (`# block`, `.ID`) parses in the original but does nothing (and corrupts its stack);
  ClaudeEPQ rejects it with `Classes are not implemented`.

Also different by design:

- **No garbage collector.** Memory is reused only through static reclamation: the compiler frees lists
  where it can prove they are dead. The original never frees anything, so every program it can finish
  also fits in memory here (and needs less).
- **Mac-only.** Apple Silicon and macOS only; no other platform is supported or tested.

## Repository layout

    src/          compiler: lexer, parser, resolver, inference, TIR + passes (src/passes/),
                  C backend (cgen*), ARM64 JIT (src/jit/), tier-up, command line
    runtime/      the run-time library shared by both engines (also embedded in the generated C)
    tests/        test suites with expected outputs; tests/check.py is `make check`
    tools/        differential runner, fuzzers, batch test runner, benchmark script, baseline builder
    bench/        benchmark programs: original/ (the original's six examples, with inputs and
                  expected outputs) and extra/
    docs/         DESIGN.md (architecture), LANGUAGE.md (behavioural specification)
    HANDOFF.md    current status and known gaps
