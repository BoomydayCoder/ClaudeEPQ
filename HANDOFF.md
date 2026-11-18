# Status (17 Nov 2026)

ClaudeEPQ is a new implementation of the EPQ language (original: BoomydayCoder/EPQ). The user guide is
`README.md`. The design is in `docs/DESIGN.md`, the language spec in `docs/LANGUAGE.md`.

## Done, on `main`

- Front end (lexer, parser, resolver) matching the original exactly, including `-e` AST dumps.
- Whole-program type inference with call-site specialisation, records (per-position types for
  fixed-shape lists, nullable records for trees), and a typed IR (TIR) with a verifier (`--verify-tir`).
- Ten optimisation passes, **all on by default** (`build/epq --list-passes`): simplify, litcmp, inline,
  recur, pairs, ifconv (scalar bodies only; element stores behind `EPQ_IFCONV_STORES=1`), appendopt,
  listopt, simplify2, reclaim. The default-off list (`kDefaultOff` in src/passes/registry.cpp) is empty.
- Shared runtime (buffered I/O, three-way deep compare, grow-in-place lists, static reclamation).
- Two engines. The **ARM64 JIT** covers every TIR op and type. The **C engine** (`--engine=aot`) emits C,
  compiles it with cc and caches the dylib. Batch mode (`--batch`) serves the test tools.
- **Default engine policy `auto`**: the JIT. A program the JIT cannot compile falls back to the C engine,
  silently (`--verbose` says why). A tier-up artefact is used when one is ready for exactly this
  source and configuration and has been measured faster than the JIT.
- **Tier-up**: a run still going after 20 ms spawns one detached background helper that compiles the
  C version into `~/.cache/claude-epq/tier/`. Runs record their best end-to-end time in
  `<key>.stats`. After two artefact runs, an artefact that is not faster than the JIT is no longer used.
  `EPQ_NO_TIERUP=1` turns tier-up off. A cache directory that cannot be written turns it off silently.
- An unusable cache directory is never fatal: the JIT needs no cache, and the forced C engine falls
  back to a private `mkdtemp` directory (tests/cache/run.py).
- `make check`: examples, conformance and core suites, and the cache test, on the default engine.
  It needs nothing outside the repository.

## Testing

- Expected outputs come from the original (baseline binary: `tools/build_reference.sh`, or
  `EPQ_BASELINE`). The full gate is tests/conformance, core, hunt, passes, records, rtio, integration
  and jit/progs. Every program must match on both engines: C through `tools/fasttest.py`, the JIT
  per process (`--also`). Also run the gate with `EPQ_POISON_FREED=1`, with all passes off and with
  `EPQ_NO_RECORDS=1`. Then run tests/jit/engine_tests.py, tests/tierup/run.py, tests/tir/run.py,
  tests/rtio/run.py and tests/cache/run.py, plus a `tools/fuzz.py --batch` differential run.
- C-engine runs must go through the batch runner. macOS scans every new binary on its first load,
  one at a time machine-wide, so one dylib per program costs seconds each. JIT runs create no binaries.

## Known gaps

- **Byte lists** (DESIGN §5.12: one byte per element for 0–255 flag arrays) are not merged yet. The
  work is on a separate branch.
- No garbage collector. Static reclamation is the only memory reuse; a program that allocates without
  bound in a long loop can run out of memory. The original has the same limit (it never frees).
- The JIT is about 0.9× the C engine's speed on its own execution time (geomean). The main remaining
  losses are bubblesort (loads of just-stored elements), heap sift-down (no load CSE) and lcg_hash.
  Tier-up recovers these for programs that run repeatedly.
- Tier-up compares best times across runs, whatever the input. A program run with very different inputs
  can be judged on an unrepresentative sample. Delete its `.stats` file or use `--tier-clear` to reset.
