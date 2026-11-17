# Where this stands (17 Nov 2026)

ClaudeEPQ is a new implementation of the EPQ language (original: BoomydayCoder/EPQ). The design is in
`docs/DESIGN.md`, the language spec in `docs/LANGUAGE.md`.

## `main` — working and verified (integration of 17 Nov, evening)

- Front end, whole-program type inference with specialisation, records (fixed-shape lists with
  per-position types), typed IR (TIR) with the low-level extension, shared runtime (fast I/O,
  three-way deep compare, grow-in-place lists), **C backend**, batch test mode, background tier-up.
- **Optimisation pipeline, all ON by default** (`build/epq --list-passes`): simplify, litcmp, inline,
  recur, pairs, ifconv (scalar only; stores behind `EPQ_IFCONV_STORES=1`), appendopt, listopt,
  simplify2, reclaim. Default-off list (`kDefaultOff` in src/passes/registry.cpp, enabled with
  `--enable-pass=NAME` / `EPQ_ENABLE_PASS`): currently empty.
- Gates on main (fasttest, `--verify-tir`, -O1 and default flags, also `EPQ_POISON_FREED=1`): 389
  conformance, 50 core, 125 hunt, every `tests/passes/*`, 26 records, 22 rtio (+10 checked by
  `tests/rtio/run.py`), 14 `tests/integration/progs` (pass-interaction cases) — all pass; TIR
  self-tests 21/21; tier-up scenarios 8/8. Differential fuzz: 3,000 programs (clean/mixed/nasty) +
  1,000 with random `EPQ_NO_PASS` subsets: 0 divergences.
- Benchmarks (C engine, `-t` self time, geomean over 24): passes off 38× the original, default 58×
  (build/integ/bench1/bench.md has the table). No benchmark is slower with the passes on.
- Rules for pass authors: `util::may_mutate_header` / `may_store_elems` (src/passes/util.hpp) list every
  op; an unlisted op counts as "may do anything". `is_pure` does NOT mean "may be moved past a store":
  `Len`, `ElemLoad`, `RecGet` read mutable memory (recur's `stable_pure` is the value-only whitelist).

## Branches

| Branch | State |
|---|---|
| `wip/pass-*` (all ten), `wip/records`, `wip/runtime-io`, `wip/tierup` | merged into `main` |
| `wip/jit` | ARM64 code generator, still in progress, NOT merged; must implement every TIR op listed in `src/tir.hpp` |

## To resume

1. Baseline: the original is expected at `../reference/EPQ` (read-only clone) and the comparison binary
   at `../refbuild/epq_ref`. If they are gone: clone BoomydayCoder/EPQ and run
   `tools/build_reference.sh <clone> <outdir>`.
2. One agent per `wip/*` branch: read the branch's commits, finish, run its tests, report.
3. Merge into `main` in pipeline order (simplify, litcmp, inline, recur, pairs, ifconv, appendopt, listopt,
   reclaim), then records, runtime-io, tierup, jit. Expect conflicts in `runtime/epq_rt.h` (comparison and
   growth functions), the end of the `Op` enum in `src/tir.hpp`, and the pass registry in `src/tir.cpp`.
4. Teach the JIT every op the passes added (`Reserve`, `AppendRange`, record ops, …).
5. Then: one-byte flag arrays (DESIGN §5.12), full fuzz of both engines against the original with random
   pass subsets, fixes, final benchmark on a quiet machine, README.

## Things learned the hard way

- **Testing fast:** run C-backend test suites with `python3 tools/fasttest.py tests/conformance
  tests/core tests/hunt` (options `-k`, `--epq-arg`, `--env`, see `tools/README.md`). It pushes
  every program through one `epq --batch` (one new dylib per run instead of one per program,
  none when rerun), so it costs one macOS first-load scan instead of hundreds: ~2 s warm,
  ~20 s cold for all 564 programs. Use `tests/conformance/run.py` per process only for
  non-C engines or to test the single-program path itself.

- Agent worktrees are cut from `origin/main`: push `main` before launching agents, and tell each agent to
  `git merge main` first.
- Do not message a running workflow agent: it starts a second copy in the same folder.
- Under load the conformance runner needs `--timeout 120`; its default produces false failures.
- The first `dlopen` of a freshly compiled dylib takes 2–3 s on macOS (one-off scan per file). The tier-up
  helper must load it once itself before marking it ready.
- Known bug on `main`: with an unwritable cache directory the C backend refuses to run the program
  (`tests/hunt/runtime/bug01_unwritable_cache`).
- **Every freshly compiled binary costs 3–5 s on its first run** (macOS scans it; the scanner handles one
  at a time). C-backend tests compile one binary per program, so a 389-test run is ~30 minutes of scanner
  time and many agents testing at once starve each other. Developer Tools was switched on for Claude on
  17 Nov; it needs an app restart to take effect (unconfirmed whether it removes the delay). Until
  confirmed: prefer the JIT for testing, keep C-backend test runs small, never run many in parallel.
- The workflow tool runs at most 8 agents at once (CPUs − 2); extra agents queue silently. Use several
  workflows of ≤ 8, and never try to redirect queued agents by editing files they read — they ignore it.
- If-converting element stores made bubblesort 4.7× slower; only scalar cases are enabled.
