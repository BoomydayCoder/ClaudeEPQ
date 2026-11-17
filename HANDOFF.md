# Where this stands (paused 17 Nov 2026, ~00:50)

ClaudeEPQ is a new implementation of the EPQ language (original: BoomydayCoder/EPQ). The design is in
`docs/DESIGN.md`, the language spec in `docs/LANGUAGE.md`.

## `main` — working and verified

- Front end, whole-program type inference with specialisation, typed IR (TIR) with the low-level
  extension, shared runtime, **C backend** (`build/epq FILE`: emits C, compiles with `cc`, caches, runs).
- 389/389 conformance tests, 50/50 core tests, 125 adversarial "hunt" programs: all match the original.
- Measured with the C backend only, execution time, on a busy machine: roughly 20×–175× faster than the
  original on the benchmarks (bubblesort 175×, sieve 56×, heapsort 51×, dijkstra 42×, mergesort 21×;
  the slowest, a sort of records, 6.8×).
- **Not on `main` yet:** the JIT and every optimisation pass (stubs only in `src/passes/`). Small programs
  are therefore not faster end to end yet — the cached-code load costs more than they take to run.

## `wip/*` branches — in progress, NOT verified, NOT merged

Each was being written by one agent when the session was paused. "Snapshot" means the last commit is an
automatic save of uncommitted files and may not even compile.

| Branch | What it is | State when paused |
|---|---|---|
| `wip/jit` | ARM64 code generator (`src/jit/codegen*`) | 14 commits + snapshot; was tuning generated code |
| `wip/pass-inline` | inlining | committed + snapshot |
| `wip/pass-pairs` | pair unboxing | committed |
| `wip/pass-ifconv` | branch removal (scalar cases only) | committed |
| `wip/pass-listopt` | list-header caching, bounds-check elision | committed |
| `wip/pass-simplify` | folding, dead code | committed + snapshot |
| `wip/pass-reclaim` | compile-time freeing of dead lists | committed |
| `wip/pass-appendopt` | capacity hints, bulk copy, grow in place | committed |
| `wip/pass-litcmp` | no-allocation list comparisons, three-way compare | committed as "WIP" |
| `wip/pass-recur` | tail calls to loops, pure-call merging | committed |
| `wip/records` | per-position types for fixed-shape lists | snapshot only; early |
| `wip/tierup` | background `cc -O3` + cache for the next run | committed |
| `wip/runtime-io` | faster number printing and reading | committed + snapshot |

None of these agents delivered a final report, so each branch needs its owner to finish, test and report
before it is merged.

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

- Agent worktrees are cut from `origin/main`: push `main` before launching agents, and tell each agent to
  `git merge main` first.
- Do not message a running workflow agent: it starts a second copy in the same folder.
- Under load the conformance runner needs `--timeout 120`; its default produces false failures.
- The first `dlopen` of a freshly compiled dylib takes 2–3 s on macOS (one-off scan per file). The tier-up
  helper must load it once itself before marking it ready.
- Known bug on `main`: with an unwritable cache directory the C backend refuses to run the program
  (`tests/hunt/runtime/bug01_unwritable_cache`).
- If-converting element stores made bubblesort 4.7× slower; only scalar cases are enabled.
