# tests/fuzzfound — divergences found by the differential fuzzer

Each `bugNN_<pass>_<short>.epq` is a reduced program on which an engine of ClaudeEPQ disagrees
with the baseline (`refbuild/epq_ref`). `.out` / `.code` are the baseline's stdout and exit
code (pyref agrees on all of them). Run them like any suite:

    python3 tools/fasttest.py tests/fuzzfound                                # C engine
    for f in tests/fuzzfound/*.epq; do build/epq --engine=jit $f; done      # JIT (per process)

Found with `tools/hunt.py` / `tools/fuzz.py` (profiles opt / mixed / nasty, seeds >= 900000).
Attribution by `EPQ_NO_PASS=<pass>` / `EPQ_NO_RECORDS=1` bisection.

| test | engine / pass | what | root cause (as far as visible) |
|---|---|---|---|
| bug01_jit_out_of_temporaries | JIT (src/jit), with records + demote-globals | `epq: jit internal error: out of temporaries (function f0_main)`, abort (exit 134) also with the default engine; baseline runs silently (exit 0). Gone with `EPQ_NO_RECORDS=1`, `EPQ_NO_PASS=demote-globals` or `=definite-assignment`; the C engine is correct. Fuzz seed 1228746 (mixed), and ~1.5% of `opt` programs since the `rec2/wide` shape was added. | src/jit/codegen.cpp `rec_lit`: all k fields of a `RecLit` are evaluated and converted (`word_of`) into live temporaries before the header is allocated (`talloc` at codegen.cpp:397 has no spill / redo path); a 10-field record literal of demoted main locals (statement position, left over after `litcmp` folded `[[...] != 0]`) exhausts the pool. |
| bug02_jit_out_of_temporaries_nopass | JIT, only with `EPQ_NO_PASS=constants,definite-assignment,ifconv,simplify,simplify2` | same abort while compiling main; baseline: `Index out of bounds` (exit 1). With the default pass set the JIT is correct on this program. Fuzz seed 1238260 (nasty, `jitrnd` engine). | same as bug01 (wide record literal `[[], [8 fields]]` in a condition). |

Known (already reported) signatures can be excluded from a campaign with
`tools/fuzz.py --known "out of temporaries"` / `tools/hunt.py --known ...` so that one open
bug does not flood the reducer.
