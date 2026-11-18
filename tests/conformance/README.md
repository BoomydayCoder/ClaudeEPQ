# Conformance corpus

Small, focused EPQ programs whose expected behaviour was captured from the baseline
(`epq_ref` from `tools/build_reference.sh`, the original implementation with the comparison-order fix, see
`docs/LANGUAGE.md` §9.7). Every program is inside the well-defined subset (`LANGUAGE.md` §9),
deterministic, and finishes on the baseline in well under 0.2 s.

## Layout

One directory per area:

| dir | what |
|---|---|
| `lex/` | comments, whitespace, CRLF, invalid characters, identifiers, literal wrap/saturation, multi-char tokens |
| `parse/` | precedence/associativity consequences (§2), `\|…\|` quirk, for-header forms, dangling else, syntax errors (`err_*`) |
| `scope/` | global discovery walk, locals/scopes, no shadowing, compile-time errors and their order (`err_*`) |
| `stmt/` | print format, if/while/for/for-each semantics (mutation during iteration, zeroed loop var, ints), returns |
| `expr/` | every operator on every type combination, wraparound, equality/ordering, short-circuit, evaluation order |
| `lists/` | aliasing, nesting, growth/shrinkage, heterogeneous lists |
| `functions/` | recursion (mutual, deep), higher-order, functions in lists, redefinition, identity, `poly_*` = same function used at many types |
| `errors/` | every run-time error message (`rt_*`), each after some output, incl. which of competing errors fires and side effects before it |
| `io/` | `<<` prompt and input parsing |
| `misc/` | exit codes, longer loops, mixed programs |
| `programs/` | the six original examples (`example_*`, with inputs) and realistic algorithms |

## Files

For a test `NAME`:

- `NAME.epq` — the program;
- `NAME.in` — stdin (optional; empty stdin if absent);
- `NAME.out` — expected stdout, byte for byte;
- `NAME.code` — expected exit code (one integer, 0 or 1).

stderr is never compared (error messages are checked indirectly: each error test makes sure the
program stops at the right point; the message text is in `LANGUAGE.md`).

`programs/example_bubblesort.epq` is the original `examples/sorting/bubblesort.txt` with the list size
reduced from 5000 to 1000 (the original takes ~2 s on the baseline); the other examples are verbatim.
`lex/cr_lf_line_endings.epq` and `io/input_crlf.in` deliberately contain `\r\n` line endings.

## Tools

```sh
# run an engine over the corpus (program path is appended to the command)
python3 tests/conformance/run.py -- ./path/to/engine [engine args...]
python3 tests/conformance/run.py -k lists/ -k poly_ -- ./engine     # filter by path substring
#   options: --timeout SEC (default 10), -j N (parallel jobs), -q (no diffs), --max-diff LINES

# regenerate .out/.code from a baseline binary (5 s timeout per test; refuses to record
# expectations for runs that time out or die from a signal)
python3 tests/conformance/regen.py "$EPQ_BASELINE"
python3 tests/conformance/regen.py "$EPQ_BASELINE" --check          # verify, write nothing
python3 tests/conformance/regen.py "$EPQ_BASELINE" tests/conformance/lists/*.epq
```

`run.py` against the baseline itself passes 100%.

## Adding tests

Keep each test small and inside the subset: no definition of a new local in an unclean position
(§9.1), no reads past the end of input or of malformed input, never print a function (also not
inside a list), no cyclic lists printed/compared, ≤255 list-literal elements and ≤256 constants per
function body, no deeply nested equal lists compared with `<`. Then run `regen.py` on the new file
and inspect the `.out` before committing.
