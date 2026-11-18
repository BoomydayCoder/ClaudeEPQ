# ClaudeEPQ — architecture

Goal: run EPQ programs (see `LANGUAGE.md`) as fast as an Apple-Silicon Mac allows, with output identical
to the original implementation. Target: macOS / ARM64 only. No portability constraints.

## 0. Where the time goes, and the ideas that remove it

The original is a stack bytecode VM. Every value is a 16-byte tagged union pushed and popped through
`std::vector`; every operator re-checks tags; every global is an `unordered_map` lookup; every call is a
hash lookup plus three vector pushes; every `<<` and `>>` is a flushed system call.

EPQ is dynamically typed, but almost every real program is *statically typeable*. Five ideas, in order
of how much they buy:

1. **Whole-program type inference + specialisation.** Infer types before running, clone functions per
   argument types, and generate code in which an integer is a bare 32-bit register and a list of
   integers is a bare `int32_t` array. Only values that genuinely cannot be typed carry tags. (§2)
2. **Compile in-process, in microseconds.** Half the example programs finish in ~20 ms in the original;
   starting a C compiler costs more than that. So the default engine is our own ARM64 JIT. (§6)
3. **Know what lists alias.** The inference already groups lists into regions that can never alias each
   other, and a tiny extra analysis finds lists with a single owner. That lets a loop keep a list's data
   pointer and length in registers instead of reloading them on every access. (§5.2)
4. **Pairs are not lists.** `[cost, v]` is this language's only way to write a record. A two-integer list
   that is never mutated is represented as one 64-bit register, laid out so that the language's
   lexicographic `<` is a single machine compare. No allocation, no pointer chase. (§5.4)
5. **Remove unpredictable branches.** Sorting code is dominated by 50/50 branches (~13 cycles per
   miss). Small conditional bodies are turned into conditional selects. (§5.5)

Then, for programs that run long enough to care, a second tier: emit C, let LLVM optimise it in the
background, cache the result, and use it on the next run. (§7)

```
source ─► lexer ─► parser ─► AST ─► resolver ─► inference + specialisation ─► TIR ─► TIR passes ─┬─► ARM64 JIT ─► run
                                                                                                └─► C ─► cc -O3 ─► cache ─► dlopen
```

Implementation: C++20, Apple clang, plain `Makefile`, no dependencies. Tooling: Python 3.9 stdlib.

## 1. Front end

- Hand-written lexer and precedence-climbing parser reproducing the bison grammar *including its
  conflict resolutions* (`LANGUAGE.md` §2). Checked by differential testing against the baseline's `-e`
  AST dump, which `epq -e` reproduces byte-for-byte.
- Resolver implements `LANGUAGE.md` §4 exactly. Output: every variable reference bound to `Global(g)` or
  `Local(fn, slot)`; every function literal gets a `DefId`.
- **Startup budget:** the whole pipeline up to the first executed instruction must cost well under
  300 µs for the example programs — an `epq` run of an empty program should be indistinguishable from
  `/usr/bin/true`. No iostreams, no static constructors, raw `read`/`write`, arenas mapped lazily.

## 2. Type inference

### 2.1 Types
```
T ::= Bot | Int | List(e) | Fn(d) | Any          (+ Pair, introduced by §5.4 after inference)
```
`e` is an *element variable* — a union-find node that itself has a `T`. `d` is a `DefId`.
`Bot` = no information yet. `Any` = must be a tagged dynamic value.

A **tvar** is anything with a type: each global; each local, parameter and return value of each function
*instance*; each element variable; each expression node of each instance.

### 2.2 Constraint: `flow(src → dst)`
"A value of src may be stored in dst". Solved to a fixpoint with `dst := join(dst, src)`:

| join | result |
|---|---|
| `Bot, t` | `t` |
| `Int, Int` | `Int` |
| `List(e1), List(e2)` | `List(e1)` and **unify(e1, e2)** — lists are mutable references, so both sides must agree on element representation |
| `Fn(d), Fn(d)` | `Fn(d)` |
| anything else | `Any` |

Dynamisation rule (same fixpoint): whenever `dst` is `Any`,
- `src = List(e)` ⇒ force `e := Any` (recursively, by this same rule);
- `src = Fn(d)` ⇒ mark `d` *escaped* (it needs a generic entry point, §2.5).

`src` keeps its static type: an `Int` stored into an `Any` slot is boxed at the store.

**Soundness invariant.** Every list object has exactly one element representation, fixed by the element
variable of its allocation site; every variable that can reference it either has that same (unified)
element variable or is `Any`, in which case the element variable is `Any`. So a list reachable from a
dynamic value always has dynamic elements, and generic runtime code needs no per-list layout tags.

**Regions.** Because every flow of a list unifies element variables, an element variable *is* a
Steensgaard points-to class: two list expressions with different element variables can never refer to
the same list. §5.2 uses this as free alias analysis.

### 2.3 Transfer rules
- literal, `<<`, arithmetic, comparisons, `!`, `|x|` → `Int`.
- `-x`: `x:Int` → `Int`; `x:List(e)` → flow(e → result); `x:Any` → `Any`.
- `a & b`, `a | b` in *value* position: flow(a → r), flow(b → r). In *condition* position no value exists.
- `x := v`: flow(v → x); the expression has `v`'s type.
- `l[i]`: `l:List(e)` → flow(e → r); `l:Any` → `Any`.
- `l[i] := v`, `l :+ v`: `l:List(e)` → flow(v → e); `l:Any` → flow(v → Any). `:+` has `l`'s type.
- `[a, b, …]`: fresh `e` per (site, instance); flow(each → e). Each `e` records its allocation sites and
  their literal lengths (needed by §5.4).
- function literal: `Fn(d)`.
- for-each: flow(elem(L) → x); plus flow(Int → x) only when `x` already existed.
- call `f(args)`: `f:Fn(d)` with matching arity → instance of `d` (§2.4), flow(arg_i → param_i),
  flow(ret → r); wrong arity → run-time `Invalid arity`; `f:Any` → args flow to `Any`, `r = Any`;
  `f:Int`/`List` → run-time `Operand must be a function`.
- `=> v`: flow(v → ret). Falling off the end: flow(Int → ret) if the end is reachable.
- A local whose first definition may be skipped gets flow(Int → x) and is zero-initialised.

All rules are monotone on a finite-height lattice; rules keyed on an operand's type re-run when it changes.

### 2.4 Specialisation
A function **instance** is one copy of a body with its own tvars. `swap(l, i, j)` called on a list of
ints and on a list of pairs gets two instances, each fully static.

- Each call site of each instance gets its own callee instance, except when the callee is already on
  the instantiation chain (recursion): then that instance is reused.
- Cap: more than 8 instances of one `d`, or 256 in total → further sites share one merged instance
  (always sound, only less precise).
- After the fixpoint, instances with identical resolved types (own tvars and, transitively, callees) are
  merged by partition refinement.

### 2.5 Dynamic calls
Each escaped `d` gets one generic instance (all parameters and the return are `Any`). Dynamic call sites
check the tag and the arity, then call through a table indexed by `DefId`.

### 2.6 Globals
One tvar per global, shared by all instances. On top of that:

- **Definite assignment** — forward must-analysis over the top level, extended across calls
  (`S(instance)` = ∩ over its call sites). Reads not proven assigned keep an `Undefined global` check.
- **Demotion** — a global referenced by no function body is a local of `main`, so it lives in a register.
  (All five variables of the sieve example are demoted.)
- **Constants** — a global assigned once, at top level, to a literal, and proven assigned at a read, is
  that literal.

## 3. Representation

| Static type | Machine representation |
|---|---|
| `Int` | `int32_t` in a `w` register |
| `List(e)` | pointer to a 16-byte header `{ data*, len, cap }`. `data` is `int32_t[]` when `e` is `Int`/`Bot`, 8-byte slots otherwise. Small lists are allocated with their first buffer directly behind the header (one cache line). |
| `Pair` (§5.4) | one 64-bit register: `(int64)first << 32 \| (uint32)second ^ 0x80000000`. Signed 64-bit compare ≡ lexicographic compare of the two signed ints; equality ≡ `cmp`. |
| `Fn(d)` | nothing at run time (the target is known) |
| `Any` | 64-bit tagged word, below |

**Dynamic values.** Integer `v` is `(v << 32) | 1`; a list is its (16-byte-aligned, non-zero) header
pointer; a function is `(DefId << 32) | 2`. This encoding is chosen so that the common dynamic operations
are inline and branch-light:
- both integers? `and t, a, b ; tbz t, #0, slow`
- add: `a + (b - 1)`; subtract: `(a - b) | 1`; multiply: `(a asr 32) * (b - 1) + 1` — the upper 32 bits
  wrap exactly like 32-bit arithmetic, the tag survives;
- compare two integers: plain signed 64-bit compare of the tagged words;
- truthiness: `word != 1`.

**Memory.** Bump-allocated arenas, no collector (the original never frees either; we use 4 bytes per
integer element where it uses 16, and none at all for pairs). An outgrown list buffer goes straight back
to a size-class free list, because buffers are only reachable through their header.

**Stack.** Generated code runs on its own lazily-committed multi-gigabyte stack (the entry stub switches
`sp`), so recursion depth is bounded by memory as in the original; a guard page turns runaway recursion
into a clean `Stack overflow` error.

## 4. TIR — typed intermediate representation

A tree per instance; the single input of both backends. Every node has a resolved static type and every
implicit operation of the source language is explicit:

- conversions: `BoxInt`, `BoxList`, `BoxFn`, `UnboxInt(err)`, …
- checks: `CheckIndex(i, len)`, `CheckDivisor`, `CheckDefined(g)`, `Fail(err)` (a statically certain type
  error must still fire at run time, in order, after the preceding side effects);
- typed operations: `AddI32`, `LtI32`, `LtList(desc)`, `EqList(desc)`, `LtPair`, `PairMake`,
  `PairField(k)`, `ElemLoad(data, i, width)`, `ElemStore`, `Append`, `PopBack`, `CallDirect(instance)`,
  `CallDynamic`, `PrintTyped(desc)`, `Select(c, a, b)` …
- generic operations on `Any` (`DynAdd`, `DynLess`, `DynIndex`, …) with inline integer fast paths;
- control: `If`, `While`, `For`, `ForEach` (desugared to index form), `Return`, `InlineBlock`;
- conditions in *branch form* (jump-true / jump-false, short-circuit); 0/1 is materialised only if used.

TIR has two levels. **High TIR** has list operations on list values (`Index(L, i)`, `Len(L)`).
**Low TIR**, produced by §5.2, exposes header fields as ordinary typed locals (`L.data`, `L.len`,
`L.cap`) and element accesses as `ElemLoad(dataLocal, i)` + `CheckIndex(i, lenLocal)`. Backends compile
Low TIR and know nothing about caching: a hoisted length is just another local competing for a register.

`desc` is a static type descriptor, so deep compare / equality / print of a statically typed nested list
is one runtime call that knows the layout.

## 5. TIR passes — where the speed beyond "typed code" comes from

Both engines consume the TIR after these passes. The pipeline (`epq --list-passes`), all on by
default, each switchable off with `--no-pass=NAME` / `EPQ_NO_PASS`: after the infrastructure steps
(definite-assignment, constants, demote-globals, fold, region-info) come **simplify** (§5.6),
**litcmp** (§5.9), **inline** (§5.1), **recur** (§5.10), **pairs** (§5.4), **ifconv** (§5.5),
**appendopt** (§5.8), **listopt** (§5.2 + §5.3), **simplify2** (§5.6 again, late), **reclaim** (§5.7)
and **bytes** (§5.12).
Records (§5.11) are part of inference, not a pass (`--no-records` / `EPQ_NO_RECORDS=1`).

### 5.1 Inlining
Small non-recursive instances (`swap`, `mod`) are inlined as `InlineBlock`s. Parameters that the callee
never assigns and whose arguments are variables or constants are substituted rather than copied.
Enables everything below (and turns `mod(x, 65537)` into a division by a constant).

### 5.2 Header caching — keeping `data`/`len` in registers
An access `l[i]` naively costs: load `len`, compare, load `data`, load element. The first and third are
redundant unless something changed the list's header in between. Only three things can: `:+`, unary-minus
pop, and a call that does either.

- **Mutation summaries.** Each instance gets the set of *regions* (element variables, §2.2) whose headers
  it may mutate, transitively through calls. Dynamic calls and `Any` appends mutate the `Any` region.
- **Unique lists.** A local assigned exactly once, from a list literal, and used only as the base of
  index / length / append / pop or as a return value, has no other name. Nothing but operations on that
  very variable can change its header.
- A cached header of variable `L` is invalidated by: assignment to `L`; append/pop on an expression in
  the same region (unless `L` or the other base is unique and they are different variables); a call
  whose summary contains `L`'s region (never, if `L` is unique). **Element stores never invalidate.**
- Placement: at a loop preheader when nothing in the loop invalidates (and `L` is definitely assigned
  there); otherwise after the defining assignment / invalidation point when at least two uses follow.
- Appends on a cached unique list update the cached `len` in place and store it through (one store, no
  reload); only the out-of-line grow path reloads `data`/`cap`.
- Loop-invariant reads of non-demoted globals are hoisted by the same machinery.

Result: the merge loop of mergesort, the sieve's inner loop and bubblesort's inner loop touch memory
only for the elements themselves.

### 5.3 Check elision
With lengths in locals, `CheckIndex(i, len)` is redundant when the same `(i, len)` pair was checked
earlier on every path with neither changed. Appends never invalidate an in-bounds fact (lists only grow);
pops do. After inlining `swap`, six checks become two. A loop guard `i < |L|` additionally discharges
`L[i]` when `i` is a non-negative induction variable whose only update in the loop is `i := i + 1`.

### 5.4 Pair unboxing
A region is a **pair region** when:
- every allocation site is a literal with exactly two `Int` elements;
- no variable or expression of the region is ever the base of `:+`, unary-minus pop, or an index *store*;
- it never flows into `Any`.

List identity is unobservable in EPQ except through mutation (`=` is deep), so such lists are values.
Every tvar of the region gets type `Pair` (§3): literals become `PairMake`, `p[0]`/`p[1]` become shifts,
`p[i]` with variable `i` is a range check plus select, `|p|` is `2`, `<`/`=` are single compares.
A list of pairs is a flat array of 64-bit words. In Dijkstra the heap, the adjacency lists and every
`[cost, v]` stop allocating altogether.

### 5.5 If-conversion
`If(c) { s1; …; sk }` with no else, k ≤ 3, where each `s` is an `Int`/`Pair` local assignment or an element
store whose bounds are already established before the `If`, and every right-hand side is pure, cheap
and cannot fail, becomes straight-line code: `c` is evaluated once into a temporary, `x := e` becomes
`x := Select(c, e, x)` and `A[i] := e` becomes `A[i] := Select(c, e, A[i])`. Sequential semantics are
preserved statement by statement. The language is single-threaded, so introducing the stores is legal —
something a C compiler is not allowed to assume.

**Revised after measurement.** Converting *element stores* is off by default: on bubblesort's swap it
measured 4.7× slower (6.5 → 30.5 ms). The unconditional stores put a store-to-load dependency on the
loop-carried path, and the branch they replace is one Apple's predictor already handles (once the
running maximum is being carried, the comparison goes the same way almost every time). The default
scope is scalar bodies only — heap child selection (`smallest := leftChild`), min/max updates — and each
shape is enabled only where a back-to-back measurement shows a win.

### 5.6 Local simplification
Constant folding and propagation, strength reduction (`* 2^k`, `/ 2^k`, other constant divisors by
multiply-high), reuse of an element value already loaded in the same statement sequence when no store to
that region intervenes, dead-store and dead-local removal.

### 5.7 Static reclamation — the part of garbage collection that makes code faster
A tracing collector would not speed up anything the original can run: it never frees, so every program
it can finish fits in memory, and ours needs a quarter of the space. What *does* cost time is that a
leak-everything bump allocator streams through fresh memory: every new 16 KB page is a page fault and
every new list is a cache miss, while the memory of lists that died microseconds ago sits unused.
Mergesort allocates O(n log n) elements to sort n.

So lists are freed at compile-time-known death points and their memory is reused at once, while it is
still in cache — no collector, no reference counts, no run-time cost beyond the free itself:

- **Ownership.** Within an instance, a list value has an origin: *fresh* (a literal here), *param*
  (borrowed from the caller), *call result*, or *unknown* (loaded from a list or a global).
- **Summaries.** Per instance and parameter: does it escape (stored into a list, a global or `Any`), and
  can it be returned? Per instance: is the result always fresh or one of the arguments?
- A fresh list — or a call result whose summary says "fresh or argument *i*", with argument *i* owned —
  that is never stored anywhere and is not returned on this path is dead when its variable's scope
  closes. It is freed there. Where two dead variables may alias (`sorted_left` may *be* `left`), the
  generated code compares the pointers and frees once.
- Freed headers and buffers go to size-class free lists that the allocator checks before bumping.
- Only the list itself is freed, never what its elements point to.

Guarded by a debug mode that poisons freed memory and never reuses it, so the fuzzer turns any unsound
free into an immediate wrong answer instead of silent corruption.

A conservative mark-sweep collector triggered only when the heap passes a large threshold would be an
optional backstop for programs the original could never run (unbounded allocation in a long loop). It
adds risk and no speed and is **not implemented**: static reclamation is the only memory reuse.

### 5.8–5.12 Optimisations added after profiling the first working compiler

With the type inference and C backend running, every benchmark's generated code was inspected and the
remaining costs measured, where possible by hand-editing the emitted C and timing it. That produced
five additions (measurements are from those experiments, on a loaded machine):

**5.8 Append loops (`appendopt`).** Appends dominate mergesort, quicksort and the sieve's set-up.
- *Capacity hints.* Before a counted loop that appends once per iteration, reserve the trip count.
  Capacity is unobservable, so a hint can never change behaviour.
- *Bulk copy.* `for (i := a; i < b; i := i + 1) L :+ M[i]` becomes one range check plus `memcpy`, with the
  original loop kept as the fallback whenever the range is not entirely in bounds, so the error fires at
  the same iteration as before.
- *Grow in place.* A buffer that is the most recent bump allocation is extended instead of copied, inside
  one large lazily-committed reservation. Measured: mergesort_big 47 → 23 ms with hints + bulk copy;
  5–20% from grow-in-place alone on every append-heavy program.

**5.9 Comparisons that should not allocate (`litcmp`).** `t != []` allocated an empty list on every
evaluation — 1.37 M of bintree's 1.73 M allocations. It becomes "is a list of length 0". The runtime's
deep `<` also becomes a three-way walk: the original's definition compares each equal element twice
(`x < y`, then `y < x`), which is exponential in nesting depth; a three-way walk visits the same pairs
in the same order and raises `Invalid comparison` at the same first invalid pairing.

**5.10 Recursion (`recur`).** The JIT gets none of what LLVM does for recursive code, so it is done in TIR.
- *Self tail calls become loops.*
- *Purity.* An instance that writes no global, does no I/O, mutates no list and calls only pure instances
  returns the same result (or the same failure) for the same arguments.
- *Duplicate pure calls are merged*, after unrolling a pure self-recursive function by one level:
  `fib(n-1) + fib(n-2)` becomes `fib(n-2) + fib(n-3) + fib(n-2)` and the two `fib(n-2)` calls collapse.
  Measured on recursive fib: 11.3 → 0.65 ms.

**5.11 Records — per-position types for fixed-shape lists.** *(implemented, on by default)* `[key, [a, b]]` mixes an integer
and a list, so the plain rule of §2.2 makes its elements `Any`; a sort of such records was the slowest
benchmark (6.8×). A region whose allocation sites are all literals of one length *k*, which is only ever
indexed by constants and never appended to or popped, gets one type variable *per position* instead of
one for all elements. Records stay heap objects with identity (a constant-index store is fine), so this
is a typing refinement, not unboxing; pair unboxing (§5.4) then applies to record fields that qualify.
Implemented as a second inference run in which candidate regions found by the first are records; a
region that turns out to be used otherwise is demoted and inference repeats. Measured by hand: 2.6×.
*Nullable records and pre-shaping.* `[]` meeting a record region becomes its null pointer. An empty
literal that is indexed and stored into by literal index *before* it meets its first node (a global
`tree := []` grown by `tree := insert(tree, v)`) collects those values in its single element type,
which then floods every field and fails the run. After such a failed run the empty sites involved are
*pre-shaped*: in the next run they are records of length k from the start (a per-site state, like the
literals' candidate state). A wrong guess is an ordinary record violation of that region and demotes
it, empty sites included, for good; every site moves only 0 → 1 → 2, so the scheme still terminates,
and a program never ends worse off than without pre-shaping. bintree: 2.6× faster.

**5.12 Byte lists — narrow element storage (`bytes`).** *(implemented, on by default)* When every value
ever stored into a region is provably in 0–255 (a sieve's flags, n-queens' column marks), the region is
laid out one byte per element. A 4-byte flag array for the 6 M sieve is 24 MB and misses cache on every
stride; one byte per flag measured 45 → 23 ms by hand. Implemented (min of 7, `-t`, machine load ~25,
so indicative): sieve_big 33.8 → 18.7 ms (JIT) / 32.3 → 17.2 ms (C); eratosthenes n = 2 M 8.8 → 7.5 /
10.1 → 8.5 ms; peak memory of sieve_big 25 → 7 MB. nqueens and unrelated programs unchanged.
- *Rule* (src/passes/bytes.cpp). A plain Int region (not pair, not record) qualifies iff every
  element-storing op into it — `ListLit` elements, `SetIndex` / `ElemStore` / `Append` values — stores a
  value that is provably in range: a constant 0–255; a 0/1 result (comparisons, `!`, `Eq`/`Ne`/`Cmp`,
  pair comparisons, `DynIsEmptyList`, `SameList`); `Select` / value-form `&` `|` of in-range values; the
  last value of a `Comma`; the value of an assignment or element store; an element loaded from (or
  popped off) a region that itself qualifies; an Int local all of whose assignments in its function are
  in range (parameters: unknown); an Int global all of whose assignments are in range; a call of an
  instance all of whose `Return` values are in range. Everything else (arithmetic, input, unboxing,
  record fields, ...) is not. Solved as a greatest fixpoint: start with every candidate, remove
  violators until nothing changes. An `AppendRange(l, m)` (a bitwise copy) couples l's and m's regions:
  both are byte regions or neither. Because inference gives every alias of a list the same region, and
  Int-element lists never flow into `Any`, every store into a region is one of these ops in some instance
  of the module, so the layout change is a whole-program property and sound.
- *Position*: last, after reclaim. Nothing earlier bakes in a width: `Ptr(r)` (listopt), `Reserve` /
  `AppendRange` (appendopt) and `Free` (reclaim) derive it from the region when code is emitted
  (`Module::elem_width`, the one place a width is derived), and only then is the set of stores final.
- *Representation.* `EU8` elements; in C a struct of `unsigned _BitInt(8)`, which (unlike `unsigned
  char`) is not a character type, so type-based alias analysis keeps byte stores apart from list
  headers. The JIT uses `ldrb`/`strb` (loads zero-extend). The region's LIST descriptor gets the element
  descriptor INT8 = `{INT, 1}`: as a value descriptor it is exactly INT, as an element descriptor it
  tells the runtime's equality, three-way comparison and printing to read bytes, so a byte list compares
  equal to an int list with the same values. `--dump-types` ends with the TIR layout
  (`List(L0: Int8)`); `--no-pass=bytes` restores the 4-byte layout. Tests: tests/passes/bytes
  (`check_fires.py` checks where it fires and where it must not).

## 6. ARM64 JIT (default engine)

`--engine=auto`, the default, runs the JIT; a program using a construct the JIT does not support
(none known today) runs on the C engine instead (`--verbose` says why); a ready tier-up artefact (§7)
is used instead of either when it has been measured to be faster. `--engine=jit` / `--engine=aot`
(`--jit` / `--aot`) force one engine and bypass tier-up.

The same holds for the code generator's own limits (out of temporaries, an encoding out of range, any
internal inconsistency): they throw `a64::Error` during compilation, and every function is compiled
before main starts, so nothing has run yet. `auto` then runs the C engine silently (`--verbose` says
why); `--engine=jit` prints one line and exits 2. Never an abort. `EPQ_JIT_TEMP_LIMIT=N` (test hook)
caps the live temporaries at N: 0 forces the fallback, 3 still compiles every test program by spilling.
Held values (`hold`) go to a callee-saved register or a frame slot when fewer than 6 temporaries are
free or a later sibling calls, so no expression shape (wide literals, many arguments, deep nesting)
needs more than a constant number of temporaries; tests/jit/stress.py generates such shapes.

One pass over Low TIR per instance, emitting into a `MAP_JIT` region (`pthread_jit_write_protect_np`,
`sys_icache_invalidate`).

- **Register allocation follows the language's scopes.** Local lifetimes are block-structured, so live
  ranges form a laminar family and a weighted greedy pass is optimal enough: each local has a weight
  (uses × 10^loop-depth); entering a scope hands its locals the free registers, heaviest first; leaving
  releases them; losers get frame slots. Pool: callee-saved `x19–x28`, plus caller-saved `x9–x15` inside
  regions that contain no calls except cold paths (which save and restore around the call).
  Only the callee-saved registers actually used are saved in the prologue.
- **Destination-driven expression codegen** into caller-saved temporaries; immediates folded into
  `add/sub/cmp`; scaled-register addressing (`ldr w0, [x_data, w_i, uxtw #2]`); `cmp`+`b.cond`, `cbz`,
  `tbz` fusion from branch-form conditions; `csel`/`csinc` for `Select`.
- Fast paths inline, cold paths (errors, grow, dynamic slow cases) out of line after the function body.
  One error stub per kind per function.
- Direct `bl` between instances (arguments in `x0–x7`); runtime helpers through `blr`.
- `sdiv` already yields `INT_MIN` for `INT_MIN / -1`; division by zero is a `cbz` to the error stub.
- The encoder is unit-tested by assembling the same instructions with the system assembler and comparing
  bytes.

## 7. Second tier: LLVM, in the background, for next time

- **C backend.** Emits one translation unit from the same Low TIR: the runtime plus one `static` function
  per instance, compiled with `cc -O3 -fwrapv -mcpu=native` into a dylib. Header fields and elements use
  distinct C types so LLVM's own alias analysis agrees with ours. Because it consumes TIR *after* §5, it
  gets pair unboxing and if-conversion that LLVM could never derive from C semantics.
- **Forced C engine** (`--engine=aot`): compiles in the foreground into `$EPQ_CACHE_DIR` (default
  `~/.cache/claude-epq`), `<hash of C source, cc flags, emitter version>.dylib`, reused by later runs. If
  that directory cannot be created or written it compiles in a private `mkdtemp` directory under
  `$TMPDIR` instead (removed after loading) and fails only if that fails too.
- **Tiering** (src/tierup.cpp; default engine only). A run starts in the JIT immediately. If it is still
  running after 20 ms (`EPQ_TIER_MS`), a detached helper process (`epq --tier-compile FILE`, utility QoS,
  stdio on /dev/null, at most two at once) compiles the C version, loads it once itself (so the first-load
  security scan is never paid in the foreground) and atomically installs
  `$EPQ_CACHE_DIR/tier/<key>.dylib` plus a `<key>.ready` marker. The key covers the source bytes, the
  `epq` build's UUID, the emitter version, every `EPQ_*` setting and the pass / inference flags, so an
  artefact is only used for exactly that source and configuration. A later run probes the marker (one
  `open`), `dlopen`s the artefact and skips the front end and the JIT.
- **Only while it pays.** JIT runs past the threshold and artefact runs record their best end-to-end
  time in `<key>.stats`; after two artefact runs, an artefact that is not faster than the JIT is no
  longer used (the C code is not faster for every program, and near the threshold the `dlopen` can cost
  as much as the front end saves). `EPQ_NO_TIERUP=1` disables tiering; so does a cache directory that
  cannot be written (silently). `--tier-status FILE` / `--tier-clear` inspect and empty the cache.
- Also the first backend to be written: it is the easiest way to validate the inference and TIR passes
  before the JIT exists, and JIT-vs-C disagreement pinpoints backend bugs afterwards.

## 8. Runtime

One C source shared by both engines: arenas, list growth, generic operations on `Any`,
descriptor-driven deep compare / equality / print, error exits (flush stdout, message to stderr, exit 1).

I/O is a first-class performance concern — the original flushes on every `>>` and performs one `write`
system call per `<<` prompt:
- output goes to a 64 KB buffer, flushed at exit, before an error, and before a read that would block on
  an interactive stdin; integers are formatted by a branch-light two-digits-at-a-time routine;
- input is read in large blocks and parsed by hand.

## 9. Correctness strategy

The original implementation is the oracle.

1. `tests/conformance/` — small programs per feature and per error path, expected output captured from
   the baseline.
2. `tools/fuzz.py` — random programs inside the well-defined subset (`LANGUAGE.md` §9), from cleanly
   typed to deliberately heterogeneous, compared on stdout + exit code across baseline / JIT / C.
   A separate Python reference interpreter written from the spec guards the generator itself.
3. Parser differential fuzzing on AST dumps.
4. Every TIR pass of §5 can be switched off individually (`--no-pass=…`); the fuzzer runs with random
   subsets disabled, so a wrong answer is attributed to one pass immediately.
5. The aggressive passes each rest on one stated invariant (§2.2 soundness; §5.2 invalidation rule; §5.4
   pair-region conditions; §5.5 purity + established bounds). Each gets targeted adversarial tests that
   try to break exactly that invariant: aliasing through arguments, globals and nested lists; mutation
   during iteration; pairs that are mutated in one distant place; conditionals whose body would fault.

## 10. Repository layout

```
docs/      LANGUAGE.md, DESIGN.md
src/       compiler (front end, inference, TIR, passes, backends, CLI)
runtime/   shared runtime
tests/     conformance programs + expected output, per-feature suites, `make check` (tests/check.py)
tools/     fasttest.py, difftest.py, fuzz.py, bench.py, build_reference.sh
bench/     benchmark programs and inputs
```
