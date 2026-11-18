# The EPQ language — behavioural specification

This describes *what programs mean*, independent of how any implementation runs them.
It was derived from the original implementation (BoomydayCoder/EPQ) and is the contract
ClaudeEPQ must satisfy: for every program in the **well-defined subset** (§9), stdout and
the exit code must be byte-identical to the original.

Every statement below has been checked by experiment against the baseline binary
(`epq_ref`, built with `tools/build_reference.sh`); `tests/conformance/` holds the corresponding test programs.

## 1. Lexical structure

- Whitespace: space, tab, newline only. Any other character not listed below — including `\r`, `\v`,
  `\f`, `_` at the start of a word, `"`, `$`, `%`, `~` — is reported on stderr as
  `FILE:LINE.COL: invalid character`, then *skipped*; lexing continues and the program still runs if it
  parses (`a$b` lexes as the two identifiers `a` `b`). ClaudeEPQ skips `\r` silently; the stderr text
  is not part of the contract.
- Comments: `//` to end of line.
- Identifiers: `[a-zA-Z][a-zA-Z_0-9]*` (case-sensitive; no reserved words).
- Integer literals: `[0-9]+` (no sign; `-5` is unary minus applied to 5; leading zeros are decimal,
  `007` is 7). Value: the digits are parsed with 64-bit `strtol`, which **saturates** at
  2^63−1, and the result is truncated to its low 32 bits. So `2147483648` → −2147483648,
  `4294967297` → 1, and every literal ≥ 2^63 (e.g. `99999999999999999999`) → −1.
  A literal directly followed by letters (`1a`) is two tokens (and a syntax error).
- Multi-character tokens (longest match wins): `:=` ASSIGN, `>>` PRINT, `<<` INPUT, `>=` GE, `<=` LE,
  `!=` NE, `:+` APPEND, `=>` DEF. Consequences: `a<<b` and `3>>2` are syntax errors (INPUT/PRINT
  tokens), `a<-1` is `a < -1`, `a<=>b` is `a <= > b`.
- Single-character tokens: `- + * / ( ) ; { } = ! ? : & | @ , . # [ ] > <`.

## 2. Grammar

The authoritative grammar is the LALR(1) grammar below with its precedence declarations.
Conflicts are resolved exactly as bison resolves them (precedence first, otherwise shift).

Precedence, lowest to highest:

```
nonassoc ID
right    ASSIGN            :=
nonassoc EXPLIST
left     APPEND            :+
nonassoc '@'
right    '?' ':'
left     '|'               logical or
left     '&'               logical and
right    '!'               logical not   (binds LOOSER than comparisons)
left     '=' GT LT GE LE NE
left     '+' '-'
left     '*' '/'
right    UMINUS            unary minus
left     '.'
left     '[' ']'
left     '(' ')'
left     ABS               |x|
```

```
program  : sequence END
sequence : /* empty */ | sequence stmt
block    : '{' sequence '}'

stmt : ';'                                       empty statement
     | exp ';'                                   expression statement
     | PRINT exp ';'                             >> e;
     | block
     | '(' exp ')' '?' stmt ':' stmt             if / else
     | '(' exp ')' '?' stmt                      if
     | '(' exp ')' '@' stmt                      while
     | '(' stmt exp ';' stmt ')' '@' stmt        for (init-stmt cond; step-stmt) body
     | '(' ID ':' exp ')' '@' stmt               for-each over a list
     | DEF exp ';'                               return   (=> e;)

explistf : exp | explistf ',' exp                (both %prec EXPLIST)
explist  : /* empty */ | explistf

idtype : ID | exp '[' exp ']' | '.' ID | exp '.' ID

exp : idtype ASSIGN exp
    | exp APPEND exp
    | exp '+' exp | exp '-' exp | exp '*' exp | exp '/' exp
    | exp '=' exp | exp GT exp | exp LT exp | exp GE exp | exp LE exp | exp NE exp
    | '|' exp '|'            %prec ABS
    | exp '|' exp | exp '&' exp
    | '!' exp
    | '-' exp                %prec UMINUS
    | '(' exp ')'
    | NUMBER
    | '[' explist ']'
    | idtype
    | INPUT
    | '(' explist ')' DEF block          function literal, any arity
    | '(' exp ')' DEF block              function literal, one parameter
    | exp '(' explist ')'                call
    | '#' block                          class literal (front end only — see §8)
```

Consequences worth knowing (all follow from the table above, all confirmed):

- `!a = b` is `!(a = b)`; `!a + b` is `!(a + b)`; `!a & b` is `(!a) & b`; `!a :+ b` is `(!a) :+ b`;
  `!a = b & c` is `(!(a = b)) & c`.
- `-a * b` is `(-a) * b`; `-a[0]` is `-(a[0])`; `-f(x)` is `-(f(x))`; `- -a` and `!!a` nest.
- Comparisons are one level, left-associative: `a < b < c` is `(a < b) < c`, `a = b < c` is `(a = b) < c`.
- `&` binds tighter than `|`: `a | b & c` is `a | (b & c)`.
- `:+` is left-associative and below everything except `:=`: `a :+ b :+ c` is `(a :+ b) :+ c`,
  `a :+ b + c` / `a :+ b = c` / `a :+ b | c` append the whole right side, `a | b :+ c` is `(a | b) :+ c`.
- An assignment can appear as the right operand of any binary operator and extends as far right as
  possible: `x := y :+ 3` is `x := (y :+ 3)`; `a + x := 3` is `a + (x := 3)`; `a * l[0] := 5` is
  `a * (l[0] := 5)`; `a :+ x := 5` is `a :+ (x := 5)`; `-x := 3` is `-(x := 3)`; `:=` is right-associative.
  Assignment targets are only `ID` and `e[e]` (`f(x) := 1`, `(x) := 1` are syntax errors); `f()[0] := 1` is fine.
- Postfix `[ ]` and `( )` chain on any expression: `f(1)(2)`, `l[0][1]`, `[1,2][0]`, `(f)(1)`,
  `(x) => {…}(5)` (a literal called immediately).
- `?`/`@` forms are statements only and need the condition parenthesised. `else` binds to the nearest `if`
  (`(a)?(b)?s1:s2` gives `s2` to the inner `if`). Each branch is a single statement: `(c) ? x := 1; : y := 2;`.
- The for-loop header contains two full *statements* (each with its own `;`) around the condition:
  `(i := 0; i < n; i := i + 1;) @ body`. Either statement may be any statement: `;` (empty), a block
  (`(; i < 3; { i := i + 1; }) @ …`), a print, an `if`, even `=> e;`.
- A function body must be a block; `=>` (return) requires an expression; no trailing commas in lists,
  calls or parameter lists; `(a, b);` is a syntax error (a parenthesised list must be followed by `=>`).
- `|…|` quirk: inside `| e |`, the second bar may also be read as logical-or. After `'|' exp '|'` the
  next token decides. If it can *follow* an expression and carries a precedence (`|`, `-`, `(`, `[`, `.`),
  ABS wins and the abs is closed: `|a|-b` is `(|a|) - b`, `|a|(b)` calls `abs(a)`, `|a| | |b|` is
  `(|a|) | (|b|)`. If it can only *start* an expression (`ID`, `NUMBER`, `!`, `<<`, `#`), the bar was
  a logical-or: `|a | b|` is `abs(a | b)`, `|x|1|` is `abs(x | 1)`, `|a|!b|` is `abs(a | !b)`,
  `|a|b|c|` is `abs((a|b)|c)`. Hence `|a|-b|`, `|a||b|`, `|a|b|` are syntax errors; `||a||` is `abs(abs(a))`.

Syntax error: `FILE:LOC: syntax error, unexpected …` on stderr, exit code 1, nothing on stdout
(the program does not run at all).

## 3. Values

| Type | Notes |
|---|---|
| integer | 32-bit two's complement. Overflow wraps. |
| list | mutable, growable, heterogeneous, **reference** semantics (assignment, argument passing, list elements, `-l`, `l[i]`, `:+`'s result all alias). |
| function | created by a function literal; identity is the literal's source site (two evaluations of the same literal — even in different calls of an enclosing function — are equal; two textually identical literals are not). No closures. |

There is no null/unit value: everything "empty" (a function falling off its end, a fresh for-each
variable) is the integer `0`. Lists are never freed by the original (no garbage collection).

## 4. Scoping

All of §4 is decided at compile time, over the whole program, before anything runs.

### 4.1 Globals
A name is **global** iff it is the target of an `ID := …` assignment reachable from the top-level
statement list by a walk that
- does not enter `{ … }` blocks, `for` statements (`( ; ; ) @`, including their init), or function literals,
- for `if`, enters only the condition (not the branches),
- enters everything else (while condition and body, for-each list expression and body, call arguments,
  list literals, all sub-expressions).

A for-each loop variable is **not** an assignment target for this walk: `(x : l) @ …` at top level does not
make `x` global (it becomes a local of the statement, invisible to functions and after the loop) unless
`x := …` appears elsewhere in the walk. Indexed assignments `l[i] := …` never create globals.
A body statement of while/for-each that is not a block is walked: `(c) @ w := 1;` makes `w` global,
`(c) @ { w := 1; }` does not.

Globals are visible everywhere, including inside every function, and are *late bound*: reading a global
before any assignment to it has executed is the run-time error `Undefined global` (e.g. a function that
reads `g`, called before `g := …` runs).
(The original allows at most 256 globals; ClaudeEPQ has no limit.)

### 4.2 Locals
Every other name is local to the function (or to the top-level program, which acts as a function with
no parameters) in which it appears. Rules, per function:

- Assigning to a name: global → sets the global (also inside functions). Else if a local of that name is
  currently in scope → sets it. Else **defines** a new local in the innermost open scope.
- Reading a name: global → global read. Else local in scope → read. Else **compile-time error**
  `Undefined variable NAME` (stderr, exit 1, nothing runs — even if the code is unreachable, and even
  if a definition appears later in the same scope or earlier in a previous loop iteration).
- There is no shadowing: an inner scope assigning an outer local's name updates the outer local.
- A local dies when the scope that defined it closes.
- Functions cannot see the locals of any enclosing function or of the top level. No closures. In
  particular a function stored in a *local* cannot call itself by name (`{ f := (n) => { => f(n); }; }`
  is `Undefined variable f`); recursion goes through globals.

Scopes are opened by:
- a block `{ … }`;
- a list literal `[ … ]` (a name first assigned inside the brackets dies at the `]`; see §9.1);
- `if`: one scope around the whole statement (covers the condition), plus one per branch;
- `while`: one around the statement (covers the condition), plus one for the body;
- `for`: one around the statement (covers init, condition, step), plus one for the body;
- for-each: one around the statement (covers the loop variable if it is new), plus one for the body.

Function parameters are locals of the function's outermost scope. The body is a block.

Compile-time errors for function literals, checked parameter by parameter left to right (first failure
wins): `Invalid argument list` (a parameter is not a plain identifier), `Argument shadows global variable`
(the parameter is a global name — anywhere in the program, even assigned later), `Duplicate argument name`.

Compile-time errors are reported in compilation order, which is evaluation order (§6): for a call the
arguments before the callee (`u(v)` with both undefined reports `v`), for `l[i] := e` first `l`, then `i`,
then `e`; for for-each the list expression, then the body. Only the first error is reported.

## 5. Statements

- `exp ;` evaluates and discards (`<<;` reads and discards a number, still printing the prompt).
- `>> exp ;` writes `>> `, the value, `\n` to stdout. Integers in decimal (with `-` if negative).
  Lists as `[a, b, c]` (`, ` separators, recursive, `[]` when empty): `[[], [[]], [1, []]]`.
- `(c) ? s1 : s2` / `(c) ? s1` — truthiness: an integer is true iff non-zero; **a list or function is
  always true, even an empty list**.
- `(c) @ s` — while (same truthiness; `([]) @ …` loops until something returns).
- `(init cond; step) @ s` — init once; loop while cond is true: body, then step. A local defined by init
  dies after the loop; an existing variable assigned by init keeps its final value.
- `(x : e) @ s` — for-each. Precisely:
  1. hidden index `i := 0`; hidden `L := e` (evaluated once; a later reassignment of the variable that
     held the list does not affect `L`, but in-place mutation of the list does);
  2. `x := 0` using the normal assignment rule (so a pre-existing global/local `x` is overwritten with 0,
     and stays 0 if no iteration runs; otherwise a new local is defined in the statement's scope);
  3. while `i < |L|` (the `|…|` operator of §6, re-evaluated every iteration):
  4. `x := L[i]` (indexing errors apply), run body, `i := i + 1`.

  Consequences: the body may grow or shrink the list and the loop follows it; assigning `x` in the body
  does not affect the iteration; after the loop `x` (if it outlives the loop) holds the last element.
  `L` a function → `Operand must be an integer or a list` at the first test (after step 2).
  `L` an integer `n`: if `|n| > 0` the first `x := L[0]` fails with `Invalid operation` (before the body
  runs); if `n = 0` or `n = −2^31` (whose `|n|` is negative) the loop runs zero times, no error.
- `=> exp ;` — return. Inside a function, returns the value (from any depth of loops/blocks). At top
  level (including inside top-level blocks and loops), **ends the program successfully** (exit 0) without
  printing. A function that falls off the end returns integer `0`.

## 6. Expressions

Operands are evaluated left to right, and *all* operands of an operator are evaluated (with their side
effects and prints) before the operator checks types. The one exception to left-to-right is the call,
which evaluates its arguments before the callee. Run-time errors print the quoted message plus newline to
stderr and exit with code 1; everything printed before the error still appears on stdout.

| Form | Meaning | Errors |
|---|---|---|
| `a + b`, `a - b`, `a * b` | 32-bit wrapping arithmetic | either operand not an integer → `Operands must be integers` |
| `a / b` | truncating division (toward zero: `-7 / 2` = −3) | if `b` is integer 0 → `Division by zero` (checked first, even if `a` is not an integer; `0 / 0` too); otherwise non-integers → `Operands must be integers`. `INT_MIN / -1` wraps to `INT_MIN`. |
| `-a` | integer: wrapping negate (`-INT_MIN` = `INT_MIN`). **List: removes and returns the last element** (in place; the element itself, aliased if it is a list). | empty list → `List is empty`; function → `Operand must be an integer or a list` |
| `\|a\|` | integer: absolute value (`\|INT_MIN\|` = `INT_MIN`). **List: its length** (top level only). | function → `Operand must be an integer or a list` |
| `a = b` | 1/0. Different types → 0. Integers by value; lists deeply, element-wise (different lengths → 0); functions by identity. | none |
| `a != b` | `!(a = b)` | none |
| `a < b`, `a > b` | integers by value; lists lexicographically (recursively, using `<` on elements; a proper prefix is smaller). `a > b` is `b < a`. | any other pairing (int/list, anything with a function, including `f < f`), at any depth actually reached → `Invalid comparison` |
| `a >= b`, `a <= b` | `!(a < b)`, `!(a > b)` | as above |
| `!a` | 1 if `a` is 0 else 0 | non-integer (list or function) → `Operand must be an integer` |
| `a & b` | short-circuit: if `a` is false the result is `a` (i.e. 0) and `b` is not evaluated, else the result is `b` (any type: `[] & 5` is 5) | — |
| `a \| b` | short-circuit: if `a` is true the result is `a` (any type: `[] \| 5` is `[]`) and `b` is not evaluated, else `b` | — |
| `x := e` | assign (see §4); value is `e` | — |
| `l[i] := e` | evaluates `l`, `i`, `e` in that order, then checks, stores; value is `e` | `l` not a list or `i` not an integer → `Invalid operation`; `i < 0` or `i ≥ \|l\|` → `Index out of bounds` |
| `l[i]` | element (evaluates `l` then `i`) | same two errors |
| `l :+ e` | evaluates `l` then `e`; appends `e` to `l` in place; **value is the list `l`** (so `l :+ 1 :+ 2` appends twice) | `l` not a list → `Operand must be a list` |
| `[e1, …, en]` | fresh list, elements evaluated left to right (original: at most 255 elements) | — |
| `<<` | writes `<< ` to stdout (flushed), then reads one integer from stdin (§7) | see §9 for EOF/garbage |
| `(p1, …, pn) => { … }` | function value | — |
| `f(a1, …, an)` | evaluates the arguments left to right, **then** the callee expression; calls | callee not a function → `Operand must be a function` (checked before arity); wrong argument count → `Invalid arity` |
| integer literal | — | — |

Lexicographic `<` on lists is exactly `std::lexicographical_compare`: walk both lists in step; at each
position, if `x < y` return true; if `y < x` return false; when one runs out, return whether the left
ran out first (and the right did not). Each `<` on elements can itself raise `Invalid comparison`, so
mixed-type elements only fail when reached: `[1, f] < [2, f]` is 1, `[3] < [2, []]` is 0, but
`[1, 5] < [1, []]` and `[[]] < [1]` fail. Note that equal elements are compared twice (`x<y`, then
`y<x`); see §9.5 for what that costs on the original.

## 7. Program-level behaviour

- stdout is the concatenation of every `>> ` line and every `<< ` prompt, in execution order.
  With input `10`, `n := <<; >> n;` prints exactly `<< >> 10\n`. A prompt is printed even when the read
  then fails.
- Input format (C++ `cin >> int`): skip whitespace (space, `\t`, `\n`, `\v`, `\f`, `\r`), then an optional
  `+` or `-`, then decimal digits; reading stops at the first non-digit, which is left for the next read.
  So `-5 +7\n\n  12` gives −5, 7, 12; several numbers per line are fine; `007` is 7; `3-4` gives 3 then −4.
  Anything else (EOF, no digits, out of range) is outside the subset (§9.2).
- Exit code 0 on normal completion (end of program or top-level return); 1 on any error (syntax,
  compile-time, run-time). An unreadable file prints `cannot open FILE` and exits 1.
- Command line: `epq [-t] FILE`. `-t` prints `Execution time: N microseconds` to stderr. (The original also
  has debugging flags such as `-e`, which dumps the parsed AST to stderr; they are not part of the contract.)

## 8. Unimplemented in the original

`# block`, `exp . ID`, `. ID` parse but have no semantics in the original (its compiler emits nothing
for them, corrupting its stack; e.g. `>> .b;` prints whatever is below the stack). ClaudeEPQ rejects them
at compile time with `Classes are not implemented`.

## 9. The well-defined subset

ClaudeEPQ matches the original exactly on programs that avoid the following, where the original's
behaviour is an accident of its stack machine rather than a language rule. In each case ClaudeEPQ does
the obvious sensible thing instead. Compile-time scoping (§4: which names are visible where, and the
compile-time errors) is well-defined for *all* programs; only run-time values are affected below.

1. **Defining a new local anywhere other than a "clean" position.** The original's locals live at
   stack positions computed at compile time, which is only right when, at the moment the defining
   assignment executes, no temporaries of the enclosing expression are pending on the stack, and the
   defining code runs exactly once per entry of the scope it defines into. Clean (in the subset):
   an expression statement `x := e;` whose direct parent is a statement list, a branch/loop body, or a
   for-loop init; chained `x := y := e;`; `>> x := e;`; and the condition of an `if` when the
   assignment is its outermost expression. Unclean (excluded): a first assignment nested inside a larger
   expression with something evaluated before it (`a + (x := 1)`, `g(1, x := 1)` — observed: `x` then
   reads the wrong slot), the right operand of `&`/`|` (may not run), anywhere inside a list literal
   (the literal's scope exit pops the wrong values: `[y := 3, 4]` evaluates to `[3, 3]`), a `while`/`for`
   condition or a `for` step (re-runs each iteration: the variable keeps its first value and the stack
   leaks). ClaudeEPQ treats all of these as ordinary definitions in the enclosing scope.
2. **Reading input at EOF, from non-numeric text, or out of the 32-bit range.** At EOF (or after any
   earlier failure) the original leaves the destination uninitialised (in practice it repeats a previous
   value); text without digits (`abc`, `- 4`, `+-5`, or the `x1F` left over after `0x1F` reads 0)
   reads 0 and puts the stream into a failed state; out-of-range numbers saturate to 2^31−1 / −2^31 and
   also fail the stream. ClaudeEPQ yields 0 for every failed read.
3. **Printing a function**, alone or inside a list (the original dumps its internal bytecode, spanning
   several lines). ClaudeEPQ prints `<function>`.
4. **Exceeding the original's size limits**: 256 globals (`Too many global variables`; the first top-level
   `l[i] := …` also silently uses one slot), 255 list-literal elements (`Too many elements in list
   initialiser`), 255 parameters/arguments (`Too many arguments`), 256 constants per function body or
   top level (each integer-literal occurrence, each function literal directly inside it, **and each
   for-each statement** count one; overflow silently wraps to the wrong constant — e.g. a for-each whose
   `1` wraps to a `0` loops forever), 65535-byte jumps (`Jump too far` / `Loop too far`), 256 local slots
   live at once in one function (silently wraps). ClaudeEPQ has no such limits.
5. **Crashing depth / cycles.** Cyclic lists (`l :+ l`, `l[0] := l`) printed or compared with `=`/`<`
   (no identity shortcut, even for `l = l`) recurse forever and crash. Lists nested about 10^6 deep crash
   when printed or compared (10^5 is fine); source expressions nested deeper than a few thousand levels
   may crash the original's compiler (50000 unary minuses do). Comparing with `<` two lists that share a
   long equal nested prefix costs 2^depth on the original (depth 28 takes ~1 s, 40 would take hours), so
   tests keep such nesting shallow. Unbounded recursion exhausts memory.
   Note: *bounded* deep recursion is well-defined and cheap on the original (the VM keeps no native stack
   per call): recursion depth 10^7 completes in under a second, so ClaudeEPQ must support very deep
   recursion (at least 10^6).
6. **Classes** (§8).
7. Building the original with a compiler that evaluates `pop() < pop()` left to right (e.g. clang)
   reverses every `<`/`>`. The baseline used for comparison is the original with those two lines made
   order-explicit; nothing else is changed.
8. **Truthiness of lists/functions and `/` by a list** are decided in the original by reading the low 32 bits
   of a pointer as an integer; they are "true" / "not zero" except with negligible probability. ClaudeEPQ
   treats lists and functions as always true and never as a zero divisor.
