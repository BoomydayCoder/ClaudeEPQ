#!/bin/sh
# Reproducibly build the EPQ baseline ("oracle") binary.
#
# usage: tools/build_reference.sh ORIGINAL_REPO OUTPUT_DIR
#
# - copies the original sources into OUTPUT_DIR/src (the original repo is only read)
# - applies the single documented patch to the COPY of vm.cc: the two lines
#       stk.push_back(Value(pop()>pop()));
#       stk.push_back(Value(pop()<pop()));
#   are made evaluation-order-explicit (clang evaluates the two pop() calls
#   left to right, which reverses every comparison; GCC evaluates right to left)
# - compiles with the original's documented flags
#       -std=c++17 -O3 -flto -march=native -DNDEBUG
#   plus -w -Wno-register (needed by clang for the bison/flex generated files)
# - result: OUTPUT_DIR/epq_ref
#
# The pre-generated lex.yy.c / parser.tab.cc from the original are used as-is
# (exactly what the original README builds); flex/bison are not re-run.
set -eu

if [ $# -ne 2 ]; then
    echo "usage: $0 ORIGINAL_REPO OUTPUT_DIR" >&2
    exit 2
fi
SRC_REPO=$1
OUT=$2
CXX=${CXX:-clang++}

SOURCES="main.cpp compiler.cc driver.cc exptree.cc lex.yy.c parser.tab.cc program.cc value.cc vm.cc"
HEADERS="compiler.hh driver.hh exptree.hh location.hh parser.tab.hh position.hh program.hh stack.hh value.hh vm.hh"

for f in $SOURCES $HEADERS; do
    [ -f "$SRC_REPO/$f" ] || { echo "missing $SRC_REPO/$f" >&2; exit 1; }
done

# refuse to write into the original repo
src_abs=$(cd "$SRC_REPO" && pwd -P)
mkdir -p "$OUT"
out_abs=$(cd "$OUT" && pwd -P)
case "$out_abs/" in
    "$src_abs"/*) echo "refusing: OUTPUT_DIR is inside the original repo" >&2; exit 1 ;;
esac

rm -rf "$out_abs/src"
mkdir -p "$out_abs/src"
for f in $SOURCES $HEADERS; do
    cp "$src_abs/$f" "$out_abs/src/$f"
done
chmod u+w "$out_abs/src/"*

# the patch: exact string replacement, each must match exactly once
python3 - "$out_abs/src/vm.cc" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
repl = [
    ("stk.push_back(Value(pop()>pop()));",
     "{ Value rhs = pop(); Value lhs = pop(); stk.push_back(Value(lhs>rhs)); }"),
    ("stk.push_back(Value(pop()<pop()));",
     "{ Value rhs = pop(); Value lhs = pop(); stk.push_back(Value(lhs<rhs)); }"),
]
for old, new in repl:
    n = s.count(old)
    if n != 1:
        sys.exit("patch: expected exactly one occurrence of %r, found %d" % (old, n))
    s = s.replace(old, new)
open(p, "w").write(s)
EOF

cd "$out_abs/src"
"$CXX" $SOURCES -std=c++17 -O3 -flto -march=native -DNDEBUG -w -Wno-register -o "$out_abs/epq_ref"
echo "built $out_abs/epq_ref"
