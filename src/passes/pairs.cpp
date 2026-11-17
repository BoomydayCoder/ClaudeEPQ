// Pass "pairs" — pair unboxing: two-int immutable list regions become Pair (DESIGN.md §5.4).
//
// STUB: does nothing yet. Registered in src/passes/registry.cpp; toggled with
// --no-pass=pairs / --only-pass=pairs / EPQ_NO_PASS=pairs. Helpers: src/passes/util.hpp.
#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

void pass_pairs(Module&) {}

}  // namespace epq::tir
