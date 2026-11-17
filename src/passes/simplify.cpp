// Pass "simplify" — local simplification: folding, propagation, strength reduction, dead code; also run late as "simplify2" (DESIGN.md §5.6).
//
// STUB: does nothing yet. Registered in src/passes/registry.cpp; toggled with
// --no-pass=simplify / --only-pass=simplify / EPQ_NO_PASS=simplify. Helpers: src/passes/util.hpp.
#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

void pass_simplify(Module&) {}

}  // namespace epq::tir
