// Pass "inline" — inlining of small non-recursive instances as InlineBlocks (DESIGN.md §5.1).
//
// STUB: does nothing yet. Registered in src/passes/registry.cpp; toggled with
// --no-pass=inline / --only-pass=inline / EPQ_NO_PASS=inline. Helpers: src/passes/util.hpp.
#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

void pass_inline(Module&) {}

}  // namespace epq::tir
