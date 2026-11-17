// Pass "reclaim" — static reclamation: Free at compile-time-known death points (DESIGN.md §5.7).
//
// STUB: does nothing yet. Registered in src/passes/registry.cpp; toggled with
// --no-pass=reclaim / --only-pass=reclaim / EPQ_NO_PASS=reclaim. Helpers: src/passes/util.hpp.
#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

void pass_reclaim(Module&) {}

}  // namespace epq::tir
