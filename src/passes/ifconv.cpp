// Pass "ifconv" — if-conversion of small conditional assignments to Select (DESIGN.md §5.5).
//
// STUB: does nothing yet. Registered in src/passes/registry.cpp; toggled with
// --no-pass=ifconv / --only-pass=ifconv / EPQ_NO_PASS=ifconv. Helpers: src/passes/util.hpp.
#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

void pass_ifconv(Module&) {}

}  // namespace epq::tir
