// Pass "listopt" — header caching (ListData/ElemLoad/CheckIndex) and check elision (DESIGN.md §5.2/§5.3).
//
// STUB: does nothing yet. Registered in src/passes/registry.cpp; toggled with
// --no-pass=listopt / --only-pass=listopt / EPQ_NO_PASS=listopt. Helpers: src/passes/util.hpp.
#include "passes.hpp"
#include "util.hpp"

namespace epq::tir {

void pass_listopt(Module&) {}

}  // namespace epq::tir
