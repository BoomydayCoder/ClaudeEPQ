// Optimisation passes (DESIGN.md §5), one file each under src/passes/.
// Registered in pipeline order in src/passes/registry.cpp (kPipeline). To add
// a pass: define `void pass_NAME(tir::Module&)` in its file, declare it here,
// add one line to kPipeline. Passes may call tir::pass_region_info(m) to
// recompute Func::mutates / Local::unique / Region flags after restructuring.
#pragma once

#include "../tir.hpp"

namespace epq::tir {

void pass_simplify(Module& m);   // §5.6 local simplification (early run "simplify", late run "simplify2")
void pass_inline(Module& m);     // §5.1 inlining into InlineBlocks
void pass_pairs(Module& m);      // §5.4 pair unboxing
void pass_ifconv(Module& m);     // §5.5 if-conversion to Select
void pass_listopt(Module& m);    // §5.2/§5.3 header caching + check elision (Low TIR)
void pass_reclaim(Module& m);    // §5.7 static reclamation (Free)

}  // namespace epq::tir
