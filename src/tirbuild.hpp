// AST + inference result -> TIR (src/tir.hpp).
#pragma once

#include "infer.hpp"
#include "tir.hpp"

namespace epq {

struct BuildOptions {
    bool definite_assignment = true;  // drop Undefined-global checks proven unnecessary
    bool demote_globals = true;       // globals referenced by no function body -> locals of main
    bool constants = true;            // single-assignment literal globals -> constants
};

// Builds one tir::Func per reachable (deduplicated) instance. funcs[0] is main.
void build_tir(const Program& prog, infer::Result& inf, const BuildOptions& opt, tir::Module& out);

}  // namespace epq
