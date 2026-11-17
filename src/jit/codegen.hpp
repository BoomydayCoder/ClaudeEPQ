// ARM64 JIT engine (DESIGN.md §6): compiles every TIR function instance to
// machine code in-process and runs main on a big stack.
#pragma once

#include <cstdint>

#include "../backend.hpp"
#include "../tir.hpp"

namespace epq::jitcg {

struct JitConfig {
    bool time = false;       // -t: print "Execution time" (execution only)
    bool dump_asm = false;   // --dump-asm: print the disassembly, do not run
    bool stats = false;      // --jit-stats: compile time and code size on stderr
    bool verbose = false;
};

// Compiles and runs the module. Returns the process exit status.
int run(const tir::Module& m, const JitConfig& cfg);

}  // namespace epq::jitcg
