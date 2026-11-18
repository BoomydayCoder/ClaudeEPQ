// ARM64 JIT engine (DESIGN.md §6): compiles every TIR function instance to
// machine code in-process and runs main on a big stack.
#pragma once

#include <cstdint>
#include <string>

#include "../backend.hpp"
#include "../tir.hpp"

namespace epq::jitcg {

struct JitConfig {
    bool time = false;       // -t: print "Execution time" (execution only)
    bool dump_asm = false;   // --dump-asm: print the disassembly, do not run
    bool stats = false;      // --jit-stats: compile time and code size on stderr
    bool verbose = false;
};

// "" if the JIT implements every op and type of the module, else a short
// description of the first unsupported one (e.g. "TIR op RecGet in function f2_g").
std::string unsupported(const tir::Module& m);

// Compiles and runs the module (which must pass unsupported()). Returns the process exit status.
// If the code generator hits one of its own limits (out of temporaries, an
// encoding out of range, any internal inconsistency), nothing has run yet:
// every function is compiled before main starts. run() then returns
// kCompileFailed with the reason in *why, and the caller picks another engine.
constexpr int kCompileFailed = -1;
int run(const tir::Module& m, const JitConfig& cfg, std::string* why);

}  // namespace epq::jitcg
