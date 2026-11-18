// Interface between the CLI / front end and the back ends (type inference,
// TIR, JIT / AOT). main() calls run_backend() once the program has been parsed
// and resolved; the back end owns everything from there on, including the
// program's exit status and the `-t` "Execution time" line.
//
// Implemented in src/driver.cpp.
#pragma once

#include <string>
#include <vector>

#include "resolver.hpp"

namespace epq {

enum class Engine { Jit, Aot };

struct Options {
    Engine engine = Engine::Jit;
    bool time = false;           // -t
    bool dump_ast = false;       // -e
    bool parse_only = false;     // --parse-only
    bool dump_resolved = false;  // --dump-resolved
    bool frontend_time = false;  // --frontend-time (diagnostic)
    bool emit_c = false;         // --emit-c: print the generated C, exit
    bool dump_tir = false;       // --dump-tir
    bool dump_types = false;     // --dump-types
    bool mono = false;           // --mono: one instance per function (no call-site specialisation)
    bool no_dedup = false;       // --no-dedup
    bool no_records = false;     // --no-records / EPQ_NO_RECORDS=1: no record typing (DESIGN.md §5.11)
    bool verbose = false;        // --verbose: compile/cache diagnostics on stderr
    bool dump_asm = false;       // --dump-asm: print the JIT's disassembly, exit
    bool jit_stats = false;      // --jit-stats: JIT compile time / code size on stderr
    // engine == Jit: 1 = --engine=jit (forced: an op the JIT lacks is an error),
    // 2 = auto, the default (JIT; the C engine for programs the JIT does not support).
    // engine == Aot (--engine=aot): the C engine.
    int jit_mode = 2;
    std::vector<std::string> no_pass;  // --no-pass=NAME (TIR passes and build options:
                                       // definite-assignment, demote-globals, constants)
    std::string file;            // source path as given on the command line
    // TIR pass control (src/passes/registry.cpp). Comma-separated lists are split
    // by the driver; EPQ_NO_PASS=a,b is appended to no_pass, EPQ_VERIFY_TIR=1 sets verify_tir.
    std::vector<std::string> only_pass;  // --only-pass=NAME[,NAME...]
    std::vector<std::string> enable_pass;  // --enable-pass=NAME / EPQ_ENABLE_PASS=a,b: run a default-off pass
    bool verify_tir = false;             // --verify-tir
    std::string dump_tir_after;          // --dump-tir-after=NAME
};

// Set by run_module when the program ran on the JIT (read by tier-up).
extern bool g_ran_on_jit;

// Runs the resolved program. Returns the process exit status.
int run_backend(const Program& prog, const Options& opts);

namespace tir { struct Module; }
// Runs passes (per opts) on an already built module, then the engine. Used by
// run_backend and by the TIR self-test (src/tir_selftest.cpp).
int run_module(tir::Module& M, const Options& opts);
// --tir-selftest=NAME: hand-built TIR programs exercising the extension ops.
int tir_selftest(const char* name, const Options& opts);

}  // namespace epq
