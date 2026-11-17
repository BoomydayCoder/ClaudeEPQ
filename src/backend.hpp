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
    bool verbose = false;        // --verbose: compile/cache diagnostics on stderr
    bool dump_asm = false;       // --dump-asm: print the JIT's disassembly, exit
    bool jit_stats = false;      // --jit-stats: JIT compile time / code size on stderr
    std::vector<std::string> no_pass;  // --no-pass=NAME (TIR passes and build options:
                                       // definite-assignment, demote-globals, constants)
    std::string file;            // source path as given on the command line
};

// Runs the resolved program. Returns the process exit status.
int run_backend(const Program& prog, const Options& opts);

}  // namespace epq
