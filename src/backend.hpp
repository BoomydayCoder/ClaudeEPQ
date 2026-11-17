// Interface between the CLI / front end and the back ends (type inference,
// TIR, JIT / AOT). main() calls run_backend() once the program has been parsed
// and resolved; the back end owns everything from there on, including the
// program's exit status and the `-t` "Execution time" line.
//
// backend_stub.cpp provides the placeholder implementation; the real back end
// replaces that file.
#pragma once

#include <string>

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
    std::string file;            // source path as given on the command line
};

// Runs the resolved program. Returns the process exit status.
int run_backend(const Program& prog, const Options& opts);

}  // namespace epq
