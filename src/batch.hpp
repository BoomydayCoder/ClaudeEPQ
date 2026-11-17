// Batch mode: `epq [options] --batch MANIFEST` runs many programs through the
// C backend with ONE freshly linked dylib (macOS scans every new binary on its
// first load, ~200 ms each, machine-wide serialised). See src/batch.cpp.
#pragma once

#include <string>

namespace epq {

// Environment variable set (only) for batch step-A children: the C backend
// writes the generated C to this path instead of compiling and running it.
constexpr const char* kBatchEmitEnv = "EPQ_BATCH_EMIT_C_TO";

// driver.cpp: child side of step A.
int batch_emit_c(const std::string& c, const char* path);

namespace batch {
// driver.cpp: the single-program path's compile flags, cache dir and cache key.
std::string cflags();
std::string cache_dir();
std::string hash(const std::string& csrc);
// driver.cpp: run a loaded entry exactly like run_module (big stack, overflow
// handler, flush on overflow, -t line). Returns the program's exit status.
int run_entry(int (*entry)(void), void (*flush)(void), bool time);
}  // namespace batch

// batch.cpp: entry point, called by main() when "--batch" is on the command line.
int batch_main(int argc, char** argv);

}  // namespace epq
