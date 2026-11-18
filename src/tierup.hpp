// Second tier (DESIGN.md §7): a background-compiled, LLVM-optimised C-backend
// dylib that later runs of the same source pick up before the front end runs.
//
//   start-up   key = hash(source bytes, build UUID, emitter version, EPQ_*
//              environment, --no-pass/--mono/--no-dedup); one open() of
//              $CACHE/tier/<key>.ready decides hit or miss.
//   hit        dlopen the (already helper-warmed) dylib, run it through the
//              driver's AOT run path (run_dylib_entry).
//   miss       the normal engine runs; a sleeping thread spawns, once, a
//              detached utility-QoS helper `epq --tier-compile FILE` if the
//              program is still running after EPQ_TIER_MS (default 20) ms.
//   helper     flock-guarded single compile: emit C, cc, rename, dlopen-warm,
//              write the marker atomically, evict to the size cap.
//
// Engine policy: active only under the default policy "auto" (no engine flag,
// or --engine=auto; --engine=jit/aot force one engine and skip tier-up);
// EPQ_NO_TIERUP=1 makes all of it inert, and so does a cache directory that
// cannot be created or written (the helper is then never spawned).
#pragma once

#include "backend.hpp"
#include "frontend.hpp"

namespace epq {

// Implemented in src/driver.cpp (tier-up hook): runs `epq_entry` of an
// already dlopen'ed C-backend dylib exactly like the AOT path (big stack,
// stack-overflow handler, -t line). Returns the exit status, or -1 if the
// handle has no entry point (nothing was run).
int run_dylib_entry(void* handle, const Options& opts);

namespace tierup {

// True for the command-line flags this module owns (main() skips them).
bool is_flag(const char* arg);

// --tier-clear (no FILE needed). Returns true and sets *rc if it handled the run.
bool command(int argc, char** argv, int* rc);

// Called once the source is loaded, before parsing. Handles --tier-compile
// and --tier-status, and a cache hit (returns true with *rc = exit status).
// On a miss arms the background trigger and returns false: the normal
// pipeline then runs.
bool start(CompileUnit& cu, const Options& opts, int argc, char** argv, int* rc);

}  // namespace tierup
}  // namespace epq
