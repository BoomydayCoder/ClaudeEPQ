// Debug disassembly of JIT code via the system toolchain (clang + objdump).
// Slow (spawns two processes, ~50 ms); for --dump-asm and inspection only.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace jit {

// `syms`: (byte offset, name) pairs marking function entries / labels.
// Addresses printed are offsets into the buffer (llvm-objdump's --adjust-vma
// does not apply to Mach-O objects); a header line names base_addr if non-zero.
// Returns the objdump listing, or an
// error message starting with "disassembly failed".
std::string disassemble(const void* code, size_t bytes,
                        const std::vector<std::pair<size_t, std::string>>& syms,
                        uint64_t base_addr = 0);

} // namespace jit
