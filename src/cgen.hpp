// C backend (DESIGN.md §7): TIR -> one C translation unit (runtime + one static
// function per instance + exported entry `int epq_entry(void)`).
#pragma once

#include <string>

#include "tir.hpp"

namespace epq {

// Bump whenever the generated code's meaning changes (part of the cache key).
constexpr const char* kEmitterVersion = "cgen-4";

std::string emit_c(const tir::Module& m);

}  // namespace epq
