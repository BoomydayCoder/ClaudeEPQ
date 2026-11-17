// Placeholder back end: replaced once type inference + code generation exist.
#include <cstdio>

#include "backend.hpp"

namespace epq {

int run_backend(const Program&, const Options&) {
    std::fputs("backend not implemented\n", stderr);
    return 2;
}

}  // namespace epq
