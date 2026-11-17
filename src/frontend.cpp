#include "frontend.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>

namespace epq {

namespace {

void write_err(const std::string& s) { std::fwrite(s.data(), 1, s.size(), stderr); }

// Reads straight into `out` (no large stack buffer: its stack probes would
// fault in several fresh pages at start-up). A regular file gets one buffer of
// its size + 1, so the usual case is two read() calls (data, then EOF); a file
// that grew since fstat, or a pipe, grows the buffer geometrically.
bool read_fd(int fd, std::string& out) {
    struct stat st;
    size_t chunk = 16384;
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) chunk = (size_t)st.st_size + 1;
    size_t got = 0;
    for (;;) {
        if (out.size() - got == 0) {
            out.resize(got + chunk);
            chunk = out.size();
        }
        ssize_t r = ::read(fd, &out[got], out.size() - got);
        if (r < 0) return false;
        if (r == 0) break;
        got += (size_t)r;
    }
    out.resize(got);
    return true;
}

// Prints the first `n` lexer diagnostics.
void print_lex_diags(const CompileUnit& cu, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        s += format_loc(cu.file, cu.lex.diags[i].loc);
        s += ": invalid character\n";
    }
    if (!s.empty()) write_err(s);
}

}  // namespace

bool load_source(CompileUnit& cu, const std::string& path) {
    cu.file = path;
    if (path == "-") return read_fd(0, cu.source);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        write_err("cannot open " + path + "\n");
        return false;
    }
    bool ok = read_fd(fd, cu.source);
    ::close(fd);
    if (!ok) write_err("cannot open " + path + "\n");
    return ok;
}

bool parse_unit(CompileUnit& cu) {
    lex(cu.source.data(), cu.source.size(), cu.syms, cu.lex);
    SyntaxError err;
    size_t depth = cu.stack_bytes / 1024;
    cu.root = parse_program(cu.lex, cu.arena, err, depth > 0xffffffffu ? 0xffffffffu : (unsigned)depth);
    if (!cu.root) {
        // The original scanner runs lazily: it has printed the diagnostics for
        // invalid characters up to (and including those just before) the token
        // at which the parser failed.
        print_lex_diags(cu, cu.lex.toks[err.tok_index].nerr_before);
        write_err(format_loc(cu.file, err.loc) + ": " + err.msg + "\n");
        return false;
    }
    print_lex_diags(cu, cu.lex.diags.size());
    return true;
}

bool resolve_unit(CompileUnit& cu) {
    ResolveError err;
    if (!resolve(cu.root, cu.syms, cu.prog, err)) {
        write_err(err.msg + "\n");
        return false;
    }
    return true;
}

}  // namespace epq
