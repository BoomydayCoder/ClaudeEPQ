#include "disasm.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unistd.h>

namespace jit {

static std::string run_capture(const std::string& cmd) {
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
    pclose(p);
    return out;
}

std::string disassemble(const void* code, size_t bytes,
                        const std::vector<std::pair<size_t, std::string>>& syms, uint64_t base_addr) {
    const char* tmp = getenv("TMPDIR");
    std::string dir = std::string(tmp && *tmp ? tmp : "/tmp") + "/epq-disasm-XXXXXX";
    std::vector<char> d(dir.begin(), dir.end());
    d.push_back(0);
    if (!mkdtemp(d.data())) return "disassembly failed: mkdtemp";
    std::string s = std::string(d.data()) + "/code.s", o = std::string(d.data()) + "/code.o";

    // symbol names -> safe, unique assembler identifiers
    std::multimap<size_t, std::string> at;
    std::map<std::string, int> seen;
    for (const auto& [off, name] : syms) {
        std::string id;
        for (char c : name) id += (isalnum((unsigned char)c) || c == '_' || c == '.') ? c : '_';
        if (id.empty() || isdigit((unsigned char)id[0])) id = "_" + id;
        if (seen[id]++) id += "_" + std::to_string(seen[id]);
        at.emplace(off, id);
    }
    FILE* f = fopen(s.c_str(), "w");
    if (!f) return "disassembly failed: cannot write temp file";
    fputs(".text\n", f);
    const uint8_t* p = static_cast<const uint8_t*>(code);
    for (size_t i = 0; i < bytes; i += 4) {
        auto [a, b] = at.equal_range(i);
        for (auto it = a; it != b; ++it) fprintf(f, "%s:\n", it->second.c_str());
        uint32_t w;
        std::memcpy(&w, p + i, 4);
        fprintf(f, "  .long 0x%08x\n", w);
    }
    fclose(f);
    char adj[64];
    snprintf(adj, sizeof adj, "--adjust-vma=0x%llx", (unsigned long long)base_addr);
    std::string q = "'" + std::string(d.data()) + "'";
    std::string out = run_capture("clang -c -x assembler " + q + "/code.s -o " + q + "/code.o 2>&1 && "
                                  "objdump -d --no-show-raw-insn " + std::string(adj) + " " + q + "/code.o 2>&1");
    unlink(s.c_str());
    unlink(o.c_str());
    rmdir(d.data());
    // drop the objdump file header
    size_t k = out.find("Disassembly of section");
    if (k == std::string::npos) return "disassembly failed: " + out;
    size_t nl = out.find('\n', k);
    return nl == std::string::npos ? out : out.substr(nl + 1);
}

} // namespace jit
