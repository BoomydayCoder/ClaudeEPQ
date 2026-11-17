// Back-end driver: inference -> TIR -> passes -> engine (C backend: emit,
// compile with cc into a cached dylib, dlopen, run on a big stack).
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "backend.hpp"
#include "cgen.hpp"
#include "infer.hpp"
#include "tirbuild.hpp"

extern char** environ;

namespace epq {

namespace {

constexpr const char* kDefaultCFlags = "-O3 -fwrapv -mcpu=native";

void write_all(int fd, const std::string& s) {
    const char* p = s.data();
    size_t n = s.size();
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w <= 0) return;
        p += w;
        n -= (size_t)w;
    }
}

uint64_t now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

std::string hash_hex(const std::string& s) {
    uint64_t h1 = 1469598103934665603ull, h2 = 0x9e3779b97f4a7c15ull;
    for (unsigned char c : s) {
        h1 = (h1 ^ c) * 1099511628211ull;
        h2 = (h2 ^ c) * 0x100000001b3ull;
        h2 ^= h2 >> 29;
    }
    char buf[40];
    std::snprintf(buf, sizeof buf, "%016llx%016llx", (unsigned long long)h1, (unsigned long long)h2);
    return buf;
}

bool mkdirs(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && ::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        if (i < path.size()) cur += path[i];
    }
    return true;
}

std::vector<std::string> split_flags(const std::string& s) {
    std::vector<std::string> v;
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!cur.empty()) v.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

int run_cmd(const std::vector<std::string>& args) {
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid;
    if (posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ) != 0) return -1;
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

// Compiles `csrc` into a cached dylib; returns its path or "" on failure.
std::string compile_cached(const std::string& csrc, bool verbose) {
    const char* fl = std::getenv("EPQ_CFLAGS");
    std::string flags = fl ? fl : kDefaultCFlags;
    std::string dir;
    if (const char* d = std::getenv("EPQ_CACHE_DIR"); d && *d) {
        dir = d;
    } else {
        const char* home = std::getenv("HOME");
        dir = std::string(home ? home : "/tmp") + "/.cache/claude-epq";
    }
    std::string key = csrc;
    key += '\0';
    key += flags;
    key += '\0';
    key += kEmitterVersion;
    std::string h = hash_hex(key);
    std::string path = dir + "/" + h + ".dylib";
    if (::access(path.c_str(), R_OK) == 0) {
        if (verbose) std::fprintf(stderr, "epq: cache hit %s\n", path.c_str());
        return path;
    }
    if (!mkdirs(dir)) {
        std::fprintf(stderr, "epq: cannot create cache directory %s\n", dir.c_str());
        return "";
    }
    char uniq[64];
    std::snprintf(uniq, sizeof uniq, ".tmp-%d-%llx", (int)getpid(), (unsigned long long)now_us());
    std::string cpath = dir + "/" + h + uniq + ".c";
    std::string tpath = dir + "/" + h + uniq + ".dylib";
    int fd = ::open(cpath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        std::fprintf(stderr, "epq: cannot write %s\n", cpath.c_str());
        return "";
    }
    write_all(fd, csrc);
    ::close(fd);
    std::vector<std::string> args = {"cc"};
    for (auto& f : split_flags(flags)) args.push_back(f);
    // -fwrapv is part of the language semantics, never optional.
    for (const char* a : {"-fwrapv", "-w", "-dynamiclib", "-o"}) args.push_back(a);
    args.push_back(tpath);
    args.push_back(cpath);
    uint64_t t0 = now_us();
    int rc = run_cmd(args);
    if (verbose) std::fprintf(stderr, "epq: cc %s: %d (%.1f ms)\n", flags.c_str(), rc, (now_us() - t0) / 1000.0);
    ::unlink(cpath.c_str());
    if (rc != 0) {
        ::unlink(tpath.c_str());
        std::fprintf(stderr, "epq: C compilation failed (rc %d)\n", rc);
        return "";
    }
    if (::rename(tpath.c_str(), path.c_str()) != 0) {
        ::unlink(tpath.c_str());
        if (::access(path.c_str(), R_OK) != 0) return "";
    }
    return path;
}

// ---- running generated code on a big, lazily committed stack ----------------

constexpr size_t kGuard = 1 << 20;
char* g_stack_lo = nullptr;   // lowest address of the mapping (guard region)
size_t g_stack_size = 0;
void (*g_flush)(void) = nullptr;

void on_fault(int sig, siginfo_t* si, void*) {
    char* a = (char*)si->si_addr;
    if (g_stack_lo && a >= g_stack_lo - 65536 && a < g_stack_lo + kGuard + 65536) {
        if (g_flush) g_flush();
        const char msg[] = "Stack overflow\n";
        (void)!::write(2, msg, sizeof msg - 1);
        _exit(1);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

struct RunJob {
    int (*entry)(void);
    int rc;
};

void* run_thread(void* p) {
    RunJob* j = static_cast<RunJob*>(p);
    static char altstack[1 << 16];
    stack_t ss;
    ss.ss_sp = altstack;
    ss.ss_size = sizeof altstack;
    ss.ss_flags = 0;
    sigaltstack(&ss, nullptr);
    j->rc = j->entry();
    return nullptr;
}

int run_on_big_stack(int (*entry)(void)) {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);

    RunJob job{entry, 1};
    void* mem = MAP_FAILED;
    size_t size = 0;
    for (size_t gb : {64, 16, 4, 1}) {
        size = gb << 30;
        mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
        if (mem != MAP_FAILED) break;
    }
    if (mem == MAP_FAILED) return entry();
    mprotect(mem, kGuard, PROT_NONE);
    g_stack_lo = (char*)mem;
    g_stack_size = size;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstack(&attr, mem, size);
    pthread_t th;
    if (pthread_create(&th, &attr, run_thread, &job) != 0) return entry();
    pthread_join(th, nullptr);
    return job.rc;
}

bool disabled(const Options& o, const char* name) {
    for (auto& s : o.no_pass)
        if (s == name) return true;
    return false;
}

}  // namespace

int run_backend(const Program& prog, const Options& opts) {
    infer::Config cfg;
    cfg.poly = !opts.mono;
    cfg.dedup = !opts.no_dedup;
    infer::Result R;
    infer::infer(prog, cfg, R);
    if (opts.dump_types) {
        std::string s;
        infer::dump_types(s, R);
        write_all(1, s);
        return 0;
    }
    BuildOptions bo;
    bo.definite_assignment = !disabled(opts, "definite-assignment");
    bo.demote_globals = !disabled(opts, "demote-globals");
    bo.constants = !disabled(opts, "constants");
    tir::Module M;
    build_tir(prog, R, bo, M);
    tir::run_passes(M, opts.no_pass, opts.verbose);
    if (opts.dump_tir) {
        std::string s;
        tir::dump(s, M);
        write_all(1, s);
        return 0;
    }
    std::string c = emit_c(M);
    if (opts.emit_c) {
        write_all(1, c);
        return 0;
    }
    // Engine::Jit falls back to the C backend until the JIT exists.
    std::string path = compile_cached(c, opts.verbose);
    if (path.empty()) return 2;
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "epq: dlopen failed: %s\n", dlerror());
        return 2;
    }
    auto entry = (int (*)(void))dlsym(h, "epq_entry");
    g_flush = (void (*)(void))dlsym(h, "epq_entry_flush");
    if (!entry) {
        std::fprintf(stderr, "epq: entry point missing\n");
        return 2;
    }
    uint64_t t0 = now_us();
    int rc = run_on_big_stack(entry);
    uint64_t t1 = now_us();
    if (opts.time) {
        char buf[80];
        int n = std::snprintf(buf, sizeof buf, "Execution time: %llu microseconds\n", (unsigned long long)(t1 - t0));
        (void)!::write(2, buf, (size_t)n);
    }
    return rc;
}

}  // namespace epq
