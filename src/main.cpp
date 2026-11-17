// ClaudeEPQ command line.
//
//   epq [--engine=jit|aot] [-t] [-e] [--parse-only] [--dump-resolved] FILE
//
// Exit status: 0 ok, 1 syntax / compile / run-time error (as the original),
// 2 usage error or missing back end.

#include <mach/mach_time.h>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <streambuf>
#include <string>

#include "backend.hpp"
#include "batch.hpp"
#include "frontend.hpp"
#include "infer.hpp"
#include "tierup.hpp"  // tier-up hook
#include "tir.hpp"
#include "tirbuild.hpp"

using namespace epq;

namespace {

void usage(FILE* f, const char* argv0) {
    std::fprintf(f,
                 "Usage: %s [options] FILE        (FILE '-' reads the program from stdin)\n"
                 "Options:\n"
                 "  --engine=jit|aot   back end (default: jit; the C backend until the JIT exists)\n"
                 "  --emit-c           print the generated C and exit\n"
                 "  --dump-tir         print the typed IR and exit\n"
                 "  --dump-types       print inferred types of globals / instances and exit\n"
                 "  --no-pass=NAME     disable a TIR pass or build step (fold, definite-assignment,\n"
                 "                     demote-globals, constants)\n"
                 "  --only-pass=A,B    run only these optimisation passes (see --list-passes)\n"
                 "  --verify-tir       check the TIR after construction and every pass (env EPQ_VERIFY_TIR=1)\n"
                 "  --dump-tir-after=P print the TIR after pass P and exit\n"
                 "  --list-passes      list the TIR passes in pipeline order\n"
                 "  --mono             one instance per function (no call-site specialisation)\n"
                 "  --no-dedup         keep identical instances separate\n"
                 "  --verbose          compile / cache diagnostics on stderr\n"
                 "  -t                 print \"Execution time: N microseconds\" to stderr\n"
                 "  -e                 dump the parsed AST to stderr (format of the original)\n"
                 "  --parse-only       parse (and dump with -e), then exit 0/1 without running\n"
                 "  --dump-resolved    print globals, functions, scopes and bindings, then exit\n"
                 "  --frontend-time    print front-end phase timings to stderr\n"
                 "  --frontend-bench=N run parse..TIR passes N times in process, report per-phase timing\n"
                 "                     (with --parse-only: lex+parse+resolve only)\n"
                 "  --tier-status FILE / --tier-clear   background-compiled cache (src/tierup.hpp)\n"
                 "  --batch MANIFEST [-j N] [--timeout SEC]\n"
                 "                     run many programs with one dylib (see src/batch.cpp, tools/README.md)\n"
                 "  -h, --help         this help\n",
                 argv0);
}

// Monotonic clock in mach ticks (no <chrono>, no libc++ clock calls).
struct Clock {
    using time_point = uint64_t;
    static time_point now() { return mach_absolute_time(); }
};
double ticks_us(uint64_t d) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)d * tb.numer / tb.denom / 1000.0;
}
double us_since(Clock::time_point t) { return ticks_us(Clock::now() - t); }
long minflt() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_minflt;
}

// Unbuffered-enough ostream onto a file descriptor, so the AST / resolver
// dumps need neither <iostream> nor std::cout / std::cerr.
class FdBuf : public std::streambuf {
public:
    explicit FdBuf(int fd) : fd_(fd) { setp(buf_, buf_ + sizeof buf_); }
    ~FdBuf() override { sync(); }
    int sync() override {
        const char* p = pbase();
        size_t n = (size_t)(pptr() - pbase());
        while (n > 0) {
            ssize_t w = ::write(fd_, p, n);
            if (w <= 0) break;
            p += w;
            n -= (size_t)w;
        }
        setp(buf_, buf_ + sizeof buf_);
        return 0;
    }
    int_type overflow(int_type c) override {
        sync();
        if (c != traits_type::eof()) {
            *pptr() = (char)c;
            pbump(1);
        }
        return 0;
    }

private:
    int fd_;
    char buf_[8192];
};

// Everything after reading the source: parse, dump, resolve, back end.
int pipeline(CompileUnit& cu, Options& opts, Clock::time_point t0, double t_read) {
    auto t1 = Clock::now();
    bool parsed = parse_unit(cu);
    double t_parse = us_since(t1);
    if (!parsed) return 1;

    if (opts.dump_ast) {
        FdBuf b(2);
        std::ostream os(&b);
        dump_ast(os, cu.root, cu.syms);
        os.flush();
    }
    if (opts.parse_only) {
        if (opts.frontend_time)
            std::fprintf(stderr, "frontend: read %.1f us, lex+parse %.1f us\n", t_read, t_parse);
        return 0;
    }

    auto t2 = Clock::now();
    bool resolved = resolve_unit(cu);
    double t_resolve = us_since(t2);
    if (opts.frontend_time)
        std::fprintf(stderr, "frontend: read %.1f us, lex+parse %.1f us, resolve %.1f us, total %.1f us\n", t_read,
                     t_parse, t_resolve, us_since(t0));
    if (!resolved) return 1;

    if (opts.dump_resolved) {
        FdBuf b(1);
        std::ostream os(&b);
        dump_resolved(os, cu.prog);
        os.flush();
        return 0;
    }

    return run_backend(cu.prog, opts);
}

}  // namespace

// --frontend-bench=N: run the whole compile front half (lex+parse, resolve,
// inference, TIR build, TIR passes) on the (already read) source N times in
// process and report the best and mean time per phase (warm caches).
int frontend_bench(const CompileUnit& src, int iters, const Options& opts) {
    enum { P_PARSE, P_RESOLVE, P_INFER, P_BUILD, P_PASSES, P_TOTAL, NP };
    static const char* const names[NP] = {"lex+parse", "resolve", "infer", "tir-build", "passes", "total"};
    const auto& plist = tir::passes();
    size_t npass = plist.size();
    std::vector<double> best(NP + npass, 1e30), sum(NP + npass, 0), first(NP + npass, 0);
    std::vector<long> flt(NP + npass, 0);  // page faults during the first iteration
    long flt_start = minflt();
    bool backend = !opts.parse_only;
    long fl = 0;
    auto acc = [&](size_t i, uint64_t d) {
        double us = ticks_us(d);
        if (us < best[i]) best[i] = us;
        if (sum[i] == 0) first[i] = us;
        sum[i] += us;
    };
    // First iteration only: faults since the previous mark.
    auto mark = [&](int it, size_t i) {
        if (it) return;
        long f = minflt();
        flt[i] = f - fl;
        fl = f;
    };
    for (int it = 0; it < iters; ++it) {
        fl = minflt();
        long fl0 = fl;
        uint64_t t0 = Clock::now();
        CompileUnit cu;
        cu.file = src.file;
        cu.source = src.source;
        if (!parse_unit(cu)) return 1;
        uint64_t t1 = Clock::now();
        mark(it, P_PARSE);
        if (!resolve_unit(cu)) return 1;
        uint64_t t2 = Clock::now();
        mark(it, P_RESOLVE);
        acc(P_PARSE, t1 - t0);
        acc(P_RESOLVE, t2 - t1);
        if (backend) {
            infer::Config cfg;
            cfg.poly = !opts.mono;
            cfg.dedup = !opts.no_dedup;
            infer::Result R;
            infer::infer(cu.prog, cfg, R);
            uint64_t t3 = Clock::now();
            mark(it, P_INFER);
            tir::Module M;
            build_tir(cu.prog, R, BuildOptions{}, M);
            uint64_t t4 = Clock::now();
            mark(it, P_BUILD);
            uint64_t tp = t4;
            for (size_t p = 0; p < npass; ++p) {
                bool off = false;
                for (auto& d : opts.no_pass)
                    if (d == plist[p].name) off = true;
                if (!off) plist[p].run(M);
                uint64_t tq = Clock::now();
                mark(it, NP + p);
                acc(NP + p, tq - tp);
                tp = tq;
            }
            acc(P_INFER, t3 - t2);
            acc(P_BUILD, t4 - t3);
            acc(P_PASSES, tp - t4);
            acc(P_TOTAL, tp - t0);
        } else {
            acc(P_TOTAL, t2 - t0);
        }
        if (it == 0) {
            for (size_t p = 0; p < npass; ++p) flt[P_PASSES] += flt[NP + p];
            flt[P_TOTAL] = minflt() - fl0;
        }
    }
    std::fprintf(stderr, "frontend-bench: %s, %zu bytes, %d iterations (best / mean / first us, first faults);"
                 " %ld faults before\n", src.file.c_str(), src.source.size(), iters, flt_start);
    for (int i = 0; i < NP; ++i) {
        if (!backend && i >= P_INFER && i < P_TOTAL) continue;
        std::fprintf(stderr, "  %-22s %9.2f %9.2f %9.2f %5ld\n", names[i], best[i], sum[i] / iters, first[i], flt[i]);
    }
    if (backend)
        for (size_t p = 0; p < npass; ++p)
            std::fprintf(stderr, "    pass %-17s %9.2f %9.2f %9.2f %5ld\n", plist[p].name, best[NP + p],
                         sum[NP + p] / iters, first[NP + p], flt[NP + p]);
    return 0;
}

int main(int argc, char** argv) {
    // ---- batch mode (src/batch.cpp) ----
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--batch") || !std::strncmp(argv[i], "--batch=", 8)) return epq::batch_main(argc, argv);
    // ---- end batch mode ----
    Options opts;
    int bench_iters = 0;
    bool have_file = false;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            usage(stdout, argv[0]);
            return 0;
        } else if (!std::strcmp(a, "-t")) {
            opts.time = true;
        } else if (!std::strcmp(a, "-e")) {
            opts.dump_ast = true;
        } else if (!std::strcmp(a, "--parse-only")) {
            opts.parse_only = true;
        } else if (!std::strcmp(a, "--dump-resolved")) {
            opts.dump_resolved = true;
        } else if (!std::strcmp(a, "--frontend-time")) {
            opts.frontend_time = true;
        } else if (!std::strncmp(a, "--frontend-bench=", 17)) {
            bench_iters = std::atoi(a + 17);
        } else if (!std::strcmp(a, "--emit-c")) {
            opts.emit_c = true;
        } else if (!std::strcmp(a, "--dump-tir")) {
            opts.dump_tir = true;
        } else if (!std::strcmp(a, "--dump-types")) {
            opts.dump_types = true;
        } else if (!std::strcmp(a, "--mono")) {
            opts.mono = true;
        } else if (!std::strcmp(a, "--no-dedup")) {
            opts.no_dedup = true;
        } else if (!std::strcmp(a, "--verbose")) {
            opts.verbose = true;
        } else if (!std::strncmp(a, "--no-pass=", 10)) {
            opts.no_pass.push_back(a + 10);
        } else if (!std::strncmp(a, "--only-pass=", 12)) {
            opts.only_pass.push_back(a + 12);
        } else if (!std::strcmp(a, "--verify-tir")) {
            opts.verify_tir = true;
        } else if (!std::strncmp(a, "--dump-tir-after=", 17)) {
            opts.dump_tir_after = a + 17;
        } else if (!std::strcmp(a, "--list-passes")) {
            for (const auto& p : epq::tir::passes())
                std::printf("%-20s %s%s\n", p.name, p.what, epq::tir::pass_is_optimisation(p.name) ? "" : "  [infrastructure]");
            return 0;
        } else if (!std::strncmp(a, "--tir-selftest=", 15)) {
            return epq::tir_selftest(a + 15, opts);
        } else if (!std::strcmp(a, "--engine=jit") || !std::strcmp(a, "--jit")) {
            opts.engine = Engine::Jit;
        } else if (!std::strcmp(a, "--engine=aot") || !std::strcmp(a, "--aot")) {
            opts.engine = Engine::Aot;
        } else if (tierup::is_flag(a)) {  // tier-up hook: --tier-compile/--tier-status/--tier-clear
        } else if (a[0] == '-' && a[1] != 0) {
            std::fprintf(stderr, "%s: unknown option %s\n", argv[0], a);
            usage(stderr, argv[0]);
            return 2;
        } else if (!have_file) {
            opts.file = a;
            have_file = true;
        } else {
            std::fprintf(stderr, "%s: more than one input file\n", argv[0]);
            return 2;
        }
    }
    if (int rc; tierup::command(argc, argv, &rc)) return rc;  // tier-up hook: --tier-clear
    if (!have_file) {
        std::fputs("No file passed\n", stderr);
        usage(stderr, argv[0]);
        return 2;
    }

    auto t0 = Clock::now();
    CompileUnit cu;
    if (!load_source(cu, opts.file)) return 1;
    double t_read = us_since(t0);
    if (bench_iters > 0) return frontend_bench(cu, bench_iters, opts);
    if (int rc; tierup::start(cu, opts, argc, argv, &rc)) return rc;  // tier-up hook: cache probe / helper

    // Small sources cannot nest deeply enough to threaten the main thread's
    // stack. Larger ones run the whole pipeline on a thread whose (lazily
    // committed) stack grows with the source, so deep nesting never crashes.
    if (cu.source.size() <= 4096) return pipeline(cu, opts, t0, t_read);
    size_t stack = (64u << 20) + cu.source.size() * 4096;
    if (stack > (size_t(32) << 30)) stack = size_t(32) << 30;
    cu.stack_bytes = stack;
    struct Job {
        CompileUnit* cu;
        Options* opts;
        Clock::time_point t0;
        double t_read;
        int rc;
    } job{&cu, &opts, t0, t_read, 1};
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, stack);
    pthread_t th;
    auto body = [](void* p) -> void* {
        Job* j = static_cast<Job*>(p);
        j->rc = pipeline(*j->cu, *j->opts, j->t0, j->t_read);
        return nullptr;
    };
    if (pthread_create(&th, &attr, body, &job) != 0) {
        cu.stack_bytes = 8u << 20;  // fall back to the main thread
        return pipeline(cu, opts, t0, t_read);
    }
    pthread_join(th, nullptr);
    return job.rc;
}
