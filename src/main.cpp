// ClaudeEPQ command line.
//
//   epq [--engine=jit|aot] [-t] [-e] [--parse-only] [--dump-resolved] FILE
//
// Exit status: 0 ok, 1 syntax / compile / run-time error (as the original),
// 2 usage error or missing back end.

#include <pthread.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "backend.hpp"
#include "frontend.hpp"
#include "tir.hpp"

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
                 "  --dump-asm         print the JIT's machine code (disassembly) and exit\n"
                 "  --jit-stats        print JIT compile time and code size to stderr\n"
                 "  --verbose          compile / cache diagnostics on stderr\n"
                 "  -t                 print \"Execution time: N microseconds\" to stderr\n"
                 "  -e                 dump the parsed AST to stderr (format of the original)\n"
                 "  --parse-only       parse (and dump with -e), then exit 0/1 without running\n"
                 "  --dump-resolved    print globals, functions, scopes and bindings, then exit\n"
                 "  --frontend-time    print front-end phase timings to stderr\n"
                 "  --frontend-bench=N run lex+parse+resolve N times in process, report timing\n"
                 "  -h, --help         this help\n",
                 argv0);
}

using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}

// Everything after reading the source: parse, dump, resolve, back end.
int pipeline(CompileUnit& cu, Options& opts, Clock::time_point t0, double t_read) {
    auto t1 = Clock::now();
    bool parsed = parse_unit(cu);
    double t_parse = us_since(t1);
    if (!parsed) return 1;

    if (opts.dump_ast) {
        dump_ast(std::cerr, cu.root, cu.syms);
        std::cerr.flush();
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
        dump_resolved(std::cout, cu.prog);
        std::cout.flush();
        return 0;
    }

    return run_backend(cu.prog, opts);
}

}  // namespace

// --frontend-bench=N: lex+parse+resolve the (already read) source N times in
// process and report the best and mean time per iteration (warm caches).
int frontend_bench(const CompileUnit& src, int iters) {
    double best = 1e30, total = 0;
    for (int i = 0; i < iters; ++i) {
        auto t = Clock::now();
        CompileUnit cu;
        cu.file = src.file;
        cu.source = src.source;
        if (!parse_unit(cu) || !resolve_unit(cu)) return 1;
        double us = us_since(t);
        best = us < best ? us : best;
        total += us;
    }
    std::fprintf(stderr, "frontend-bench: %d iterations, best %.2f us, mean %.2f us (lex+parse+resolve, %zu bytes)\n",
                 iters, best, total / iters, src.source.size());
    return 0;
}

int main(int argc, char** argv) {
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
        } else if (!std::strcmp(a, "--dump-asm")) {
            opts.dump_asm = true;
        } else if (!std::strcmp(a, "--jit-stats")) {
            opts.jit_stats = true;
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
    if (!have_file) {
        std::fputs("No file passed\n", stderr);
        usage(stderr, argv[0]);
        return 2;
    }

    auto t0 = Clock::now();
    CompileUnit cu;
    if (!load_source(cu, opts.file)) return 1;
    double t_read = us_since(t0);
    if (bench_iters > 0) return frontend_bench(cu, bench_iters);

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
