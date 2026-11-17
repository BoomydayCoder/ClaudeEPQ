// Batch mode for the C backend:
//
//   epq [options] --batch MANIFEST [-j N] [--timeout SEC]
//
// MANIFEST: one test per line, five TAB-separated fields:
//   PROGRAM  STDIN|-  STDOUT_OUT  STDERR_OUT  EXITCODE_OUT
// (empty lines and lines starting with '#' are ignored). All other options and
// the environment apply to every program exactly as in single-program mode.
//
// Why: macOS security-scans every new binary on its first exec/dlopen (~200 ms,
// serialised machine-wide). The single-program path makes one new dylib per
// program; this makes ONE per batch:
//
//   A. per program: run this same executable (already scanned) on it, with
//      EPQ_BATCH_EMIT_C_TO set: the normal pipeline runs and, where it would
//      compile the C, writes it to a file instead (src/driver.cpp). Compile-time
//      errors therefore produce exactly the normal stdout / stderr / exit code.
//   B. per distinct generated C: `cc -c` (normal flags) into <cache>/<hash>.o,
//      <hash> = the single-program cache key; entry symbols renamed to
//      epq_entry_<hash> / epq_flush_<hash> with -D (everything else in the
//      generated C is already static). Parallel, cached across runs.
//   C. link all objects into <cache>/batch-<sethash>.dylib (reused when the same
//      object set comes again: zero new binaries), dlopen it once.
//   D. per test: fork (no exec), redirect stdio, run the entry through the same
//      code as a normal run (batch::run_entry), exit with its status. Up to -j
//      at a time, per-test timeout (exit code 124, "TIMEOUT" on stderr), a child
//      killed by a signal gets 128+signal and a note on stderr.
//
// Step A's stdout/stderr go to the test's files (truncated); step D appends.
// The batch itself exits 0 unless the manifest or the link/dlopen failed.
#include <dlfcn.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "batch.hpp"

extern char** environ;

namespace epq {

namespace {

uint64_t now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

std::string hex64(const std::string& s) {
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

std::vector<std::string> split_ws(const std::string& s) {
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

bool read_file(const std::string& path, std::string& out) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    out.clear();
    char buf[65536];
    for (;;) {
        ssize_t r = ::read(fd, buf, sizeof buf);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        out.append(buf, (size_t)r);
    }
    ::close(fd);
    return true;
}

void write_fd_all(int fd, const char* p, size_t n) {
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return;
        p += w;
        n -= (size_t)w;
    }
}

void write_file(const std::string& path, const std::string& s, bool append) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC), 0644);
    if (fd < 0) return;
    write_fd_all(fd, s.data(), s.size());
    ::close(fd);
}

struct Test {
    std::string prog, in, out, err, code;
    std::string cfile;   // step A output
    std::string hash;    // "" = finished in step A (or failed before running)
    int rc = -1;         // final exit status once known
    int (*entry)(void) = nullptr;
    void (*flush)(void) = nullptr;
};

void finish(Test& t, int rc, const std::string& err_note) {
    t.rc = rc;
    t.hash.clear();
    if (!err_note.empty()) write_file(t.err, err_note, true);
    write_file(t.code, std::to_string(rc) + "\n", false);
}

// Decodes a wait status into an exit code (+ a stderr note for signals).
int decode_status(int st, bool timed_out, std::string& note) {
    if (timed_out) {
        note = "TIMEOUT\n";
        return 124;
    }
    if (WIFEXITED(st)) return WEXITSTATUS(st);
    if (WIFSIGNALED(st)) {
        int sig = WTERMSIG(st);
        note = "epq-batch: killed by signal " + std::to_string(sig) + " (" + strsignal(sig) + ")\n";
        return 128 + sig;
    }
    note = "epq-batch: unknown wait status\n";
    return 125;
}

// Runs launch(i) for i in [0, n) with at most `jobs` children alive; launch
// returns a pid, or <= 0 if it handled item i synchronously. done(i, status,
// timed_out) is called once per reaped child. timeout_us 0 = none.
template <class Launch, class Done>
void pool(size_t n, int jobs, uint64_t timeout_us, Launch launch, Done done) {
    struct Running {
        pid_t pid;
        size_t i;
        uint64_t t0;
        bool killed;
    };
    std::vector<Running> run;
    size_t next = 0;
    while (next < n || !run.empty()) {
        while (next < n && (int)run.size() < jobs) {
            size_t i = next++;
            pid_t p = launch(i);
            if (p > 0) run.push_back({p, i, now_us(), false});
        }
        if (run.empty()) continue;
        int st = 0;
        pid_t p = waitpid(-1, &st, WNOHANG);
        if (p > 0) {
            for (size_t k = 0; k < run.size(); ++k)
                if (run[k].pid == p) {
                    Running r = run[k];
                    run.erase(run.begin() + (long)k);
                    done(r.i, st, r.killed);
                    break;
                }
            continue;
        }
        if (p < 0 && errno == ECHILD) {  // should not happen; avoid spinning forever
            for (auto& r : run) done(r.i, 0, false);
            run.clear();
            continue;
        }
        if (timeout_us) {
            uint64_t t = now_us();
            for (auto& r : run)
                if (!r.killed && t - r.t0 > timeout_us) {
                    ::kill(r.pid, SIGKILL);
                    r.killed = true;
                }
        }
        struct timespec ts = {0, 200000};
        nanosleep(&ts, nullptr);
    }
}

pid_t spawn(const std::vector<std::string>& args, posix_spawn_file_actions_t* fa, char** envp) {
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid;
    if (posix_spawnp(&pid, argv[0], fa, nullptr, argv.data(), envp) != 0) return -1;
    return pid;
}

std::string self_path(const char* argv0) {
    char buf[PATH_MAX];
    uint32_t sz = sizeof buf;
    if (_NSGetExecutablePath(buf, &sz) == 0) {
        char real[PATH_MAX];
        if (realpath(buf, real)) return real;
        return buf;
    }
    return argv0;
}

int usage_error(const char* msg) {
    std::fprintf(stderr, "epq --batch: %s\nusage: epq [options] --batch MANIFEST [-j N] [--timeout SEC]\n", msg);
    return 2;
}

}  // namespace

int batch_main(int argc, char** argv) {
    uint64_t T0 = now_us();
    std::string manifest;
    int jobs = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (jobs < 1) jobs = 4;
    double timeout_s = 30;
    bool time_flag = false;
    std::vector<std::string> pass;  // options handed to every step-A child
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "epq --batch: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--batch") manifest = val("--batch");
        else if (a.rfind("--batch=", 0) == 0) manifest = a.substr(8);
        else if (a == "-j" || a == "--jobs") jobs = std::atoi(val("-j").c_str());
        else if (a.rfind("--jobs=", 0) == 0) jobs = std::atoi(a.c_str() + 7);
        else if (a.size() > 2 && a.rfind("-j", 0) == 0) jobs = std::atoi(a.c_str() + 2);
        else if (a == "--timeout") timeout_s = std::atof(val("--timeout").c_str());
        else if (a.rfind("--timeout=", 0) == 0) timeout_s = std::atof(a.c_str() + 10);
        else {
            if (a == "-t") time_flag = true;
            if (a[0] != '-' || a.size() == 1) return usage_error(("unexpected argument " + a).c_str());
            pass.push_back(a);
        }
    }
    if (manifest.empty()) return usage_error("no manifest");
    if (jobs < 1) jobs = 1;
    uint64_t timeout_us = timeout_s > 0 ? (uint64_t)(timeout_s * 1e6) : 0;

    std::string text;
    if (!read_file(manifest, text)) return usage_error(("cannot read manifest " + manifest).c_str());
    std::vector<Test> tests;
    {
        size_t pos = 0, lineno = 0;
        while (pos < text.size()) {
            size_t e = text.find('\n', pos);
            if (e == std::string::npos) e = text.size();
            std::string line = text.substr(pos, e - pos);
            pos = e + 1;
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> f;
            size_t s = 0;
            for (;;) {
                size_t t = line.find('\t', s);
                f.push_back(line.substr(s, t == std::string::npos ? std::string::npos : t - s));
                if (t == std::string::npos) break;
                s = t + 1;
            }
            if (f.size() != 5) {
                std::fprintf(stderr, "epq --batch: %s:%zu: expected 5 tab-separated fields, got %zu\n", manifest.c_str(),
                             lineno, f.size());
                return 2;
            }
            Test t;
            t.prog = f[0];
            t.in = f[1];
            t.out = f[2];
            t.err = f[3];
            t.code = f[4];
            tests.push_back(t);
        }
    }

    ::unsetenv(kBatchEmitEnv);
    std::string self = self_path(argv[0]);
    std::string dir = batch::cache_dir();
    std::string flags = batch::cflags();
    char uniq[64];
    std::snprintf(uniq, sizeof uniq, "batch-work-%d-%llx", (int)getpid(), (unsigned long long)now_us());
    std::string work = dir + "/" + uniq;
    bool have_work = mkdirs(work);

    // ---- A: front end + passes + C emission, one child of this executable per program ----
    std::vector<std::string> env_base;
    for (char** e = environ; *e; ++e) env_base.push_back(*e);
    size_t n_front_fail = 0;
    uint64_t TA = now_us();
    pool(tests.size(), jobs, timeout_us,
        [&](size_t i) -> pid_t {
            Test& t = tests[i];
            t.cfile = work + "/" + std::to_string(i) + ".c";
            ::unlink(t.cfile.c_str());
            posix_spawn_file_actions_t fa;
            posix_spawn_file_actions_init(&fa);
            posix_spawn_file_actions_addopen(&fa, 0, t.in == "-" ? "/dev/null" : t.in.c_str(), O_RDONLY, 0);
            posix_spawn_file_actions_addopen(&fa, 1, t.out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            posix_spawn_file_actions_addopen(&fa, 2, t.err.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            std::vector<std::string> args = {self};
            args.insert(args.end(), pass.begin(), pass.end());
            args.push_back(t.prog);
            std::vector<std::string> envs = env_base;
            if (have_work) envs.push_back(std::string(kBatchEmitEnv) + "=" + t.cfile);
            std::vector<char*> envp;
            for (auto& s : envs) envp.push_back(const_cast<char*>(s.c_str()));
            envp.push_back(nullptr);
            pid_t p = spawn(args, &fa, envp.data());
            posix_spawn_file_actions_destroy(&fa);
            if (p <= 0) {
                write_file(t.err, "epq-batch: cannot start front end for " + t.prog + " (bad stdin/output path?)\n",
                           false);
                write_file(t.out, "", false);
                finish(t, 2, "");
                ++n_front_fail;
            }
            return p;
        },
        [&](size_t i, int st, bool timed_out) {
            Test& t = tests[i];
            std::string note;
            int rc = decode_status(st, timed_out, note);
            std::string c;
            if (!timed_out && rc == 0 && read_file(t.cfile, c)) {
                t.hash = batch::hash(c);
                return;
            }
            ++n_front_fail;
            finish(t, rc, note);
        });

    // ---- B: compile each distinct C to <cache>/<hash>.o, in parallel ----
    uint64_t TB = now_us();
    std::map<std::string, size_t> first_of;  // hash -> first test index with it
    for (size_t i = 0; i < tests.size(); ++i)
        if (!tests[i].hash.empty() && !first_of.count(tests[i].hash)) first_of[tests[i].hash] = i;
    std::vector<std::string> todo;
    size_t n_cached = 0;
    for (auto& kv : first_of) {
        if (::access((dir + "/" + kv.first + ".o").c_str(), R_OK) == 0) ++n_cached;
        else todo.push_back(kv.first);
    }
    std::map<std::string, std::string> failed_cc;  // hash -> stderr text
    std::vector<std::string> tmp_obj(todo.size()), cc_log(todo.size());
    pool(todo.size(), jobs, 0,
        [&](size_t k) -> pid_t {
            const std::string& h = todo[k];
            tmp_obj[k] = dir + "/" + h + "." + uniq + ".o";
            cc_log[k] = work + "/" + h + ".cclog";
            std::vector<std::string> args = {"cc"};
            for (auto& f : split_ws(flags)) args.push_back(f);
            for (const char* a : {"-fwrapv", "-w", "-c"}) args.push_back(a);
            args.push_back("-Depq_entry=epq_entry_" + h);
            args.push_back("-Depq_entry_flush=epq_flush_" + h);
            args.push_back("-o");
            args.push_back(tmp_obj[k]);
            args.push_back(tests[first_of[h]].cfile);
            posix_spawn_file_actions_t fa;
            posix_spawn_file_actions_init(&fa);
            posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
            posix_spawn_file_actions_addopen(&fa, 1, cc_log[k].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            posix_spawn_file_actions_adddup2(&fa, 1, 2);
            pid_t p = spawn(args, &fa, environ);
            posix_spawn_file_actions_destroy(&fa);
            if (p <= 0) failed_cc[h] = "epq: C compilation failed (rc -1)\n";
            return p;
        },
        [&](size_t k, int st, bool) {
            const std::string& h = todo[k];
            int rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
            std::string log;
            read_file(cc_log[k], log);
            if (rc != 0 || ::rename(tmp_obj[k].c_str(), (dir + "/" + h + ".o").c_str()) != 0) {
                ::unlink(tmp_obj[k].c_str());
                if (rc == 0) rc = -1;
                failed_cc[h] = log + "epq: C compilation failed (rc " + std::to_string(rc) + ")\n";
            }
        });
    for (auto& t : tests) {
        if (t.hash.empty()) continue;
        auto it = failed_cc.find(t.hash);
        if (it != failed_cc.end()) finish(t, 2, it->second);
    }

    // ---- C: one dylib for the whole object set ----
    uint64_t TC = now_us();
    std::vector<std::string> objs;
    for (auto& kv : first_of)
        if (!failed_cc.count(kv.first)) objs.push_back(kv.first);
    std::sort(objs.begin(), objs.end());
    bool linked_new = false;
    void* handle = nullptr;
    std::string dylib;
    uint64_t TD0 = TC, TD1 = TC;
    auto fail_all = [&](const std::string& msg) {
        for (auto& t : tests)
            if (!t.hash.empty()) finish(t, 2, msg);
    };
    if (!objs.empty()) {
        std::string setkey = flags + '\n';
        for (auto& h : objs) setkey += h + '\n';
        dylib = dir + "/batch-" + hex64(setkey) + ".dylib";
        if (::access(dylib.c_str(), R_OK) != 0) {
            std::string list = work + "/objs.txt", lst;
            for (auto& h : objs) lst += dir + "/" + h + ".o\n";
            write_file(list, lst, false);
            std::string tmp = dir + "/batch-" + uniq + ".dylib";
            std::vector<std::string> args = {"cc"};
            for (auto& f : split_ws(flags)) args.push_back(f);
            for (const char* a : {"-w", "-dynamiclib", "-o"}) args.push_back(a);
            args.push_back(tmp);
            args.push_back("-Wl,-filelist," + list);
            pid_t p = spawn(args, nullptr, environ);
            int st = 0;
            if (p > 0)
                while (waitpid(p, &st, 0) < 0 && errno == EINTR) {
                }
            if (p <= 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0 || ::rename(tmp.c_str(), dylib.c_str()) != 0) {
                ::unlink(tmp.c_str());
                std::fprintf(stderr, "epq --batch: linking %zu objects failed\n", objs.size());
                fail_all("epq-batch: link failed\n");
                dylib.clear();
            } else {
                linked_new = true;
            }
        }
        TD0 = now_us();
        if (!dylib.empty()) {
            handle = dlopen(dylib.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle) {
                std::string m = std::string("epq: dlopen failed: ") + dlerror() + "\n";
                std::fputs(m.c_str(), stderr);
                fail_all(m);
            }
        }
        TD1 = now_us();
        if (handle)
            for (auto& t : tests) {
                if (t.hash.empty()) continue;
                t.entry = (int (*)(void))dlsym(handle, ("epq_entry_" + t.hash).c_str());
                t.flush = (void (*)(void))dlsym(handle, ("epq_flush_" + t.hash).c_str());
                if (!t.entry) finish(t, 2, "epq: entry point missing\n");
            }
    }

    // ---- D: run every compiled test in a forked child of this (pristine) process ----
    uint64_t TR = now_us();
    std::vector<size_t> to_run;
    for (size_t i = 0; i < tests.size(); ++i)
        if (!tests[i].hash.empty()) to_run.push_back(i);
    std::fflush(nullptr);
    pool(to_run.size(), jobs, timeout_us,
        [&](size_t k) -> pid_t {
            Test& t = tests[to_run[k]];
            pid_t p = fork();
            if (p == 0) {
                int fi = ::open(t.in == "-" ? "/dev/null" : t.in.c_str(), O_RDONLY);
                int fo = ::open(t.out.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
                int fe = ::open(t.err.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
                if (fi < 0 || fo < 0 || fe < 0) _exit(2);
                dup2(fi, 0);
                dup2(fo, 1);
                dup2(fe, 2);
                if (fi > 2) ::close(fi);
                if (fo > 2) ::close(fo);
                if (fe > 2) ::close(fe);
                int rc = batch::run_entry(t.entry, t.flush, time_flag);
                _exit(rc);
            }
            if (p < 0) finish(t, 2, "epq-batch: fork failed\n");
            return p;
        },
        [&](size_t k, int st, bool timed_out) {
            Test& t = tests[to_run[k]];
            std::string note;
            int rc = decode_status(st, timed_out, note);
            finish(t, rc, note);
        });
    uint64_t TE = now_us();

    // ---- cleanup + summary ----
    if (have_work) {
        for (auto& t : tests) ::unlink(t.cfile.c_str());
        for (auto& h : todo) ::unlink((work + "/" + h + ".cclog").c_str());
        ::unlink((work + "/objs.txt").c_str());
        ::rmdir(work.c_str());
    }
    std::fprintf(stderr,
                 "epq-batch: %zu programs (%zu ended before running), %zu objects (%zu compiled, %zu cached, %zu failed),"
                 " dylib %s; front end %.2f s, cc %.2f s, link %.2f s, dlopen %.2f s, run %.2f s, total %.2f s\n",
                 tests.size(), n_front_fail, first_of.size(), todo.size() - failed_cc.size(), n_cached,
                 failed_cc.size(), dylib.empty() ? "none" : (linked_new ? "new" : "cached"), (TB - TA) / 1e6,
                 (TC - TB) / 1e6, (TD0 - TC) / 1e6, (TD1 - TD0) / 1e6, (TE - TR) / 1e6, (TE - T0) / 1e6);
    return 0;
}

}  // namespace epq
