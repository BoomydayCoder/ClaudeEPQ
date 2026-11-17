// Second tier: background C-backend compile, picked up by later runs.
// See tierup.hpp for the protocol; DESIGN.md §7 for the rationale.
//
// Files in $EPQ_CACHE_DIR/tier/ (default ~/.cache/claude-epq/tier/):
//   <key>.ready    marker, written last (atomically): names the dylib and its
//                  size/inode/mtime; present => the dylib was compiled and
//                  dlopen-warmed by the helper.
//   <key>.dylib    the compiled program.
//   <key>.lock     flock()ed by the one helper compiling <key> (the kernel drops
//                  the lock when the helper dies, so a crashed helper can never
//                  leave a lock that blocks later compiles).
//   <key>.tmp-*    helper temporaries (removed on every exit path; leftovers of a
//                  SIGKILLed helper are swept after an hour).
#include "tierup.hpp"

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <pthread/spawn.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cgen.hpp"

extern char** environ;

namespace epq {
namespace tierup {
namespace {

// Same default as the driver's AOT path (driver.cpp kDefaultCFlags).
constexpr const char* kDefaultCFlags = "-O3 -fwrapv -mcpu=native";
constexpr const char* kMarkerMagic = "epq-tier 1";
constexpr unsigned kDefaultTierMs = 20;
constexpr unsigned long long kDefaultCapMB = 256;

uint64_t now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

// ---- key ---------------------------------------------------------------------

// Two-lane 64-bit multiply/rotate hash, 16 bytes per step (~3 us for 10 KB).
// Not cryptographic; collisions only matter between sources of one user.
struct Hasher {
    uint64_t a = 0x243f6a8885a308d3ull, b = 0x13198a2e03707344ull, n = 0;
    static uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
    static uint64_t fmix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= x >> 33;
        return x;
    }
    void step(uint64_t w0, uint64_t w1) {
        a = rotl((a ^ w0) * 0x87c37b91114253d5ull, 31) + b;
        b = rotl((b ^ w1) * 0x4cf5ad432745937full, 33) + a;
    }
    void feed(const void* p, size_t len) {
        const unsigned char* s = static_cast<const unsigned char*>(p);
        n += len;
        while (len >= 16) {
            uint64_t w0, w1;
            std::memcpy(&w0, s, 8);
            std::memcpy(&w1, s + 8, 8);
            step(w0, w1);
            s += 16;
            len -= 16;
        }
        unsigned char t[16] = {0};
        std::memcpy(t, s, len);
        t[15] = (unsigned char)(len + 1);  // distinguishes the tail length
        uint64_t w0, w1;
        std::memcpy(&w0, t, 8);
        std::memcpy(&w1, t + 8, 8);
        step(w0, w1);
    }
    void field(const std::string& s) { feed(s.data(), s.size()); }
    void field(const char* s) { feed(s, std::strlen(s)); }
    std::string hex() const {
        uint64_t x = fmix(a ^ n), y = fmix(b + x);
        x = fmix(x + y);
        char buf[40];
        std::snprintf(buf, sizeof buf, "%016llx%016llx", (unsigned long long)x, (unsigned long long)y);
        return buf;
    }
};

// Identity of this epq build: the main executable's LC_UUID (read from the
// mapped image, no syscall). Changes on every relink with different code.
void feed_build_id(Hasher& h) {
    const auto* mh = reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(0));
    if (mh && mh->magic == MH_MAGIC_64) {
        const char* p = reinterpret_cast<const char*>(mh + 1);
        for (uint32_t i = 0; i < mh->ncmds; ++i) {
            const auto* lc = reinterpret_cast<const load_command*>(p);
            if (lc->cmd == LC_UUID) {
                h.feed(reinterpret_cast<const uuid_command*>(lc)->uuid, 16);
                return;
            }
            p += lc->cmdsize;
        }
    }
    // No UUID (unusual linker): fall back to the executable's size + mtime.
    char exe[PATH_MAX];
    uint32_t sz = sizeof exe;
    struct stat st;
    if (_NSGetExecutablePath(exe, &sz) == 0 && ::stat(exe, &st) == 0) {
        int64_t v[3] = {(int64_t)st.st_size, (int64_t)st.st_mtimespec.tv_sec, (int64_t)st.st_mtimespec.tv_nsec};
        h.feed(v, sizeof v);
    }
}

// Environment variables that steer tier-up itself (or only say where the cache
// lives) are not part of the key; every other EPQ_* variable is (EPQ_CFLAGS
// today, any future pass toggle tomorrow).
bool env_in_key(const char* e) {
    if (std::strncmp(e, "EPQ_", 4) != 0) return false;
    return std::strncmp(e, "EPQ_CACHE_DIR=", 14) != 0 && std::strncmp(e, "EPQ_TIER_", 9) != 0 &&
           std::strncmp(e, "EPQ_NO_TIERUP=", 14) != 0;
}

std::string compute_key(const std::string& source, const Options& o) {
    Hasher h;
    h.field(source);
    feed_build_id(h);
    h.field(kEmitterVersion);
    h.field(kMarkerMagic);
    const char* envs[64];
    size_t ne = 0;
    for (char** e = environ; *e && ne < 64; ++e)
        if (env_in_key(*e)) envs[ne++] = *e;
    std::sort(envs, envs + ne, [](const char* x, const char* y) { return std::strcmp(x, y) < 0; });
    for (size_t i = 0; i < ne; ++i) h.field(envs[i]);
    h.field(o.mono ? "mono" : "poly");
    h.field(o.no_dedup ? "nodedup" : "dedup");
    for (auto& s : o.no_pass) h.field("-" + s);
    for (auto& s : o.only_pass) h.field("+" + s);
    for (auto& s : o.enable_pass) h.field("^" + s);  // integration: default-off passes
    h.field(o.no_records ? "norec" : "rec");          // integration: --no-records changes the code
    return h.hex();
}

// ---- paths -------------------------------------------------------------------

std::string tier_dir() {
    std::string dir;
    if (const char* d = std::getenv("EPQ_CACHE_DIR"); d && *d) {
        dir = d;
    } else {
        const char* home = std::getenv("HOME");
        dir = std::string(home ? home : "/tmp") + "/.cache/claude-epq";
    }
    return dir + "/tier";
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

void tier_log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
// EPQ_TIER_LOG=FILE: append one line per tier-up event (tests use this).
void tier_log(const char* fmt, ...) {
    const char* f = std::getenv("EPQ_TIER_LOG");
    if (!f || !*f) return;
    char buf[512];
    int n = std::snprintf(buf, sizeof buf, "%d ", (int)getpid());
    va_list ap;
    va_start(ap, fmt);
    n += std::vsnprintf(buf + n, sizeof buf - (size_t)n, fmt, ap);
    va_end(ap);
    if (n >= (int)sizeof buf - 1) n = (int)sizeof buf - 2;
    buf[n++] = '\n';
    int fd = ::open(f, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return;
    (void)!::write(fd, buf, (size_t)n);
    ::close(fd);
}

// ---- marker ------------------------------------------------------------------

struct Marker {
    std::string dylib;  // basename inside the tier directory
    long long size = -1, ino = -1, mtime_s = -1, mtime_ns = -1;
};

std::string marker_text(const std::string& key, const std::string& dylib, const struct stat& st) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s\n%s\n%s\n%lld %lld %lld %lld\n", kMarkerMagic, key.c_str(), dylib.c_str(),
                  (long long)st.st_size, (long long)st.st_ino, (long long)st.st_mtimespec.tv_sec,
                  (long long)st.st_mtimespec.tv_nsec);
    return buf;
}

// Parses an already read marker; false if it is not one for `key`.
bool parse_marker(const char* buf, size_t n, const std::string& key, Marker& m) {
    std::string s(buf, n);
    size_t p1 = s.find('\n'), p2 = p1 == std::string::npos ? p1 : s.find('\n', p1 + 1),
           p3 = p2 == std::string::npos ? p2 : s.find('\n', p2 + 1);
    if (p3 == std::string::npos) return false;
    if (s.compare(0, p1, kMarkerMagic) != 0 || p1 != std::strlen(kMarkerMagic)) return false;
    if (s.compare(p1 + 1, p2 - p1 - 1, key) != 0 || p2 - p1 - 1 != key.size()) return false;
    m.dylib = s.substr(p2 + 1, p3 - p2 - 1);
    if (m.dylib.empty() || m.dylib.find('/') != std::string::npos || m.dylib[0] == '.') return false;
    return std::sscanf(s.c_str() + p3 + 1, "%lld %lld %lld %lld", &m.size, &m.ino, &m.mtime_s, &m.mtime_ns) == 4;
}

bool stat_matches(const struct stat& st, const Marker& m) {
    return S_ISREG(st.st_mode) && st.st_size == m.size && (long long)st.st_ino == m.ino &&
           st.st_mtimespec.tv_sec == m.mtime_s && st.st_mtimespec.tv_nsec == m.mtime_ns;
}

// Reads and validates <dir>/<key>.ready. Returns 1 valid, 0 absent, -1 present
// but invalid (unparsable, wrong key, dylib missing or changed).
int read_marker(const std::string& dir, const std::string& key, Marker& m, std::string* dylib_path) {
    std::string mpath = dir + "/" + key + ".ready";
    int fd = ::open(mpath.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    char buf[512];
    ssize_t n = ::read(fd, buf, sizeof buf);
    ::close(fd);
    if (n <= 0 || !parse_marker(buf, (size_t)n, key, m)) return -1;
    *dylib_path = dir + "/" + m.dylib;
    struct stat st;
    if (::stat(dylib_path->c_str(), &st) != 0 || !stat_matches(st, m)) return -1;
    return 1;
}

void drop_artefact(const std::string& dir, const std::string& key, const Marker* m) {
    ::unlink((dir + "/" + key + ".ready").c_str());
    if (m && !m->dylib.empty() && m->dylib.find('/') == std::string::npos) ::unlink((dir + "/" + m->dylib).c_str());
}

// ---- spawning ----------------------------------------------------------------

struct SpawnOpts {
    bool new_session = false;
    bool devnull_stdio = false;
};

// posix_spawn with every fd except 0-2 closed, utility QoS, default signals.
pid_t spawn(const char* path, bool search, const std::vector<std::string>& args, const SpawnOpts& so) {
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    posix_spawnattr_t attr;
    posix_spawn_file_actions_t fa;
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_init(&fa);
    short flags = POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF;
    if (so.new_session) flags |= POSIX_SPAWN_SETSID;
    posix_spawnattr_setflags(&attr, flags);
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    posix_spawnattr_setsigmask(&attr, &none);
    posix_spawnattr_setsigdefault(&attr, &all);
    const char* q = std::getenv("EPQ_TIER_QOS");
    posix_spawnattr_set_qos_class_np(&attr, q && !std::strcmp(q, "background") ? QOS_CLASS_BACKGROUND
                                            : q && !std::strcmp(q, "default")  ? QOS_CLASS_DEFAULT
                                                                               : QOS_CLASS_UTILITY);
    if (so.devnull_stdio) {
        posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
        posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
        posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    } else {
        for (int fd = 0; fd <= 2; ++fd) posix_spawn_file_actions_addinherit_np(&fa, fd);
    }
    pid_t pid = -1;
    int err = search ? posix_spawnp(&pid, path, &fa, &attr, argv.data(), environ)
                     : posix_spawn(&pid, path, &fa, &attr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&attr);
    return err == 0 ? pid : -1;
}

int wait_rc(pid_t pid) {
    int st = 0;
    while (waitpid(pid, &st, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

// True if some helper currently holds <key>.lock.
bool lock_held(const std::string& lock_path) {
    int fd = ::open(lock_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    bool held = ::flock(fd, LOCK_EX | LOCK_NB) != 0 && errno == EWOULDBLOCK;
    ::close(fd);  // releases our lock if we got it
    return held;
}

// ---- foreground trigger ------------------------------------------------------

struct Trigger {
    std::string file, dir, key;
    std::vector<std::string> no_pass, only_pass, enable_pass;
    bool mono = false, no_dedup = false, verbose = false, no_records = false;
    unsigned ms = kDefaultTierMs;
    uint64_t t0 = 0;
    pid_t pid = 0;  // helper already launched (EPQ_TIER_MS=0)
};

pid_t launch_helper(const Trigger& t) {
    if (lock_held(t.dir + "/" + t.key + ".lock")) {
        tier_log("trigger-skip-locked %s", t.key.c_str());
        return -1;
    }
    char exe[PATH_MAX];
    uint32_t sz = sizeof exe;
    if (_NSGetExecutablePath(exe, &sz) != 0) return -1;
    std::vector<std::string> args = {exe, "--tier-compile"};
    for (auto& s : t.no_pass) args.push_back("--no-pass=" + s);
    for (auto& s : t.only_pass) args.push_back("--only-pass=" + s);
    for (auto& s : t.enable_pass) args.push_back("--enable-pass=" + s);
    if (t.no_records) args.push_back("--no-records");
    if (t.mono) args.push_back("--mono");
    if (t.no_dedup) args.push_back("--no-dedup");
    args.push_back(t.file);
    SpawnOpts so;
    so.new_session = true;
    so.devnull_stdio = true;
    pid_t pid = spawn(exe, false, args, so);
    tier_log("spawn %s after %.1f ms pid %d", t.key.c_str(), (now_us() - t.t0) / 1000.0, (int)pid);
    if (t.verbose && pid > 0) {
        char buf[160];
        int n = std::snprintf(buf, sizeof buf, "epq: tier-up helper %d started after %.1f ms\n", (int)pid,
                              (now_us() - t.t0) / 1000.0);
        (void)!::write(2, buf, (size_t)n);
    }
    return pid;
}

void* trigger_thread(void* p) {
    Trigger* t = static_cast<Trigger*>(p);
    if (t->pid == 0) {
        struct timespec ts = {(time_t)(t->ms / 1000), (long)(t->ms % 1000) * 1000000L};
        while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
        }
        t->pid = launch_helper(*t);
    }
    // Reap the helper if we are still alive when it finishes (no zombie); if
    // we exit first it is reparented to launchd. Its own session either way.
    if (t->pid > 0) wait_rc(t->pid);
    return nullptr;
}

// ---- helper ------------------------------------------------------------------

// Removes files on scope exit unless released.
struct Cleanup {
    std::vector<std::string> paths;
    ~Cleanup() {
        for (auto& p : paths) ::unlink(p.c_str());
    }
};

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

bool write_file(const std::string& path, const std::string& s) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return false;
    const char* p = s.data();
    size_t n = s.size();
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w <= 0) {
            ::close(fd);
            return false;
        }
        p += w;
        n -= (size_t)w;
    }
    return ::close(fd) == 0;
}

// Front end + inference + TIR + passes + emit, with fd 1 redirected to `cpath`.
struct EmitJob {
    CompileUnit* cu;
    const Options* opts;
    int rc;
};

void* emit_body(void* p) {
    EmitJob* j = static_cast<EmitJob*>(p);
    j->rc = 1;
    if (!parse_unit(*j->cu)) return nullptr;
    // Programs with lexer diagnostics print them on every run; a cached dylib
    // would skip the front end and lose them, so such programs never tier up.
    if (!j->cu->lex.diags.empty()) return nullptr;
    if (!resolve_unit(*j->cu)) return nullptr;
    j->rc = run_backend(j->cu->prog, *j->opts);
    return nullptr;
}

bool emit_c_file(CompileUnit& cu, const Options& base, const std::string& cpath) {
    Options o = base;
    o.emit_c = true;
    o.time = o.verbose = o.dump_ast = o.dump_tir = o.dump_types = false;
    int fd = ::open(cpath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return false;
    int saved = ::dup(1);
    ::dup2(fd, 1);
    ::close(fd);
    size_t stack = (64u << 20) + cu.source.size() * 4096;
    if (stack > (size_t(32) << 30)) stack = size_t(32) << 30;
    cu.stack_bytes = stack;
    EmitJob job{&cu, &o, 1};
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, stack);
    pthread_t th;
    if (pthread_create(&th, &attr, emit_body, &job) == 0) {
        pthread_join(th, nullptr);
    } else {
        cu.stack_bytes = 8u << 20;
        emit_body(&job);
    }
    ::dup2(saved, 1);
    ::close(saved);
    struct stat st;
    return job.rc == 0 && ::stat(cpath.c_str(), &st) == 0 && st.st_size > 0;
}

// Oldest-first eviction of compiled programs beyond the cap; also sweeps
// temporaries of helpers that were killed more than an hour ago.
void housekeeping(const std::string& dir, const std::string& keep_key) {
    unsigned long long cap_mb = kDefaultCapMB;
    if (const char* c = std::getenv("EPQ_TIER_CACHE_MB"); c && *c) cap_mb = std::strtoull(c, nullptr, 10);
    DIR* d = ::opendir(dir.c_str());
    if (!d) return;
    struct Ent {
        std::string key;
        long long size;
        time_t mtime;
    };
    std::vector<Ent> dylibs;
    unsigned long long total = 0;
    time_t now = time(nullptr);
    while (struct dirent* e = ::readdir(d)) {
        std::string n = e->d_name;
        if (n.size() < 33 || n[0] == '.') continue;
        std::string path = dir + "/" + n;
        struct stat st;
        if (::lstat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (n.find(".tmp-") != std::string::npos) {
            if (now - st.st_mtimespec.tv_sec > 3600) ::unlink(path.c_str());
        } else if (n.size() == 32 + 6 && n.compare(32, 6, ".dylib") == 0) {
            dylibs.push_back({n.substr(0, 32), (long long)st.st_size, st.st_mtimespec.tv_sec});
            total += (unsigned long long)st.st_size;
        }
    }
    ::closedir(d);
    const unsigned long long cap = cap_mb << 20;
    if (total <= cap) return;
    std::sort(dylibs.begin(), dylibs.end(), [](const Ent& x, const Ent& y) { return x.mtime < y.mtime; });
    for (auto& e : dylibs) {
        if (total <= cap) break;
        if (e.key == keep_key) continue;
        ::unlink((dir + "/" + e.key + ".ready").c_str());  // marker first: no reader sees a dangling one
        ::unlink((dir + "/" + e.key + ".dylib").c_str());
        total -= (unsigned long long)e.size;
        tier_log("evict %s", e.key.c_str());
    }
}

int helper_main(CompileUnit& cu, const Options& opts) {
    // Never touch the terminal, whoever started us.
    int dn = ::open("/dev/null", O_RDWR | O_CLOEXEC);
    if (dn >= 0) {
        ::dup2(dn, 0);
        ::dup2(dn, 1);
        ::dup2(dn, 2);
        if (dn > 2) ::close(dn);
    }
    signal(SIGPIPE, SIG_IGN);
    setpriority(PRIO_PROCESS, 0, 10);
    uint64_t t0 = now_us();
    std::string dir = tier_dir();
    std::string key = compute_key(cu.source, opts);
    if (!mkdirs(dir)) return 1;
    Marker m;
    std::string dpath;
    if (read_marker(dir, key, m, &dpath) == 1) return 0;

    std::string lock = dir + "/" + key + ".lock";
    int lfd = ::open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (lfd < 0) return 1;
    if (::flock(lfd, LOCK_EX | LOCK_NB) != 0) {
        tier_log("helper-locked %s", key.c_str());
        ::close(lfd);
        return 0;
    }
    // Under the lock: someone may have finished while we were starting.
    if (read_marker(dir, key, m, &dpath) == 1) {
        ::unlink(lock.c_str());
        ::close(lfd);
        return 0;
    }
    // At most EPQ_TIER_JOBS (default 2) compiles of different programs run at
    // once. No free slot: give up quietly (a later long run retries) rather
    // than queue, so bursts of runs never pile up sleeping helpers.
    int jobs = 2;
    if (const char* j = std::getenv("EPQ_TIER_JOBS"); j && *j) jobs = std::max(1, std::atoi(j));
    int sfd = -1;
    for (int i = 0; i < jobs && sfd < 0; ++i) {
        std::string sp = dir + "/.slot-" + std::to_string((i + (int)getpid()) % jobs);
        int fd = ::open(sp.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) continue;
        if (::flock(fd, LOCK_EX | LOCK_NB) == 0)
            sfd = fd;
        else
            ::close(fd);
    }
    if (sfd < 0) {
        tier_log("helper-busy %s", key.c_str());
        ::unlink(lock.c_str());
        ::close(lfd);
        return 0;
    }
    tier_log("compile-start %s after %.1f ms", key.c_str(), (now_us() - t0) / 1000.0);
    t0 = now_us();
    Cleanup tmp;
    char uniq[64];
    std::snprintf(uniq, sizeof uniq, ".tmp-%d-%llx", (int)getpid(), (unsigned long long)t0);
    std::string cpath = dir + "/" + key + uniq + ".c";
    std::string tdylib = dir + "/" + key + uniq + ".dylib";
    std::string tmarker = dir + "/" + key + uniq + ".ready";
    tmp.paths = {cpath, tdylib, tmarker, lock};  // the lock goes last, after the marker exists

    bool ok = emit_c_file(cu, opts, cpath);
    uint64_t t_emit = now_us();
    int rc = -1;
    if (ok) {
        const char* fl = std::getenv("EPQ_CFLAGS");
        std::vector<std::string> args = {"cc"};
        for (auto& f : split_flags(fl ? fl : kDefaultCFlags)) args.push_back(f);
        for (const char* a : {"-fwrapv", "-w", "-dynamiclib", "-o"}) args.push_back(a);
        args.push_back(tdylib);
        args.push_back(cpath);
        pid_t pid = spawn("cc", true, args, SpawnOpts{});
        rc = pid > 0 ? wait_rc(pid) : -1;
        ok = rc == 0;
    }
    uint64_t t_cc = now_us();
    std::string final_dylib = key + ".dylib";
    std::string fpath = dir + "/" + final_dylib;
    if (ok) ok = ::rename(tdylib.c_str(), fpath.c_str()) == 0;
    // Warm: the first dlopen of a new dylib pays the code-signature / policy
    // scan; do it here so no foreground run ever does.
    uint64_t t_dl0 = now_us();
    if (ok) {
        void* h = dlopen(fpath.c_str(), RTLD_NOW | RTLD_LOCAL);
        ok = h && dlsym(h, "epq_entry");
        if (h) dlclose(h);
        if (!ok) ::unlink(fpath.c_str());
    }
    uint64_t t_dl1 = now_us();
    struct stat st;
    if (ok) ok = ::stat(fpath.c_str(), &st) == 0;
    if (ok) ok = write_file(tmarker, marker_text(key, final_dylib, st));
    if (ok) ok = ::rename(tmarker.c_str(), (dir + "/" + key + ".ready").c_str()) == 0;
    if (!ok) {
        ::unlink(fpath.c_str());
        tier_log("compile-failed %s cc-rc %d", key.c_str(), rc);
    } else {
        tier_log("compile-done %s emit %.1f ms cc %.1f ms warm-dlopen %.1f ms total %.1f ms", key.c_str(),
                 (t_emit - t0) / 1000.0, (t_cc - t_emit) / 1000.0, (t_dl1 - t_dl0) / 1000.0,
                 (now_us() - t0) / 1000.0);
        housekeeping(dir, key);
    }
    // Cleanup removes the temporaries and then the lock file while we still
    // hold the lock: anyone locking a fresh lock file afterwards sees the marker.
    {
        Cleanup done;
        done.paths.swap(tmp.paths);
    }
    ::close(lfd);
    if (sfd >= 0) ::close(sfd);
    return ok ? 0 : 1;
}

bool tier_disabled() { return env_on("EPQ_NO_TIERUP"); }

bool has_arg(int argc, char** argv, const char* a) {
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], a)) return true;
    return false;
}

bool engine_forced(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (!std::strncmp(argv[i], "--engine=", 9) || !std::strcmp(argv[i], "--jit") || !std::strcmp(argv[i], "--aot"))
            return true;
    return false;
}

int status_main(CompileUnit& cu, const Options& opts) {
    std::string dir = tier_dir();
    std::string key = compute_key(cu.source, opts);
    Marker m;
    std::string dpath;
    int r = read_marker(dir, key, m, &dpath);
    if (r == 1) {
        std::printf("ready %s\n", dpath.c_str());
        return 0;
    }
    if (lock_held(dir + "/" + key + ".lock"))
        std::printf("compiling %s\n", key.c_str());
    else
        std::printf("%s %s\n", r < 0 ? "invalid" : "absent", key.c_str());
    return 1;
}

}  // namespace

bool is_flag(const char* a) {
    return !std::strcmp(a, "--tier-compile") || !std::strcmp(a, "--tier-status") || !std::strcmp(a, "--tier-clear");
}

bool command(int argc, char** argv, int* rc) {
    if (!has_arg(argc, argv, "--tier-clear")) return false;
    std::string dir = tier_dir();
    int removed = 0;
    if (DIR* d = ::opendir(dir.c_str())) {
        while (struct dirent* e = ::readdir(d)) {
            if (e->d_name[0] == '.') continue;
            if (::unlink((dir + "/" + e->d_name).c_str()) == 0) ++removed;
        }
        ::closedir(d);
    }
    std::printf("tier cache %s: removed %d files\n", dir.c_str(), removed);
    *rc = 0;
    return true;
}

bool start(CompileUnit& cu, const Options& opts, int argc, char** argv, int* rc) {
    if (has_arg(argc, argv, "--tier-compile")) {
        *rc = helper_main(cu, opts);
        return true;
    }
    if (has_arg(argc, argv, "--tier-status")) {
        *rc = status_main(cu, opts);
        return true;
    }
    // Engine policy "auto" only, a real run of a named file only.
    if (opts.file == "-" || opts.emit_c || opts.dump_tir || opts.dump_types || opts.dump_ast || opts.parse_only ||
        opts.dump_resolved || opts.frontend_time || opts.verify_tir || !opts.dump_tir_after.empty() || tier_disabled() ||
        engine_forced(argc, argv))
        return false;

    uint64_t t0 = now_us();
    std::string dir = tier_dir();
    std::string key = compute_key(cu.source, opts);
    uint64_t tk = now_us();
    Marker m;
    std::string dpath;
    int r = read_marker(dir, key, m, &dpath);
    uint64_t t1 = now_us();
    if (r == 1) {
        void* h = dlopen(dpath.c_str(), RTLD_NOW | RTLD_LOCAL);
        uint64_t t2 = now_us();
        if (opts.verbose)
            std::fprintf(stderr, "epq: tier-up hit %s (probe %llu us, dlopen %llu us)\n", dpath.c_str(),
                         (unsigned long long)(t1 - t0), (unsigned long long)(t2 - t1));
        if (h) {
            tier_log("hit %s", key.c_str());
            int st = run_dylib_entry(h, opts);
            if (st >= 0) {
                *rc = st;
                return true;
            }
        }
        if (opts.verbose) std::fprintf(stderr, "epq: tier-up artefact unusable, falling back\n");
        tier_log("bad-artefact %s", key.c_str());
        drop_artefact(dir, key, &m);
    } else if (r < 0) {
        tier_log("bad-marker %s", key.c_str());
        drop_artefact(dir, key, m.dylib.empty() ? nullptr : &m);
    } else if (opts.verbose) {
        std::fprintf(stderr, "epq: tier-up miss (probe %llu us: key %llu us, open %llu us)\n",
                     (unsigned long long)(t1 - t0), (unsigned long long)(tk - t0), (unsigned long long)(t1 - tk));
    }

    // Miss: arm the one-shot trigger; the normal engine runs meanwhile.
    t1 = now_us();
    auto* t = new Trigger;  // owned by the trigger thread for the process lifetime
    t->file = opts.file;
    t->dir = dir;
    t->key = key;
    t->no_pass = opts.no_pass;
    t->only_pass = opts.only_pass;
    t->enable_pass = opts.enable_pass;
    t->no_records = opts.no_records;
    t->mono = opts.mono;
    t->no_dedup = opts.no_dedup;
    t->verbose = opts.verbose;
    t->t0 = t0;
    if (const char* ms = std::getenv("EPQ_TIER_MS"); ms && *ms) t->ms = (unsigned)std::strtoul(ms, nullptr, 10);
    // EPQ_TIER_MS=0 (testing): launch before the program starts, so even a
    // program that ends at once tiers up; the thread then only reaps.
    if (t->ms == 0) t->pid = launch_helper(*t);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    // A static stack: saves pthread_create's stack mapping (measured ~5 us).
    alignas(16384) static char trigger_stack[64 * 1024];
    pthread_attr_setstack(&attr, trigger_stack, sizeof trigger_stack);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_t th;
    unsigned ms = t->ms;
    if (pthread_create(&th, &attr, trigger_thread, t) != 0) delete t;
    pthread_attr_destroy(&attr);
    if (opts.verbose)
        std::fprintf(stderr, "epq: tier-up trigger armed (%u ms) in %llu us\n", ms,
                     (unsigned long long)(now_us() - t1));
    return false;
}

}  // namespace tierup
}  // namespace epq
