"""Shared helpers for the ClaudeEPQ test tools (Python 3.9 stdlib only).

- engine specs:  NAME=COMMAND  (COMMAND is shell-split; flags and FILE are appended)
- run_engine():  run one program with a timeout, stdin from a file/bytes, in its
                 own process group (killed as a group on timeout), output capped.
"""
import os
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
BUILD = os.path.join(REPO, "build")
NEW_BIN = os.path.join(BUILD, "epq")

# The baseline (the original implementation, the oracle): $EPQ_BASELINE, else
# build/tools/ref/epq_ref (where tools/build_reference.sh puts it by default).
BASELINE_HELP = ("the baseline binary (the original EPQ implementation) was not found.\n"
                 "  Build it:  git clone https://github.com/BoomydayCoder/EPQ ../EPQ && "
                 "tools/build_reference.sh ../EPQ build/tools/ref\n"
                 "  or point EPQ_BASELINE at an existing epq_ref binary.")


def find_baseline():
    for c in (os.environ.get("EPQ_BASELINE"), os.path.join(BUILD, "tools", "ref", "epq_ref")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def require_baseline():
    """The baseline's path, or exit 2 with instructions."""
    b = find_baseline()
    if not b:
        sys.stderr.write("error: " + BASELINE_HELP + "\n")
        sys.exit(2)
    return b


class Engine:
    def __init__(self, name, argv):
        self.name = name
        self.argv = list(argv)

    def __repr__(self):
        return "%s=%s" % (self.name, " ".join(shlex.quote(a) for a in self.argv))

    def is_new(self):
        """True if this engine is the ClaudeEPQ binary (understands --parse-only)."""
        return any(os.path.basename(a) == "epq" for a in self.argv[:1])


def parse_engine(spec):
    if "=" not in spec:
        raise ValueError("engine spec must be NAME=COMMAND: %r" % spec)
    name, cmd = spec.split("=", 1)
    # a bare path (possibly containing spaces) is one argv element; otherwise shell-split
    argv = [cmd] if os.path.isfile(cmd.strip()) else shlex.split(cmd)
    if not argv:
        raise ValueError("empty command in %r" % spec)
    return Engine(name.strip(), argv)


def default_engines(include_pyref=False):
    out = []
    b = find_baseline()
    if b:
        out.append(Engine("baseline", [b]))
    else:
        sys.stderr.write("warning: " + BASELINE_HELP + "\n  Comparing the other engines with each other only.\n")
    if os.path.isfile(NEW_BIN) and os.access(NEW_BIN, os.X_OK):
        out.append(Engine("jit", [NEW_BIN, "--engine=jit"]))
        out.append(Engine("aot", [NEW_BIN, "--engine=aot"]))
    if include_pyref:
        out.append(Engine("pyref", [sys.executable, os.path.join(TOOLS, "pyref.py")]))
    return out


def engines_from_args(specs, include_pyref=False):
    """specs: list of NAME=CMD strings (may be empty -> defaults).  The first engine
    is the reference that others are compared against; 'baseline' is moved first."""
    engs = [parse_engine(s) for s in specs] if specs else default_engines(include_pyref)
    if not engs:
        raise SystemExit("no engines (baseline not found; pass --engine NAME=CMD)")
    names = [e.name for e in engs]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate engine names: %s" % names)
    for i, e in enumerate(engs):
        if e.name == "baseline" and i:
            engs.insert(0, engs.pop(i))
            break
    return engs


class RunResult:
    __slots__ = ("stdout", "stderr", "rc", "timed_out", "signal", "wall", "truncated", "spawn_error")

    def __init__(self):
        self.stdout = b""
        self.stderr = b""
        self.rc = None
        self.timed_out = False
        self.signal = None
        self.wall = 0.0
        self.truncated = False
        self.spawn_error = None

    @property
    def crashed(self):
        return self.signal is not None and not self.timed_out

    def status(self):
        if self.spawn_error:
            return "spawn-error"
        if self.timed_out:
            return "timeout"
        if self.truncated:
            return "output-limit"
        if self.signal is not None:
            return "signal %d" % self.signal
        return "exit %d" % self.rc

    def to_dict(self):
        return {"rc": self.rc, "timed_out": self.timed_out, "signal": self.signal,
                "wall": self.wall, "truncated": self.truncated, "status": self.status()}


def _kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_cmd(argv, stdin_data=None, stdin_path=None, timeout=10.0, max_output=64 << 20,
            discard_stdout=False):
    """Run argv; stdin from bytes or a file path (else /dev/null).  Returns RunResult.
    The child is the leader of a new process group; on timeout or output overflow the
    whole group is SIGKILLed.  Never leaves processes behind."""
    r = RunResult()
    stdin_f = None
    try:
        if stdin_path is not None:
            stdin_f = open(stdin_path, "rb")
            stdin_arg = stdin_f
        elif stdin_data is not None:
            stdin_arg = subprocess.PIPE
        else:
            stdin_arg = subprocess.DEVNULL
        t0 = time.perf_counter()
        try:
            p = subprocess.Popen(argv, stdin=stdin_arg,
                                 stdout=subprocess.DEVNULL if discard_stdout else subprocess.PIPE,
                                 stderr=subprocess.PIPE, start_new_session=True)
        except OSError as e:
            r.spawn_error = str(e)
            r.rc = 127
            return r
        deadline = t0 + timeout
        sel = selectors.DefaultSelector()
        bufs = {}
        if p.stdout is not None:
            sel.register(p.stdout, selectors.EVENT_READ)
            bufs[p.stdout] = []
        sel.register(p.stderr, selectors.EVENT_READ)
        bufs[p.stderr] = []
        if stdin_data is not None and stdin_path is None:
            def _feed(pipe=p.stdin, data=stdin_data):
                try:
                    pipe.write(data)
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
            threading.Thread(target=_feed, daemon=True).start()
        total = 0
        open_pipes = len(bufs)
        try:
            while open_pipes:
                left = deadline - time.perf_counter()
                if left <= 0:
                    r.timed_out = True
                    break
                for key, _ in sel.select(timeout=min(left, 0.5)):
                    chunk = os.read(key.fd, 1 << 16)
                    if not chunk:
                        sel.unregister(key.fileobj)
                        open_pipes -= 1
                        continue
                    bufs[key.fileobj].append(chunk)
                    total += len(chunk)
                    if total > max_output:
                        r.truncated = True
                        open_pipes = 0
                        break
            if r.timed_out or r.truncated:
                _kill_group(p)
                p.wait()
            else:
                left = deadline - time.perf_counter()
                try:
                    p.wait(timeout=max(left, 0.01))
                except subprocess.TimeoutExpired:
                    r.timed_out = True
                    _kill_group(p)
                    p.wait()
        finally:
            sel.close()
            if p.poll() is None:
                _kill_group(p)
                p.wait()
            # stray grandchildren of a finished leader
            _kill_group(p)
        r.wall = time.perf_counter() - t0
        if p.stdout is not None:
            r.stdout = b"".join(bufs[p.stdout])
            p.stdout.close()
        r.stderr = b"".join(bufs[p.stderr])
        p.stderr.close()
        rc = p.returncode
        if rc is not None and rc < 0:
            r.signal = -rc
        r.rc = rc
        return r
    finally:
        if stdin_f is not None:
            stdin_f.close()


def run_engine(engine, prog_path, stdin_path=None, stdin_data=None, timeout=10.0, flags=(),
               max_output=64 << 20):
    argv = engine.argv + list(flags) + [prog_path]
    return run_cmd(argv, stdin_data=stdin_data, stdin_path=stdin_path, timeout=timeout,
                   max_output=max_output)


def input_for(prog_path):
    """Sibling .in file used as stdin, if present."""
    base, _ = os.path.splitext(prog_path)
    for cand in (base + ".in", prog_path + ".in"):
        if os.path.isfile(cand):
            return cand
    return None


def collect_programs(paths, exts=(".epq", ".txt")):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs.sort()
                for f in sorted(files):
                    if f.endswith(exts) and not f.endswith(".in"):
                        out.append(os.path.join(root, f))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit("no such program or directory: %s" % p)
    return out


def first_line(b):
    return b.split(b"\n", 1)[0]


def short(b, n=400):
    s = b.decode("utf-8", "replace")
    if len(s) > n:
        s = s[:n] + "...[%d bytes]" % len(b)
    return s


def jobs_default():
    return os.cpu_count() or 4


# ---------------------------------------------------------------------------
# batch mode (epq --batch, src/batch.cpp): many programs, one new dylib
# ---------------------------------------------------------------------------

BATCH_TIMEOUT_NOTE = b"TIMEOUT\n"
BATCH_SIGNAL_NOTE = b"epq-batch: killed by signal "


def is_batchable(engine):
    """True for a ClaudeEPQ C-backend engine (possibly behind `env K=V` or
    tools/fake_engine.py): its programs can go through `epq --batch`."""
    a = engine.argv
    if any(x in ("--engine=jit", "--jit") for x in a):
        return False
    return any(os.path.basename(x) == "epq" for x in a)


def _batch_result(out, err, code_txt, max_output):
    r = RunResult()
    r.stdout = out
    try:
        rc = int(code_txt.strip())
    except ValueError:
        r.spawn_error = "batch produced no exit code"
        r.rc = 127
        r.stderr = err
        return r
    if rc == 124 and err.endswith(BATCH_TIMEOUT_NOTE):
        r.timed_out = True
        err = err[:-len(BATCH_TIMEOUT_NOTE)]
    elif rc > 128:
        i = err.rfind(BATCH_SIGNAL_NOTE)
        if i >= 0 and (i == 0 or err[i - 1:i] == b"\n"):
            err = err[:i]
            r.signal = rc - 128
            rc = -r.signal
    r.rc = rc
    r.stderr = err
    if len(out) + len(err) > max_output:
        r.truncated = True
    return r


def run_batch(engine, items, timeout=10.0, jobs=None, flags=(), chunk=300, max_output=64 << 20, env=None,
              log=None):
    """Run items = [(prog_path, stdin_path_or_None), ...] on a C-backend engine through
    `epq --batch`, `chunk` programs per invocation (one new dylib per chunk, none when
    the same chunk is run again).  Returns one RunResult per item, in order (wall = 0:
    batch mode does not time single programs).  Raises RuntimeError if a batch fails."""
    import shutil
    import tempfile
    jobs = jobs or jobs_default()
    results = []
    base = os.path.join(BUILD, "batchrun")
    os.makedirs(base, exist_ok=True)
    for c0 in range(0, len(items), chunk):
        part = items[c0:c0 + chunk]
        d = tempfile.mkdtemp(prefix="b%d-" % os.getpid(), dir=base)
        try:
            lines = []
            for i, (prog, stdin) in enumerate(part):
                stem = os.path.join(d, "%05d" % i)
                for s in (prog, stdin or ""):
                    if "\t" in s or "\n" in s:
                        raise RuntimeError("tab/newline in path %r" % s)
                lines.append("\t".join([os.path.abspath(prog), os.path.abspath(stdin) if stdin else "-",
                                        stem + ".o", stem + ".e", stem + ".c"]))
            man = os.path.join(d, "manifest.tsv")
            with open(man, "w") as f:
                f.write("\n".join(lines) + "\n")
            argv = engine.argv + list(flags) + ["--batch", man, "-j", str(jobs), "--timeout", str(timeout)]
            p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=env)
            if p.returncode != 0:
                raise RuntimeError("%s --batch failed (exit %d): %s" % (engine.name, p.returncode,
                                                                         short(p.stderr, 2000)))
            if log:
                log("  [%s] %s" % (engine.name, p.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]))
            for i in range(len(part)):
                stem = os.path.join(d, "%05d" % i)
                def rd(x):
                    try:
                        with open(stem + x, "rb") as f:
                            return f.read()
                    except OSError:
                        return b""
                results.append(_batch_result(rd(".o"), rd(".e"), rd(".c"), max_output))
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return results
