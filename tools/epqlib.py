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

# baseline search order: $EPQ_BASELINE, ../refbuild/epq_ref (next to the repo), build/tools/ref/epq_ref
def find_baseline():
    cands = [os.environ.get("EPQ_BASELINE"),
             os.path.join(os.path.dirname(REPO), "refbuild", "epq_ref"),
             os.path.join(BUILD, "tools", "ref", "epq_ref")]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


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
