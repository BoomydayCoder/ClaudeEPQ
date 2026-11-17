#!/usr/bin/env python3
"""Interactive-stdin test: pending output must be flushed before a read that
would block, so a driver that waits for each prompt before answering works.

usage: tests/rtio/interactive.py -- CMD [ARGS...]
"""
import os
import select
import subprocess
import sys
import time

PROG = """>> 1;
a := <<;
>> a * 2;
b := <<;
>> [a, b];
"""


def read_until(fd, want, buf, timeout=20.0):
    end = time.time() + timeout
    while want not in buf:
        left = end - time.time()
        if left <= 0:
            raise SystemExit("FAIL: timed out waiting for %r; got %r" % (want, buf))
        r, _, _ = select.select([fd], [], [], left)
        if r:
            chunk = os.read(fd, 65536)
            if not chunk:
                raise SystemExit("FAIL: EOF waiting for %r; got %r" % (want, buf))
            buf += chunk
    return buf


def main():
    args = sys.argv[1:]
    if "--" not in args:
        print(__doc__)
        return 2
    cmd = args[args.index("--") + 1:]
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "build", "rtio")
    os.makedirs(d, exist_ok=True)
    prog = os.path.join(d, "interactive.epq")
    with open(prog, "w") as f:
        f.write(PROG)
    p = subprocess.Popen(cmd + [prog], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        fd = p.stdout.fileno()
        buf = read_until(fd, b">> 1\n<< ", b"")
        p.stdin.write(b"21\n")
        p.stdin.flush()
        buf = read_until(fd, b">> 42\n<< ", buf)
        p.stdin.write(b"5\n")
        p.stdin.close()
        buf = read_until(fd, b">> [21, 5]\n", buf)
        p.wait(timeout=20)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    if buf != b">> 1\n<< >> 42\n<< >> [21, 5]\n" or p.returncode != 0:
        print("FAIL: got %r exit %r" % (buf, p.returncode))
        return 1
    print("interactive: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
