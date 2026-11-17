#!/usr/bin/env python3
"""Fast suite runner for the C backend: all selected tests through ONE `epq --batch`.

usage: fasttest.py [--epq PATH] [--epq-arg ARG ...] [--env K=V ...] [-k SUBSTR ...]
                   [-j N] [--timeout SEC] [--max-diff LINES] [-q] [--keep]
                   [--also CMD] SUITE_DIR [SUITE_DIR ...]

A suite dir uses the conformance layout (recursive): X.epq, optional X.in (stdin),
expected X.out (stdout bytes) and X.code (exit code; 0 if absent). Tests without
X.out are skipped. stdout and exit code are compared; stderr is shown for failures.

Why: macOS scans every new binary on first load (~200 ms, one at a time machine-wide);
the per-process runners create one dylib per C-backend program, this creates one per
run (and none at all when the same set of programs is run again). See tools/README.md.

  --epq PATH       epq binary (default build/epq)
  --epq-arg ARG    extra option for every program (repeatable), e.g. --epq-arg=--verify-tir;
                   --engine=aot is added unless an --engine= argument is given
  --env K=V        environment for the batch (repeatable), e.g. --env EPQ_NO_PASS=listopt
  -k SUBSTR        only tests whose path (relative to the suite dir) contains SUBSTR (repeatable)
  -j N             parallel compiles / runs (default: number of cores)
  --timeout SEC    per-test limit (default 30); a timeout is exit code 124
  --keep           keep the output files of passing tests too (default: failures only)
  --also CMD       also run every test through CMD (shell-split, FILE appended) the slow
                   per-process way and compare it against the expected files as well
Output files: build/fasttest/<run-id>/. Exit status 0 iff every selected test passes.
"""
import difflib
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import epqlib  # noqa: E402


def find_tests(dirs, filt):
    out = []
    for d in dirs:
        d = os.path.abspath(d)
        if os.path.isfile(d) and d.endswith('.epq'):
            out.append((os.path.dirname(d), d))
            continue
        if not os.path.isdir(d):
            raise SystemExit('fasttest: no such suite dir: %s' % d)
        for root, subdirs, files in os.walk(d):
            subdirs.sort()
            for f in sorted(files):
                if f.endswith('.epq'):
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, d)
                    if not filt or any(s in rel for s in filt):
                        out.append((d, p))
    return out


def read(path, default=None):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except OSError:
        return default


def show_failure(label, exp_out, exp_code, out, code, err, quiet, maxdiff):
    print('FAIL %s' % label)
    if quiet:
        return
    if code != exp_code:
        print('  exit code: expected %s, got %s' % (exp_code, code))
    if out != exp_out:
        d = list(difflib.unified_diff(
            exp_out.decode('utf-8', 'replace').splitlines(),
            out.decode('utf-8', 'replace').splitlines(),
            'expected', 'actual', lineterm='', n=2))
        for line in d[:maxdiff]:
            print('  ' + line[:200])
        if len(d) > maxdiff:
            print('  ... (%d more diff lines)' % (len(d) - maxdiff))
    if err:
        e = err.decode('utf-8', 'replace').strip().splitlines()
        print('  stderr: ' + ' | '.join(e[:3])[:200])


def main():
    args = sys.argv[1:]
    epq = os.path.join(REPO, 'build', 'epq')
    epq_args, env_over, filt, dirs = [], {}, [], []
    jobs, timeout, maxdiff, quiet, keep, also = os.cpu_count() or 4, 30.0, 40, False, False, None
    while args:
        a = args.pop(0)
        def val():
            if '=' in a and a.startswith('--'):
                return a.split('=', 1)[1]
            if not args:
                raise SystemExit('fasttest: %s needs a value' % a)
            return args.pop(0)
        name = a.split('=', 1)[0] if a.startswith('--') else a
        if name in ('-h', '--help'):
            print(__doc__)
            return 0
        elif name == '--epq':
            epq = val()
        elif name == '--epq-arg':
            epq_args.append(val())
        elif name == '--env':
            kv = val()
            if '=' not in kv:
                raise SystemExit('fasttest: --env needs K=V')
            k, v = kv.split('=', 1)
            env_over[k] = v
        elif name == '-k':
            filt.append(val())
        elif name == '-j':
            jobs = int(val())
        elif name == '--timeout':
            timeout = float(val())
        elif name == '--max-diff':
            maxdiff = int(val())
        elif name == '-q':
            quiet = True
        elif name == '--keep':
            keep = True
        elif name in ('--also', '--engine-check'):
            also = shlex.split(val())
        elif a.startswith('-'):
            print('fasttest: unknown option %s' % a)
            return 2
        else:
            dirs.append(a)
    if not dirs:
        print(__doc__)
        return 2
    if not any(x.startswith('--engine') for x in epq_args):
        epq_args = ['--engine=aot'] + epq_args

    tests = []
    skipped = 0
    for root, p in find_tests(dirs, filt):
        base = p[:-4]
        exp_out = read(base + '.out')
        if exp_out is None:
            skipped += 1
            continue
        code_txt = read(base + '.code', b'0')
        exp_code = int(code_txt.strip() or b'0')
        stdin = base + '.in' if os.path.isfile(base + '.in') else None
        label = os.path.relpath(p, os.getcwd())
        if '\t' in p or '\n' in p or (stdin and ('\t' in stdin or '\n' in stdin)):
            raise SystemExit('fasttest: tab/newline in path: %r' % p)
        tests.append({'path': p, 'stdin': stdin, 'exp_out': exp_out, 'exp_code': exp_code, 'label': label})
    if not tests:
        print('0/0 passed, 0 failed' + (' (%d skipped: no .out)' % skipped if skipped else ''))
        return 0

    run_id = time.strftime('%Y%m%d-%H%M%S') + '-%d' % os.getpid()
    outdir = os.path.join(REPO, 'build', 'fasttest', run_id)
    os.makedirs(outdir)
    lines = []
    for i, t in enumerate(tests):
        stem = os.path.join(outdir, '%05d' % i)
        t['files'] = (stem + '.stdout', stem + '.stderr', stem + '.rc')
        lines.append('\t'.join([t['path'], t['stdin'] or '-'] + list(t['files'])))
    manifest = os.path.join(outdir, 'manifest.tsv')
    with open(manifest, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    env = dict(os.environ)
    env.update(env_over)
    cmd = [epq] + epq_args + ['--batch', manifest, '-j', str(jobs), '--timeout', str(timeout)]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall = time.time() - t0
    summary = r.stderr.decode('utf-8', 'replace').strip()
    if r.returncode != 0:
        print('fasttest: epq --batch failed (exit %d):\n%s' % (r.returncode, summary))
        return 2

    also_res = {}
    if also:
        def run_also(i):
            t = tests[i]
            return i, epqlib.run_cmd(also + [t['path']], stdin_path=t['stdin'], timeout=timeout)
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            for i, rr in ex.map(run_also, range(len(tests))):
                also_res[i] = rr

    failed = 0
    for i, t in enumerate(tests):
        fo, fe, fc = t['files']
        out = read(fo, b'')
        err = read(fe, b'')
        code_b = read(fc, b'')
        try:
            code = int(code_b.strip())
        except ValueError:
            code = 'missing'
        ok = out == t['exp_out'] and code == t['exp_code']
        if not ok:
            failed += 1
            show_failure(t['label'], t['exp_out'], t['exp_code'], out, code, err, quiet, maxdiff)
            if not quiet:
                print('  files: %s' % os.path.splitext(fo)[0] + '.{stdout,stderr,rc}')
        elif not keep:
            for p in t['files']:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        if also:
            rr = also_res[i]
            acode = 'timeout' if rr.timed_out else (128 + rr.signal if rr.signal else rr.rc)
            if rr.stdout != t['exp_out'] or acode != t['exp_code']:
                failed += 1
                show_failure('%s [also: %s]' % (t['label'], ' '.join(also)), t['exp_out'], t['exp_code'],
                             rr.stdout, acode, rr.stderr, quiet, maxdiff)
    n = len(tests) * (2 if also else 1)
    if not quiet:
        print(summary.splitlines()[-1] if summary else '(no batch summary)')
    print('%d/%d passed, %d failed%s  [%.1f s, outputs in %s]' % (
        n - failed, n, failed, ' (%d skipped: no .out)' % skipped if skipped else '', wall,
        os.path.relpath(outdir, os.getcwd())))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
