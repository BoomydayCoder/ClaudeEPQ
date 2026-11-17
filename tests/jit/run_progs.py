#!/usr/bin/env python3
"""Run an engine over tests/jit/progs/ (same format and options as tests/conformance/run.py).

usage: tests/jit/run_progs.py [-k SUBSTR] [-j N] [-q] -- CMD [ARGS...]
Expected outputs were captured from the baseline with
  tests/conformance/regen.py BASELINE tests/jit/progs/*.epq
"""
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progs")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conformance"))
import run as R  # noqa: E402

R.HERE = HERE
sys.exit(R.main())
