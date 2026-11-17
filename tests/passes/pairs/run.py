#!/usr/bin/env python3
"""Run an engine over tests/passes/pairs/progs (same format and options as tests/conformance/run.py).

usage: tests/passes/pairs/run.py [-k SUBSTR] [-j N] [-q] -- CMD [ARGS...]
Expected outputs are captured from the baseline with
  tests/conformance/regen.py BASELINE tests/passes/pairs/progs/*.epq
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "conformance"))
import run as R  # noqa: E402

R.HERE = os.path.join(HERE, "progs")
sys.exit(R.main())
