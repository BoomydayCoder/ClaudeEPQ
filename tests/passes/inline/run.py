#!/usr/bin/env python3
"""Run an engine over tests/passes/inline/ (same format and options as tests/conformance/run.py).

usage: tests/passes/inline/run.py [-k SUBSTR] [-j N] [-q] -- CMD [ARGS...]
Expected outputs were captured from the baseline with
  tests/conformance/regen.py BASELINE tests/passes/inline/*.epq
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "conformance"))
import run as R  # noqa: E402

R.HERE = HERE
sys.exit(R.main())
