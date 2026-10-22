#!/usr/bin/env python3
"""Compatibility command for exploratory FROC evaluation."""
from pathlib import Path
import importlib
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/"scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def __getattr__(name):
    return getattr(importlib.import_module("scripts.exploratory_froc.evaluate_froc_vindr"), name)


if __name__ == "__main__":
    print("[Exploratory] FROC utilities do not establish a final thesis localization ranking.", file=sys.stderr)
    runpy.run_path(str(ROOT/"scripts/exploratory_froc/evaluate_froc_vindr.py"), run_name="__main__")
