"""Preserve bare historical ``src`` helper imports in the new source layout.

Some trainers catch ModuleNotFoundError for ``src`` and fall back to scripts.src.
Without this bridge the new src directory becomes a namespace package and changes
that failure into ImportError. Re-export the same historical helper functions.
"""
from scripts.src import *  # noqa: F401,F403
