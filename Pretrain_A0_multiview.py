#!/usr/bin/env python3
"""Unified multiview entry point; select fusion by YAML or --fusion_type."""
from training.cli import main_for_fusion

if __name__ == "__main__":
    main_for_fusion()
