#!/usr/bin/env python3
"""Historical training command; delegates to the shared multiview workflow."""
from training.cli import main_for_fusion, run_legacy_main, legacy_export
EXPERIMENT_NAME = 'A0_multiview_residual_fusion_dual_auc'


def main(args, config):
    return run_legacy_main('residual', args, config)


def __getattr__(name):
    return legacy_export(name, 'residual')


if __name__ == "__main__":
    main_for_fusion('residual')
