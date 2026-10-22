#!/usr/bin/env python3
"""Historical training command; delegates to the shared multiview workflow."""
from training.cli import main_for_fusion, run_legacy_main, legacy_export
EXPERIMENT_NAME = 'A0_multiview_transformer2_mean_fusion_dual_auc'


def main(args, config):
    return run_legacy_main('transformer', args, config)


def __getattr__(name):
    return legacy_export(name, 'transformer')


if __name__ == "__main__":
    main_for_fusion('transformer')
