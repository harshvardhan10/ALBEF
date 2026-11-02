#!/usr/bin/env python3
"""Inspect registered notebook paths and historical cwd; never execute notebook code."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook_context(notebook):
    """Return context for a registered canonical or historical notebook path."""
    with (ROOT / 'docs/audit/notebook_catalog.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    registered = {path: row for row in rows
                  for path in (row['legacy_path'], row['canonical_path'])}
    key = Path(notebook).as_posix()
    if key not in registered:
        raise ValueError('Not a registered notebook: ' + notebook)
    row = registered[key]
    return {
        'canonical_path': str(ROOT / row['canonical_path']),
        'open_legacy_path': str(ROOT / row['legacy_path']),
        'kernel_working_directory': str(ROOT / row['legacy_working_directory']),
        'category': row['category'],
        'status': row['status'],
        'executed': False,
        'context_basis': 'Original file parent; not verified HPC execution provenance',
        'instruction': 'Open the legacy path and verify the kernel working directory '
                       'before running existing cells; no cells were executed here.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', help='Registered canonical or historical notebook path')
    args = parser.parse_args()
    try:
        context = notebook_context(args.notebook)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(context, indent=2))


if __name__ == '__main__':
    main()
