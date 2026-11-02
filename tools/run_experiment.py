#!/usr/bin/env python3
"""Run a registered canonical entry point in its historical module context.

No experiment source is rewritten, no directory is silently changed, and data or
checkpoint arguments are forwarded untouched. Run from the repository root.
"""
import argparse
import csv
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', help='Canonical or legacy registered Python entrypoint path')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the repository root: ' + str(ROOT))
    with (ROOT / 'docs/audit/applied_migration.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['kind'] == 'entrypoint']
    registered = {name: row for row in rows
                  for name in (row['canonical_path'], row['legacy_path'])}
    key = Path(args.experiment).as_posix()
    if key not in registered:
        parser.error('Not a registered experiment entrypoint: ' + args.experiment)
    legacy = ROOT / registered[key]['legacy_path']
    # Legacy sibling imports take precedence, followed by root-level historical
    # modules and the reusable source namespace. Preserve existing extra paths.
    for directory in reversed((legacy.parent, ROOT, ROOT / 'src')):
        sys.path.insert(0, str(directory))
    sys.argv = [str(legacy), *args.arguments]
    runpy.run_path(str(legacy), run_name='__main__')


if __name__ == '__main__':
    main()
