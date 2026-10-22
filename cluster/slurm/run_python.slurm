#!/bin/bash -l
#SBATCH --job-name=thesis-python
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Activate the workflow-specific environment before sbatch. All arguments after
# this script are forwarded literally to Python; no eval or shell expansion.
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
if [[ -n "${WORKING_DIRECTORY:-}" ]]; then
    cd "$WORKING_DIRECTORY"
fi
ENTRYPOINT="${1:?Supply a repository-relative Python entry point}"
shift
test -f "$ENTRYPOINT"
python -u "$ENTRYPOINT" "$@"
