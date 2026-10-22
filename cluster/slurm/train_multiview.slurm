#!/bin/bash -l
#SBATCH --job-name=albef-multiview
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the cleanup repository root}"
test -f Pretrain_A0_multiview.py
: "${CONFIG:?Set CONFIG to an experiment or smoke YAML}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the training output directory}"
: "${FUSION_TYPE:?Set FUSION_TYPE to one of the six architectures}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

python - <<'PY'
import os
from training.configuration import load_run_config
config = load_run_config(os.environ['CONFIG'], os.environ.get('ENVIRONMENT_CONFIG'))
debug = config.get('debug_max_batches') is not None or config.get('train_subset_size') is not None
if debug and os.environ.get('SMOKE_TEST') != '1':
    raise ValueError('Subset/debug configuration requires explicit SMOKE_TEST=1')
if config.get('view_fusion_type') not in (None, os.environ['FUSION_TYPE']):
    raise ValueError('FUSION_TYPE does not match the experiment YAML')
print('Preflight:', os.environ['FUSION_TYPE'], 'diagnostic' if debug else 'full dataset')
PY

extra=()
if [[ -n "${ENVIRONMENT_CONFIG:-}" ]]; then
    extra+=(--environment_config "$ENVIRONMENT_CONFIG")
fi
if [[ -n "${TEXT_ENCODER:-}" ]]; then
    extra+=(--text_encoder "$TEXT_ENCODER")
fi
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
python -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE:-2}" \
    --master_addr=127.0.0.1 --master_port="$MASTER_PORT" --use_env \
    Pretrain_A0_multiview.py --config "$CONFIG" \
    --fusion_type "$FUSION_TYPE" --output_dir "$OUTPUT_DIR" \
    --device cuda --distributed true --dist_url env:// \
    --auto_resume "${AUTO_RESUME:-true}" "${extra[@]}"
