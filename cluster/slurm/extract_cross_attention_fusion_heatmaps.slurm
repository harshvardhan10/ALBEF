#!/bin/bash -l
#SBATCH --job-name=xattn-heatmap-check
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Activate the known working conda environment BEFORE sbatch. SLURM copies the
# script into its spool, so use --chdir rather than BASH_SOURCE to locate code.
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit with sbatch --chdir=/absolute/ALBEF ...}"
cd "${PROJECT_ROOT}"
test -f scripts_new/extract_cross_attention_fusion_itc_margin_gradcam_heatmaps.py

: "${CONFIG:?Set CONFIG to the saved resolved cross-attention config}"
: "${LABELS_CSV:?Set LABELS_CSV to the fixed 4-8 case CSV}"
: "${IMAGES_ROOT:?Set IMAGES_ROOT}"
: "${LUNG_MASK_ROOT:?Set LUNG_MASK_ROOT}"
: "${HEART_MASK_ROOT:?Set HEART_MASK_ROOT}"
: "${TOKENIZER_PATH:?Set TOKENIZER_PATH to the local BERT snapshot}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to a new run directory}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/vault/iwi5/iwi5362h/outputs/A0_multiview_cross_attention_fusion_dual_auc}"
TEXT_ENCODER="${TEXT_ENCODER:-bert-base-uncased}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

python - <<'PY'
import os
from pathlib import Path
import pandas as pd
frame = pd.read_csv(os.environ['LABELS_CSV'])
if not 1 <= len(frame) <= 8:
    raise ValueError('This validation job requires a fixed CSV with 1-8 cases, not the full test set')
if Path(os.environ['OUTPUT_BASE']).exists():
    raise FileExistsError('OUTPUT_BASE already exists; choose a new run directory')
PY

for role in best_cardiomegaly_auc best_macro_auc_stable; do
    python -u scripts_new/extract_cross_attention_fusion_itc_margin_gradcam_heatmaps.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT_DIR}/checkpoint_${role}.pth" \
        --checkpoint_role "${role}" \
        --training_commit "${TRAINING_COMMIT:-unconfirmed}" \
        --labels_csv "${LABELS_CSV}" \
        --images_root "${IMAGES_ROOT}" \
        --lung_mask_root "${LUNG_MASK_ROOT}" \
        --heart_mask_root "${HEART_MASK_ROOT}" \
        --tokenizer_name_or_path "${TOKENIZER_PATH}" \
        --text_encoder "${TEXT_ENCODER}" \
        --output_dir "${OUTPUT_BASE}/${role}" \
        --target_labels Cardiomegaly 'Pleural effusion' \
        --vit_layer -2 --device cuda
done
