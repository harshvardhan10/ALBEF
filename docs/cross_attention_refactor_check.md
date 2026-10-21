# Check the consolidated model against your saved baseline

The author confirmed that the original baseline and repeat runs both passed comparisons, with six cases per checkpoint role and no all-zero CAM warnings. Training commit remains `unconfirmed`. This new check compares that unchanged baseline against the consolidated model, not against another copy of the old model.

## 1. Update the separate server checkout

Run after SSH. `git status --short` should print nothing. If it shows local edits, stop and preserve them before pulling.

```bash
cd /home/woody/iwi5/iwi5362h/ALBEF-cleanup-validation
git status --short
```

This update stops tracking generated Python files and `data/mimic_cxr.json`. Preserve your server copies before pulling. The backup is outside the repository, and uses a new directory:

```bash
export CHECKOUT_BACKUP="/home/vault/iwi5/iwi5362h/outputs/cleanup_validation/repository_backups/before_pull_$(date +%Y%m%d_%H%M%S)"
python - <<'PY_BACKUP'
import os
import shutil
import subprocess
from pathlib import Path

backup = Path(os.environ["CHECKOUT_BACKUP"])
backup.mkdir(parents=True, exist_ok=False)
paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
count = 0
for name in paths:
    path = Path(name)
    selected = name == "data/mimic_cxr.json" or path.suffix == ".pyc"
    if selected and path.is_file():
        target = backup/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
print("Backed up", count, "tracked local files to", backup)
PY_BACKUP
```

If the backup command fails, stop before pulling. Otherwise:

```bash
git pull --ff-only origin thesis-repository-cleanup
python - <<'PY_RESTORE'
import os
import shutil
from pathlib import Path

backup = Path(os.environ["CHECKOUT_BACKUP"])
for source in backup.rglob("*"):
    if source.is_file():
        target = Path.cwd()/source.relative_to(backup)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
print("Local copies restored where absent; backup retained at", backup)
PY_RESTORE
git rev-parse HEAD
module load python
conda activate albef-notebook
```

The restored generated files and real manifest are now ignored by Git; the backup remains available. Neither source files nor Git history are removed.

No patch is needed. Do not recreate or change the existing case CSV, saved training config or baseline outputs.

## 2. Restore the same inputs and choose a fresh candidate output

```bash
export RUN_DIR='/home/vault/iwi5/iwi5362h/outputs/A0_multiview_cross_attention_fusion_dual_auc'
export CONFIG="$RUN_DIR/config.yaml"
export CHECKPOINT_DIR="$RUN_DIR"
export VALIDATION_DIR='/home/vault/iwi5/iwi5362h/outputs/cleanup_validation'
export LABELS_CSV="$VALIDATION_DIR/validation-cases.csv"
export BASELINE_DIR="$VALIDATION_DIR/baseline_20261022_003021"
export IMAGES_ROOT='/home/woody/iwi5/iwi5362h/data/vindr_cxr/test'
export LUNG_MASK_ROOT='/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/lung'
export HEART_MASK_ROOT='/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/heart'
export TOKENIZER_PATH='/home/woody/iwi5/iwi5362h/.cache/huggingface/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
export TRAINING_COMMIT='unconfirmed'
export OUTPUT_BASE="$VALIDATION_DIR/refactored_$(date +%Y%m%d_%H%M%S)"
export LOG_DIR='/home/vault/iwi5/iwi5362h/logs/cleanup_validation'
mkdir -p "$LOG_DIR"
printf 'Candidate outputs: %s\n' "$OUTPUT_BASE"
```

Use the exact paths/tokenizer/text encoder from the successful baseline if you previously corrected any defaults above. If you used a local `TEXT_ENCODER` override, export that same value again. Keep the same Python/torch environment and temperature/layer settings. Save the printed candidate path in case you reconnect before the job finishes.

## 3. Submit the same bounded extraction job

```bash
sbatch --export=ALL --chdir="$PWD" \
  --output="$LOG_DIR/xattn_%j.out" \
  --error="$LOG_DIR/xattn_%j.err" \
  cluster/slurm/extract_cross_attention_fusion_heatmaps.slurm
```

Wait for completion. Check both checkpoint loads and extraction completion in the logs, and report any all-zero warnings or traceback. The job uses the same checkpoints and cases; it does not train the model.

## 4. Compare both roles against the original baseline

Run in the same terminal. After reconnecting, restore `OUTPUT_BASE` to the exact printed candidate path; do not generate a new timestamp for this step.

```bash
python scripts_new/compare_cross_attention_heatmaps.py \
  --baseline "$BASELINE_DIR/best_cardiomegaly_auc" \
  --candidate "$OUTPUT_BASE/best_cardiomegaly_auc" \
  --report "$OUTPUT_BASE/comparison_cardiomegaly.json"

python scripts_new/compare_cross_attention_heatmaps.py \
  --baseline "$BASELINE_DIR/best_macro_auc_stable" \
  --candidate "$OUTPUT_BASE/best_macro_auc_stable" \
  --report "$OUTPUT_BASE/comparison_macro.json"
```

Both must print PASS. A failure should be investigated without changing tolerances or deleting outputs. These checks compare the scores, raw branch/composite maps, visual maps and scientific input metadata. Passing is a checkpoint/extraction equivalence check, not evidence of better localization or full DDP training equivalence.

## 5. Render the candidate panels

```bash
python scripts_new/render_cross_attention_heatmaps.py \
  --input_dir "$OUTPUT_BASE/best_cardiomegaly_auc" \
  --output_dir "$OUTPUT_BASE/overlays_cardiomegaly"

python scripts_new/render_cross_attention_heatmaps.py \
  --input_dir "$OUTPUT_BASE/best_macro_auc_stable" \
  --output_dir "$OUTPUT_BASE/overlays_macro"
```

Compare a few panels to the corresponding original baseline panels. Report both PASS/FAIL results, whether any warnings occurred, and the candidate checkout commit. Keep manifests/logs/reports on the server; provide their contents only if needed to investigate a discrepancy. No checkpoint files or patient images need to be sent.
