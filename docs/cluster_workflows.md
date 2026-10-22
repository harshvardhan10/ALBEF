# Configuration and cluster workflows

Submit from a separate cleanup checkout, with the existing workflow-specific environment activated. Keep running-job directories unchanged. Portable jobs have no email directives and inherit the activated environment. Historical jobs remain byte-identical under cluster/historical; use them to recover scientific arguments, not as portable commands. Some archival jobs reference external MedKLIP/other installations not supplied by these three repositories.

## Resolve inputs without editing methodology

Copy configs/environments/cluster.example.yaml to an ignored `cluster.local.yaml`, or write an equivalent file outside Git. Set only the fields needed by the chosen experiment; remove irrelevant example fields. Use absolute paths for external manifests, masks, weights and the cached text encoder. Dataset records may also contain relative paths: inspect their loader and ensure those resolve in this checkout. The overlay rejects methodological overrides such as image_res or optimizer settings. Historical configs/configs_new YAML remains unchanged.

Multiview accepts the overlay directly:

```bash
python Pretrain_A0_multiview.py --help
python Pretrain_A0_multiview.py \
  --config configs/experiments/stage_05_multiview/cross_attention.yaml \
  --environment_config /absolute/private/cluster.local.yaml \
  --fusion_type cross_attention --output_dir /absolute/new/output
```

This is a training command, not an instruction to retrain the completed experiment. It uses the original distributed defaults; use the launcher below for a scheduled run. Old fusion-specific commands retain their flags and select the same shared trainer. CLI single-view checkpoint paths override config paths; explicit --text_encoder overrides the overlay. The resolved config, source hash and Git state are saved with future multiview runs. Choose a new output directory unless intentionally resuming; auto_resume defaults to true as before.

For anatomy/patch/single-view trainers, first resolve a new external config:

```bash
python -m training.configuration \
  --experiment configs/experiments/stage_02_anatomy_guidance/a4.yaml \
  --environment /absolute/private/cluster.local.yaml \
  --output /absolute/new/resolved-a4.yaml
```

Then pass it to that historical trainer's --config flag, retaining its checkpoint/text-encoder/distributed CLI inputs. Resolution refuses an existing destination. Anatomy/patch loops and objectives are retained; helper extraction does not make their CLI/config requirements interchangeable. A4 additionally needs its classification_teacher_checkpoint; single-view masked runs need mask_root. These are path overlay fields, not new methodology.

## Optional bounded shared-trainer check

The completed six-case heatmap comparison validates inference compatibility. The following optional check exercises dataset loading, optimizer steps and recovery checkpoint writing in the legacy environment. It does not establish full-training or DDP equivalence.

1. In the cleanup checkout, create a private environment overlay from the saved training config. For the confirmed server paths:

```bash
export RUN_DIR=/home/vault/iwi5/iwi5362h/outputs/A0_multiview_cross_attention_fusion_dual_auc
export PRIVATE_DIR=/home/vault/iwi5/iwi5362h/outputs/cleanup_validation
export ENVIRONMENT_CONFIG="$PRIVATE_DIR/training-smoke.local.yaml"
export TEXT_ENCODER=/home/woody/iwi5/iwi5362h/.cache/huggingface/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594
python - <<'PYCODE'
import os
from pathlib import Path
import yaml
source = yaml.safe_load((Path(os.environ['RUN_DIR']) / 'config.yaml').read_text())
# Historical relative manifest paths refer to the original training checkout.
original = Path('/home/woody/iwi5/iwi5362h/ALBEF')
def absolute(value):
    if isinstance(value, list):
        return [absolute(item) for item in value]
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else original / path)
keys = ('lung_train_file', 'heart_train_file', 'lung_mask_root', 'heart_mask_root')
overlay = {key: absolute(source[key]) for key in keys}
overlay['initial_checkpoints'] = {
    key: absolute(source['initial_checkpoints'][key])
    for key in ('original', 'lung', 'heart')
}
overlay['text_encoder'] = os.environ['TEXT_ENCODER']
for key in ('lung_train_file', 'heart_train_file'):
    values = overlay[key] if isinstance(overlay[key], list) else [overlay[key]]
    for value in values:
        assert Path(value).is_file(), value
for value in overlay['initial_checkpoints'].values():
    assert Path(value).is_file(), value
with Path(os.environ['ENVIRONMENT_CONFIG']).open('x') as handle:
    yaml.safe_dump(overlay, handle, sort_keys=False)
print(os.environ['ENVIRONMENT_CONFIG'])
PYCODE
```

If the saved config used checkpoint CLI overrides, substitute their actual paths before running. Verify masks and image paths inside the manifests too. The smoke config selects eight aligned records, batch size two, at most two batches and one epoch; validation is disabled and the recovery checkpoint is retained. Required pretrained caches must already exist. Do not use this preset for thesis results.

2. Activate the existing ALBEF environment; export a fresh output path and submit the bounded job:

```bash
export CONFIG=configs/smoke/multiview_cross_attention.yaml
export FUSION_TYPE=cross_attention
export SMOKE_TEST=1 AUTO_RESUME=false NPROC_PER_NODE=1
export OUTPUT_DIR="$PRIVATE_DIR/trainer_smoke_$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="$PRIVATE_DIR/logs"
mkdir -p "$LOG_DIR"
test ! -e "$OUTPUT_DIR"
sbatch --export=ALL --gres=gpu:a40:1 --time=00:15:00 \
  --output="$LOG_DIR/trainer_%j.out" --error="$LOG_DIR/trainer_%j.err" \
  cluster/slurm/train_multiview.slurm
```

This schedules one GPU/process. For a distinct optional two-process DDP check, use a new OUTPUT_DIR, NPROC_PER_NODE=2 and --gres=gpu:a40:2; two batches still bound the run. Site scheduling overrides may be necessary. Never loosen scientific parameters merely to make an equivalence check pass.

3. Check the job log for finite training losses, completion without traceback, and successful checkpoint writing. Check OUTPUT_DIR/config.yaml has cross_attention and your resolved inputs, and checkpoint_last.pth contains model/optimizer/scheduler/epoch state. A smoke run with validation disabled should not be expected to create either best-AUC checkpoint. Preserve logs/config/checkpoint outside Git. Stop and report the first failure rather than broadening the run.

Other fusion smoke presets use the same launcher with matching FUSION_TYPE. Anatomy/A4/A5/A6.1 runtime checks remain optional and not executed: use a private two-record authorized manifest, batch size two and one epoch in a new diagnostic resolved config, keep their distinct objectives/freeze/checkpoint inputs, and inspect that trainer's --help before scheduling through run_python.slurm. Do not treat successful shared-trainer testing as certification of those separate loops.

## Generic evaluation/baseline launch

`cluster/slurm/run_python.slurm` forwards a Python entrypoint and literal arguments, with ALBEF on PYTHONPATH. For CheXzero, export WORKING_DIRECTORY="$PWD/baselines/chexzero" and pass its local entrypoint after activating its separate environment. Provide real method-specific arguments from the retained baseline README/catalogue; the launcher supplies no invented dataset or checkpoint defaults. Review resource/time overrides per job. No agent-submitted jobs are part of this cleanup.
