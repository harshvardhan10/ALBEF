# Cross-attention fusion: server validation

This is a baseline checkpoint smoke test before model refactoring. Training is complete; do not retrain. The actual model/trainer files remain unchanged. Both checkpoint roles are tested independently. The new extractor needs the established ALBEF server environment and offline BERT cache.

## 1. Transfer and select the cleanup checkout

No changes have been pushed. Transfer the supplied patch to your server, or authorize a branch push separately. In a separate ALBEF checkout on the server (not a running-job directory), check out `thesis-repository-cleanup`. Verify that the branch starts at `6320a2e93ce63bf2a77f3c716278c4379e9114f1` before applying the patch. Run `git status --short` and ensure it is clean. Then:

```bash
git apply --check /path/to/thesis-cleanup-baseline.patch
git apply /path/to/thesis-cleanup-baseline.patch
```

If either check fails, stop and reconcile the checkout; do not force application. Read `docs/cleanup_validation.md` for exactly what has been verified locally. Record `git rev-parse HEAD`; the manifest also hashes the extractor and model source and records a dirty checkout.

## 2. Establish the run inputs

Use the actual resolved YAML saved with the completed cross-attention training run. Confirm its image_res, BERT config, fusion block and preprocessing match the checkpoint. The checked-in fallback is `configs_new/Pretrain_A0_multiview_cross_attention_fusion_dual_auc.yaml`; using it without comparing to the run record leaves training provenance unconfirmed.

Known checkpoint directory, confirmed by the author:

```text
/home/vault/iwi5/iwi5362h/outputs/A0_multiview_cross_attention_fusion_dual_auc
```

Required files: checkpoint_best_cardiomegaly_auc.pth and checkpoint_best_macro_auc_stable.pth. Supply the training Git commit from the run log; use `unconfirmed` only if it cannot be recovered and report that limitation.

Create a **local** `validation-cases.csv` containing 4–8 cases selected before looking at the new heatmaps: positives for Cardiomegaly and Pleural effusion plus negative controls. Use canonical VinDr labels; do not invent image IDs or copy the entire test split. Example schema only:

```csv
image_id,Cardiomegaly,Pleural effusion
YOUR_REAL_IMAGE_ID_1,1,0
YOUR_REAL_IMAGE_ID_2,0,1
```

Replace the IDs and labels with real values. Freeze this CSV for both baseline and refactored runs. Images are `<IMAGES_ROOT>/<id>.png`; masks are `<MASK_ROOT>/<id[:2]>/<id>.png`. Both masks must match image dimensions. IDs must be unique and labels binary. Keep patient data and outputs outside the Git checkout.

## 3. Activate the environment and set paths

The following commands run on your server. Replace the three `/absolute/path/...` values and NEW_RUN_NAME. Verify other defaults against the actual server layout. Create LOG_DIR before sbatch because SLURM opens logs before the script runs.

```bash
module load python
conda activate albef-notebook
cd /absolute/path/to/cleanup-ALBEF
export CONFIG='/absolute/path/to/saved-cross-attention-config.yaml'
export LABELS_CSV='/absolute/path/to/validation-cases.csv'
export IMAGES_ROOT='/home/woody/iwi5/iwi5362h/data/vindr_cxr/test'
export LUNG_MASK_ROOT='/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/lung'
export HEART_MASK_ROOT='/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/heart'
export CHECKPOINT_DIR='/home/vault/iwi5/iwi5362h/outputs/A0_multiview_cross_attention_fusion_dual_auc'
export TOKENIZER_PATH='/home/woody/iwi5/iwi5362h/.cache/huggingface/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
export TRAINING_COMMIT='unconfirmed'
export OUTPUT_BASE='/home/vault/iwi5/iwi5362h/outputs/cleanup_validation/NEW_RUN_NAME'
export LOG_DIR='/home/vault/iwi5/iwi5362h/logs/cleanup_validation'
mkdir -p "$LOG_DIR"
python -c 'import sys,torch,torchvision; print(sys.version); print(torch.__version__, torch.version.cuda, torchvision.__version__)'
```

The old launch script supplied the tokenizer path above. Confirm it exists; do not download a new tokenizer. BERT initialization also needs cached model weights; if `bert-base-uncased` cannot resolve offline, export TEXT_ENCODER to the local complete BERT model snapshot. The full fusion checkpoint must match exactly: positional embeddings are not interpolated and missing keys are not ignored. A failure is useful evidence, not a reason to relax strict loading.

## 4. Submit the bounded validation job

```bash
sbatch --export=ALL --chdir="$PWD" \
  --output="$LOG_DIR/xattn_%j.out" \
  --error="$LOG_DIR/xattn_%j.err" \
  cluster/slurm/extract_cross_attention_fusion_heatmaps.slurm
```

This allocates one A40 for at most 30 minutes and checks both roles on at most eight CSV rows. Adjust scheduling resources if your site requires it; do not change method parameters for the baseline comparison. The launcher has no email directives. It inherits your activated environment. It refuses an existing OUTPUT_BASE. On an allocated GPU node you can instead invoke the Python extractor directly with the same arguments shown in the launcher.

## 5. Check the result

Each role's subdirectory must contain one `<id>.pt` per selected case and these files:

- cross_attention_fusion_itc_margin_gradcam_index.csv
- cross_attention_fusion_itc_margin_gradcam_manifest.json

Read both job logs. Expect `<All keys matched successfully>` and the final extraction-complete line twice (once per role). There must be no traceback, missing files, silent skips or nonfinite values. All-zero positive CAM warnings require investigation and must be reported even if the process exits zero.

The manifest records case IDs, checkpoint hash/role, config bytes and parsed values, extraction revision/source hashes, prompts, layer, environment and fusion description. The model must be `two_layer_original_anchored_cross_attention_lung_heart`, with original anchor and ordered lung/heart memories, and uses_itm=false. `--vit_layer -2` resolves to index 10 for the 12-block model. Per case/label, original/lung/heart/fused dictionaries hold cam_signed_raw, cam_positive_raw, cam_vis and cam_vis_up. The normal config produces 16x16 patch maps and 256x256 upsampled maps.

Render side-by-side original and fused-map overlays using the delivered fixed-palette utility. It uses the original image paths in the .pt metadata, so run it on the server where the images remain available:

```bash
python scripts_new/render_cross_attention_heatmaps.py \
  --input_dir "$OUTPUT_BASE/best_cardiomegaly_auc" \
  --output_dir "$OUTPUT_BASE/overlays_cardiomegaly"
python scripts_new/render_cross_attention_heatmaps.py \
  --input_dir "$OUTPUT_BASE/best_macro_auc_stable" \
  --output_dir "$OUTPUT_BASE/overlays_macro"
```

Each panel contains the original image and one fused overlay per label, at 256x256 display size, with a fixed blue/red palette and opacity 0.4. Output directories must be new. Apply the same settings to every comparison. Never interpret normalized color intensity as a comparable confidence score.

## 6. Repeat and compare

First repeat this baseline using the same case CSV and a **new** OUTPUT_BASE; this measures reproducibility on your actual environment. Then, after model consolidation, generate a candidate run using the identical checkpoint files, config, cases, prompts, layer and environment. Compare each role separately:

```bash
python scripts_new/compare_cross_attention_heatmaps.py \
  --baseline '/absolute/path/to/baseline/best_cardiomegaly_auc' \
  --candidate '/absolute/path/to/candidate/best_cardiomegaly_auc' \
  --report '/absolute/path/to/new-cardiomegaly-comparison.json'
```

Repeat for best_macro_auc_stable with a separate report. PASS means matching recorded scientific inputs and scores/raw and visual maps within rtol=1e-5, atol=1e-6; it does not mean good localization. Reports include per-field absolute differences. If it fails, inspect the report before changing any tolerance or implementation. A missing baseline permits only a post-change smoke-test claim, not equivalence.

## 7. Evidence needed to resume model consolidation

Return the training commit and saved-config location, exact command and extraction revision, both logs, both manifests, the comparison reports and a few permitted matched overlays (or descriptions if images cannot be shared). Keep credentials and patient images on the server. Do not send checkpoint files. The agent cannot independently certify GPU/checkpoint/DDP equivalence from local CPU tests.
