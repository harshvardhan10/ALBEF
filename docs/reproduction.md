# Reproduction

1. Use the [experiment catalogue](experiment_catalogue.md) to select the model, checkpoint role and method. Historical run evidence is distinct from new intended-full-data presets.
2. Activate the existing successful server environment: recorded Python 3.8.20, torch 1.8.1+cu111 and CUDA 11.1. Record installed versions. Root requirements.txt now references requirements_albef_legacy.txt; PyTorch/CUDA remain environment-specific. The original notebook freeze is preserved byte-for-byte at requirements/notebooks-historical.txt. None is a newly certified environment lock; no packages were installed or upgraded during cleanup.
3. Supply private image/mask/manifest/checkpoint paths and cached BERT/DeiT assets. Keep patient data, credentials and outputs outside the checkout. Training retains original pretrained initialization behavior; ensure required pretrained assets are cached before any offline job.
4. The author completed both-role baseline/refactored heatmap comparisons and reviewed overlays. [Original baseline instructions](cross_attention_validation.md) and [refactor comparison instructions](cross_attention_refactor_check.md) are retained for audit/repetition, not a request to repeat completed work.
5. Use [cluster workflows](cluster_workflows.md) for resolved configs, portable launch commands and optional bounded training checks. Historical cluster scripts preserve provenance and must not be submitted unchanged.

CheXzero uses its own environment and working directory: `cd baselines/chexzero`. Its unqualified model/clip imports must resolve there. Keep the retained README, LICENSE and NOTICE. BioViL-T also needs its own existing environment; baseline import is not runtime certification.

Local checks: `OMP_NUM_THREADS=1 python -m unittest discover -s tests -q`. Tests use real CPU tensors and frozen original source references with small encoder stand-ins. They cover model/training/helper behavior and artifacts, not a full ViT/BERT/GPU/DDP training run. No server jobs, downloads or emails were submitted by the agent.
