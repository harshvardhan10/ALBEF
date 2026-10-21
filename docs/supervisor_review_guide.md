# Supervisor review guide

1. Read the repository README and experiment catalogue to connect each stage to its hypothesis and implementation.
2. Compare the six fusion modules via model_architecture.md. Existing source files are unchanged pending the baseline checkpoint check.
3. Inspect the new cross-attention extractor and its CPU tests. It uses the actual anchored-fusion import and records case/checkpoint provenance; it does not repurpose a single-view cross-attention script.
4. Inspect source_provenance.md and source_imports.json to trace every imported orchestration/baseline file to its original commit and hash.
5. Review limitations.md, checkpoints.md and cleanup_validation.md before interpreting a result. Training completion alone supplies no new localization conclusion.

Review artifacts are local to thesis-repository-cleanup. No merge or push has occurred. Shared models/trainers, broad launcher portability and destructive hygiene remain staged behind the baseline/approval gates documented in the progress ledger.
