# Checkpoints and result provenance

The author confirmed cross-attention training completed. Both files are reported present under:

```text
/home/vault/iwi5/iwi5362h/outputs/A0_multiview_cross_attention_fusion_dual_auc/
  checkpoint_best_cardiomegaly_auc.pth
  checkpoint_best_macro_auc_stable.pth
```

The first selects cardiomegaly validation AUC; the second selects stable-label macro validation AUC. A last checkpoint is for recovery, not a third scientific selection role. Checkpoint existence/content has not been independently inspected. Exact training commit and saved resolved config are still required from server records.

For each result record: experiment identifier, training and extraction commits, checkpoint SHA-256/role, resolved config, case/split identity, label order, prompts, threshold source, extraction/evaluation command and output manifest. Other experiments' checkpoint paths and evaluation coverage remain pending inventory; do not invent them from this run's names.

Do not commit weights, raw patient images or generated heatmaps. Keep these in server storage; curate only permitted summaries. No best-checkpoint result is inferred from training completion.
