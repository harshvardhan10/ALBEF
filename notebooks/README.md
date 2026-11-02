# Notebook catalog

All 28 historical notebooks are preserved. Seven contain no stored cell outputs or embedded attachments and now have grouped canonical locations. Twenty-one with stored outputs remain at their original locations pending data/output review. No cells, outputs or notebook metadata were rewritten or cleared.

## Execution context

Canonical paths below are for viewing and organization. Before running, inspect the context:

```bash
python -B tools/notebook_context.py notebooks/evaluation/localization_pointing_game.ipynb
```

Open the reported legacy path and verify the kernel cwd. The reported directory is derived from the original file parent; it is not a recovered HPC launch record. Some notebooks call `os.chdir("..")` or read `../results`; starting from a different directory can change behavior. This inspector does not start a kernel, execute cells, load data or overwrite outputs. Legacy symlinks require symlink support.

A5-containing comparisons remain historical work, not a thesis A5 experiment requirement. Four qualitative thesis radiographs, larger extraction galleries and quantitative pointing cohorts are distinct units of evidence.

## Organized and retained analyses

| Category | Notebook (canonical view) | Status | Legacy execution path |
|---|---|---|---|
| comparisons | [compare_original_albef_fc_gated_mean_residual_transformer_heatmaps.ipynb](comparisons/compare_original_albef_fc_gated_mean_residual_transformer_heatmaps.ipynb) | Moved; byte-identical | `notebooks/compare_original_albef_fc_gated_mean_residual_transformer_heatmaps.ipynb` |
| comparisons | [compare_original_albef_vs_fc_fused_heatmaps.ipynb](comparisons/compare_original_albef_vs_fc_fused_heatmaps.ipynb) | Moved; byte-identical | `notebooks/compare_original_albef_vs_fc_fused_heatmaps.ipynb` |
| comparisons | [visualize_chexzero_checkpoints_and_model_comparison.ipynb](comparisons/visualize_chexzero_checkpoints_and_model_comparison.ipynb) | Moved; byte-identical | `notebooks/visualize_chexzero_checkpoints_and_model_comparison.ipynb` |
| comparisons | [visualize_transformer_vs_original_albef_gradcams.ipynb](comparisons/visualize_transformer_vs_original_albef_gradcams.ipynb) | Moved; byte-identical | `notebooks/visualize_transformer_vs_original_albef_gradcams.ipynb` |
| evaluation | [evaluate_vindr_heatmaps.ipynb](evaluate_vindr_heatmaps.ipynb) | Held; stored outputs | `notebooks/evaluate_vindr_heatmaps.ipynb` |
| evaluation | [localization_pointing_game.ipynb](evaluation/localization_pointing_game.ipynb) | Moved; byte-identical | `notebooks/localization_pointing_game.ipynb` |
| exploration | [eda.ipynb](eda.ipynb) | Held; stored outputs | `notebooks/eda.ipynb` |
| exploration | [zero_shot_vind_testing.ipynb](zero_shot_vind_testing.ipynb) | Held; stored outputs | `notebooks/zero_shot_vind_testing.ipynb` |
| exploratory_refinement | [CLARity_style_anatomy_gated_diagnostic.ipynb](CLARity_style_anatomy_gated_diagnostic.ipynb) | Held; stored outputs | `notebooks/CLARity_style_anatomy_gated_diagnostic.ipynb` |
| exploratory_refinement | [end-to-end cardiomegaly evaluation.ipynb](end-to-end%20cardiomegaly%20evaluation.ipynb) | Held; stored outputs | `notebooks/end-to-end cardiomegaly evaluation.ipynb` |
| exploratory_refinement | [stress_test_CLARity_results.ipynb](stress_test_CLARity_results.ipynb) | Held; stored outputs | `notebooks/stress_test_CLARity_results.ipynb` |
| extraction | [extract_heatmap_att.ipynb](extract_heatmap_att.ipynb) | Held; stored outputs | `notebooks/extract_heatmap_att.ipynb` |
| extraction | [extract_heatmap_gradcam_with_hooks.ipynb](extract_heatmap_gradcam_with_hooks.ipynb) | Held; stored outputs | `notebooks/extract_heatmap_gradcam_with_hooks.ipynb` |
| extraction | [extract_heatmaps.ipynb](extract_heatmaps.ipynb) | Held; stored outputs | `notebooks/extract_heatmaps.ipynb` |
| extraction | [extract_heatmaps_crossattn.ipynb](extract_heatmaps_crossattn.ipynb) | Held; stored outputs | `notebooks/extract_heatmaps_crossattn.ipynb` |
| extraction | [extract_heatmaps_gradcam.ipynb](extract_heatmaps_gradcam.ipynb) | Held; stored outputs | `notebooks/extract_heatmaps_gradcam.ipynb` |
| extraction | [visualization.ipynb](../visualization.ipynb) | Held; stored outputs | `visualization.ipynb` |
| figures | [generate_thesis_anatomy_and_vindr_figures.ipynb](figures/generate_thesis_anatomy_and_vindr_figures.ipynb) | Moved; byte-identical | `notebooks/generate_thesis_anatomy_and_vindr_figures.ipynb` |
| non_thesis_comparison | [compare_original_A1_A2_A3_A4_A5_qualitative_heatmaps.ipynb](../archive/non_thesis_experiments/notebooks/compare_original_A1_A2_A3_A4_A5_qualitative_heatmaps.ipynb) | Moved; byte-identical | `notebooks/compare_original_A1_A2_A3_A4_A5_qualitative_heatmaps.ipynb` |
| visualization | [visualize_biovil_t_heatmaps.ipynb](visualize_biovil_t_heatmaps.ipynb) | Held; stored outputs | `notebooks/visualize_biovil_t_heatmaps.ipynb` |
| visualization | [visualize_biovil_t_same_50_albef_itc_cases.ipynb](visualize_biovil_t_same_50_albef_itc_cases.ipynb) | Held; stored outputs | `notebooks/visualize_biovil_t_same_50_albef_itc_cases.ipynb` |
| visualization | [visualize_cardiomegaly_grad_cams.ipynb](visualize_cardiomegaly_grad_cams.ipynb) | Held; stored outputs | `notebooks/visualize_cardiomegaly_grad_cams.ipynb` |
| visualization | [visualize_cardiomegaly_pleural_effusion_itc_heatmaps.ipynb](visualize_cardiomegaly_pleural_effusion_itc_heatmaps.ipynb) | Held; stored outputs | `notebooks/visualize_cardiomegaly_pleural_effusion_itc_heatmaps.ipynb` |
| visualization | [visualize_cardiomegaly_pleural_effusion_itc_heatmaps_margin.ipynb](visualize_cardiomegaly_pleural_effusion_itc_heatmaps_margin.ipynb) | Held; stored outputs | `notebooks/visualize_cardiomegaly_pleural_effusion_itc_heatmaps_margin.ipynb` |
| visualization | [visualize_heatmaps.ipynb](visualize_heatmaps.ipynb) | Held; stored outputs | `notebooks/visualize_heatmaps.ipynb` |
| visualization | [visualize_itc_margin_gradcam_heatmaps.ipynb](visualize_itc_margin_gradcam_heatmaps.ipynb) | Held; stored outputs | `notebooks/visualize_itc_margin_gradcam_heatmaps.ipynb` |
| visualization | [visualize_itc_margin_gradcam_heatmaps_layer_10.ipynb](visualize_itc_margin_gradcam_heatmaps_layer_10.ipynb) | Held; stored outputs | `notebooks/visualize_itc_margin_gradcam_heatmaps_layer_10.ipynb` |
| visualization | [visualizing_loss_curves.ipynb](visualizing_loss_curves.ipynb) | Held; stored outputs | `notebooks/visualizing_loss_curves.ipynb` |

The [machine-readable catalog](../docs/audit/notebook_catalog.csv) records hashes, status, categories and original-parent context. Stored outputs may contain patient images or private identifiers; this index does not certify redistribution rights.
