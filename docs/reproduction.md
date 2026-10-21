# Reproduction

1. Read the experiment catalogue and identify the exact model/trainer/config and checkpoint role.
2. Use the documented successful server environment: Python 3.8.20, torch 1.8.1+cu111 and CUDA 11.1. Record the actual installed versions. requirements_albef_legacy.txt is a dependency reference, not a complete GPU environment lock; requirements.txt is a historical notebook freeze and includes incompatible modern packages for Python 3.8. Neither is silently rewritten as a newly verified environment.
3. Supply local dataset/mask paths, checkpoint files and an offline cached BERT model/tokenizer. New extraction forces offline mode; no new accounts are needed.
4. For completed cross-attention training, follow [cross_attention_validation.md](cross_attention_validation.md). This does not launch training again.
5. For historical experiments, inspect the pinned original launch record in cluster/historical and original configs. Those scripts are site-specific and may reference other checkouts. Do not submit them unchanged. The source map documents their origin.

Run CheXzero from its own working directory (`cd baselines/chexzero`) and use its own requirements/environment. Its unqualified imports such as model and clip must resolve there; do not combine its dependencies blindly with ALBEF. Original README, LICENSE and NOTICE remain in that directory. No baseline runtime reproduction has been claimed.

Local focused checks: `python -m unittest discover -s tests -v`. They exercise extraction/comparison behavior without loading the legacy model stack or server weights. This is not a full pretraining or DDP check. No experiments, download jobs or emails are submitted by the agent.
