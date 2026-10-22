# Dependency records

The root requirements.txt selects the existing legacy runtime pins, without automatically installing torch/CUDA. Keep the working server environment. requirements-eval.txt and requirements-dev.txt explicitly reuse that stack; unittest needs no new testing dependency.

notebooks-historical.txt preserves the original notebook/environment freeze byte-for-byte. It mixes newer packages with a different torch version and is historical evidence, not a Python 3.8 training lock. Do not install it into the training environment. Baselines retain their own dependency files and environments. No dependency installation or upgrade was performed during cleanup.
