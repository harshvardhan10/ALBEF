# Anatomy-Aware Weak Localization in Chest X-Ray Vision–Language Models

Research repository for **Improving Weak Localization in Chest X-Ray Vision–Language Models Using Anatomy-Aware Refinement**, a Master's thesis by **Harshvardhan Agarwal**, Friedrich-Alexander-Universität Erlangen-Nürnberg, 2026.

## Primary research branch: `work`

**[`work`](https://github.com/harshvardhan10/ALBEF/tree/work) is the main branch for the thesis research code, documentation and ongoing development.**

The Git branch named `main` retains the earlier repository state. Use `work` for the organized repository and current instructions.

- **[Read the full project README](https://github.com/harshvardhan10/ALBEF/blob/work/README.md)**
- [Find the model code for each main experiment](https://github.com/harshvardhan10/ALBEF/blob/work/docs/model_code.md)
- [Explore the experiment mapping](https://github.com/harshvardhan10/ALBEF/blob/work/docs/experiments.md)

## Get the research code

```bash
git clone --branch work --single-branch https://github.com/harshvardhan10/ALBEF.git
cd ALBEF
```

For an existing checkout, save any local changes before switching:

```bash
git fetch origin
git switch work
```

The `work` branch contains single-view ALBEF, anatomical guidance, original/lung/heart representations, late score fusion, six joint fusion architectures, heatmap extraction and evaluation. Its documentation distinguishes thesis-reported results from independently verified behavior and records the assets still required for full reproduction.
