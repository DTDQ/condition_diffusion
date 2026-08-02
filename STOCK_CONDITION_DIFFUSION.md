# Scheme A + GLM Conditional Diffusion-TS

The diffusion target is the 19 Scheme A features. At every forecast origin the model reads:

- 60 trading days of normalized 19-dimensional Scheme A history;
- 48 GLM scores split into company (20), sector (12), macro (10), and meta (6) tokens;
- the corresponding 48-dimensional observed mask;
- a noisy future trajectory (18 days for the May-2026 experiment), with 19 features.

The denoiser keeps Diffusion-TS's direct reconstruction and trend/season decomposition. Its decoder cross-attends to the encoded noisy future, history, and four GLM tokens. GLM scores and masks are conditions, never prediction targets.

## Train

```bash
.miniconda3/envs/condition-diffusion/bin/python -m stock_diffusion.train \
  --config Config/stock_condition_diffusion.yaml
```

Run a two-step end-to-end verification first:

```bash
.miniconda3/envs/condition-diffusion/bin/python -m stock_diffusion.train \
  --config Config/stock_condition_diffusion.yaml --smoke-test
```

Resume with `--resume outputs/stock_condition_diffusion_may2026/checkpoint-N.pt`. The May-2026 experiment trains through March 2026, validates on April, and tests on all 18 available May trading days. Robust normalization is fitted only on the training period and saved with the run.
