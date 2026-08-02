# 48-dimensional conditional Diffusion-TS run

## Data

- Period: 2023-01-03 through 2026-07-31
- Balanced universe: 3,574 stocks
- Trading dates: 866
- Panel keys: 3,095,084
- Per-step input: 19 Scheme-A values + 48 GLM condition values + 48 observed masks = 115 features
- Sequence tensor: `(3574, 866, 115)`, ordered as stock × time × feature
- Split: train through 2026-05-29, validation in 2026-06, test in 2026-07
- The 43 monthly condition files passed checks for complete stock/date keys, identical 98-column headers, finite values, binary masks, and no duplicates.

## Training

- Device: NVIDIA GeForce RTX 4070 Ti (CUDA)
- Steps: 45,000
- Best checkpoint: step 35,000
- Best validation loss: 0.034170763976871965
- History length: 60 trading days
- Forecast length: 18 trading days

## Test setup

- Test period: 2026-07-01 through 2026-07-31
- Forecast origins: 6
- Stocks per origin: 3,574
- Test windows: 21,444
- Probabilistic samples per window: 5

## Main metrics

| Metric | Result |
|---|---:|
| MAE | 0.174785 |
| RMSE | 0.492555 |
| CRPS | 0.143552 |
| 80% interval empirical coverage | 37.1946% |
| Mean 80% interval width | 0.214312 |
| Persistence MAE | 0.202136 |
| Persistence RMSE | 0.555221 |
| MAE improvement over persistence | 13.5310% |

The point forecast beats persistence, but the probabilistic interval is substantially under-covered and therefore too narrow.

## VWAP direction

For direction from each forecast origin to each future horizon:

- Accuracy: 67.7632%
- Balanced accuracy: 68.5027%
- Up recall: 92.9624%
- Down recall: 44.0430%
- Predicted-up rate: 73.9003%
- Actual-up rate: 48.4882%
- Confusion counts: true up 173,110; true down 87,129; false up 110,698; false down 13,105

Daily VWAP direction accuracy is 58.6223%. The model has a strong upward prediction bias, so raw directional accuracy should be read together with balanced accuracy and down recall.

## Metrics matching the upstream Diffusion-TS evaluation style

- Single generated trajectory MSE on original scale: 0.280200
- Correlational score: 0.565064
- Discriminative score: 0.375752 ± 0.021544 (five runs)
- Predictive score: 0.0161166 ± 0.0000424 (five runs)
- Context-FID: 0.416047 ± 0.063726 (five runs)

These upstream-style scores use one generated trajectory rather than the five-sample ensemble mean, so their MSE is not directly identical to the main RMSE squared.

No shuffled-condition or GLM ablation result is included.
