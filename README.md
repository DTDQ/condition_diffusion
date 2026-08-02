# Conditional Diffusion-TS for Stock Panels

基于 Diffusion-TS 改造的股票面板条件扩散模型。模型以每只股票独立的 Scheme A 历史序列为预测对象，通过 cross-attention 注入 48 维 GLM 条件特征及其 48 维 observed mask。

## 最新正式实验

- 数据时间：2023-01 至 2026-07
- 训练截止：2026-05-29
- 验证集：2026-06
- 测试集：2026-07
- 股票数：3,574
- 历史长度：60 个交易日
- 预测长度：18 个交易日
- 最佳检查点：`outputs/expanded_2023_2026_glm/checkpoint-best.pt`
- 中文测试报告：[`outputs/expanded_2023_2026_glm/TEST_RESULTS_2026-07.md`](outputs/expanded_2023_2026_glm/TEST_RESULTS_2026-07.md)

正式测试的 MAE 为 0.174785，较 persistence 基线改善 13.53%；VWAP 相对预测起点的方向准确率为 67.76%。详细的概率校准、方向偏置和原 Diffusion-TS 风格指标见测试报告。

## 仓库内容

- `stock_diffusion/`：条件编码器、扩散模型、训练与评估代码
- `scripts/`：特征生成、面板审计和股票序列构建脚本
- `Config/`：正式训练与实验配置
- `outputs/`：模型检查点、指标、日志、图表和报告
- `third_party/Diffusion-TS/`：参考的 Diffusion-TS 实现
- `tests/`：模型和数据管线单元测试
- `STOCK_CONDITION_DIFFUSION.md`：模型结构与早期实验说明

## 数据说明

本仓库不包含原始行情、GLM 材料、事件文件、训练 CSV、构建后的 NumPy 序列或测试预测样本。运行训练前，需要自行生成与配置文件对应的数据目录。

正式版本每个时间点包含：

- 19 维 Scheme A 特征；
- 48 维 GLM 条件值；
- 48 维 observed mask；
- 合计 115 维。

数据张量的轴顺序为 `stock × time × feature`，不能将不同股票首尾连接成一条时间序列。

## 训练

```bash
python -m stock_diffusion.train \
  --config Config/expanded_2023_2026_glm.yaml
```

正式训练前可以运行两步烟雾测试：

```bash
python -m stock_diffusion.train \
  --config Config/expanded_2023_2026_glm.yaml \
  --smoke-test
```

## 测试

```bash
python -m stock_diffusion.evaluate \
  --config Config/expanded_2023_2026_glm.yaml \
  --checkpoint outputs/expanded_2023_2026_glm/checkpoint-best.pt \
  --samples 5 \
  --batch-size 64
```

方向指标：

```bash
python -m stock_diffusion.directional_metrics \
  --output-dir outputs/expanded_2023_2026_glm \
  --data-dir train_data/scheme_a_glm_sequences_2023-01_2026-07
```

## 单元测试

```bash
python -m unittest discover -s tests -v
```

## 上游项目

模型设计参考了 ICLR 2024 的 Diffusion-TS。上游代码和许可证保留在 `third_party/Diffusion-TS/`。
