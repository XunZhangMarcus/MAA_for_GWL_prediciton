# MAA-GWLP · Multi-Agent Adversarial Groundwater-Level Prediction

地下水位预测的多代理对抗框架，聚合 GRU / LSTM / Transformer 生成器与判别器，支持多时间窗、知识蒸馏与交叉微调。当前版本已移除金融回测逻辑，内置水文指标（NSE、KGE、R²、Bias、RMSE 等）与水位可视化。

---

## What’s Inside

- `run_multi_gan.py`：完整的多代理 MAA 训练/推理入口（多窗口、多生成器、多判别器、蒸馏与交叉微调）。
- `run_baseframe.py`：单模型基线（默认 Transformer，可切换 GRU/LSTM）。
- `time_series_maa.py` / `time_series_baseframe.py`：数据预处理（按列选择特征/目标、归一化、滑窗、涨跌/平分类标签）与数据加载器构建。
- `models/`：生成器（回归 + 分类双头）与判别器。
- `utils/evaluate_visualization.py`：水文指标计算（MSE/MAE/RMSE/MAPE/NSE/KGE/R²/Bias）、拟合/残差曲线、GAN 训练过程可视化。
- `utils/multiGAN_trainer_disccls.py`：多代理对抗训练循环（自适应权重、蒸馏、交叉微调、AMP 支持）。
- `configs/`：示例配置模板（填入你的数据路径、特征/目标列、窗口长度等）。
- `out_put/`：示例输出目录结构（日志/图表）。

---

## Data Expectations

请准备按时间排序的 CSV，至少包含：

- `timestamp`（或日期列，用于排序/切分）
- 目标：地下水位列（单井可用 `gwl`，多井可包含 `station_id` 用于分组）
- 驱动因子示例：降水 `precip`、气温 `temp`、蒸散发/ET0、抽水量 `pumping`、河水位、土壤湿度、土地利用 one-hot 等

使用时在脚本参数里指定 `--feature_columns`、`--target_columns` 对应的列索引（0-based）。默认分类标签为水位涨/平/跌（基于数值差分）；若想关闭分类头，可在模型中自行调整。

---

## Quickstart (CLI)

```bash
# 单模型基线（默认 Transformer）
python run_baseframe.py \
  --data_path data/groundwater/sample.csv \
  --feature_columns 1 2 3 4 5 \
  --target_columns "[[0]]" \
  --window_sizes 30 \
  --predict_step 1

# 多代理 MAA（默认 GRU/LSTM/Transformer 三生成器 + 判别器）
python run_multi_gan.py \
  --data_path data/groundwater/sample.csv \
  --feature_columns 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 \
  --target_columns "[[0]]" \
  --window_sizes 15 30 45 \
  --num_epochs 200
```

> 提示：`--feature_columns` 可按 “每个窗口一组” 传入，例如 `a b c d e a b c d e a b c d e` 对应三个窗口。示例命令仅为占位，请根据实际列数修改。

---

## Key Changes for Groundwater

- **指标**：移除金融回撤/夏普等，新增 NSE、KGE、R²、Bias，沿用 MSE/MAE/RMSE/MAPE 与分目标 MSE。
- **可视化**：输出观测 vs 预测曲线与残差曲线；GAN 训练损失、MSE 收敛曲线保持可用。
- **标签**：默认使用涨/平/跌三分类标签；若需要“超限预警”可在 `generate_labels` 逻辑中替换。
- **默认路径**：输出目录调整为 `out_put/groundwater_*`，示例数据路径指向 `data/groundwater/sample.csv`。

---

## Suggested Workflow

1. **准备数据列**：确认目标列与特征列索引，填入命令行或 `configs/` YAML。多井场景可先按井分组运行。
2. **训练**：先跑 `run_baseframe.py` 做基线，再跑 `run_multi_gan.py` 获得多代理集成结果。
3. **查看结果**：在 `out_put/groundwater_*` 下查看日志、拟合/残差图、GAN 损失曲线；CSV 中包含 NSE/KGE/R²/Bias 等指标。
4. **调优要点**：
   - 窗口长度与预测步长：短窗捕捉快速响应，长窗建模补给-开采记忆。
   - 生成器组合：风格多样（GRU/LSTM/Transformer）提升稳健性。
   - 蒸馏与交叉微调：保留 `--distill_epochs`、`--cross_finetune_epochs` 以提升弱生成器。

---

## Repo Structure (excerpt)

```
├── run_multi_gan.py           # 多代理入口
├── run_baseframe.py           # 基线入口
├── time_series_maa.py         # 多代理数据管线
├── time_series_baseframe.py   # 基线数据管线
├── models/                    # 生成器/判别器
├── utils/                     # 训练、评估、可视化工具
├── configs/                   # 示例配置
└── out_put/                   # 输出示例
```

---

## License

MIT License. Use responsibly for groundwater resource management and research.
