# MAA-GWLP · Multi-Agent Adversarial Groundwater-Level Prediction

Multi-agent adversarial learning for groundwater-level forecasting with GRU/LSTM/Transformer generators and paired discriminators. The framework supports multi-window inputs, knowledge distillation, cross-finetuning, and hydrology-focused metrics (NSE, KGE, R², Bias, RMSE, MAE, MAPE).
<img width="3685" height="2182" alt="图片1" src="https://github.com/user-attachments/assets/967067b8-aaea-4eae-9eb6-7ee67919cf46" />

---

## Components

- `run_multi_gan.py`: multi-agent training/inference entry (multi-window, multi-generator/discriminator, distillation, cross-finetune).
- `run_baseframe.py`: single-model baseline (Transformer by default; GRU/LSTM optional).
- `time_series_maa.py` / `time_series_baseframe.py`: data prep (column selection, scaling, sliding windows, rise/flat/fall labels) and dataloader construction.
- `models/`: generators (regression + classification heads) and discriminators.
- `utils/evaluate_visualization.py`: hydrology metrics (MSE/MAE/RMSE/MAPE/NSE/KGE/R²/Bias), fit/residual plots, GAN loss visualizations.
- `utils/multiGAN_trainer_disccls.py`: multi-agent adversarial training loop with AMP support.
- `configs/`: sample config template.
- `out_put/`: sample output folder structure (logs/plots).

---

## Data Expectations

Provide a time-ordered CSV with at least:

- `timestamp` (or sortable date)
- Target: groundwater level column (single well `gwl`; for multiple wells include `station_id` to group beforehand)
- Driver examples: precipitation `precip`, temperature `temp`, ET0, pumping `pumping`, river stage, soil moisture, land-use one-hot, etc.

Set `--feature_columns` / `--target_columns` (0-based indices) in CLI or config. Default classification labels are rise/flat/fall from level differences. Adjust `generate_labels` if you prefer a different scheme (e.g., exceedance warning).

---

## Quickstart

```bash
# Baseline (Transformer)
python run_baseframe.py \
  --data_path data/groundwater/sample.csv \
  --feature_columns 1 2 3 4 5 \
  --target_columns "[[0]]" \
  --window_sizes 30 \
  --predict_step 1

# Multi-agent MAA (GRU/LSTM/Transformer)
python run_multi_gan.py \
  --data_path data/groundwater/sample.csv \
  --feature_columns 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 \
  --target_columns "[[0]]" \
  --window_sizes 15 30 45 \
  --num_epochs 200
```

Tip: `--feature_columns` is grouped per window; duplicate the same set for each window length you use.

---

## Workflow

1. Define target/feature columns (or edit `configs/gwl_example.yaml`), then split by well if needed.
2. Train baseline with `run_baseframe.py`; then train full MAA with `run_multi_gan.py`.
3. Inspect outputs under `out_put/groundwater_*`: logs, fit/residual plots, GAN loss curves; CSV with NSE/KGE/R²/Bias and standard errors.
4. Tune windows (memory vs responsiveness), generator mix, and distillation/cross-finetune settings for robustness.

---

## Repo Structure (excerpt)

```
├── run_multi_gan.py
├── run_baseframe.py
├── time_series_maa.py
├── time_series_baseframe.py
├── models/
├── utils/
├── configs/
└── out_put/
```

---

## License

MIT License. Use responsibly for groundwater management and research.
