
# MAA‑GWLP: Multi‑Agent Adversarial Groundwater‑Level Prediction  
_A Robust Framework for Complex Hydro‑Temporal Forecasting_

---

## 1 · Overview  
**MAA‑GWLP** (Multi‑Agent Adversarial **G**round**w**ater‑**L**evel **P**rediction) is an advanced deep‑learning framework that unifies *multi‑agent adversarial learning*, *elite‑guided refinement*, and *knowledge‑distillation alignment* to deliver high‑fidelity groundwater‑level forecasts in heterogeneous, multi‑factor hydro‑systems. By jointly optimising regression (water height) and classification (hydrologic‑state) tasks, the framework captures regional heterogeneity, abrupt anthropogenic impacts, and long‑lag responses that routinely confound traditional models.

---

## 2 · Repository Components  

| Script / Module | Purpose | Algorithms Included | Typical Output |
|-----------------|---------|---------------------|----------------|
| **`run_baseframe.py`** | Baseline training / prediction | LSTM, GRU, Transformer | Deterministic forecasts, evaluation metrics |
| **`run_multi_gan.py`** | Full adversarial training / prediction | **MAA‑GWLP** (multi‑agent generators + discriminators) | Probabilistic or point forecasts, attention maps, discriminator scores |

Both pipelines share the same data loaders, preprocessing utilities, and configuration file (`config.yml`). Switching between single‑agent and multi‑agent modes therefore requires only a change of entry script.

---

## 3 · Key Contributions
* **Multi‑Agent Generators & Discriminators** – collective \(G=\{G_i\}\) and \(D=\{D_j\}\) trained with adaptive adversarial losses.  
* **Adaptive Window Alignment** – guarantees \(\min_i w_i > h_{\text{max}}\) for cross‑model discrimination.  
* **Dynamic Weight Matrix \(W\)** – softmax‑scaled performance weights yield a self‑organising adversarial ecosystem.  
* **Elite‑Guided Fine‑Tuning** – best generator–discriminator pair intensively refined every \(\kappa\) epochs.  
* **Multi‑Task Knowledge Distillation** – teacher–student transfer of regression and classification “soft targets”.  
* **Hydrology‑Centric Optimisation** – loss functions and encoders tailored to precipitation, pumping, ET, and land‑use drivers.

---

## 4 · Methodology Snapshot
```text
Generators G₁ … G_N  ─▶  Cross‑Model Alignment ─▶  Discriminators D₁ … D_N
        ▲                     │                         │
        │ Dynamic W           ▼                         ▼
        └─── Elite‑Guided Refinement ◀── Knowledge Distillation
````

---

## 5 · Typical Groundwater Workflow

1. **Data Preparation** – merge groundwater levels with exogenous forcings, normalise, and package into multi‑channel tensors.
2. **Unsupervised Pre‑training** – auto‑encoder reconstruction loss with optional WGAN‑GP robustness.
3. **Multi‑Agent Adversarial Training** – alternate optimisation of $G_i$ and $D_j$ using dynamic $W$; elite fine‑tuning every $\kappa$ epochs.
4. **Knowledge‑Distillation Phase** – teacher–student transfer with temperature $T$ plus ground‑truth supervision.
5. **Inference & Interpretation** – single‑ or multi‑step forecasts; attention maps or discriminator feedback for explainability.

---

## 6 · Installation

```bash
conda env create -f environment.yml   # Python ≥ 3.9, PyTorch ≥ 2.0 (CUDA 11.x optional)
conda activate maa_gwlp
```

---

## 7 · Quick Start

```bash
# ---- Baseline models (LSTM / GRU / Transformer) ----
python run_baseframe.py --config config.yml         # training + evaluation
python run_baseframe.py --config config.yml --predict  # inference only

# ---- Full MAA‑GWLP multi‑agent framework ----
python run_multi_gan.py   --config config.yml       # training + evaluation
python run_multi_gan.py   --config config.yml --predict  # inference only
```

---

## 8 · Adapting to Other Hydro‑Domains

Replace the dataset with any hydro‑meteorological time‑series conforming to the loader interface and adjust `config.yml` (input dims, window sizes, horizons).

---

## 9 · License

Distributed under the MIT License. See `LICENSE` for details.

```
