# MAA‑GWLP: Multi‑Agent Adversarial Groundwater‑Level Prediction

*A Robust Framework for Complex Hydro‑Temporal Forecasting*

<img width="1030" height="495" alt="MAA‑GWLP architecture" src="https://github.com/user-attachments/assets/85022e2c-316c-4e94-9e7c-9c288af6ba5d" />

---

## 1 · Overview

**MAA‑GWLP** (Multi‑Agent Adversarial **G**round**w**ater‑**L**evel **P**rediction) is a next‑generation deep‑learning framework for groundwater‑level forecasting.

It combines three synergistic mechanisms:

1. **Multi‑Agent Adversarial Learning** – a collective of generators <i>G<sub>i</sub></i> and discriminators <i>D<sub>j</sub></i> trained on staggered historical windows, enforcing realism through adaptive, cross‑model discrimination.
2. **Elite‑Guided Refinement** – periodic, intensive fine‑tuning of the best generator–discriminator pair to accelerate convergence.
3. **Multi‑Task Knowledge Distillation** – teacher‑student transfer of both regression and classification knowledge, enabling agents to share complementary skills and to model interactions among heterogeneous drivers (e.g., precipitation ↔ pumping).

The framework currently ingests **precipitation, temperature, pumping‑rate, and land‑use indicators** (coastal, plain, karst) alongside groundwater levels; additional factors can be added with minimal code changes. A Bayesian extension for quantitative **forecast‑uncertainty estimation** is on the roadmap.

---

## 2 · Repository Components

| Entry Script           | Purpose                                            | Algorithms Included                                                  | Output                                                                 |
| ---------------------- | -------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **`run_baseframe.py`** | Baseline training / prediction                     | LSTM, GRU, Transformer                                               | Deterministic forecasts, standard metrics                              |
| **`run_multi_gan.py`** | Full multi‑agent adversarial training / prediction | **MAA‑GWLP** (generators + discriminators, distillation, elite loop) | Probabilistic or point forecasts, attention maps, discriminator scores |

Both pipelines share data loaders, preprocessing utilities, and a unified configuration file. Switching between single‑agent and multi‑agent modes only requires changing the entry script.

---

## 3 · Key Innovations

* **Cross‑Model Discrimination with Adaptive Windows** <em>For each pair (G<sub>i</sub>, D<sub>j</sub>) inputs are temporally aligned according to<br>
  (X,Y)<sub>G<sub>i</sub>→D<sub>j</sub></sub> under the constraint <strong>min<sub>i</sub> w<sub>i</sub> > h<sub>max</sub></strong>.</em>

* **Dynamic Weight Matrix** **W<sub>ij</sub>** – modulates the influence of each generator on each discriminator based on validation performance (temperature parameter **β**).

* **Multi‑Objective Generator Loss** – combines regression (L<sub>2</sub>), classification (cross‑entropy), and adversarial deception; hyper‑parameters **λ<sub>1</sub>**, **λ<sub>2</sub>**, **λ<sub>3</sub>** control trade‑offs.

* **Elite‑Guided Loop**  – after every **κ** epochs, the best generator–discriminator pair undergoes **τ** epochs of focused adversarial refinement.

* **Multi‑Task Knowledge Distillation** – transfers softened logits (temperature **T**) and hard targets from the best to the weakest agent, explicitly modelling factor interactions and reducing over‑fitting.

* **Hydrology‑Centric Design** – losses include Nash–Sutcliffe Efficiency (NSE) and Huber; encoders accommodate spatial land‑use categories and multi‑factor couplings.

---

## 4 · Methodology Summary

```text
Generators G₁ … Gₙ  ─▶  Adaptive Alignment ─▶  Discriminators D₁ … Dₙ
       ▲                     │                           │
       │  Dynamic Weight W   ▼                           ▼
       └── Elite‑Guided Refinement  ◀──  Knowledge Distillation
```

---

## 5 · Typical Workflow

1. **Data Preparation**

   * Merge groundwater levels with precipitation, temperature, pumping, and land‑use factors.
   * Normalise / standardise; encode land‑use as one‑hot (coastal, plain, karst).

2. **Unsupervised Pre‑training**

   * Initialise each <i>G<sub>i</sub></i> via auto‑encoder reconstruction loss; optional WGAN‑GP for robustness.

3. **Multi‑Agent Adversarial Training**

   * Train <i>G<sub>i</sub></i>, <i>D<sub>j</sub></i> with losses L<sub>G</sub>, L<sub>D</sub> and adaptive <i>W</i>; perform elite refinement every **κ** epochs.

4. **Knowledge Distillation Phase**

   * Teacher‑student transfer with temperature‑controlled KL + ground‑truth supervision, capturing cross‑factor interactions.

5. **Inference & Interpretation**

   * Output single‑ or multi‑step forecasts; discriminator scores and attention maps highlight driver importance.

**Planned extension:** incorporate Bayesian posterior sampling (e.g., SVGD) on top of the trained agent ensemble to quantify predictive uncertainty.

---

## 6 · Installation

```bash
conda env create -f environment.yml     # Python ≥3.9, PyTorch ≥2.0, CUDA 11.x optional
conda activate maa_gwlp
```

---

## 7 · Quick Start

```bash
# --- Baseline LSTM / GRU / Transformer ---
python run_baseframe.py --config config.yml               # train + eval
python run_baseframe.py --config config.yml --predict     # inference

# --- Full MAA‑GWLP Architecture ---
python run_multi_gan.py --config config.yml               # train + eval
python run_multi_gan.py --config config.yml --predict     # inference
```

---

## 8 · Extending to Other Hydro‑Domains

Replace the dataset with any hydro‑meteorological time‑series compatible with `loaders/` and update `config.yml` (input dimensions, window sizes, horizons, driver list).

---

## 9 · License

Distributed under the MIT License. See `LICENSE` for details.
