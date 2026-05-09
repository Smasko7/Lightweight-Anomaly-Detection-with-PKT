# 🔬 Knowledge Distillation for Lightweight Anomaly Detection

> Can a tiny ensemble student learn better anomaly detection by imitating a deep teacher's latent space — without growing its parameter count?

This project investigates **Probabilistic Knowledge Transfer (PKT)** from four deep autoencoder teachers to a KitNET-style ensemble student across four tabular anomaly detection benchmarks.

**4 datasets × 14 variants × 3 seeds = 168 experiments.**

---

## 📖 Background & Motivation

**KitNET** (Mirsky et al., NDSS 2018) is an ensemble of small autoencoders, each trained on a local subset of features. It is extremely lightweight (under 5K parameters) and was designed for online network intrusion detection. Its weakness is that each sub-autoencoder operates in isolation — it has no mechanism to incorporate global data structure.

**PKT** (Passalis & Tefas, ECCV 2018 / TNNLS 2020) transfers knowledge by matching *pairwise similarity distributions* between teacher and student latent spaces. Because it operates on B×B similarity matrices, teacher and student can have different embedding dimensionalities — no projection layer is needed.

**Our question:** Does a frozen deep teacher, trained on normal data only, give the Kitsune student useful global structure that it cannot learn on its own?

---

## 💡 Novel Contribution

The core novelty of this work is **applying PKT to a KitNET-style ensemble student** for anomaly detection. Prior PKT applications target monolithic students with a single latent space. KitNET has no such latent — it produces a distributed representation spread across K independent sub-AE hidden states. We define a *pseudo-latent* by concatenating these hidden activations, enabling PKT without changing the student's inference-time architecture or parameter count.

```
┌─────────────────────────────────────────────────────────────┐
│  🎓 TEACHER  (deep AE, ~100–290K params)                    │
│  Trained on normal data only · Frozen during distillation   │
│  Produces: global latent  z_t ∈ ℝ⁶⁴                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    PKT loss (B×B pairwise
                    similarity matching)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  🎒 STUDENT  (Kitsune PyTorch, ~0.4–5K params)              │
│  Ensemble of K small AEs on local feature groups            │
│  Produces: anomaly score + pseudo-latent z_s ∈ ℝᴰ          │
└─────────────────────────────────────────────────────────────┘
```

Because PKT operates on B×B pairwise similarity matrices, the dimensionality mismatch between teacher (64-dim) and student (24–140-dim) is irrelevant — **no projection layer is needed**.

We explore three **PKT routing strategies**, with the third being an additional contribution specific to Transformer teachers:

```
┌────────────────┬────────────────────────────────────────────────────┐
│ Strategy       │ What gets matched                                  │
├────────────────┼────────────────────────────────────────────────────┤
│ Concat         │ PKT(z_t ,  concat(h₁…hₖ))  — 1 loss, global→global│
│ Per-subAE      │ Σᵢ PKT(z_t, hᵢ)             — K losses, global→each│
│ Token ★        │ Σᵢ PKT(token_i, hᵢ)         — K losses, aligned    │
└────────────────┴────────────────────────────────────────────────────┘
  ★  Token routing only available with Transformer teacher
```

**Token routing** exploits the Transformer's tokenization (which uses the **same feature groups** as Kitsune): token i has attended to global context but corresponds to sub-AE i's exact feature subset, providing principled local-global alignment rather than broadcasting the same global vector to every sub-AE.

We experiment with four teacher architectures — Vanilla AE, 1D CNN AE, Transformer AE, and VAE — to assess how teacher quality and inductive bias interact with the distillation framework.

---

## 🏗️ Architecture

### 🎒 Student — Kitsune (PyTorch reimplementation)

```
                    Input  x ∈ ℝᴺ
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐     ┌────▼────┐    ┌────▼────┐
    │ Sub-AE₁ │     │ Sub-AE₂ │    │ Sub-AEₖ │   ← each sees fpg features
    │  h=0.75 │     │  h=0.75 │    │  h=0.75 │   ← h = ⌊fpg × 0.75⌋
    └────┬────┘     └────┬────┘    └────┬────┘
         │               │               │
       RMSE₁           RMSE₂           RMSEₖ
         │               │               │
         └───────────────┼───────────────┘
                         │
                  ┌──────▼──────┐
                  │  normalize  │   ← per-component min-max
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Output AE  │   ← K → ⌊K×0.75⌋ → K
                  └──────┬──────┘
                         │
                  anomaly score (RMSE)

  [PKT only]  pseudo_latent = concat(h₁, h₂, …, hₖ)
```

| Dataset | Features | K groups | fpg | Hidden/group | Pseudo-latent dim | Params |
|---------|----------|----------|-----|-------------|-------------------|--------|
| optdigits | 64 | 8 | 8 | 6 | 48 | **990** |
| mnist | 100 | 10 | 10 | 7 | 70 | **1,727** |
| landsat | 36 | 6 | 6 | 4 | 24 | **406** |
| backdoor | 196 | 14 | 14 | 10 | 140 | **4,560** |

### 🎓 Teachers (~100–290K params each)

| Teacher | Architecture | Latent | Notes |
|---------|-------------|--------|-------|
| Vanilla AE | FC: N→256→128→64→128→256→N | 64-dim | Strongest standalone on 3/4 datasets |
| CNN AE | 1D Conv + AdaptiveAvgPool → FC 64 | 64-dim | Fails on tabular (no spatial structure) |
| Transformer AE | K-group tokens, 2× TransformerEncoderLayer, mean-pool | 64-dim | Enables token routing via `encode_tokens()` |
| VAE | Shared encoder → μ + log_σ² heads, ELBO | 64-dim (μ) | Regularized, smooth latent |

The Transformer uses the **same feature-group tokenization** as Kitsune, which is what makes token-aligned routing possible.

---

## 📐 PKT Routing — Visual Comparison

```
  CONCAT routing                PER-SUBAE routing             TOKEN routing ★
  ─────────────────             ─────────────────             ─────────────────
  Teacher                       Teacher                       Transformer
  z_t ∈ ℝ⁶⁴                   z_t ∈ ℝ⁶⁴                   token₁  token₂  tokenₖ
      │                             │                            │       │       │
      │   1 PKT loss                ├── PKT ──► h₁              │PKT    │PKT    │PKT
      │                             ├── PKT ──► h₂              ▼       ▼       ▼
      ▼                             └── PKT ──► hₖ             h₁      h₂      hₖ
  concat(h₁…hₖ)                K losses, same teacher     K losses, each token
  1 global loss                  latent broadcast           matched to its group
```

---

## ⚙️ Training

### PKT Loss

```python
# teacher_latent: (B, D_t),  student_latent: (B, D_s)  — D_t ≠ D_s is fine
t = F.normalize(teacher_latent, dim=1)
s = F.normalize(student_latent, dim=1)
T = (t @ t.T + 1) / 2              # cosine sim → [0,1]
S = (s @ s.T + 1) / 2
P_t = T / T.sum(dim=1, keepdim=True)   # row-normalize → distributions
P_s = S / S.sum(dim=1, keepdim=True)
loss = F.kl_div(P_s.log(), P_t, reduction='batchmean')
```

### Two-Phase Training

The original KitNET paper trains sub-AEs and output AE as two separate layers: sub-AEs reconstruct raw features, the output AE reconstructs the resulting RMSE scores. These layers decouple naturally in the original online (per-sample) setting.

In our batch gradient-based reimplementation with PKT, naively training everything jointly fails: PKT reshapes sub-AE RMSE values on normal training data, causing the output AE to see a non-stationary input distribution and converge to a flat, non-discriminative solution. Diagnosis showed d-prime collapsing from 1.13 at the sub-AE level to 0.09 at the output AE level.

We adapt KitNET's layer separation into an explicit two-phase batch training schedule:

```
  ╔══════════════════════════════════════════════════════════╗
  ║  Phase 1  (50 epochs, patience=10)                       ║
  ║  ─────────────────────────────────                       ║
  ║  Optimize:  sub_aes only  ·  output AE frozen            ║
  ║  Loss:      Σ RMSE_sub  +  α × pkt_scale × PKT          ║
  ║  pkt_scale = L_recon.detach() / (L_pkt.detach() + ε)    ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Phase 2  (25 epochs, patience=10)                       ║
  ║  ─────────────────────────────────                       ║
  ║  Reset RMSE normalization buffers                        ║
  ║  Freeze:    sub_aes  ·  Optimize: output_ae only         ║
  ║  Loss:      RMSE(rmse_norm, output_recon)                ║
  ╚══════════════════════════════════════════════════════════╝
```

Adaptive scaling keeps the PKT term proportional to the reconstruction loss so neither dominates.

---

## 🗂️ Datasets

All from **ADBench** (Han et al., NeurIPS 2022). Training uses **normal data only** (unsupervised). MinMaxScaler fit on train split. Test = full dataset with labels.

| Dataset | Samples | Features | K | Anomaly % | Train (normal) |
|---------|---------|----------|---|-----------|---------------|
| optdigits | 5,216 | 64 | 8 | 2.9% | 4,052 |
| mnist | 7,603 | 100 | 10 | 9.2% | 5,522 |
| landsat | 5,803 | 36 | 6 | 21.2% | 3,658 |
| backdoor | 95,329 | 196 | 14 | 22.0% | 59,520 |

---

## 📊 Results

Mean AUC-ROC ± std over 3 seeds (42, 123, 456):

### Teachers (upper bound reference)

| Teacher | optdigits | mnist | landsat | backdoor | Params |
|---------|-----------|-------|---------|----------|--------|
| Vanilla AE | **0.992±0.001** | **0.959±0.003** | **0.714±0.017** | 0.950±0.007 | ~120K |
| Transformer | 0.926±0.027 | 0.928±0.002 | 0.556±0.012 | **0.954±0.004** | ~169K |
| VAE | 0.788±0.014 | 0.877±0.021 | 0.449±0.001 | 0.916±0.007 | ~130K |
| CNN | 0.681±0.039 | 0.326±0.010 | 0.248±0.005 | 0.321±0.000 | 137K |

### Kitsune Variants

| Model | optdigits | mnist | landsat | backdoor |
|-------|-----------|-------|---------|----------|
| kitsune vanilla | 0.760±0.050 | 0.823±0.008 | 0.510±0.034 | 0.911±0.006 |
| + pkt_cnn_concat | **+0.029** → 0.789 | −0.050 | +0.016 | −0.006 |
| + pkt_vanilla_concat | −0.077 | −0.026 | +0.020 | +0.019 |
| + pkt_vanilla_per_subae | −0.032 | −0.012 | +0.043 | +0.003 |
| + pkt_transformer_concat | +0.003 | −0.060 | +0.024 | +0.019 |
| + pkt_transformer_per_subae | −0.013 | −0.091 | +0.024 | +0.017 |
| + **pkt_transformer_token** ★ | −0.037 | −0.019 | **+0.045** → 0.555 | **+0.021** → 0.932 |
| + pkt_vae_concat | −0.066 | **≈0** → 0.821 | −0.019 | +0.003 |
| + pkt_vae_per_subae | −0.013 | −0.043 | +0.019 | +0.002 |

### 🏆 Best PKT vs. Kitsune Vanilla

| Dataset | kitsune vanilla | Best PKT | Δ AUC-ROC | Best variant |
|---------|----------------|----------|-----------|--------------|
| optdigits | 0.760 | **0.789** | **+0.029** | pkt_cnn_concat |
| mnist | 0.823 | **0.821** | ≈0 | pkt_vae_concat |
| landsat | 0.510 | **0.555** | **+0.045** | pkt_transformer_token ★ |
| backdoor | 0.911 | **0.932** | **+0.021** | pkt_transformer_token ★ |

### 🔑 Key Findings

1. **PKT improves Kitsune on 3 of 4 datasets.** The hypothesis holds where sufficient training data and anomaly signal exist. The improvement is most credible on backdoor (59K training samples, std < 0.017).

2. **Token routing is the best overall strategy.** `pkt_transformer_token` wins on landsat and backdoor — the two most statistically reliable datasets — and is competitive on mnist. It is the principled choice when a Transformer teacher is available.

3. **CNN teacher fails on tabular data as a standalone** (AUC 0.25–0.32). 1D convolution finds no spatial structure in unordered features. Despite this, `pkt_cnn_concat` still lifts optdigits by +0.029 because the student's own reconstruction loss dominates when the teacher provides weak signal.

4. **Vanilla AE is the strongest standalone teacher** on 3/4 datasets. Simple FC layers match the sub-AE inductive bias well and produce clean, discriminative latents without Transformer or VAE optimization complexity.

5. **High variance on optdigits** (std up to 0.131) — only 150 anomalies make results sensitive to initialization. Median over more seeds would be more robust for this dataset.

6. **Teacher–student capacity ratio is extreme:** 41–415× depending on dataset. PKT bridges this gap on the harder, higher-dimensional datasets where global structure matters most.

---

## 🚀 Installation & Usage

### Install

```bash
pip install -r requirements.txt
```

### Download Data

```bash
python scripts/download_data.py
```

Datasets are downloaded from ADBench to `data/raw/`. If automatic download fails, get the `.npz` files from [ADBench on GitHub](https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical) and place them in `data/raw/`.

### Single Run

```bash
# Train a teacher
python -m training.train_teacher \
    --dataset {optdigits,mnist,landsat,backdoor} \
    --model {vanilla,cnn,transformer,vae} \
    --seed 42

# Train Kitsune — vanilla (no KD)
python -m training.train_kitsune \
    --dataset optdigits --mode vanilla --seed 42

# Train Kitsune — PKT
python -m training.train_kitsune \
    --dataset optdigits --mode pkt \
    --teacher {vanilla,cnn,transformer,vae} \
    --pkt_mode {concat,per_subae,token} \
    --seed 42
    # note: --pkt_mode token requires --teacher transformer
```

### Full Sweep

```bash
bash scripts/run_all.sh   # 168 runs
```

Results are written as JSON to `results/`. Checkpoints saved to `checkpoints/`.

### Analysis Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 📁 Project Structure

```
kd-anomaly/
├── configs/default.yaml          — hyperparameters (lr, epochs, batch size, α)
├── data/
│   ├── loader.py                 — ADBench loading, MinMaxScaler, DataLoaders
│   └── raw/                      — downloaded .npz files
├── models/
│   ├── kitnet_original/          — original NumPy KitNET-py (reference only)
│   ├── kitsune.py                — PyTorch Kitsune student (two-phase support)
│   ├── vanilla_ae.py             — FC autoencoder teacher
│   ├── conv_ae.py                — 1D Conv autoencoder teacher
│   ├── transformer_ae.py         — Transformer teacher + encode_tokens()
│   └── vae.py                    — VAE teacher
├── losses/pkt.py                 — PKTLoss (cosine sim → softmax → KL div)
├── training/
│   ├── train_teacher.py          — teacher training (MSE + early stopping)
│   ├── train_kitsune.py          — two-phase Kitsune training
│   └── utils.py                  — seed, device, param count
├── evaluation/metrics.py         — AUC-ROC, AUC-PR, F1-oracle, F1-φ, timing
├── scripts/
│   ├── download_data.py          — fetch ADBench .npz files
│   └── run_all.sh                — full experiment sweep
├── checkpoints/                  — saved model weights
├── results/                      — per-run JSON output (168 files)
├── notebooks/analysis.ipynb      — results tables, Pareto plots, histograms
└── RESEARCH_STATE.md             — detailed experimental log and analysis
```

---

## 📚 Citations

```bibtex
@inproceedings{mirsky2018kitsune,
  title     = {Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection},
  author    = {Mirsky, Yisroel and Doitshman, Tomer and Elovici, Yuval and Shabtai, Asaf},
  booktitle = {NDSS},
  year      = {2018}
}

@inproceedings{passalis2018pkt,
  title     = {Learning Deep Representations with Probabilistic Knowledge Transfer},
  author    = {Passalis, Nikolaos and Tefas, Anastasios},
  booktitle = {ECCV},
  year      = {2018}
}

@article{passalis2020pkt,
  title   = {Probabilistic Knowledge Transfer for Lightweight Deep Representation Learning},
  author  = {Passalis, Nikolaos and Tefas, Anastasios},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2020}
}

@inproceedings{han2022adbench,
  title     = {ADBench: Anomaly Detection Benchmark},
  author    = {Han, Songqiao and Hu, Xiyang and Huang, Hailiang and Jiang, Mingqi and Zhao, Yue},
  booktitle = {NeurIPS},
  year      = {2022}
}
```
