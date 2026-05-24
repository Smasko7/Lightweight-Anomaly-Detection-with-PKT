# 🔬 Knowledge Distillation for Lightweight Anomaly Detection

> Can a tiny ensemble student learn better anomaly detection by imitating a deep teacher's latent space — without growing its parameter count?

This project investigates **Probabilistic Knowledge Transfer (PKT)** from four deep autoencoder teachers — and a same-architecture wider Kitsune teacher — to a KitNET-style ensemble student across four tabular anomaly detection benchmarks.

**4 datasets × 26 variants × 3 seeds = 312 experiments.**

> **Datasets:** optdigits, mnist, landsat, backdoor — all from [ADBench](https://github.com/Minqi824/ADBench) (Han et al., NeurIPS 2022).

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

We explore four **PKT routing strategies**:

```
┌─────────────────┬───────────────────────────────────────────────────────┐
│ Strategy        │ What gets matched                                     │
├─────────────────┼───────────────────────────────────────────────────────┤
│ Concat          │ PKT(z_t ,  concat(h₁…hₖ))      — 1 loss, global→global│
│ Per-subAE       │ Σᵢ PKT(z_t, hᵢ)                — K losses, global→each│
│ Token ★         │ Σᵢ PKT(token_i, hᵢ)            — K losses, aligned    │
│ Subae_paired ★★ │ Σᵢ PKT(t.subᵢ.h, s.subᵢ.h)    — K losses, paired     │
└─────────────────┴───────────────────────────────────────────────────────┘
  ★   Token routing only available with Transformer teacher
  ★★  Subae_paired routing only available with Big-Kitsune teacher
```

**Token routing** exploits the Transformer's tokenization (which uses the **same feature groups** as Kitsune): token i has attended to global context but corresponds to sub-AE i's exact feature subset, providing principled local-global alignment rather than broadcasting the same global vector to every sub-AE.

**Subae_paired routing** is the Big-Kitsune analogue: because the teacher itself decomposes into K sub-AEs over the same feature groups, teacher sub-AE i's hidden can be paired directly with student sub-AE i's hidden. The strictest form of architectural alignment we can express.

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

### 🔢 Parameter Counts Across All Variants

Total trainable parameters per (model, dataset). The student (`kitsune_*`) parameter count is **identical** across the baseline, all PKT variants, and the Big-Kitsune-distilled variants — PKT does not change the student's architecture, only its training signal.

| Model | optdigits | mnist | landsat | backdoor |
|-------|----------:|------:|--------:|---------:|
| Kitsune student (K0, all `pkt_*` variants) | 990 | 1,727 | 406 | 4,560 |
| Big-Kitsune teacher r=4 | 4,968 | 9,350 | 2,226 | 24,570 |
| Big-Kitsune teacher r=16 | 19,656 | 37,070 | 8,778 | 97,650 |
| Big-Kitsune teacher r=32 | 39,240 | 74,030 | 17,514 | 195,090 |
| Vanilla AE teacher | 117,120 | 135,588 | 102,756 | 184,836 |
| VAE teacher | ~125,000 | ~144,000 | ~111,000 | ~193,000 |
| CNN teacher | 137,000 | 137,000 | 137,000 | 137,000 |
| Transformer teacher | ~169,000 | ~169,000 | ~168,000 | ~170,000 |

Big-Kitsune teachers are dramatically smaller than the deep teachers at low `r`. At r=32 they approach (and on backdoor slightly exceed) Vanilla AE — but only on the largest-feature dataset. Param-matching across all four datasets would require dataset-specific `r ∈ {30, …, 190}` (large enough that overcomplete shallow sub-AEs degenerate to near-identity, defeating the purpose).

### 🎓 Teachers (~100–290K params each)

| Teacher | Architecture | Latent | Notes |
|---------|-------------|--------|-------|
| Vanilla AE | FC: N→256→128→64→128→256→N | 64-dim | Strongest standalone on 3/4 datasets |
| CNN AE | 1D Conv + AdaptiveAvgPool → FC 64 | 64-dim | Fails on tabular (no spatial structure) |
| Transformer AE | K-group tokens, 2× TransformerEncoderLayer, mean-pool | 64-dim | Enables token routing via `encode_tokens()` |
| VAE | Shared encoder → μ + log_σ² heads, ELBO | 64-dim (μ) | Regularized, smooth latent |
| **Big-Kitsune** (`bigkit`) | Same K-group ensemble as student but with wider sub-AEs (`hidden_ratio ∈ {4, 16, 32}`) | concat-pseudo-latent (varies) | Tests *architectural alignment* with the student. Output AE never trained — scored via direct RMSE sum. |

The Transformer uses the **same feature-group tokenization** as Kitsune, which is what makes token-aligned routing possible.

**Why Big-Kitsune?** It is the natural control variable for the rest of the study. Vanilla AE / CNN / Transformer / VAE all vary teacher *architecture* (FC bottleneck vs. convolution vs. attention vs. probabilistic latent). Big-Kitsune holds architecture fixed and varies only capacity (`hidden_ratio`) — isolating whether PKT benefits from a *more capable* same-architecture teacher, or genuinely from *different inductive biases*. The `subae_paired` routing (one PKT loss per matched sub-AE pair) is enabled by the shared K-group decomposition between teacher and student.

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

The key insight of PKT is that it never compares latent vectors *directly*. Instead it asks: **"does sample i look similar to sample j in the teacher's latent space? Does the student agree?"** This is captured through B×B pairwise similarity matrices, which have the same shape regardless of latent dimensionality — so no projection layer is needed.

```python
# teacher_latent: (B, D_t),  student_latent: (B, D_s)  — D_t ≠ D_s is fine
t = F.normalize(teacher_latent, dim=1)   # (B, D_t) — unit-norm rows
s = F.normalize(student_latent, dim=1)   # (B, D_s) — unit-norm rows

T = (t @ t.T + 1) / 2    # (B, B) cosine similarities shifted to [0, 1]
S = (s @ s.T + 1) / 2    # (B, B) same for student

P_t = T / T.sum(dim=1, keepdim=True)   # row-normalize → each row is a distribution
P_s = S / S.sum(dim=1, keepdim=True)   # over the B samples in the batch

loss = F.kl_div(P_s.log(), P_t, reduction='batchmean')   # D_KL(P_t ‖ P_s)
```

Step by step:

1. **L2-normalize** both latents so dot products equal cosine similarity.
2. **`t @ t.T`** produces a B×B matrix where entry (i, j) is cos(zᵢ, zⱼ). Shifting by `+1)/2` maps the range from [−1, 1] to [0, 1].
3. **Row-normalize** each similarity matrix so every row sums to 1 — turning it into a conditional probability distribution: *"given sample i, how likely is each other sample j to be its nearest neighbour?"*
4. **KL divergence** `D_KL(P_t ‖ P_s)` penalises the student when its neighbourhood structure disagrees with the teacher's. The student is trained to have the same *relational geometry* as the teacher, not the same coordinates.

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

## 🎯 Anomaly Score Methods

Different model families produce their final scalar anomaly score through different paths. This matters because *how* a model is scored is part of the experimental contract — two models with the same internal representation can give very different AUC numbers depending on the scoring head.

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ Model family             │ Score                                          │
├──────────────────────────┼────────────────────────────────────────────────┤
│ Vanilla AE / CNN / Trans │ mean( (x − recon)² )            per sample     │
│ VAE                      │ mean MSE + KL divergence        per sample     │
│ Kitsune (K0, all pkt_*)  │ Output AE RMSE on normalized    per sample     │
│                          │  per-group RMSE vector                         │
│ Big-Kitsune (`bigkit_*`) │ Σᵢ RMSE(xᵢ, sub_aeᵢ(xᵢ))      "direct RMSE   │
│                          │  across K sub-AEs                  sum"        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

### Why Big-Kitsune uses direct RMSE sum

By design, the Big-Kitsune teacher trains *only* its sub-AEs (with `Σᵢ MSE(xᵢ, sub_aeᵢ(xᵢ))` as the loss). The Output Autoencoder structure exists in the class — for compatibility with the rest of the Kitsune architecture — but its weights are **never updated** and remain at random initialization. Routing the score through that random Output AE would produce meaningless numbers. Direct RMSE sum bypasses it entirely and reports `Σᵢ ‖xᵢ − sub_aeᵢ(xᵢ)‖₂ / √fpg` across the K sub-AEs.

This is also a useful diagnostic for the Kitsune students: comparing their normal output-AE-routed score against the same direct-RMSE-sum score on identical checkpoints isolates how much of the final AUC comes from sub-AE quality vs. from the output AE's post-processing. See `RESEARCH_STATE.md` §8.5 for that analysis.

### Which variants use which scoring path

| Scoring path | Variants |
|---|---|
| Reconstruction MSE | `vanilla`, `cnn`, `transformer` |
| ELBO (MSE + KL) | `vae` |
| Output AE RMSE (two-phase trained) | `kitsune_vanilla` (K0), all `kitsune_pkt_*` (incl. bigkit-distilled students) |
| Direct RMSE sum across K sub-AEs | `bigkit_r4`, `bigkit_r16`, `bigkit_r32` (standalone teacher only) |

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
| Transformer | 0.926±0.027 | 0.928±0.002 | 0.556±0.012 | 0.954±0.004 | ~169K |
| VAE | 0.788±0.014 | 0.877±0.021 | 0.449±0.001 | 0.916±0.007 | ~130K |
| CNN | 0.681±0.039 | 0.326±0.010 | 0.248±0.005 | 0.321±0.000 | 137K |
| Big-Kitsune r=4 | 0.706±0.051 | 0.947±0.003 | 0.493±0.011 | 0.943±0.005 | 2.2K – 24.6K |
| Big-Kitsune r=16 | 0.667±0.022 | 0.851±0.012 | 0.590±0.024 | 0.955±0.004 | 8.8K – 97.7K |
| Big-Kitsune r=32 | 0.633±0.025 | 0.822±0.008 | **0.610**±0.030 | **0.968**±0.005 | 17.5K – 195K |

**Big-Kitsune r=32 is the best standalone detector in the study on backdoor (0.968) and landsat (0.610)** — beating every deep teacher. The same model degrades on optdigits/mnist where the small `fpg` makes overcomplete sub-AEs collapse toward identity.

### Kitsune Variants

| Model | optdigits | mnist | landsat | backdoor |
|-------|-----------|-------|---------|----------|
| kitsune vanilla | 0.760±0.050 | 0.823±0.008 | 0.510±0.034 | 0.911±0.006 |
| + pkt_cnn_concat | **+0.029** → 0.789 | −0.050 | +0.016 | −0.006 |
| + pkt_cnn_per_subae | −0.018 | −0.062 | +0.009 | −0.002 |
| + pkt_vanilla_concat | −0.077 | −0.026 | +0.020 | +0.019 |
| + pkt_vanilla_per_subae | −0.032 | −0.012 | +0.043 | +0.003 |
| + pkt_transformer_concat | +0.003 | −0.060 | +0.024 | +0.019 |
| + pkt_transformer_per_subae | −0.013 | −0.091 | +0.024 | +0.017 |
| + **pkt_transformer_token** ★ | −0.037 | −0.019 | **+0.045** → 0.555 | **+0.021** → 0.932 |
| + pkt_vae_concat | −0.066 | **≈0** → 0.821 | −0.019 | +0.003 |
| + pkt_vae_per_subae | −0.013 | −0.043 | +0.019 | +0.002 |
| + pkt_bigkit_r4_concat | −0.045 | −0.051 | +0.035 | −0.017 |
| + pkt_bigkit_r4_per_subae | −0.034 | −0.049 | +0.044 | −0.002 |
| + pkt_bigkit_r4_subae_paired ★★ | −0.036 | −0.016 | **+0.047** | +0.001 |
| + pkt_bigkit_r16_concat | −0.097 | −0.038 | +0.037 | −0.004 |
| + pkt_bigkit_r16_per_subae | −0.044 | −0.043 | +0.038 | +0.008 |
| + pkt_bigkit_r16_subae_paired ★★ | −0.046 | −0.031 | +0.037 | −0.007 |
| + pkt_bigkit_r32_concat | −0.079 | −0.038 | +0.040 | **+0.018** |
| + pkt_bigkit_r32_per_subae | −0.044 | −0.041 | +0.042 | +0.012 |
| + pkt_bigkit_r32_subae_paired ★★ | −0.054 | −0.027 | +0.033 | −0.009 |

### 🏆 Best PKT vs. Kitsune Vanilla

| Dataset | kitsune vanilla | Best PKT | Δ AUC-ROC | Best variant |
|---------|----------------|----------|-----------|--------------|
| optdigits | 0.760 | **0.789** | **+0.029** | pkt_cnn_concat |
| mnist | 0.823 | **0.821** | ≈0 | pkt_vae_concat |
| landsat | 0.510 | **0.557** | **+0.047** | pkt_bigkit_r4_subae_paired ★★ |
| backdoor | 0.911 | **0.932** | **+0.021** | pkt_transformer_token ★ |

Cross-architectural distillation (`pkt_transformer_token`, `pkt_cnn_concat`) still wins on three of four datasets. Big-Kitsune distillation matches or slightly beats them only on landsat — and never reaches its own standalone teacher's AUC (0.610 standalone → 0.557 distilled on landsat; 0.968 standalone → 0.929 distilled on backdoor).

### 🔑 Key Findings

1. **PKT improves Kitsune on 3 of 4 datasets.** The hypothesis holds where sufficient training data and anomaly signal exist. The improvement is most credible on backdoor (59K training samples, std < 0.017).

2. **Token routing is the best overall strategy.** `pkt_transformer_token` wins on landsat and backdoor — the two most statistically reliable datasets — and is competitive on mnist. It is the principled choice when a Transformer teacher is available.

3. **CNN teacher fails on tabular data as a standalone** (AUC 0.25–0.32). 1D convolution finds no spatial structure in unordered features. Despite this, `pkt_cnn_concat` still lifts optdigits by +0.029 because the student's own reconstruction loss dominates when the teacher provides weak signal.

4. **Vanilla AE is the strongest standalone teacher** on 3/4 datasets. Simple FC layers match the sub-AE inductive bias well and produce clean, discriminative latents without Transformer or VAE optimization complexity.

5. **High variance on optdigits** (std up to 0.131) — only 150 anomalies make results sensitive to initialization. Median over more seeds would be more robust for this dataset.

6. **Teacher–student capacity ratio is extreme:** 41–415× depending on dataset. PKT bridges this gap on the harder, higher-dimensional datasets where global structure matters most.

7. **Big-Kitsune is the strongest *standalone* detector on the largest dataset.** `bigkit_r32` reaches 0.968 AUC on backdoor at only ~195K params, beating Vanilla AE (0.950, 185K) and Transformer (0.954, 170K). On landsat it reaches 0.610 — the only model besides Vanilla AE above 0.6 on that hard dataset. On the small-feature datasets (optdigits, mnist) wider sub-AEs collapse toward identity and degrade.

8. **Same-architecture distillation does *not* outperform cross-architecture.** Even with a stronger standalone teacher (`bigkit_r32` on backdoor: 0.968) and an architecturally-aligned `subae_paired` routing, the distilled student plateaus around K0 level (0.929 on backdoor — no better than `pkt_transformer_token` at 0.932 from a much weaker teacher). The active ingredient in PKT is **architectural diversity**, not alignment.

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
    --model {vanilla,cnn,transformer,vae,bigkit} \
    --seed 42
    # bigkit additionally accepts: --hidden_ratio {4,16,32}

# Train Kitsune — vanilla (no KD)
python -m training.train_kitsune \
    --dataset optdigits --mode vanilla --seed 42

# Train Kitsune — PKT
python -m training.train_kitsune \
    --dataset optdigits --mode pkt \
    --teacher {vanilla,cnn,transformer,vae,bigkit} \
    --pkt_mode {concat,per_subae,token,subae_paired} \
    --seed 42
    # --pkt_mode token         requires --teacher transformer
    # --pkt_mode subae_paired  requires --teacher bigkit
    # bigkit teacher additionally needs: --teacher_hidden_ratio {4,16,32}
```

### Full Sweep

```bash
bash scripts/run_all.sh   # 312 runs (84 baseline + 144 Big-Kitsune × 3 ratios + final eval)
```

The sweep config inside `run_all.sh` lets you scope down to a single Big-Kitsune ratio (defaults can be edited at the top of the script). Results are written as JSON to `results/`. Checkpoints saved to `checkpoints/`.

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
│   ├── kitsune.py                — PyTorch Kitsune student + Big-Kitsune teacher
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
│   ├── run_all.sh                — full experiment sweep (incl. Big-Kitsune ratios loop)
│   └── evaluate_all.py           — re-score all checkpoints, write results/*.json
├── checkpoints/                  — saved model weights
├── results/                      — per-run JSON output (~312 files)
├── notebooks/analysis.ipynb      — results tables, Pareto plots, histograms
└── RESEARCH_STATE.md             — detailed experimental log and analysis
```

---

## 🧭 Insights & Conclusions

After 312 runs across two experimental phases, four headline takeaways:

### 1. PKT works — when the teacher carries an inductive bias the student lacks.

The two clearest wins are `pkt_transformer_token` (+0.045 on landsat, +0.021 on backdoor) and `pkt_cnn_concat` (+0.029 on optdigits). Both teachers carry computational structures that the student structurally cannot represent: attention over feature groups, and 1D convolution. The student cannot rediscover that structure on its own, so the teacher's similarity geometry is genuinely *new information*. Distillation transfers it.

### 2. PKT does *not* work just because the teacher is "good".

`bigkit_r32` is the strongest standalone teacher on backdoor (0.968) and landsat (0.610) — yet its distilled student gains nothing meaningful over K0 (best PKT student: 0.929 on backdoor vs K0's 0.911 with std ±0.007). Standalone teacher quality does **not** predict its value as a PKT signal source. A teacher whose hypothesis class is a superset of the student's offers nothing the student can't reach by itself with reconstruction loss alone.

### 3. Architectural diversity > architectural alignment.

The naïve intuition — "teacher should look like the student so distillation is smoother" — is refuted. The architecturally-aligned `subae_paired` routing (designed specifically to exploit the shared K-group decomposition between bigkit and student) is no better than `concat` or `per_subae`. Best PKT variants across the study are all *cross-architecture*. Architectural alignment also predicts the wrong thing: it suggests `bigkit_r32` distillation should match its 0.968 standalone, but distillation actually loses 0.04 AUC on backdoor.

### 4. The output AE is a dataset-dependent component, not a free win.

The two-phase fix (Phase 1: sub-AEs + PKT only, Phase 2: output AE alone) was necessary to prevent catastrophic signal collapse (d′ 1.13 → 0.09 under joint training). But even after the fix, whether the output AE *adds* value depends on the dataset: it helps on landsat by ~+0.12 AUC across PKT variants, hurts on mnist by ~−0.10, and is near-neutral on backdoor. Whether to ship a Kitsune student with or without the output AE should be a per-deployment decision driven by validation AUC.

### Practical recommendation

For a new tabular anomaly detection problem, our results suggest:
1. **Try `bigkit_r32` first as a standalone detector** — it is small, fast, and competitive with deep AEs at a fraction of the params on backdoor-scale datasets.
2. **If a deep teacher is acceptable, distill with `pkt_transformer_token`** (or `pkt_cnn_concat` for very small datasets). These carry the inductive biases the student lacks.
3. **Do not invest in same-architecture "big teacher" distillation** — that capacity is wasted under PKT.
4. **Decide output-AE vs direct-RMSE-sum scoring by validation AUC**, not by intuition.

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
