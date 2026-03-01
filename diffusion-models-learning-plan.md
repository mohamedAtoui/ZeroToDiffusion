# 💥 ZeroToDiffusion

> From pure noise to deep understanding — a hands-on journey through the math, theory, and code behind diffusion models, with comparisons to what came before.

---

## 📁 Suggested GitHub Repo Structure

```
ZeroToDiffusion/
├── README.md                        # Overview, motivation, how to navigate
├── ROADMAP.md                       # This plan (track progress with checkboxes)
│
├── 01-foundations/
│   ├── probability-review.md        # Gaussian, Bayes, KL divergence, ELBO
│   ├── generative-models-landscape.md  # VAEs, GANs, Flows, Autoregressive
│   └── notebooks/
│       ├── gaussian_basics.ipynb
│       └── kl_divergence_demo.ipynb
│
├── 02-before-diffusion/
│   ├── vae/
│   │   ├── README.md                # Theory notes on VAEs
│   │   └── vae_mnist.py             # Simple VAE on MNIST
│   ├── gan/
│   │   ├── README.md                # Theory notes on GANs
│   │   └── dcgan_mnist.py           # Simple DCGAN on MNIST
│   └── comparison.md                # Side-by-side: strengths, weaknesses, samples
│
├── 03-diffusion-theory/
│   ├── forward-process.md           # Math of noise scheduling
│   ├── reverse-process.md           # Denoising, parameterization
│   ├── training-objective.md        # ELBO → simplified loss
│   ├── sampling.md                  # DDPM vs DDIM sampling
│   └── notebooks/
│       ├── noise_schedule_viz.ipynb  # Visualize forward process
│       └── math_walkthrough.ipynb   # Step-by-step derivations
│
├── 04-implementation/
│   ├── v1-simple-ddpm/
│   │   ├── README.md                # Design decisions, architecture notes
│   │   ├── model.py                 # U-Net with time embeddings
│   │   ├── diffusion.py             # Forward/reverse process
│   │   ├── train.py                 # Training loop
│   │   ├── sample.py                # Generate images
│   │   └── configs/                 # Hyperparameter configs
│   ├── v2-conditional/
│   │   ├── README.md
│   │   ├── model.py                 # + class conditioning
│   │   ├── cfg.py                   # Classifier-free guidance
│   │   └── train.py
│   └── v3-latent-diffusion/         # (Stretch goal)
│       └── README.md
│
├── 05-comparison/
│   ├── metrics.md                   # FID, IS, visual quality
│   ├── benchmark.py                 # Run all models, compute metrics
│   ├── results/
│   │   ├── vae_samples.png
│   │   ├── gan_samples.png
│   │   ├── ddpm_samples.png
│   │   └── comparison_table.md
│   └── analysis.md                  # What you learned, tradeoffs
│
├── 06-attention-and-beyond/
│   ├── attention-in-diffusion.md    # Role of self-attention in U-Net
│   ├── transformers-vs-unet.md      # DiT (Diffusion Transformers)
│   └── dit_exploration.ipynb        # Optionally explore DiT
│
├── readings/
│   ├── paper-notes/
│   │   ├── ddpm-2020.md             # Annotated notes on each paper
│   │   ├── ddim-2021.md
│   │   ├── improved-ddpm-2021.md
│   │   ├── cfg-2022.md
│   │   ├── latent-diffusion-2022.md
│   │   └── dit-2023.md
│   └── reading-list.md              # Full curated list with links
│
├── blog-posts/                      # (Optional) write-ups for each phase
│   ├── 01-why-diffusion.md
│   ├── 02-building-my-first-ddpm.md
│   └── 03-what-i-learned.md
│
└── requirements.txt
```

---

## 🗓️ Weekly Learning Plan (8–10 Weeks)

### Phase 1: Foundations (Week 1–2)

**Goal:** Build the math and context you need before touching diffusion.

#### Week 1 — Math & Generative Model Landscape

| Day | Task |
|-----|------|
| 1–2 | Review probability basics: Gaussians, Bayes' theorem, marginal/conditional distributions. Write `probability-review.md`. |
| 3   | Deep dive into **KL divergence** and **ELBO** (Evidence Lower Bound). Create a notebook visualizing KL divergence between two distributions. |
| 4–5 | Survey of generative models: read overviews of **VAEs**, **GANs**, **Normalizing Flows**, **Autoregressive models**. Write `generative-models-landscape.md`. |
| 6–7 | Start coding: implement a **simple VAE** on MNIST in PyTorch. Document what works and what doesn't (blurry samples, posterior collapse). |

**Key Resources:**
- Lilian Weng's blog: "What are Diffusion Models?" — best single overview
- Stanford CS236 (Deep Generative Models) lecture notes
- Bishop's Pattern Recognition Ch. 1–2 for probability review

#### Week 2 — GANs & Pre-Diffusion Baselines

| Day | Task |
|-----|------|
| 1–3 | Read GAN paper (Goodfellow 2014). Implement **DCGAN on MNIST/CIFAR-10**. Experience training instability firsthand. |
| 4–5 | Write `comparison.md`: compare VAE vs GAN samples, training stability, mode collapse, sample diversity. Save sample grids. |
| 6–7 | Read "Auto-Encoding Variational Bayes" (Kingma 2014) more carefully. Understand the reparameterization trick. Annotate in `paper-notes/`. |

**Deliverable:** `02-before-diffusion/` folder complete with working VAE + GAN code and comparison notes.

---

### Phase 2: Diffusion Theory Deep Dive (Week 3–4)

**Goal:** Understand the math of diffusion models thoroughly before implementing.

#### Week 3 — DDPM Paper & Forward Process

| Day | Task |
|-----|------|
| 1–2 | Read the **DDPM paper** (Ho et al., 2020) end to end. Don't worry about understanding everything yet. |
| 3–4 | Work through the **forward process** math: noise schedule (β_t), cumulative products (ᾱ_t), closed-form q(x_t\|x_0). Write `forward-process.md`. Build a notebook that visualizes an image being progressively noised. |
| 5–7 | Derive the **reverse process**: p_θ(x_{t-1}\|x_t), the mean/variance parameterization. Write `reverse-process.md`. |

#### Week 4 — Training Objective & Sampling

| Day | Task |
|-----|------|
| 1–3 | Derive the **ELBO for diffusion**, understand how it simplifies to the MSE noise-prediction loss. Write `training-objective.md`. This is the hardest part — take your time. |
| 4–5 | Read the **DDIM paper** (Song et al., 2021). Understand deterministic sampling and how fewer steps work. Write `sampling.md`. |
| 6–7 | Create `math_walkthrough.ipynb`: a clean notebook walking through all key derivations with LaTeX and code verification. |

**Key Papers to Annotate:**
1. "Denoising Diffusion Probabilistic Models" — Ho et al., 2020
2. "Denoising Diffusion Implicit Models" — Song et al., 2021
3. "Improved Denoising Diffusion Probabilistic Models" — Nichol & Dhariwal, 2021

**Deliverable:** `03-diffusion-theory/` complete. You should be able to explain the full pipeline on a whiteboard.

---

### Phase 3: Build Your Own DDPM (Week 5–7)

**Goal:** Implement a working diffusion model from scratch in PyTorch.

#### Week 5 — U-Net Architecture + Diffusion Scaffold

| Day | Task |
|-----|------|
| 1–3 | Build the **U-Net** with sinusoidal time embeddings, residual blocks, and self-attention at lower resolutions. This is your `model.py`. |
| 4–5 | Implement `diffusion.py`: forward process (add noise), reverse process (denoise), noise schedule. |
| 6–7 | Write `train.py`: training loop on MNIST or CIFAR-10 (start with 32×32 or 64×64). Get it running even if results are bad. |

#### Week 6 — Training, Debugging, First Samples

| Day | Task |
|-----|------|
| 1–3 | Train on CIFAR-10 or a simple dataset (e.g., CelebA 64×64). Debug: check loss curves, visualize intermediate denoising steps. |
| 4–5 | Implement `sample.py` with both DDPM (1000 steps) and DDIM (50–100 steps) sampling. Compare speed and quality. |
| 6–7 | Save sample grids at different training checkpoints. Document everything in `v1-simple-ddpm/README.md`. |

#### Week 7 — Conditional Generation & CFG

| Day | Task |
|-----|------|
| 1–3 | Add **class conditioning** to your U-Net (embed class label, add to time embedding). |
| 4–5 | Implement **classifier-free guidance** (CFG). Train with random label dropout. |
| 6–7 | Generate class-conditional samples. Experiment with guidance scale. Document in `v2-conditional/`. |

**Deliverable:** Working unconditional + conditional DDPM with saved samples and training logs.

---

### Phase 4: Compare & Analyze (Week 8)

**Goal:** Rigorous comparison across all models you've built.

| Day | Task |
|-----|------|
| 1–2 | Implement or use a library for **FID score** (Fréchet Inception Distance). Run on all your models: VAE, GAN, DDPM. |
| 3–4 | Create visual comparison grids. Measure: sample quality, diversity, training time, inference time, training stability. |
| 5–6 | Write `analysis.md`: your honest findings. What surprised you? Where does each model shine? |
| 7   | Build `comparison_table.md` with all metrics side by side. |

**Comparison Axes:**
- **Quality** (FID, visual inspection)
- **Diversity** (mode coverage)
- **Training stability** (loss curves, failure modes)
- **Speed** (training time, sampling time)
- **Controllability** (conditional generation quality)

---

### Phase 5: Attention & Modern Directions (Week 9–10)

**Goal:** Connect diffusion to the attention/transformer revolution.

#### Week 9 — Attention in Diffusion

| Day | Task |
|-----|------|
| 1–3 | Study the role of **self-attention layers** in your U-Net. Ablation: remove attention, compare results. Write `attention-in-diffusion.md`. |
| 4–7 | Read the **DiT paper** (Peebles & Xie, 2023) — Diffusion Transformers. Understand how transformers replace U-Nets entirely. Write `transformers-vs-unet.md`. |

#### Week 10 — Wrap Up & Polish

| Day | Task |
|-----|------|
| 1–3 | (Stretch) Explore **latent diffusion** concepts: encode to latent space with a pretrained VAE, run diffusion there. Even a toy version teaches a lot. |
| 4–5 | Write blog posts / final README. Make the repo navigable for others. |
| 6–7 | Polish code, add docstrings, clean notebooks. Share! |

---

## 📚 Curated Reading List

### Must-Read Papers (in order)
1. **VAE** — "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014)
2. **GAN** — "Generative Adversarial Networks" (Goodfellow et al., 2014)
3. **DDPM** — "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) ⭐
4. **DDIM** — "Denoising Diffusion Implicit Models" (Song et al., 2021)
5. **Improved DDPM** — (Nichol & Dhariwal, 2021)
6. **Classifier-Free Guidance** — (Ho & Salimans, 2022)
7. **Latent Diffusion / Stable Diffusion** — (Rombach et al., 2022)
8. **DiT** — "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)

### Best Blog Posts & Tutorials
- Lilian Weng: "What are Diffusion Models?" — lilianweng.github.io
- Hugging Face Diffusion Models Course — free, hands-on
- Calvin Luo: "Understanding Diffusion Models: A Unified Perspective" (monograph)
- Outlier YouTube channel: DDPM visual explainer
- AssemblyAI: "Diffusion Models from Scratch in PyTorch"

### Video Courses
- Stanford CS236: Deep Generative Models (free lectures)
- Hugging Face Diffusion Course (free, with code)

---

## 💡 Tips for the Journey

1. **Don't skip the math.** The derivations in weeks 3–4 are the hardest part but give you real understanding. Everything else flows from there.

2. **Start small.** Train on MNIST first (fast iteration), then move to CIFAR-10 or CelebA. Don't jump to 256×256 images until your pipeline works.

3. **Document as you go.** Write your theory notes and READMEs *while* you're learning, not after. Future-you will thank present-you.

4. **Git commit often.** Commit at every milestone: "VAE working on MNIST", "forward process visualized", "first DDPM samples". This tells a story.

5. **Compare honestly.** Your GAN might produce sharper images than your DDPM on MNIST. That's fine — document *why* and what changes at scale.

6. **Use Weights & Biases or TensorBoard.** Log everything: loss curves, sample grids at intervals, hyperparameters. This makes your repo much more impressive.

---

## ✅ Progress Tracker

Use this in your `ROADMAP.md`:

```markdown
- [ ] Phase 1: Foundations
  - [ ] Probability review notes
  - [ ] KL divergence notebook
  - [ ] Generative models landscape doc
  - [ ] VAE implementation (MNIST)
  - [ ] GAN implementation (MNIST)
  - [ ] Comparison document
- [ ] Phase 2: Diffusion Theory
  - [ ] DDPM paper annotated
  - [ ] Forward process notes + notebook
  - [ ] Reverse process notes
  - [ ] Training objective derivation
  - [ ] DDIM paper annotated
  - [ ] Sampling notes
  - [ ] Math walkthrough notebook
- [ ] Phase 3: Implementation
  - [ ] U-Net architecture
  - [ ] Diffusion process code
  - [ ] Training on CIFAR-10
  - [ ] DDPM + DDIM sampling
  - [ ] Class-conditional model
  - [ ] Classifier-free guidance
- [ ] Phase 4: Comparison
  - [ ] FID scores computed
  - [ ] Visual comparison grids
  - [ ] Analysis write-up
- [ ] Phase 5: Attention & Beyond
  - [ ] Attention ablation study
  - [ ] DiT paper notes
  - [ ] Final polish & blog posts
```

---

*From zero to diffusion — let's go! 💥*
