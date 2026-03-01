# Generative Models Landscape

Generative models learn to produce new data that resembles a training distribution. Formally, given observed data $x \sim p_{\text{data}}(x)$, the goal is to learn a model $p_\theta(x)$ that approximates $p_{\text{data}}$ well enough to draw realistic samples or evaluate likelihoods.

**Taxonomy.** Generative models split along one key axis:

- **Explicit density** -- the model defines $p_\theta(x)$ directly (tractable or approximate).
- **Implicit density** -- the model can *sample* from $p_\theta(x)$ without ever writing down the density.

This document surveys six major families, ending with a preview of diffusion models -- the focus of this repository.

---

## 1. Variational Autoencoders (VAEs)

### Core Idea

Introduce a latent variable $z$ and jointly learn:

- An **encoder** (recognition model) $q_\phi(z \mid x)$ that maps data to a distribution over latents.
- A **decoder** (generative model) $p_\theta(x \mid z)$ that reconstructs data from latents.

Because the true posterior $p_\theta(z \mid x)$ is intractable, we optimize a surrogate: the **Evidence Lower Bound (ELBO)**.

### Architecture

```
x  -->  Encoder  -->  mu, sigma  -->  z = mu + sigma * eps  -->  Decoder  -->  x_hat
             q_phi(z|x)            (reparameterization trick)       p_theta(x|z)
```

### Training

Maximize the ELBO with respect to both $\phi$ and $\theta$:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}\!\big[\log p_\theta(x|z)\big] - D_{\mathrm{KL}}\!\big(q_\phi(z|x) \,\|\, p(z)\big)$$

- **First term**: reconstruction quality (how well the decoder recovers $x$).
- **Second term**: regularization (keeps the approximate posterior close to the prior $p(z)$, typically $\mathcal{N}(0, I)$).

Gradients flow through the stochastic layer via the **reparameterization trick**: sample $\varepsilon \sim \mathcal{N}(0,I)$, set $z = \mu + \sigma \odot \varepsilon$.

### Strengths

- Stable, straightforward training (single loss, no adversarial dynamics).
- Principled probabilistic framework with a well-defined ELBO.
- Smooth, continuous latent space that supports interpolation and manipulation.

### Weaknesses

- Samples tend to be **blurry** because the decoder averages over posterior uncertainty.
- **Posterior collapse**: the model ignores the latent $z$ when the decoder is too powerful.
- Limited expressiveness of the approximate posterior $q_\phi(z|x)$ (typically diagonal Gaussian).

### Notable Variants

| Variant | Key Contribution |
|---|---|
| **beta-VAE** | Disentangled representations via $\beta > 1$ on the KL term |
| **VQ-VAE** | Discrete latent codes using vector quantization; avoids posterior collapse |
| **NVAE** | Deep hierarchical VAE with residual cells; competitive image quality |
| **Hierarchical VAEs** | Multiple stochastic layers for richer posteriors |

---

## 2. Generative Adversarial Networks (GANs)

### Core Idea

Two networks play a **minimax game**:

- A **generator** $G(z)$ maps noise $z \sim p(z)$ to fake data.
- A **discriminator** $D(x)$ tries to distinguish real data from generated data.

The generator learns to fool the discriminator; the discriminator learns to not be fooled.

### Architecture

```
z ~ N(0,I)  -->  Generator G(z)  -->  x_fake
                                         |
                                    Discriminator D  -->  real / fake
                                         |
x_real  ---------------------------------+
```

### Training

Alternating gradient updates on the value function:

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\!\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\!\big[\log(1 - D(G(z)))\big]$$

At the Nash equilibrium, $G$ produces samples indistinguishable from real data and $D$ outputs $\tfrac{1}{2}$ everywhere.

### Strengths

- Produces **sharp, high-fidelity** samples (no blurring from likelihood averaging).
- Fast single-pass inference (one forward pass through $G$).
- No explicit density required -- purely implicit.

### Weaknesses

- **Mode collapse**: the generator may cover only a subset of the data distribution.
- **Training instability**: balancing $G$ and $D$ is notoriously fragile.
- No density estimation -- cannot compute $p_\theta(x)$.
- Hard to evaluate -- FID, IS, and other metrics are imperfect proxies.

### Notable Variants

| Variant | Key Contribution |
|---|---|
| **DCGAN** | Convolutional architecture guidelines that stabilized GAN training |
| **WGAN / WGAN-GP** | Wasserstein distance loss; gradient penalty for Lipschitz constraint |
| **StyleGAN (1/2/3)** | Style-based generator with progressive detail; state-of-the-art faces |
| **BigGAN** | Large-scale class-conditional generation on ImageNet |
| **Progressive GAN** | Grow resolution during training for stable high-res synthesis |

---

## 3. Normalizing Flows

### Core Idea

Start with a simple base distribution $z \sim p_0(z)$ (e.g., Gaussian) and apply a chain of **invertible, differentiable** transformations $f = f_K \circ \cdots \circ f_1$ to obtain $x = f(z)$. The density of $x$ is computed exactly via the **change of variables** formula:

$$\log p_\theta(x) = \log p_0\!\big(f^{-1}(x)\big) + \sum_{k=1}^{K} \log \left|\det \frac{\partial f_k^{-1}}{\partial f_{k-1}^{-1}}\right|$$

### Architecture

```
z ~ N(0,I)  <-->  f_1  <-->  f_2  <-->  ...  <-->  f_K  <-->  x
                 (each f_k is invertible with tractable Jacobian)
```

Every layer must be: (1) invertible, and (2) have a cheaply computable log-determinant of its Jacobian.

### Training

Maximize the **exact** log-likelihood directly -- no bounds, no adversaries:

$$\max_\theta \; \mathbb{E}_{x \sim p_{\text{data}}}\!\big[\log p_\theta(x)\big]$$

### Strengths

- **Exact** log-likelihood computation (not a bound).
- **Exact** sampling and density evaluation in both directions.
- Invertibility enables direct latent-space manipulation.

### Weaknesses

- Strict **architectural constraints** -- every layer must be invertible with a tractable Jacobian.
- Jacobian determinant computation can be expensive ($O(d^3)$ in general; coupling layers reduce this).
- Often less expressive per parameter compared to unconstrained architectures.

### Notable Variants

| Variant | Key Contribution |
|---|---|
| **RealNVP** | Affine coupling layers with cheap triangular Jacobians |
| **Glow** | Invertible 1x1 convolutions; high-res face synthesis |
| **Neural Spline Flows** | Monotonic rational-quadratic splines for flexible coupling |
| **Continuous NFs (FFJORD)** | ODE-based flows; free-form Jacobian via Hutchinson trace estimator |

---

## 4. Autoregressive Models

### Core Idea

Factorize the joint distribution using the **chain rule of probability**:

$$p(x) = \prod_{i=1}^{d} p(x_i \mid x_1, \ldots, x_{i-1})$$

Each conditional $p(x_i \mid x_{<i})$ is parameterized by a neural network with **causal masking** so that prediction of $x_i$ only sees preceding elements.

### Architecture

```
x_1  -->  x_2  -->  x_3  -->  ...  -->  x_d
 |          |          |                   |
p(x_1)   p(x_2|x_1)  p(x_3|x_1,x_2)    p(x_d|x_{<d})

Implemented via masked convolutions (PixelCNN) or causal self-attention (Transformers).
```

### Training

**Teacher forcing**: at each position, the model predicts $x_i$ given the ground-truth prefix $x_{<i}$. The loss is the exact negative log-likelihood:

$$\mathcal{L} = -\sum_{i=1}^{d} \log p_\theta(x_i \mid x_{<i})$$

### Strengths

- **Exact** tractable likelihood -- no bounds or approximations.
- High-quality, detailed samples (especially for text and audio).
- Simple maximum-likelihood training; no adversarial dynamics or posterior issues.

### Weaknesses

- **Sequential sampling**: generating $x_i$ requires all previous $x_{<i}$, making sampling $O(d)$ serial steps.
- No learned latent representation (each token is conditioned on raw previous tokens).

### Notable Variants

| Variant | Key Contribution |
|---|---|
| **PixelCNN / PixelCNN++** | Masked convolutions for autoregressive image generation |
| **PixelRNN** | Row-LSTM and diagonal BiLSTM for spatial dependencies |
| **WaveNet** | Dilated causal convolutions for raw audio waveform generation |
| **GPT family** | Causal Transformer for text; scales to billions of parameters |
| **Image GPT** | GPT-style autoregressive model applied to image pixel sequences |

---

## 5. Energy-Based Models (EBMs)

### Core Idea

Define an **unnormalized** log-density (energy function) $E_\theta(x)$ such that:

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}, \qquad Z_\theta = \int \exp(-E_\theta(x))\, dx$$

The partition function $Z_\theta$ is generally intractable, so training and sampling require specialized techniques.

### Training

Since $\nabla_\theta \log Z_\theta$ is intractable, several strategies exist:

- **Contrastive Divergence (CD)**: approximate the gradient of $\log Z_\theta$ using short-run MCMC.
- **Score Matching**: bypass $Z_\theta$ entirely by matching $\nabla_x \log p_\theta(x)$ to $\nabla_x \log p_{\text{data}}(x)$.
- **Noise-Contrastive Estimation (NCE)**: train a classifier to distinguish data from a known noise distribution.

### Strengths

- **Flexible architecture**: $E_\theta(x)$ can be any neural network mapping $x \to \mathbb{R}$; no invertibility or autoregressive constraints.
- Can in principle model **arbitrary** distributions.
- Composable: energies of independent constraints can be summed.

### Weaknesses

- **Intractable partition function** makes likelihood evaluation impossible without approximation.
- **Slow sampling**: MCMC methods (Langevin dynamics, HMC) require many steps to mix.
- Training can be unstable, especially when MCMC chains are too short.

### Notable Variants

| Variant | Key Contribution |
|---|---|
| **Restricted Boltzmann Machines** | Bipartite structure enabling efficient block Gibbs sampling |
| **Deep Boltzmann Machines** | Multiple hidden layers for richer representations |
| **Deep EBMs (Du & Mordatch)** | Modern ConvNet energy functions trained with Langevin MCMC |
| **Score-Based Models** | Estimate $\nabla_x \log p(x)$ directly; bridge to diffusion models |

---

## 6. Diffusion Models (Preview)

### Core Idea

Diffusion models define a two-phase process:

1. **Forward process** (fixed): gradually corrupt data $x_0$ by adding Gaussian noise over $T$ steps, producing a sequence $x_0, x_1, \ldots, x_T$ where $x_T \approx \mathcal{N}(0, I)$.

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I)$$

2. **Reverse process** (learned): a neural network learns to **denoise** step by step, recovering $x_0$ from $x_T$.

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I)$$

### Why Diffusion Models Are Exciting

- **Sample quality** rivaling or surpassing GANs, without adversarial training.
- **Stable training** with a simple denoising objective -- no mode collapse, no balancing act.
- **Full mode coverage** -- likelihood-based training avoids the partial coverage problem of GANs.
- Naturally supports **conditional generation**, inpainting, super-resolution, and editing.

### Connection to Score Matching and SDEs

Diffusion models are deeply connected to:

- **Score matching**: the denoising network effectively learns the **score function** $\nabla_{x_t} \log p(x_t)$ at each noise level.
- **Stochastic Differential Equations (SDEs)**: the discrete forward/reverse processes generalize to continuous-time SDEs, unifying DDPM-style and score-based models under one framework (Song et al., 2021).

### The ELBO for Diffusion

The variational bound decomposes into $T$ denoising terms:

$$\log p_\theta(x_0) \geq \mathbb{E}_q\!\bigg[\underbrace{\log p_\theta(x_0 \mid x_1)}_{L_0\text{ (reconstruction)}} - \underbrace{D_{\mathrm{KL}}(q(x_T \mid x_0) \,\|\, p(x_T))}_{L_T\text{ (prior matching)}} - \sum_{t=2}^{T} \underbrace{D_{\mathrm{KL}}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}\text{ (denoising matching)}}\bigg]$$

Each $L_{t-1}$ term asks the model to match the tractable posterior $q(x_{t-1} \mid x_t, x_0)$ -- a Gaussian whose mean and variance we can compute in closed form. This is what we will implement step by step in this repository.

### What We Will Build

This repo walks through diffusion models from scratch:

1. The math behind the forward and reverse processes.
2. Deriving and simplifying the training loss.
3. Implementing DDPM in PyTorch.
4. Noise schedules, samplers, and practical tricks.
5. Conditional generation and guidance.

---

## Comparison Table

| Property | VAE | GAN | Normalizing Flow | Autoregressive | EBM | Diffusion |
|---|---|---|---|---|---|---|
| **Likelihood** | Approximate (ELBO) | None | Exact | Exact | Intractable | Approximate (ELBO) |
| **Sample Quality** | Moderate | High | Moderate | High | Moderate | High |
| **Training Stability** | Stable | Unstable | Stable | Stable | Unstable | Stable |
| **Sampling Speed** | Fast | Fast | Fast | Slow | Slow | Slow |
| **Latent Space** | Meaningful | Fixed (noise input) | Meaningful | None | None | Fixed (noise input) |
| **Mode Coverage** | Full | Partial | Full | Full | Full | Full |

---

## Key Takeaways

1. **No single model dominates** across all axes. VAEs offer stability and latent spaces but blur; GANs produce sharp images but collapse and are hard to train; autoregressive models give exact likelihoods but sample slowly.

2. **Diffusion models** achieve a compelling balance: stable likelihood-based training, high sample quality, and full mode coverage -- at the cost of slow iterative sampling (which is an active area of research: DDIM, distillation, consistency models).

3. **Connections run deep.** Score-based EBMs, continuous normalizing flows, and diffusion SDEs are different views of the same underlying mathematics. Understanding one illuminates the others.

This landscape sets the stage for everything that follows. Next, we dive into the math and code of diffusion models.
