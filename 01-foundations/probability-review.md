# Probability and Information Theory Review

A rigorous but accessible reference covering the mathematical foundations needed for understanding diffusion models and variational inference.

---

## 1. Probability Distributions

### Discrete vs Continuous

A **probability mass function (PMF)** describes the probability of a discrete random variable $X$ taking each possible value:

$$P(X = x) = p(x), \quad \sum_{x} p(x) = 1, \quad p(x) \geq 0$$

A **probability density function (PDF)** describes the density of a continuous random variable. Note that $f(x)$ is *not* a probability itself --- it can exceed 1 --- but it integrates to 1:

$$f(x) \geq 0, \quad \int_{-\infty}^{\infty} f(x) \, dx = 1$$

The probability that $X$ falls in an interval is:

$$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$

### Cumulative Distribution Function (CDF)

The **CDF** $F(x) = P(X \leq x)$ works for both discrete and continuous variables:

- Discrete: $F(x) = \sum_{x_i \leq x} p(x_i)$
- Continuous: $F(x) = \int_{-\infty}^{x} f(t) \, dt$

The CDF is always non-decreasing, with $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$. For continuous distributions, $f(x) = F'(x)$.

### Expectation and Variance

The **expectation** (mean) of a random variable is its probability-weighted average:

$$\mathbb{E}[X] = \begin{cases} \sum_x x \, p(x) & \text{(discrete)} \\ \int_{-\infty}^{\infty} x \, f(x) \, dx & \text{(continuous)} \end{cases}$$

Key properties of expectation (linearity):

$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

This holds regardless of whether $X$ and $Y$ are independent.

The **variance** measures spread around the mean:

$$\text{Var}(X) = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

Key properties of variance:

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

If $X$ and $Y$ are **independent**: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.

### Common Distributions

**Bernoulli** --- a single binary trial with success probability $p$:

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$
$$\mathbb{E}[X] = p, \quad \text{Var}(X) = p(1-p)$$

**Categorical** --- generalization of Bernoulli to $K$ outcomes with probabilities $p_1, \ldots, p_K$:

$$P(X = k) = p_k, \quad \sum_{k=1}^{K} p_k = 1$$

This is used in discrete diffusion models and classifier guidance.

**Uniform** --- constant density over an interval $[a, b]$:

$$f(x) = \frac{1}{b - a} \quad \text{for } x \in [a, b]$$
$$\mathbb{E}[X] = \frac{a+b}{2}, \quad \text{Var}(X) = \frac{(b-a)^2}{12}$$

**Gaussian (Normal)** --- the most important distribution for diffusion models:

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

We write $X \sim \mathcal{N}(\mu, \sigma^2)$. The Gaussian appears everywhere in diffusion models because the forward process adds Gaussian noise at each step.

---

## 2. The Gaussian Distribution

### Univariate Gaussian

The PDF of a Gaussian with mean $\mu$ and variance $\sigma^2$ is:

$$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Properties:**

- $\mathbb{E}[X] = \mu$
- $\text{Var}(X) = \sigma^2$
- The distribution is symmetric about $\mu$
- Completely determined by its first two moments

**The 68-95-99.7 Rule** --- for a Gaussian distribution:

- $P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 0.683$ (about 68%)
- $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 0.954$ (about 95%)
- $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 0.997$ (about 99.7%)

**Standard Normal** --- when $\mu = 0$ and $\sigma^2 = 1$, we get $\mathcal{N}(0, 1)$. Any Gaussian can be standardized:

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

This is the **reparameterization trick** in its simplest form: we can write $X = \mu + \sigma Z$ where $Z \sim \mathcal{N}(0,1)$. This is critical for VAEs and diffusion models because it makes sampling differentiable with respect to $\mu$ and $\sigma$.

### Multivariate Gaussian

For a $d$-dimensional random vector $\mathbf{x} \in \mathbb{R}^d$ with mean vector $\boldsymbol{\mu} \in \mathbb{R}^d$ and covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$:

$$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where $|\boldsymbol{\Sigma}|$ is the determinant of the covariance matrix.

**Geometric interpretation:** The covariance matrix $\boldsymbol{\Sigma}$ determines the shape and orientation of the probability contours (ellipsoids of constant density). The eigenvectors of $\boldsymbol{\Sigma}$ give the principal axes of the ellipsoid, and the eigenvalues give the variance along each axis. The term $(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$ is the **Mahalanobis distance** --- it measures how far $\mathbf{x}$ is from $\boldsymbol{\mu}$, accounting for the shape of the distribution.

**Special case --- diagonal covariance:** When $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$, the dimensions are independent and the joint density factors:

$$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \prod_{i=1}^{d} \mathcal{N}(x_i; \mu_i, \sigma_i^2)$$

**Special case --- isotropic (spherical) covariance:** When $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$, all dimensions have the same variance and the contours are spheres. Diffusion models typically use isotropic Gaussian noise: $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

### Marginal Distributions

Given a joint Gaussian over partitioned variables $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2)^\top$:

$$\begin{pmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix} \right)$$

The **marginal distributions** are obtained by simply reading off the relevant sub-blocks:

$$\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11}), \quad \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_{22})$$

Intuitively, to "integrate out" $\mathbf{x}_2$, we just drop $\mathbf{x}_2$ and keep the corresponding mean and covariance block for $\mathbf{x}_1$. This is a remarkable property --- marginalization of Gaussians is trivial.

### Conditional Distributions

The **conditional distribution** of $\mathbf{x}_1$ given $\mathbf{x}_2 = \mathbf{a}$ is also Gaussian:

$$\mathbf{x}_1 | \mathbf{x}_2 = \mathbf{a} \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$$

where:

$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} (\mathbf{a} - \boldsymbol{\mu}_2)$$
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21}$$

**Intuition:** Observing $\mathbf{x}_2$ shifts the mean of $\mathbf{x}_1$ by an amount proportional to how far $\mathbf{x}_2$ deviates from its own mean, scaled by the correlation. The conditional covariance is always smaller than (or equal to) the marginal covariance --- observing $\mathbf{x}_2$ can only reduce uncertainty about $\mathbf{x}_1$.

This formula is directly used in deriving the posterior $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ in diffusion models.

### Sum of Independent Gaussians

If $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are **independent**, then:

$$X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \, \sigma_1^2 + \sigma_2^2)$$

More generally, for constants $a, b$:

$$aX + bY \sim \mathcal{N}(a\mu_1 + b\mu_2, \, a^2\sigma_1^2 + b^2\sigma_2^2)$$

**Why this is critical for diffusion:** The forward process applies repeated Gaussian noise steps. Because the sum of Gaussians is Gaussian, we can collapse $T$ sequential noise steps into a single step. Starting from $\mathbf{x}_0$:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0, \, (1 - \bar{\alpha}_t) \mathbf{I})$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Without the additivity property of Gaussians, we could not derive this closed-form expression, and training diffusion models would require simulating all $T$ steps sequentially.

---

## 3. Bayes' Theorem

### The Formula

For random variables $\mathbf{x}$ (observed data) and $\mathbf{z}$ (latent/hidden variables):

$$p(\mathbf{z} | \mathbf{x}) = \frac{p(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z})}{p(\mathbf{x})}$$

Each term has a name and a role:

| Term | Name | Role |
|---|---|---|
| $p(\mathbf{z} \| \mathbf{x})$ | **Posterior** | What we believe about $\mathbf{z}$ after observing $\mathbf{x}$ |
| $p(\mathbf{x} \| \mathbf{z})$ | **Likelihood** | How probable $\mathbf{x}$ is if $\mathbf{z}$ were the true latent |
| $p(\mathbf{z})$ | **Prior** | What we believe about $\mathbf{z}$ before seeing any data |
| $p(\mathbf{x})$ | **Evidence** (marginal likelihood) | Total probability of $\mathbf{x}$ across all possible $\mathbf{z}$ |

### Connection to Generative Models

In generative modeling, we assume data $\mathbf{x}$ (images, text, audio) is generated by some latent process involving unobserved variables $\mathbf{z}$:

1. Sample a latent code: $\mathbf{z} \sim p(\mathbf{z})$ (e.g., Gaussian prior)
2. Generate data from the code: $\mathbf{x} \sim p(\mathbf{x} | \mathbf{z})$ (e.g., neural network decoder)

After observing $\mathbf{x}$, we want to infer $\mathbf{z}$ --- this requires the posterior $p(\mathbf{z} | \mathbf{x})$.

### The Intractability Problem

The evidence $p(\mathbf{x})$ is computed by marginalizing over all possible latent variables:

$$p(\mathbf{x}) = \int p(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}$$

This integral is almost always **intractable** for interesting models because:

- The latent space is high-dimensional (often hundreds or thousands of dimensions)
- The likelihood $p(\mathbf{x}|\mathbf{z})$ is a complex nonlinear function (a neural network)
- There is no closed-form solution, and naive Monte Carlo estimation has extremely high variance

Since $p(\mathbf{x})$ appears in the denominator of Bayes' theorem, computing the exact posterior $p(\mathbf{z} | \mathbf{x})$ is also intractable.

**This is the central problem of latent variable modeling.** It motivates two major families of approximate methods:

- **Variational inference** --- approximate $p(\mathbf{z}|\mathbf{x})$ with a tractable distribution $q(\mathbf{z}|\mathbf{x})$ (used in VAEs)
- **MCMC sampling** --- draw samples from $p(\mathbf{z}|\mathbf{x})$ without computing it explicitly (used in score-based models)

Diffusion models cleverly sidestep some of these issues by using a fixed forward process $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ rather than learning the approximate posterior, but the ELBO derivation still underpins their training objective.

---

## 4. KL Divergence

### Definition

The **Kullback-Leibler divergence** from distribution $q$ to distribution $p$ measures how much $q$ differs from $p$:

$$D_{\text{KL}}(q \| p) = \mathbb{E}_{\mathbf{x} \sim q}\left[\log \frac{q(\mathbf{x})}{p(\mathbf{x})}\right] = \int q(\mathbf{x}) \log \frac{q(\mathbf{x})}{p(\mathbf{x})} \, d\mathbf{x}$$

**Intuition:** KL divergence measures the expected "surprise" from using $p$ to model data that actually comes from $q$. It quantifies the information lost when $p$ is used to approximate $q$.

### Properties

1. **Non-negativity (Gibbs' inequality):** $D_{\text{KL}}(q \| p) \geq 0$ for all $q, p$.

   *Proof sketch:* By Jensen's inequality applied to the concave function $\log$:
   $$-D_{\text{KL}}(q \| p) = \mathbb{E}_q\left[\log \frac{p(\mathbf{x})}{q(\mathbf{x})}\right] \leq \log \mathbb{E}_q\left[\frac{p(\mathbf{x})}{q(\mathbf{x})}\right] = \log \int p(\mathbf{x}) \, d\mathbf{x} = \log 1 = 0$$

2. **Zero if and only if $q = p$:** $D_{\text{KL}}(q \| p) = 0 \iff q(\mathbf{x}) = p(\mathbf{x})$ almost everywhere.

3. **NOT symmetric:** $D_{\text{KL}}(q \| p) \neq D_{\text{KL}}(p \| q)$ in general. This is why KL divergence is not a true "distance."

### Forward KL vs Reverse KL

The asymmetry of KL divergence gives rise to two distinct optimization problems with very different behavior:

**Forward KL: $D_{\text{KL}}(p \| q)$ --- "mode-covering" / "mean-seeking"**

$$D_{\text{KL}}(p \| q) = \mathbb{E}_{\mathbf{x} \sim p}\left[\log \frac{p(\mathbf{x})}{q(\mathbf{x})}\right]$$

Minimizing forward KL over $q$ forces $q(\mathbf{x}) > 0$ wherever $p(\mathbf{x}) > 0$ (otherwise the KL blows up to $+\infty$). This means $q$ must *cover all modes* of $p$, even if it puts mass in regions where $p$ is small. This leads to **over-dispersed** approximations.

- Equivalent to **maximum likelihood estimation**: if $p$ is the empirical data distribution, minimizing $D_{\text{KL}}(p \| q_\theta)$ over parameters $\theta$ is the same as maximizing $\mathbb{E}_{\mathbf{x} \sim p}[\log q_\theta(\mathbf{x})]$.
- Used when we can sample from $p$ (the data distribution) but want to fit $q_\theta$.

**Reverse KL: $D_{\text{KL}}(q \| p)$ --- "mode-seeking" / "zero-avoiding"**

$$D_{\text{KL}}(q \| p) = \mathbb{E}_{\mathbf{x} \sim q}\left[\log \frac{q(\mathbf{x})}{p(\mathbf{x})}\right]$$

Minimizing reverse KL forces $q(\mathbf{x}) \to 0$ wherever $p(\mathbf{x}) \approx 0$ (otherwise $\log q/p \to \infty$). This means $q$ avoids regions where $p$ has no mass and tends to **lock onto a single mode** of $p$. This leads to **under-dispersed** approximations.

- Used in **variational inference**: we minimize $D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$ to approximate the true posterior.
- We can evaluate $\log q$ and $\log p$ (up to a constant) but cannot sample from $p(\mathbf{z}|\mathbf{x})$ directly.

### KL Divergence Between Two Gaussians (Derivation)

Let $q = \mathcal{N}(\mu_1, \sigma_1^2)$ and $p = \mathcal{N}(\mu_2, \sigma_2^2)$. We want to compute $D_{\text{KL}}(q \| p)$.

**Step 1:** Write out the definition.

$$D_{\text{KL}}(q \| p) = \mathbb{E}_{x \sim q}\left[\log q(x) - \log p(x)\right]$$

**Step 2:** Expand the log-densities.

$$\log q(x) = -\frac{1}{2}\log(2\pi\sigma_1^2) - \frac{(x - \mu_1)^2}{2\sigma_1^2}$$

$$\log p(x) = -\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{(x - \mu_2)^2}{2\sigma_2^2}$$

**Step 3:** Take the difference.

$$\log q(x) - \log p(x) = \frac{1}{2}\log\frac{\sigma_2^2}{\sigma_1^2} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}$$

**Step 4:** Take expectation under $q$. We use $\mathbb{E}_q[x] = \mu_1$ and $\mathbb{E}_q[(x - \mu_1)^2] = \sigma_1^2$.

For the third term, expand $(x - \mu_2)^2 = (x - \mu_1 + \mu_1 - \mu_2)^2$:

$$\mathbb{E}_q[(x - \mu_2)^2] = \mathbb{E}_q[(x - \mu_1)^2] + 2(\mu_1 - \mu_2)\mathbb{E}_q[x - \mu_1] + (\mu_1 - \mu_2)^2 = \sigma_1^2 + (\mu_1 - \mu_2)^2$$

**Step 5:** Combine.

$$D_{\text{KL}}(q \| p) = \frac{1}{2}\log\frac{\sigma_2^2}{\sigma_1^2} - \frac{\sigma_1^2}{2\sigma_1^2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2}$$

$$\boxed{D_{\text{KL}}\big(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}}$$

**Multivariate generalization** for $q = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and $p = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$ in $d$ dimensions:

$$D_{\text{KL}}(q \| p) = \frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} - d + \text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right]$$

**Important special case:** KL from $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ to the standard normal $\mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$D_{\text{KL}}\big(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \| \mathcal{N}(\mathbf{0}, \mathbf{I})\big) = \frac{1}{2}\left[-\log|\boldsymbol{\Sigma}| - d + \text{tr}(\boldsymbol{\Sigma}) + \boldsymbol{\mu}^\top\boldsymbol{\mu}\right]$$

This is the regularization term in the VAE loss. With diagonal $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$:

$$= \frac{1}{2}\sum_{i=1}^{d}\left[-\log\sigma_i^2 - 1 + \sigma_i^2 + \mu_i^2\right]$$

### Connection to Maximum Likelihood

Suppose $p_{\text{data}}$ is the true data distribution and $p_\theta$ is our model. Then:

$$D_{\text{KL}}(p_{\text{data}} \| p_\theta) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\text{data}}(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_\theta(\mathbf{x})]$$

The first term is the negative entropy of $p_{\text{data}}$, which is a constant with respect to $\theta$. Therefore:

$$\arg\min_\theta D_{\text{KL}}(p_{\text{data}} \| p_\theta) = \arg\max_\theta \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_\theta(\mathbf{x})]$$

**Minimizing forward KL is equivalent to maximum likelihood estimation.** This is the theoretical justification for training generative models by maximizing the log-likelihood of the data.

---

## 5. Evidence Lower Bound (ELBO)

The ELBO is the central objective for training latent variable models, including VAEs and diffusion models. It provides a tractable surrogate for the intractable log-likelihood $\log p(\mathbf{x})$.

### Derivation from First Principles

We want to maximize $\log p(\mathbf{x})$ but cannot compute it directly. Introduce an approximate posterior $q(\mathbf{z}|\mathbf{x})$ and write:

**Step 1:** Start with the log-evidence and introduce $q$.

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$$

Multiply and divide by $q(\mathbf{z}|\mathbf{x})$:

$$= \log \int q(\mathbf{z}|\mathbf{x}) \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \, d\mathbf{z}$$

$$= \log \, \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]$$

**Step 2:** Apply Jensen's inequality. Since $\log$ is concave:

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]$$

The right-hand side is the **ELBO** (Evidence Lower BOund):

$$\text{ELBO}(q) = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]$$

### The Exact Relationship: ELBO + KL = log p(x)

We can derive the exact relationship (not just an inequality) by a different route.

**Step 1:** Write $\log p(\mathbf{x})$ as an expectation under $q$ (it does not depend on $\mathbf{z}$):

$$\log p(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x})]$$

**Step 2:** Add and subtract $\log q(\mathbf{z}|\mathbf{x})$ and $\log p(\mathbf{z}|\mathbf{x})$:

$$= \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right] + \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})}\right]$$

We used $\log p(\mathbf{x}) = \log p(\mathbf{x}, \mathbf{z}) - \log p(\mathbf{z}|\mathbf{x})$ (from the chain rule of probability).

**Step 3:** Recognize the two terms:

$$\boxed{\log p(\mathbf{x}) = \underbrace{\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]}_{\text{ELBO}} + \underbrace{D_{\text{KL}}\big(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x})\big)}_{\geq \, 0}}$$

Since $D_{\text{KL}} \geq 0$, we immediately get $\log p(\mathbf{x}) \geq \text{ELBO}$, confirming it is a lower bound.

**Key insight:** The gap between the ELBO and the true log-likelihood is exactly $D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$. The better $q$ approximates the true posterior $p(\mathbf{z}|\mathbf{x})$, the tighter the bound becomes. When $q = p(\mathbf{z}|\mathbf{x})$ exactly, the KL is zero and the ELBO equals the log-evidence.

### Decomposition 1: Reconstruction Minus Regularization

Factor the joint as $p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x}|\mathbf{z})p(\mathbf{z})$:

$$\text{ELBO} = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]$$

$$= \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] + \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]$$

$$\boxed{\text{ELBO} = \underbrace{\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction term}} - \underbrace{D_{\text{KL}}\big(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})\big)}_{\text{Regularization term}}}$$

**Interpretation:**

- **Reconstruction term** $\mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})]$: Measures how well we can reconstruct $\mathbf{x}$ from latent codes $\mathbf{z}$ sampled from $q$. Encourages the model to encode useful information.

- **Regularization term** $D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$: Penalizes $q(\mathbf{z}|\mathbf{x})$ for deviating from the prior $p(\mathbf{z})$. Encourages the latent space to be structured (close to the prior).

The ELBO elegantly balances these two objectives: encode enough to reconstruct, but not so specifically that the latent space becomes unstructured.

### Decomposition 2: Expected Joint plus Entropy

Alternatively, split the log-ratio differently:

$$\text{ELBO} = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log p(\mathbf{x}, \mathbf{z})\right] - \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log q(\mathbf{z}|\mathbf{x})\right]$$

$$\boxed{\text{ELBO} = \underbrace{\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}, \mathbf{z})]}_{\text{Expected complete-data log-likelihood}} + \underbrace{\mathcal{H}[q(\mathbf{z}|\mathbf{x})]}_{\text{Entropy of } q}}$$

where $\mathcal{H}[q] = -\mathbb{E}_q[\log q]$ is the entropy.

**Interpretation:** Maximize the expected log-probability of the data and latent variables under $q$, while keeping $q$ as spread out (high-entropy) as possible. The entropy term prevents $q$ from collapsing to a point mass.

### Connection to VAEs

In a **Variational Autoencoder (VAE)**:

- $q_\phi(\mathbf{z}|\mathbf{x})$: The **encoder** --- a neural network that maps data $\mathbf{x}$ to an approximate posterior over latents $\mathbf{z}$, typically $\mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$
- $p_\theta(\mathbf{x}|\mathbf{z})$: The **decoder** --- a neural network that maps latent $\mathbf{z}$ to a distribution over data $\mathbf{x}$
- $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$: The **prior** over latents

The VAE is trained by maximizing the ELBO with respect to both $\phi$ (encoder) and $\theta$ (decoder):

$$\max_{\phi, \theta} \, \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

The KL term has a closed-form solution (see Section 4), and the reconstruction term is estimated via Monte Carlo sampling using the reparameterization trick.

### Connection to Diffusion Models

In a **diffusion model**, the latent variables are the noisy intermediate states $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$. The ELBO for diffusion decomposes into $T$ terms:

$$\text{ELBO} = \underbrace{\mathbb{E}_q[\log p_\theta(\mathbf{x}_0 | \mathbf{x}_1)]}_{\text{reconstruction term}} - \underbrace{D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T))}_{\text{prior matching}} - \underbrace{\sum_{t=2}^{T} \mathbb{E}_q\left[D_{\text{KL}}\big(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\big)\right]}_{\text{denoising matching terms}}$$

where:

- **Reconstruction term:** How well the model recovers $\mathbf{x}_0$ from the first noisy version $\mathbf{x}_1$.
- **Prior matching:** Ensures the fully noised distribution $q(\mathbf{x}_T|\mathbf{x}_0)$ is close to the prior $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$. This term has no learnable parameters (it is determined by the noise schedule) and is approximately zero for large $T$.
- **Denoising matching terms:** Each term asks the learned reverse process $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ to match the tractable posterior $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$. Since both are Gaussian, this KL has a closed form (see Section 4).

The key insight is that each denoising matching term is a KL between two Gaussians, which reduces to a simple weighted mean squared error --- this is why the simplified diffusion training objective is just a denoising loss:

$$L_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

### Why Maximizing the ELBO Minimizes KL to the True Posterior

Recall the exact decomposition:

$$\log p(\mathbf{x}) = \text{ELBO}(q) + D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$$

Since $\log p(\mathbf{x})$ is a constant with respect to $q$ (it depends only on the generative model):

$$\max_q \text{ELBO}(q) \iff \min_q D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$$

**Maximizing the ELBO with respect to the variational distribution $q$ is equivalent to minimizing the reverse KL divergence from $q$ to the true posterior.** This is the fundamental principle of variational inference: we turn an intractable Bayesian inference problem into a tractable optimization problem.

Note that this is the *reverse* KL $D_{\text{KL}}(q \| p)$, which is mode-seeking (see Section 4). This means variational inference tends to under-estimate the posterior variance and may miss some modes of the true posterior. Understanding this trade-off is important for appreciating the limitations of variational methods.

---

## Summary: The Path to Diffusion Models

The mathematical concepts in this document connect as follows:

1. **Gaussians** provide closed-form operations (conditioning, marginalization, addition) that make diffusion tractable.
2. **Bayes' theorem** frames the inference problem: we want the posterior but cannot compute the evidence.
3. **KL divergence** gives us a way to measure and minimize the gap between approximate and true posteriors.
4. **The ELBO** turns intractable likelihood maximization into a tractable optimization objective.
5. **Diffusion models** decompose the ELBO into $T$ simple denoising terms, each of which is a KL between Gaussians, reducing to a mean squared error loss.

Each concept builds on the previous one. Mastering these foundations makes the derivation of diffusion models feel natural rather than mysterious.
