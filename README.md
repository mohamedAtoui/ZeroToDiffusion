# ZeroToDiffusion

From pure noise to deep understanding — a hands-on journey through the math, theory, and code behind diffusion models, with comparisons to what came before.

## Motivation

Diffusion models power state-of-the-art image generation (Stable Diffusion, DALL-E, Imagen), but their math can feel opaque. This repo builds understanding from the ground up: starting with probability fundamentals, implementing pre-diffusion baselines (VAEs, GANs), and then deriving and coding diffusion models from scratch.

Every concept has both a theory writeup and working code.

## Repo Structure

```
ZeroToDiffusion/
├── 01-foundations/               # Week 1: Math & generative model landscape
│   ├── probability-review.md     # Gaussians, Bayes, KL divergence, ELBO
│   ├── generative-models-landscape.md
│   └── notebooks/
│       ├── gaussian_basics.ipynb
│       └── kl_divergence_demo.ipynb
├── 02-before-diffusion/          # Pre-diffusion baselines
│   └── vae/
│       ├── README.md             # VAE theory notes
│       └── vae_mnist.py          # Working VAE on MNIST
├── 03-diffusion-theory/          # (Week 3-4)
├── 04-implementation/            # (Week 5-7)
├── 05-comparison/                # (Week 8)
├── 06-attention-and-beyond/      # (Week 9-10)
└── readings/                     # Paper notes & reading list
```

## Getting Started

### Local Setup (recommended)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/<your-username>/ZeroToDiffusion.git
cd ZeroToDiffusion
uv sync

# Run the VAE
uv run python 02-before-diffusion/vae/vae_mnist.py

# Run with options
uv run python 02-before-diffusion/vae/vae_mnist.py --latent-dim 2 --epochs 30 --kl-anneal --visualize-latent
```

### Google Colab

The notebooks in this repo include Colab setup cells. Open any `.ipynb` file in GitHub and click "Open in Colab", or upload directly to [colab.research.google.com](https://colab.research.google.com).

## Week 1 Progress

- [x] Probability review (Gaussians, Bayes, KL, ELBO)
- [x] Gaussian basics notebook
- [x] KL divergence & ELBO notebook
- [x] Generative models landscape survey
- [x] VAE theory writeup
- [x] VAE implementation on MNIST

## Key Resources

- [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Stanford CS236: Deep Generative Models](https://deepgenerativemodels.github.io/)
- [Calvin Luo: Understanding Diffusion Models (monograph)](https://arxiv.org/abs/2208.11970)
- [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course)
- [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114)
- [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)

## License

This project is for educational purposes.
