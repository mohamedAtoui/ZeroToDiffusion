"""
Variational Autoencoder (VAE) on MNIST

A self-contained implementation of a VAE with MLP encoder/decoder.
Trains on MNIST and produces reconstructions, samples, and visualizations.

Usage:
    uv run python 02-before-diffusion/vae/vae_mnist.py
    uv run python 02-before-diffusion/vae/vae_mnist.py --latent-dim 2 --visualize-latent
    uv run python 02-before-diffusion/vae/vae_mnist.py --kl-anneal --epochs 30
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)

    def forward(self, z: torch.Tensor):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss = BCE reconstruction + KL divergence."""
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n = 0

    for batch, _ in loader:
        batch = batch.view(-1, 784).to(device)
        recon, mu, log_var = model(batch)
        loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, log_var, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n += batch.size(0)

    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "kl": total_kl / n,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_reconstructions(model: VAE, loader: DataLoader, device: torch.device, out_dir: Path):
    model.eval()
    batch, _ = next(iter(loader))
    batch = batch[:16].to(device)
    flat = batch.view(-1, 784)

    with torch.no_grad():
        recon, _, _ = model(flat)

    orig = batch.cpu().numpy().reshape(-1, 28, 28)
    recon_imgs = recon.cpu().numpy().reshape(-1, 28, 28)

    fig, axes = plt.subplots(2, 16, figsize=(20, 3))
    for i in range(16):
        axes[0, i].imshow(orig[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_imgs[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstructed", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "reconstructions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_samples(model: VAE, device: torch.device, out_dir: Path, n: int = 64):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim, device=device)
        samples = model.decoder(z).cpu().numpy().reshape(-1, 28, 28)

    rows = int(np.sqrt(n))
    fig, axes = plt.subplots(rows, rows, figsize=(8, 8))
    for i in range(rows):
        for j in range(rows):
            axes[i, j].imshow(samples[i * rows + j], cmap="gray")
            axes[i, j].axis("off")
    plt.suptitle("Samples from Prior z ~ N(0, I)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "samples.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(history: dict[str, list[float]], out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["loss"], label="Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss per sample")
    ax1.set_title("Total Loss")
    ax1.legend()

    ax2.plot(history["recon"], label="Reconstruction", color="tab:blue")
    ax2.plot(history["kl"], label="KL Divergence", color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss per sample")
    ax2.set_title("Loss Components")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_space(model: VAE, loader: DataLoader, device: torch.device, out_dir: Path):
    model.eval()
    mus = []
    labels = []

    with torch.no_grad():
        for batch, lbl in loader:
            flat = batch.view(-1, 784).to(device)
            mu, _ = model.encoder(flat)
            mus.append(mu.cpu())
            labels.append(lbl)

    mus = torch.cat(mus).numpy()
    labels = torch.cat(labels).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mus[:, 0], mus[:, 1], c=labels, cmap="tab10", s=1, alpha=0.5)
    plt.colorbar(scatter, label="Digit class")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("Latent Space (Encoder Means)")
    plt.tight_layout()
    plt.savefig(out_dir / "latent_space.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_manifold(model: VAE, device: torch.device, out_dir: Path, n: int = 20):
    model.eval()
    # Grid of z values from the 2D latent space (using quantiles of N(0,1))
    from scipy.stats import norm

    grid = norm.ppf(np.linspace(0.05, 0.95, n))
    canvas = np.zeros((28 * n, 28 * n))

    with torch.no_grad():
        for i, yi in enumerate(grid):
            for j, xi in enumerate(grid):
                z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
                sample = model.decoder(z).cpu().numpy().reshape(28, 28)
                canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = sample

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("Latent Manifold (Decoded Grid)")
    plt.tight_layout()
    plt.savefig(out_dir / "latent_manifold.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST")
    parser.add_argument("--latent-dim", type=int, default=20, help="Latent space dimension")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--kl-anneal", action="store_true", help="Enable KL annealing")
    parser.add_argument("--visualize-latent", action="store_true", help="Plot latent space (requires --latent-dim 2)")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Output directory
    out_dir = Path("outputs/vae")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"VAE with latent_dim={args.latent_dim}, params={sum(p.numel() for p in model.parameters()):,}")

    # Train
    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}

    for epoch in range(1, args.epochs + 1):
        # KL annealing: linearly increase weight from 0 to 1 over training
        if args.kl_anneal:
            kl_weight = min(1.0, epoch / (args.epochs * 0.5))
        else:
            kl_weight = 1.0

        metrics = train_epoch(model, train_loader, optimizer, device, kl_weight)

        history["loss"].append(metrics["loss"])
        history["recon"].append(metrics["recon"])
        history["kl"].append(metrics["kl"])

        kl_info = f", kl_weight={kl_weight:.3f}" if args.kl_anneal else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {metrics['loss']:.2f} | "
            f"Recon: {metrics['recon']:.2f} | "
            f"KL: {metrics['kl']:.2f}{kl_info}"
        )

    # Visualizations
    print("\nGenerating visualizations...")
    plot_reconstructions(model, test_loader, device, out_dir)
    plot_samples(model, device, out_dir)
    plot_loss_curves(history, out_dir)

    if args.visualize_latent and args.latent_dim == 2:
        plot_latent_space(model, test_loader, device, out_dir)
        plot_latent_manifold(model, device, out_dir)
    elif args.visualize_latent and args.latent_dim != 2:
        print("Warning: --visualize-latent requires --latent-dim 2, skipping latent plots")

    # Save checkpoint
    checkpoint_path = out_dir / "vae_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "history": history,
        },
        checkpoint_path,
    )
    print(f"\nCheckpoint saved to {checkpoint_path}")
    print(f"Visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
