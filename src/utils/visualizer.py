"""
Visualization utilities for GAN training and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def save_sample_grid(images, filepath, nrow=8, normalize=True):
    """
    Save a grid of images.
    
    Args:
        images: Tensor of images (N, C, H, W)
        filepath: Path to save the image
        nrow: Number of images per row
        normalize: Whether to normalize from [-1, 1] to [0, 1]
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    # Create grid
    if normalize:
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    
    images = torch.clamp(images, 0, 1)
    grid = make_grid(images, nrow=nrow, padding=2, pad_value=1)
    
    # Convert to numpy and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # If grayscale, squeeze
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
        plt.imsave(filepath, grid_np, cmap='gray')
    else:
        plt.imsave(filepath, grid_np)


def plot_losses(epochs, d_losses, g_losses, save_path):
    """
    Plot training losses.
    
    Args:
        epochs: List of epoch numbers
        d_losses: List of discriminator losses
        g_losses: List of generator losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, d_losses, label='Discriminator Loss', color='blue')
    plt.plot(epochs, g_losses, label='Generator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under curve
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title('ROC Curve - Signature Verification')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_det_curve(far, frr, save_path):
    """
    Plot DET (Detection Error Tradeoff) curve.
    
    Args:
        far: False acceptance rates
        frr: False rejection rates
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    plt.plot(far, frr, color='blue', lw=2)
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Curve - Signature Verification')
    plt.grid(True, alpha=0.3)
    
    # Mark EER point
    eer_idx = np.argmin(np.abs(np.array(far) - np.array(frr)))
    eer = (far[eer_idx] + frr[eer_idx]) / 2
    plt.scatter([far[eer_idx]], [frr[eer_idx]], color='red', s=100, zorder=5, label=f'EER = {eer:.4f}')
    plt.legend()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_comparison(real_images, fake_images, save_path, n_samples=8):
    """
    Create side-by-side comparison of real and fake images.
    
    Args:
        real_images: Tensor of real images
        fake_images: Tensor of fake images
        save_path: Path to save the comparison
        n_samples: Number of samples to show
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i in range(n_samples):
        # Real images
        real_img = real_images[i].cpu().numpy()
        if real_img.shape[0] == 1:
            real_img = real_img.squeeze(0)
        real_img = (real_img + 1) / 2  # [-1, 1] -> [0, 1]
        axes[0, i].imshow(real_img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Real', fontsize=12)
        
        # Fake images
        fake_img = fake_images[i].cpu().numpy()
        if fake_img.shape[0] == 1:
            fake_img = fake_img.squeeze(0)
        fake_img = (fake_img + 1) / 2
        axes[1, i].imshow(fake_img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Generated', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
