"""
Training script for Vanilla GAN on signature images.
Implements the complete training loop with logging and checkpointing.
Supports resuming from checkpoint.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import glob

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_signatures import get_data_loader
from vanilla_gan_model import VanillaGAN
from utils.logger import Logger
from utils.visualizer import save_sample_grid, plot_losses


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None, 0
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
    latest = checkpoints[-1]
    epoch = int(latest.split('_')[-1].replace('.pth', ''))
    return latest, epoch


def train_gan(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize data loader
    print(f"\nLoading data from: {args.data_dir}")
    dataloader = get_data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    print(f"Number of batches: {len(dataloader)}")
    
    # Initialize GAN
    print("\nInitializing Vanilla GAN...")
    gan = VanillaGAN(
        latent_dim=args.latent_dim,
        img_channels=1,
        feature_maps=args.feature_maps,
        device=device
    )
    
    # Check for resume from checkpoint
    start_epoch = 1
    if args.resume:
        if args.resume_from:
            # Resume from specific checkpoint
            checkpoint_path = args.resume_from
        else:
            # Find latest checkpoint
            checkpoint_path, _ = find_latest_checkpoint(args.checkpoint_dir)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n>>> RESUMING from checkpoint: {checkpoint_path}")
            start_epoch = gan.load_checkpoint(checkpoint_path) + 1
            print(f">>> Starting from epoch {start_epoch}")
        else:
            print("\nNo checkpoint found, starting from scratch...")
    
    # Initialize logger
    logger = Logger(args.log_dir)
    
    # Fixed noise for consistent samples
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch_idx, real_images in enumerate(pbar):
            batch_size = real_images.size(0)
            
            # Add small Gaussian noise to real images (stabilization)
            if args.noise_std > 0:
                noise = torch.randn_like(real_images) * args.noise_std
                real_images = real_images + noise
                real_images = torch.clamp(real_images, -1, 1)
            
            # Train Discriminator
            d_loss, d_real_loss, d_fake_loss = gan.train_discriminator(real_images)
            
            # Train Generator
            g_loss = gan.train_generator(batch_size)
            
            # Accumulate losses
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            epoch_d_real += d_real_loss
            epoch_d_fake += d_fake_loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss:.4f}',
                'G_loss': f'{g_loss:.4f}'
            })
        
        # Average losses
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_real = epoch_d_real / num_batches
        avg_d_fake = epoch_d_fake / num_batches
        
        # Log epoch
        logger.log_epoch(epoch, avg_d_loss, avg_g_loss, avg_d_real, avg_d_fake)
        
        # Print epoch summary
        print(f"Epoch {epoch}: D_loss={avg_d_loss:.4f} (real={avg_d_real:.4f}, fake={avg_d_fake:.4f}), G_loss={avg_g_loss:.4f}")
        
        # Save samples
        if epoch % args.sample_interval == 0:
            gan.generator.eval()
            with torch.no_grad():
                fake_samples = gan.generator(fixed_noise)
            gan.generator.train()
            
            sample_path = os.path.join(args.sample_dir, f"samples_epoch_{epoch:04d}.png")
            save_sample_grid(fake_samples, sample_path, nrow=8)
            print(f"Saved samples to {sample_path}")
        
        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            # Save generator
            g_path = os.path.join(args.checkpoint_dir, f"G_epoch_{epoch:04d}.pth")
            torch.save(gan.generator.state_dict(), g_path)
            
            # Save discriminator
            d_path = os.path.join(args.checkpoint_dir, f"D_epoch_{epoch:04d}.pth")
            torch.save(gan.discriminator.state_dict(), d_path)
            
            # Save full checkpoint (for resuming)
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth")
            gan.save_checkpoint(checkpoint_path, epoch)
            
            # Also save as "latest" for easy resuming
            latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
            gan.save_checkpoint(latest_path, epoch)
            print(f"Saved checkpoint at epoch {epoch}")
        
        # Check for mode collapse (all signatures look identical)
        if avg_g_loss < 0.1 or avg_d_loss > 10:
            print("Warning: Possible training instability detected!")
    
    # Save final model
    final_g_path = os.path.join(args.checkpoint_dir, "G_final.pth")
    torch.save(gan.generator.state_dict(), final_g_path)
    print(f"\nSaved final generator to {final_g_path}")
    
    # Plot training curves
    epochs, d_losses, g_losses = logger.get_losses()
    plot_path = os.path.join(args.log_dir, "training_curves.png")
    plot_losses(epochs, d_losses, g_losses, plot_path)
    print(f"Saved training curves to {plot_path}")
    
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Vanilla GAN for Signature Generation")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default='../../archive/data/data/real',
                        help='Directory containing training images')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Dimension of latent vector z (default: 100)')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Size of images (default: 64)')
    parser.add_argument('--feature_maps', type=int, default=64,
                        help='Base number of feature maps (default: 64)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers (default: 0)')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for training')
    
    # Resume arguments
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from latest checkpoint')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to specific checkpoint to resume from')
    
    # Stabilization arguments
    parser.add_argument('--noise_std', type=float, default=0.05,
                        help='Std of noise added to real images (default: 0.05)')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--sample_dir', type=str, default='../samples',
                        help='Directory to save sample images')
    parser.add_argument('--log_dir', type=str, default='../logs',
                        help='Directory to save logs')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='Save samples every N epochs (default: 10)')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("Vanilla GAN Training Configuration")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 60)
    
    # Train
    train_gan(args)


if __name__ == "__main__":
    main()
