"""
Evaluation script for Vanilla GAN.
Performs visual inspection and computes image-level metrics.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_signatures import get_data_loader
from generator_vanilla_gan import Generator
from utils.metrics import calculate_stroke_density
from utils.visualizer import save_sample_grid, visualize_comparison


def evaluate_gan(args):
    """
    Evaluate trained GAN model.
    
    Args:
        args: Command line arguments
    """
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load generator
    print(f"\nLoading generator from: {args.model_path}")
    generator = Generator(latent_dim=args.latent_dim, img_channels=1).to(device)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()
    
    # Load real data for comparison
    print(f"\nLoading real data from: {args.data_dir}")
    dataloader = get_data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=0
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # 4A. Visual Inspection
    print("\n--- 4A. Visual Inspection ---")
    
    # Generate large batch
    print(f"Generating {args.num_samples} synthetic signatures...")
    all_fake = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            z = torch.randn(args.batch_size, args.latent_dim, device=device)
            fake = generator(z)
            all_fake.append(fake.cpu())
    
    all_fake = torch.cat(all_fake, dim=0)[:args.num_samples]
    
    # Save sample grid
    grid_path = os.path.join(args.output_dir, "generated_samples_grid.png")
    save_sample_grid(all_fake[:64], grid_path, nrow=8)
    print(f"Saved sample grid to {grid_path}")
    
    # Get real samples for comparison
    real_batch = next(iter(dataloader))
    
    # Save comparison
    comparison_path = os.path.join(args.output_dir, "real_vs_generated.png")
    visualize_comparison(real_batch[:8], all_fake[:8], comparison_path)
    print(f"Saved comparison to {comparison_path}")
    
    # 4B. Image-Level Metrics & Diversity
    print("\n--- 4B. Image-Level Metrics ---")
    
    # Convert to numpy for metrics
    fake_np = all_fake.numpy()
    real_list = []
    for batch in dataloader:
        real_list.append(batch.numpy())
        if len(real_list) * args.batch_size >= args.num_samples:
            break
    real_np = np.concatenate(real_list, axis=0)[:args.num_samples]
    
    # Stroke density statistics
    print("\nStroke Density Statistics:")
    fake_density = calculate_stroke_density(fake_np)
    real_density = calculate_stroke_density(real_np)
    
    print(f"  Generated signatures:")
    print(f"    Mean density: {fake_density['mean_density']:.4f}")
    print(f"    Std density:  {fake_density['std_density']:.4f}")
    print(f"    Min density:  {fake_density['min_density']:.4f}")
    print(f"    Max density:  {fake_density['max_density']:.4f}")
    
    print(f"  Real signatures:")
    print(f"    Mean density: {real_density['mean_density']:.4f}")
    print(f"    Std density:  {real_density['std_density']:.4f}")
    print(f"    Min density:  {real_density['min_density']:.4f}")
    print(f"    Max density:  {real_density['max_density']:.4f}")
    
    # Diversity check (check if all generated images are different)
    print("\nDiversity Check:")
    
    # Calculate variance across generated images
    variance = np.var(fake_np, axis=0).mean()
    print(f"  Average pixel variance: {variance:.6f}")
    
    # Check for mode collapse
    if variance < 0.01:
        print("  WARNING: Low variance may indicate mode collapse!")
    else:
        print("  Diversity looks acceptable.")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Vanilla GAN Evaluation Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write("Generated Signatures:\n")
        f.write(f"  Mean density: {fake_density['mean_density']:.4f}\n")
        f.write(f"  Std density:  {fake_density['std_density']:.4f}\n\n")
        f.write("Real Signatures:\n")
        f.write(f"  Mean density: {real_density['mean_density']:.4f}\n")
        f.write(f"  Std density:  {real_density['std_density']:.4f}\n\n")
        f.write(f"Pixel Variance: {variance:.6f}\n")
    
    print(f"\nSaved metrics to {metrics_path}")
    print("\nEvaluation complete!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vanilla GAN")
    
    parser.add_argument('--model_path', type=str, 
                        default='../checkpoints/G_final.pth',
                        help='Path to trained generator')
    parser.add_argument('--data_dir', type=str,
                        default='../archive/data/data/real',
                        help='Directory with real signatures')
    parser.add_argument('--output_dir', type=str, default='../figures',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU')
    
    args = parser.parse_args()
    evaluate_gan(args)


if __name__ == "__main__":
    main()
