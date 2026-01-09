"""
Generate synthetic signatures using trained Vanilla GAN.
Command-line tool for batch signature generation.
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator_vanilla_gan import Generator


def generate_signatures(args):
    """
    Generate synthetic signature images.
    
    Args:
        args: Command line arguments
    """
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Load generator
    print(f"\nLoading generator from: {args.model_path}")
    generator = Generator(latent_dim=args.latent_dim, img_channels=1).to(device)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()
    
    # Generate signatures
    print(f"\nGenerating {args.num_signatures} synthetic signatures...")
    
    batch_size = min(args.batch_size, args.num_signatures)
    num_batches = (args.num_signatures + batch_size - 1) // batch_size
    
    generated_count = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            # Calculate batch size for last batch
            current_batch_size = min(batch_size, args.num_signatures - generated_count)
            
            # Sample noise
            z = torch.randn(current_batch_size, args.latent_dim, device=device)
            
            # Generate
            fake_images = generator(z)
            
            # Convert to numpy and save
            fake_images = fake_images.cpu().numpy()
            
            for i in range(current_batch_size):
                # Denormalize from [-1, 1] to [0, 255]
                img = fake_images[i, 0]  # Get grayscale channel
                img = ((img + 1) * 127.5).astype(np.uint8)
                
                # Save image
                img_pil = Image.fromarray(img, mode='L')
                filename = f"synthetic_signature_{generated_count + i + 1:05d}.png"
                filepath = os.path.join(args.output_dir, filename)
                img_pil.save(filepath)
                
                generated_count += 1
    
    print(f"\nGenerated {generated_count} signatures")
    print(f"Saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Signatures")
    
    parser.add_argument('--model_path', type=str, 
                        default='../checkpoints/G_final.pth',
                        help='Path to trained generator model')
    parser.add_argument('--output_dir', type=str, 
                        default='../data/synthetic',
                        help='Output directory for generated signatures')
    parser.add_argument('--num_signatures', '-n', type=int, default=100,
                        help='Number of signatures to generate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent vector dimension (must match trained model)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for generation')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 50)
    print("Signature Generation Configuration")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Number of signatures: {args.num_signatures}")
    print(f"Latent dimension: {args.latent_dim}")
    print("=" * 50)
    
    generate_signatures(args)


if __name__ == "__main__":
    main()
