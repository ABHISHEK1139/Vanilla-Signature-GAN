"""
Ablation studies for Vanilla GAN on signatures.
Compares different configurations: latent dimensions, activations.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_signatures import get_data_loader
from utils.visualizer import save_sample_grid
from utils.metrics import calculate_stroke_density


class GeneratorAblation(nn.Module):
    """Generator with configurable latent dimension and activations."""
    
    def __init__(self, latent_dim=100, img_channels=1, feature_maps=64, use_leaky=False):
        super(GeneratorAblation, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Choose activation
        if use_leaky:
            activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            activation = nn.ReLU(True)
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 256),
            nn.BatchNorm1d(4 * 4 * 256),
            activation
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            activation,
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            activation,
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            activation,
            
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = self.conv_blocks(x)
        return x


class DiscriminatorAblation(nn.Module):
    """Discriminator with configurable activations."""
    
    def __init__(self, img_channels=1, feature_maps=64):
        super(DiscriminatorAblation, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps * 8 * 4 * 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.conv_blocks(x)
        return self.classifier(features)


def run_ablation_experiment(config, dataloader, device, num_epochs=50, output_dir='../figures'):
    """Run a single ablation experiment."""
    
    # Create models
    generator = GeneratorAblation(
        latent_dim=config['latent_dim'],
        use_leaky=config.get('use_leaky', False)
    ).to(device)
    
    discriminator = DiscriminatorAblation().to(device)
    
    # Optimizers
    opt_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    # Training
    g_losses = []
    d_losses = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        for real_images in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.full((batch_size, 1), 0.0, device=device)
            
            # Train D
            opt_D.zero_grad()
            d_real = discriminator(real_images)
            d_real_loss = criterion(d_real, real_labels)
            
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake_images = generator(z)
            d_fake = discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            opt_D.step()
            
            # Train G
            opt_G.zero_grad()
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake_images = generator(z)
            g_output = discriminator(fake_images)
            g_loss = criterion(g_output, torch.ones_like(g_output))
            g_loss.backward()
            opt_G.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
        
        g_losses.append(epoch_g_loss / num_batches)
        d_losses.append(epoch_d_loss / num_batches)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: D_loss={d_losses[-1]:.4f}, G_loss={g_losses[-1]:.4f}")
    
    # Generate samples for evaluation
    generator.eval()
    with torch.no_grad():
        z = torch.randn(100, config['latent_dim'], device=device)
        samples = generator(z)
    
    # Calculate diversity
    samples_np = samples.cpu().numpy()
    variance = np.var(samples_np, axis=0).mean()
    density = calculate_stroke_density(samples_np)
    
    # Save sample grid
    exp_name = f"ablation_z{config['latent_dim']}_{'leaky' if config.get('use_leaky') else 'relu'}"
    sample_path = os.path.join(output_dir, f"{exp_name}_samples.png")
    save_sample_grid(samples[:64], sample_path, nrow=8)
    
    return {
        'config': config,
        'final_g_loss': g_losses[-1],
        'final_d_loss': d_losses[-1],
        'variance': float(variance),
        'mean_density': density['mean_density'],
        'training_stable': d_losses[-1] < 5 and g_losses[-1] < 5
    }


def run_ablation_studies(args):
    """Run all ablation studies."""
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {args.data_dir}")
    dataloader = get_data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=64,
        num_workers=0
    )
    
    # Define experiments
    experiments = [
        {'latent_dim': 50, 'use_leaky': False, 'name': 'z=50, ReLU'},
        {'latent_dim': 100, 'use_leaky': False, 'name': 'z=100, ReLU (baseline)'},
        {'latent_dim': 200, 'use_leaky': False, 'name': 'z=200, ReLU'},
        {'latent_dim': 100, 'use_leaky': True, 'name': 'z=100, LeakyReLU'},
    ]
    
    results = []
    
    print("\n" + "=" * 60)
    print("ABLATION STUDIES")
    print("=" * 60)
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        result = run_ablation_experiment(
            exp, dataloader, device,
            num_epochs=args.epochs,
            output_dir=args.output_dir
        )
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'G_Loss':<10} {'D_Loss':<10} {'Variance':<10} {'Stable':<8}")
    print("-" * 60)
    
    for r in results:
        stable = "Yes" if r['training_stable'] else "No"
        print(f"{r['config']['name']:<25} {r['final_g_loss']:<10.4f} {r['final_d_loss']:<10.4f} {r['variance']:<10.6f} {stable:<8}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Studies for Vanilla GAN")
    
    parser.add_argument('--data_dir', type=str,
                        default='../archive/data/data/real',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='../figures',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU')
    
    args = parser.parse_args()
    run_ablation_studies(args)


if __name__ == "__main__":
    main()
