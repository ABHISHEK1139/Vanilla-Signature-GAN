"""
Signature Verification Model - Training Script
Siamese CNN for signature verification (genuine vs forgery).
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_signatures import SignatureVerificationDataset
from generator_vanilla_gan import Generator


class SiameseNetwork(nn.Module):
    """
    Siamese CNN for signature verification.
    Takes two signatures and outputs similarity score.
    """
    
    def __init__(self, img_channels=1):
        super(SiameseNetwork, self).__init__()
        
        # Shared CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )
        
        # Final classification layers
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x):
        """Extract features from one image."""
        return self.cnn(x)
    
    def forward(self, x1, x2):
        """
        Forward pass for pair of images.
        
        Args:
            x1: First signature
            x2: Second signature
        
        Returns:
            Similarity score (0 = different, 1 = same)
        """
        # Extract features
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # L1 distance
        diff = torch.abs(feat1 - feat2)
        
        # Classification
        out = self.fc(diff)
        return out


def generate_synthetic_data(generator, num_samples, latent_dim, device):
    """Generate synthetic signatures using trained GAN."""
    generator.eval()
    synthetic = []
    
    batch_size = 64
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in range(num_batches):
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = generator(z)
            synthetic.append(fake.cpu())
    
    return torch.cat(synthetic, dim=0)[:num_samples]


def train_verifier(args):
    """Train signature verification model."""
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("\nLoading verification dataset...")
    dataset = SignatureVerificationDataset(
        args.real_dir,
        args.forged_dir,
        image_size=args.image_size
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Load GAN for augmentation if specified
    synthetic_data = None
    if args.use_augmentation and args.generator_path:
        print(f"\nLoading generator for augmentation from: {args.generator_path}")
        generator = Generator(latent_dim=args.latent_dim, img_channels=1).to(device)
        generator.load_state_dict(torch.load(args.generator_path, map_location=device))
        
        # Generate synthetic data
        print(f"Generating {args.num_synthetic} synthetic signatures...")
        synthetic_data = generate_synthetic_data(
            generator, args.num_synthetic, args.latent_dim, device
        )
        print(f"Synthetic data shape: {synthetic_data.shape}")
    
    # Initialize model
    print("\nInitializing Siamese Network...")
    model = SiameseNetwork(img_channels=1).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Forward
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                outputs = model(img1, img2)
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.output_dir, "best_verifier.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with val acc: {val_acc:.4f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "verifier_final.pth")
    torch.save(model.state_dict(), final_path)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Signature Verification Model")
    
    parser.add_argument('--real_dir', type=str,
                        default='../archive/data/data/real',
                        help='Directory with real signatures')
    parser.add_argument('--forged_dir', type=str,
                        default='../archive/data/data/forged',
                        help='Directory with forged signatures')
    parser.add_argument('--output_dir', type=str, default='../checkpoints',
                        help='Output directory')
    
    # Augmentation
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use GAN-generated signatures for augmentation')
    parser.add_argument('--generator_path', type=str,
                        default='../checkpoints/G_final.pth',
                        help='Path to trained generator')
    parser.add_argument('--num_synthetic', type=int, default=1000,
                        help='Number of synthetic samples')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent dimension of generator')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU')
    
    args = parser.parse_args()
    train_verifier(args)


if __name__ == "__main__":
    main()
