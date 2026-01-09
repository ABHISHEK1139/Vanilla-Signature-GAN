"""
Vanilla GAN model combining Generator and Discriminator.
"""

import torch
import torch.nn as nn
from generator_vanilla_gan import Generator
from discriminator_vanilla_gan import Discriminator


class VanillaGAN:
    """
    Vanilla GAN model for signature generation.
    
    Combines Generator and Discriminator with training utilities.
    Uses Binary Cross-Entropy loss as specified.
    """
    
    def __init__(self, latent_dim=100, img_channels=1, feature_maps=64, device='cuda'):
        """
        Initialize the Vanilla GAN.
        
        Args:
            latent_dim: Dimension of latent vector z
            img_channels: Number of image channels (1 for grayscale)
            feature_maps: Base number of feature maps
            device: Device to use ('cuda' or 'cpu')
        """
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.device = device
        
        # Initialize networks
        self.generator = Generator(latent_dim, img_channels, feature_maps).to(device)
        self.discriminator = Discriminator(img_channels, feature_maps).to(device)
        
        # Loss function: Binary Cross-Entropy
        self.criterion = nn.BCELoss()
        
        # Optimizers: Adam with specified hyperparameters
        # lr = 2e-4, β1 = 0.5, β2 = 0.999
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999)
        )
        
        # Label smoothing values
        self.real_label = 0.9  # Use 0.9 instead of 1.0 for label smoothing
        self.fake_label = 0.0
    
    def sample_noise(self, batch_size):
        """
        Sample noise from standard normal distribution.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Noise tensor of shape (batch_size, latent_dim)
        """
        return torch.randn(batch_size, self.latent_dim, device=self.device)
    
    def train_discriminator(self, real_images):
        """
        Train discriminator for one step.
        
        Args:
            real_images: Batch of real images
        
        Returns:
            Tuple of (d_loss, d_real_loss, d_fake_loss)
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Labels with smoothing
        real_labels = torch.full((batch_size, 1), self.real_label, device=self.device)
        fake_labels = torch.full((batch_size, 1), self.fake_label, device=self.device)
        
        # Reset gradients
        self.optimizer_D.zero_grad()
        
        # Train on real images
        # D_real_loss = BCE(D(x_real), 1)
        real_output = self.discriminator(real_images)
        d_real_loss = self.criterion(real_output, real_labels)
        
        # Generate fake images
        noise = self.sample_noise(batch_size)
        fake_images = self.generator(noise)
        
        # Train on fake images
        # D_fake_loss = BCE(D(x_fake_detached), 0)
        fake_output = self.discriminator(fake_images.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        
        # Backpropagation
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item(), d_real_loss.item(), d_fake_loss.item()
    
    def train_generator(self, batch_size):
        """
        Train generator for one step.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Generator loss
        """
        # Labels (generator wants discriminator to output 1)
        real_labels = torch.full((batch_size, 1), 1.0, device=self.device)
        
        # Reset gradients
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        noise = self.sample_noise(batch_size)
        fake_images = self.generator(noise)
        
        # Generator wants D(x_fake) ≈ 1
        # G_loss = BCE(D(x_fake), 1)
        fake_output = self.discriminator(fake_images)
        g_loss = self.criterion(fake_output, real_labels)
        
        # Backpropagation
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()
    
    def generate(self, num_samples):
        """
        Generate synthetic signatures.
        
        Args:
            num_samples: Number of signatures to generate
        
        Returns:
            Generated images tensor
        """
        self.generator.eval()
        with torch.no_grad():
            noise = self.sample_noise(num_samples)
            fake_images = self.generator(noise)
        self.generator.train()
        return fake_images
    
    def save_checkpoint(self, filepath, epoch):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'latent_dim': self.latent_dim,
            'img_channels': self.img_channels
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint
        
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        return checkpoint['epoch']


def test_vanilla_gan():
    """Test the Vanilla GAN model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create GAN
    gan = VanillaGAN(latent_dim=100, device=device)
    
    # Test with dummy data
    dummy_real = torch.randn(16, 1, 64, 64)
    
    # Train discriminator
    d_loss, d_real, d_fake = gan.train_discriminator(dummy_real)
    print(f"D Loss: {d_loss:.4f} (Real: {d_real:.4f}, Fake: {d_fake:.4f})")
    
    # Train generator
    g_loss = gan.train_generator(16)
    print(f"G Loss: {g_loss:.4f}")
    
    # Generate samples
    samples = gan.generate(8)
    print(f"Generated samples shape: {samples.shape}")


if __name__ == "__main__":
    test_vanilla_gan()
