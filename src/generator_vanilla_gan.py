"""
Generator network for Vanilla GAN.
Generates 64x64 grayscale signature images from latent vectors.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for Vanilla GAN.
    
    Architecture:
        - Input: latent vector z ∈ R^100 sampled from N(0, I)
        - Dense layer -> reshape to small feature map (4×4×256)
        - Sequence of upsampling blocks with ConvTranspose2D
        - BatchNorm and ReLU activations
        - Output: 64×64×1 image with tanh activation
    """
    
    def __init__(self, latent_dim=100, img_channels=1, feature_maps=64):
        """
        Initialize the Generator.
        
        Args:
            latent_dim: Dimension of input latent vector (default: 100)
            img_channels: Number of output image channels (default: 1 for grayscale)
            feature_maps: Base number of feature maps (default: 64)
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.feature_maps = feature_maps
        
        # Initial dense layer: latent_dim -> 4*4*256
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 256),
            nn.BatchNorm1d(4 * 4 * 256),
            nn.ReLU(True)
        )
        
        # Convolutional layers
        # Input: 4x4x256
        self.conv_blocks = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(256, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # Block 4: 32x32 -> 64x64
            nn.ConvTranspose2d(feature_maps * 2, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1] to match normalized input
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using normal distribution."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        """
        Forward pass.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            Generated images of shape (batch_size, img_channels, 64, 64)
        """
        # Dense layer
        x = self.fc(z)
        
        # Reshape to 4x4x256
        x = x.view(-1, 256, 4, 4)
        
        # Convolutional blocks
        x = self.conv_blocks(x)
        
        return x


def test_generator():
    """Test the Generator network."""
    # Create generator
    gen = Generator(latent_dim=100, img_channels=1)
    
    # Test with random input
    z = torch.randn(16, 100)
    output = gen(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in gen.parameters())
    print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    test_generator()
