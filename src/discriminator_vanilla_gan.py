"""
Discriminator network for Vanilla GAN.
Classifies 64x64 grayscale images as real or fake.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator network for Vanilla GAN.
    
    Architecture:
        - Input: 64×64×1 image (real or fake)
        - Sequence of Conv2D + LeakyReLU(0.2) + Dropout
        - Downsampling with stride 2
        - Flatten + Dense(1) with sigmoid
        - Output: scalar probability of being "real"
    """
    
    def __init__(self, img_channels=1, feature_maps=64):
        """
        Initialize the Discriminator.
        
        Args:
            img_channels: Number of input image channels (default: 1 for grayscale)
            feature_maps: Base number of feature maps (default: 64)
        """
        super(Discriminator, self).__init__()
        
        self.img_channels = img_channels
        self.feature_maps = feature_maps
        
        # Convolutional layers
        # Input: 64x64x1
        self.conv_blocks = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        # Final classification layer: 4x4x512 -> 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps * 8 * 4 * 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images of shape (batch_size, img_channels, 64, 64)
        
        Returns:
            Probability of being real, shape (batch_size, 1)
        """
        # Convolutional blocks
        features = self.conv_blocks(x)
        
        # Classification
        output = self.classifier(features)
        
        return output


def test_discriminator():
    """Test the Discriminator network."""
    # Create discriminator
    disc = Discriminator(img_channels=1)
    
    # Test with random input
    x = torch.randn(16, 1, 64, 64)
    output = disc(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in disc.parameters())
    print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    test_discriminator()
