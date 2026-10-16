"""
Data loader for signature images.
Loads preprocessed signature images for GAN training.
"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SignatureDataset(Dataset):
    """Dataset class for loading signature images."""
    
    def __init__(self, data_dir, image_size=64, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing signature images
            image_size: Size to resize images to
            transform: Optional additional transforms
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_paths = []
        
        if not os.path.isdir(data_dir):
            raise RuntimeError(
                f"Data directory not found: {data_dir}. "
                "Pass a valid --data_dir (e.g. a directory of preprocessed signature images)."
            )
        
        # Collect all image paths from subdirectories (cropped signatures)
        self._collect_image_paths()
        
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No signature images (*.jpg/*.jpeg/*.png) found under {data_dir}."
            )
        
        # Default transform: resize, convert to tensor, normalize to [-1, 1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
        
        print(f"Found {len(self.image_paths)} signature images")
    
    def _collect_image_paths(self):
        """Collect all image paths from the data directory."""
        # Pattern to match image files in subdirectories
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        
        for ext in extensions:
            # Search ONLY in subdirectories (cropped signatures in folders)
            # Skip files in main directory as they are actually folders on OneDrive
            for subdir in os.listdir(self.data_dir):
                subdir_path = os.path.join(self.data_dir, subdir)
                if os.path.isdir(subdir_path):
                    self.image_paths.extend(glob.glob(os.path.join(subdir_path, ext)))
        
        # Remove duplicates and filter only actual files
        self.image_paths = [p for p in set(self.image_paths) if os.path.isfile(p)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single image.
        
        Args:
            idx: Index of the image
        
        Returns:
            Transformed image tensor
        """
        img_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fail loudly instead of recursing to the next image: unbounded
            # recursion over a dataset with several corrupt files ends in
            # RecursionError, and an empty dataset ends in ZeroDivisionError.
            raise RuntimeError(f"Failed to load image: {img_path}") from e


class SignatureVerificationDataset(Dataset):
    """Dataset for signature verification (pairs of genuine and forged)."""
    
    def __init__(self, real_dir, forged_dir, image_size=64, transform=None):
        """
        Initialize the dataset.
        
        Args:
            real_dir: Directory containing real signatures
            forged_dir: Directory containing forged signatures
            image_size: Size to resize images to
            transform: Optional additional transforms
        """
        self.image_size = image_size
        self.pairs = []
        
        for label, data_dir in (("real", real_dir), ("forged", forged_dir)):
            if not os.path.isdir(data_dir):
                raise RuntimeError(
                    f"{label} data directory not found: {data_dir}. "
                    f"Pass a valid --{label}_dir."
                )
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform
        
        # Collect genuine signatures
        self.real_images = self._collect_images(real_dir)
        self.forged_images = self._collect_images(forged_dir)
        
        # Create pairs
        self._create_pairs()
        
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No verification pairs could be created from {real_dir} and {forged_dir}: "
                "need at least 2 real signature images."
            )
        
        print(f"Created {len(self.pairs)} verification pairs")
    
    def _collect_images(self, data_dir):
        """Collect images from directory."""
        images = []
        extensions = ['*.jpg', '*.jpeg', '*.png']
        
        for ext in extensions:
            images.extend(glob.glob(os.path.join(data_dir, ext)))
            images.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        return list(set(images))
    
    def _create_pairs(self):
        """Create genuine-genuine and genuine-forged pairs."""
        # Genuine pairs (label = 1)
        for i in range(0, len(self.real_images) - 1, 2):
            self.pairs.append((self.real_images[i], self.real_images[i+1], 1))
        
        # Forged pairs (label = 0)
        for i in range(min(len(self.real_images), len(self.forged_images))):
            self.pairs.append((self.real_images[i % len(self.real_images)], 
                             self.forged_images[i], 0))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get a pair of images and label."""
        img1_path, img2_path, label = self.pairs[idx]
        
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label


def get_data_loader(data_dir, batch_size=64, image_size=64, num_workers=4, shuffle=True):
    """
    Create a DataLoader for signature images.
    
    Args:
        data_dir: Directory containing signature images
        batch_size: Batch size
        image_size: Size to resize images to
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader instance
    """
    dataset = SignatureDataset(data_dir, image_size=image_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_verification_loader(real_dir, forged_dir, batch_size=32, image_size=64, 
                            num_workers=4, shuffle=True):
    """
    Create a DataLoader for signature verification.
    
    Args:
        real_dir: Directory containing real signatures
        forged_dir: Directory containing forged signatures
        batch_size: Batch size
        image_size: Size to resize images to
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader instance
    """
    dataset = SignatureVerificationDataset(real_dir, forged_dir, image_size=image_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test data loader
    import sys
    
    # Anchor to the repo root so the script works from any cwd
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = os.path.join(REPO_ROOT, "data", "real")
    
    loader = get_data_loader(data_dir, batch_size=16, image_size=64)
    
    # Test loading a batch
    for batch in loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Value range: [{batch.min():.2f}, {batch.max():.2f}]")
        break
