"""
Logger utility for tracking training progress and metrics.
"""

import os
import json
from datetime import datetime


class Logger:
    """Simple logger for GAN training metrics."""
    
    def __init__(self, log_dir):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")
        
        self.logs = {
            "start_time": timestamp,
            "epochs": []
        }
    
    def log_epoch(self, epoch, d_loss, g_loss, d_real_loss=None, d_fake_loss=None):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            d_loss: Discriminator loss
            g_loss: Generator loss
            d_real_loss: Discriminator loss on real images
            d_fake_loss: Discriminator loss on fake images
        """
        epoch_data = {
            "epoch": epoch,
            "d_loss": float(d_loss),
            "g_loss": float(g_loss),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if d_real_loss is not None:
            epoch_data["d_real_loss"] = float(d_real_loss)
        if d_fake_loss is not None:
            epoch_data["d_fake_loss"] = float(d_fake_loss)
        
        self.logs["epochs"].append(epoch_data)
        
        # Save to file
        self._save()
    
    def _save(self):
        """Save logs to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def get_losses(self):
        """
        Get all recorded losses.
        
        Returns:
            Tuple of (epochs, d_losses, g_losses)
        """
        epochs = [e["epoch"] for e in self.logs["epochs"]]
        d_losses = [e["d_loss"] for e in self.logs["epochs"]]
        g_losses = [e["g_loss"] for e in self.logs["epochs"]]
        return epochs, d_losses, g_losses
