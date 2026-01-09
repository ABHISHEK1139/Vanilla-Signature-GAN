"""
Metrics for evaluating GAN performance and signature verification.
"""

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc


def calculate_fid_simple(real_features, fake_features):
    """
    Calculate simplified FID (Fréchet Inception Distance) score.
    
    Args:
        real_features: Features from real images (numpy array)
        fake_features: Features from fake images (numpy array)
    
    Returns:
        FID score (lower is better)
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    
    # Product of covariances
    covmean = np.sqrt(sigma_real @ sigma_fake)
    
    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return fid


def calculate_verification_metrics(y_true, y_scores):
    """
    Calculate verification metrics: FAR, FRR, EER.
    
    Args:
        y_true: Ground truth labels (1 for genuine, 0 for forgery)
        y_scores: Predicted similarity scores
    
    Returns:
        Dictionary with FAR, FRR, EER, and threshold
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # FAR = FPR (False Positive Rate)
    # FRR = 1 - TPR (False Negative Rate)
    far = fpr
    frr = 1 - tpr
    
    # Find EER (where FAR = FRR)
    eer_idx = np.argmin(np.abs(far - frr))
    eer = (far[eer_idx] + frr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    return {
        "far": far,
        "frr": frr,
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "auc": float(roc_auc),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def calculate_stroke_density(images):
    """
    Calculate stroke density statistics for signature images.
    
    Args:
        images: Batch of images (numpy array, values in [-1, 1])
    
    Returns:
        Dictionary with density statistics
    """
    # Convert to binary (stroke vs background)
    # Assuming dark strokes on light background
    binary = (images < 0).astype(float)
    
    # Calculate density per image
    densities = np.mean(binary, axis=(1, 2, 3)) if len(images.shape) == 4 else np.mean(binary, axis=(1, 2))
    
    return {
        "mean_density": float(np.mean(densities)),
        "std_density": float(np.std(densities)),
        "min_density": float(np.min(densities)),
        "max_density": float(np.max(densities))
    }
