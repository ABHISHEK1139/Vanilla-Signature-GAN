"""
Signature Verification Model - Evaluation Script
Computes FAR, FRR, EER, Accuracy and generates ROC/DET curves.
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_signatures import SignatureVerificationDataset
from signature_verifier_train import SiameseNetwork
from utils.metrics import calculate_verification_metrics, calculate_accuracy
from utils.visualizer import plot_roc_curve, plot_det_curve


def evaluate_verifier(args):
    """Evaluate signature verification model."""
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("\nLoading test dataset...")
    dataset = SignatureVerificationDataset(
        args.real_dir,
        args.forged_dir,
        image_size=args.image_size
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset)}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = SiameseNetwork(img_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Collect predictions
    all_labels = []
    all_scores = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            outputs = model(img1, img2)
            
            all_labels.extend(labels.numpy())
            all_scores.extend(outputs.cpu().numpy().flatten())
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    metrics = calculate_verification_metrics(all_labels, all_scores)
    
    # Accuracy at EER threshold
    predictions = (all_scores > metrics['eer_threshold']).astype(int)
    accuracy = calculate_accuracy(all_labels, predictions)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"EER (Equal Error Rate): {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # FAR and FRR at specific thresholds
    print("\nFAR/FRR at different thresholds:")
    for thresh in [0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(metrics['thresholds'] - thresh))
        print(f"  Threshold {thresh}: FAR={metrics['far'][idx]:.4f}, FRR={metrics['frr'][idx]:.4f}")
    
    # Plot ROC curve
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'], roc_path)
    print(f"\nSaved ROC curve to {roc_path}")
    
    # Plot DET curve
    det_path = os.path.join(args.output_dir, "det_curve.png")
    plot_det_curve(metrics['far'], metrics['frr'], det_path)
    print(f"Saved DET curve to {det_path}")
    
    # Save results to file
    results_path = os.path.join(args.output_dir, "verification_results.txt")
    with open(results_path, 'w') as f:
        f.write("Signature Verification Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test samples: {len(dataset)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"EER: {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n\n")
        f.write("FAR/FRR at different thresholds:\n")
        for thresh in [0.3, 0.5, 0.7]:
            idx = np.argmin(np.abs(metrics['thresholds'] - thresh))
            f.write(f"  Threshold {thresh}: FAR={metrics['far'][idx]:.4f}, FRR={metrics['frr'][idx]:.4f}\n")
    
    print(f"Saved results to {results_path}")
    print("\nEvaluation complete!")
    
    return {
        'accuracy': accuracy,
        'eer': metrics['eer'],
        'auc': metrics['auc']
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Signature Verification Model")
    
    parser.add_argument('--model_path', type=str,
                        default='../checkpoints/best_verifier.pth',
                        help='Path to trained model')
    parser.add_argument('--real_dir', type=str,
                        default='../archive/data/data/real',
                        help='Directory with real signatures')
    parser.add_argument('--forged_dir', type=str,
                        default='../archive/data/data/forged',
                        help='Directory with forged signatures')
    parser.add_argument('--output_dir', type=str, default='../figures',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU')
    
    args = parser.parse_args()
    evaluate_verifier(args)


if __name__ == "__main__":
    main()
