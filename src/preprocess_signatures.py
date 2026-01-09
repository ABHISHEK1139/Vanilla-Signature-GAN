"""
Preprocessing module for signature images.
Converts, crops, resizes, and normalizes signature images.
"""

import os
import cv2
import numpy as np
from PIL import Image


def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image: Input image (PIL Image or numpy array)
    
    Returns:
        Grayscale image as numpy array
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    return gray


def binarize_image(image, threshold=None):
    """
    Binarize image using Otsu's method or given threshold.
    
    Args:
        image: Grayscale image
        threshold: Optional threshold value (uses Otsu if None)
    
    Returns:
        Binary image
    """
    if threshold is None:
        # Use Otsu's method
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    return binary


def crop_signature(image, padding=10):
    """
    Crop image around signature (remove margins).
    
    Args:
        image: Grayscale image
        padding: Padding around the signature
    
    Returns:
        Cropped image
    """
    # Invert if background is white
    if np.mean(image) > 127:
        inv = 255 - image
    else:
        inv = image
    
    # Find contours
    _, binary = cv2.threshold(inv, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get bounding box of all contours
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)
    
    # Crop
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped


def resize_image(image, size=(64, 64)):
    """
    Resize image to fixed size while maintaining aspect ratio.
    
    Args:
        image: Input image
        size: Target size (width, height)
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas and center image
    if len(resized.shape) == 2:
        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    else:
        canvas = np.ones((target_h, target_w, resized.shape[2]), dtype=np.uint8) * 255
    
    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def normalize_image(image):
    """
    Normalize pixel values to [-1, 1] range (for tanh output).
    
    Args:
        image: Input image with values in [0, 255]
    
    Returns:
        Normalized image with values in [-1, 1]
    """
    return (image.astype(np.float32) / 127.5) - 1.0


def denormalize_image(image):
    """
    Denormalize pixel values from [-1, 1] to [0, 255].
    
    Args:
        image: Normalized image with values in [-1, 1]
    
    Returns:
        Image with values in [0, 255]
    """
    return ((image + 1.0) * 127.5).astype(np.uint8)


def preprocess_signature(image_path, target_size=(64, 64), do_crop=True):
    """
    Full preprocessing pipeline for a single signature image.
    
    Args:
        image_path: Path to the signature image
        target_size: Target size (width, height)
        do_crop: Whether to crop around signature
    
    Returns:
        Preprocessed and normalized image
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Crop if requested
    if do_crop:
        image = crop_signature(image)
    
    # Resize
    image = resize_image(image, target_size)
    
    # Normalize to [-1, 1]
    image = normalize_image(image)
    
    return image


def preprocess_dataset(input_dir, output_dir, target_size=(64, 64)):
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir: Input directory with raw images
        output_dir: Output directory for preprocessed images
        target_size: Target size (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    errors = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                
                # Create relative output path
                rel_path = os.path.relpath(root, input_dir)
                out_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(out_subdir, exist_ok=True)
                
                output_path = os.path.join(out_subdir, file)
                
                try:
                    # Load and preprocess
                    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image = crop_signature(image)
                        image = resize_image(image, target_size)
                        cv2.imwrite(output_path, image)
                        processed += 1
                except Exception as e:
                    errors += 1
                    print(f"Error processing {input_path}: {e}")
    
    print(f"Processed: {processed}, Errors: {errors}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess signature images")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=64, help="Target image size")
    
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.output, (args.size, args.size))
