"""
REST API for Signature Generation.
Provides endpoints to generate synthetic signatures.
"""

import os
import sys
import io
import base64
import zipfile
from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator_vanilla_gan import Generator

app = Flask(__name__)

# Global model variables
generator = None
device = None
latent_dim = 100


def load_model(model_path, z_dim=100):
    """Load the generator model."""
    global generator, device, latent_dim
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_dim = z_dim
    
    generator = Generator(latent_dim=latent_dim, img_channels=1).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    print(f"Model loaded on {device}")


def generate_images(num_samples):
    """Generate signature images."""
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_images = generator(z)
    return fake_images


def tensor_to_base64(tensor):
    """Convert tensor to base64 encoded PNG."""
    img = tensor.cpu().numpy()
    img = ((img + 1) * 127.5).astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L')
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': generator is not None,
        'device': device
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate synthetic signatures.
    
    Request body:
        {
            "n": 10,           # Number of signatures (default: 1)
            "format": "base64" # "base64" or "zip" (default: "base64")
        }
    
    Response:
        {
            "success": true,
            "count": 10,
            "images": ["base64_string_1", "base64_string_2", ...]
        }
        
        or ZIP file download
    """
    if generator is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    # Parse request
    data = request.get_json() or {}
    num_signatures = data.get('n', 1)
    output_format = data.get('format', 'base64')
    
    # Validate
    num_signatures = min(max(1, num_signatures), 1000)  # Limit to 1-1000
    
    # Generate
    fake_images = generate_images(num_signatures)
    
    if output_format == 'zip':
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i in range(num_signatures):
                img = fake_images[i, 0].cpu().numpy()
                img = ((img + 1) * 127.5).astype(np.uint8)
                pil_img = Image.fromarray(img, mode='L')
                
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                zip_file.writestr(f'signature_{i+1:04d}.png', img_buffer.getvalue())
        
        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='signatures.zip'
        )
    
    else:
        # Return base64 encoded images
        images_base64 = []
        for i in range(num_signatures):
            img_b64 = tensor_to_base64(fake_images[i, 0])
            images_base64.append(img_b64)
        
        return jsonify({
            'success': True,
            'count': num_signatures,
            'images': images_base64
        })


@app.route('/generate/<int:n>', methods=['GET'])
def generate_get(n):
    """
    Simple GET endpoint to generate signatures.
    
    Returns ZIP file with n signatures.
    """
    if generator is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    n = min(max(1, n), 100)  # Limit to 1-100 for GET
    
    fake_images = generate_images(n)
    
    # Create ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i in range(n):
            img = fake_images[i, 0].cpu().numpy()
            img = ((img + 1) * 127.5).astype(np.uint8)
            pil_img = Image.fromarray(img, mode='L')
            
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            zip_file.writestr(f'signature_{i+1:04d}.png', img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='signatures.zip'
    )


def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Signature Generation API")
    parser.add_argument('--model_path', type=str, 
                        default='../checkpoints/G_final.pth',
                        help='Path to trained generator')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent dimension')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to bind')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Load model
    if os.path.exists(args.model_path):
        load_model(args.model_path, args.latent_dim)
    else:
        print(f"Warning: Model not found at {args.model_path}")
        print("Start training first or provide correct model path")
    
    # Run server
    print(f"\nStarting API server at http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /health - Health check")
    print("  POST /generate - Generate signatures (JSON body: {n: 10})")
    print("  GET  /generate/<n> - Generate n signatures (returns ZIP)")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
