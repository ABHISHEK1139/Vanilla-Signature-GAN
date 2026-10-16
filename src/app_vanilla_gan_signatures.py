"""
Streamlit Web Application for Signature Generation.
Simple UI for generating synthetic signatures using trained Vanilla GAN.
"""

import os
import sys
import io
import zipfile
import torch
import numpy as np
from PIL import Image
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator_vanilla_gan import Generator


# Page configuration
st.set_page_config(
    page_title="Signature Generator",
    page_icon="✍️",
    layout="wide"
)


@st.cache_resource
def load_generator(model_path, latent_dim=100):
    """Load and cache the generator model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator(latent_dim=latent_dim, img_channels=1).to(device)
    # weights_only=True: the file must be a plain state_dict saved by
    # torch.save(model.state_dict()); no arbitrary unpickling needed.
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    generator.eval()
    return generator, device


def generate_batch(generator, device, num_samples, latent_dim):
    """Generate a batch of signatures."""
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_images = generator(z)
    return fake_images


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    img = tensor.cpu().numpy()
    img = ((img + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(img, mode='L')


def create_zip(images):
    """Create a ZIP file from images."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            zip_file.writestr(f'signature_{i+1:04d}.png', img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer


def main():
    st.title("✍️ Synthetic Signature Generator")
    st.markdown("Generate realistic synthetic signatures using Vanilla GAN")
    
    # Sidebar settings
    st.sidebar.header("Settings")
    
    # Dynamic Model Selection
    
    # helper to find checkpoints
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(root_dir, 'checkpoints')
    
    # If not found (running locally maybe?), try default path
    if not os.path.exists(checkpoints_dir):
        checkpoints_dir = os.path.join(root_dir, '..', 'checkpoints')
        
    model_files = []
    if os.path.exists(checkpoints_dir):
        model_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
    
    if model_files:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            model_files,
            index=0,
            help="Choose a trained model from the checkpoints folder"
        )
        model_path = os.path.join(checkpoints_dir, selected_model)
    else:
        st.sidebar.error("No .pth models found in checkpoints folder!")
        model_path = st.sidebar.text_input(
            "Model Path (Manual Entry)",
            value=os.path.join(checkpoints_dir, "G_final.pth")
        )
    
    latent_dim = st.sidebar.number_input(
        "Latent Dimension",
        min_value=10,
        max_value=500,
        value=100,
        help="Must match the trained model"
    )
    
    # Load model
    if st.sidebar.button("Load Model") or 'generator' not in st.session_state:
        if os.path.exists(model_path):
            with st.spinner(f"Loading {os.path.basename(model_path)}..."):
                try:
                    generator, device = load_generator(model_path, latent_dim)
                    st.session_state.generator = generator
                    st.session_state.device = device
                    st.session_state.latent_dim = latent_dim
                    st.sidebar.success(f"Model loaded! Device: {device}")
                except Exception as e:
                    st.sidebar.error(f"Error loading model: {str(e)}")
        else:
            st.sidebar.error("Model file not found!")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Generation Settings")
        
        num_signatures = st.number_input(
            "Number of Signatures",
            min_value=1,
            max_value=1000,
            value=16,
            step=1
        )
        
        seed = st.number_input(
            "Random Seed (optional)",
            min_value=0,
            max_value=999999,
            value=42,
            help="Set for reproducible results"
        )
        
        use_seed = st.checkbox("Use fixed seed", value=False)
        
        generate_btn = st.button("🎨 Generate Signatures", type="primary")
    
    with col2:
        st.subheader("Generated Signatures")
        
        if generate_btn and 'generator' in st.session_state:
            if use_seed:
                torch.manual_seed(seed)
            
            with st.spinner(f"Generating {num_signatures} signatures..."):
                fake_images = generate_batch(
                    st.session_state.generator,
                    st.session_state.device,
                    num_signatures,
                    st.session_state.latent_dim
                )
                
                # Convert to PIL images
                pil_images = []
                for i in range(num_signatures):
                    img = tensor_to_pil(fake_images[i, 0])
                    pil_images.append(img)
                
                st.session_state.generated_images = pil_images
            
            st.success(f"Generated {num_signatures} signatures!")
        
        # Display images
        if 'generated_images' in st.session_state:
            images = st.session_state.generated_images
            
            # Display in grid
            cols_per_row = 4
            for i in range(0, len(images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(images):
                        col.image(images[i + j], caption=f"Signature {i+j+1}")
            
            # Download button
            st.markdown("---")
            zip_buffer = create_zip(images)
            st.download_button(
                label="📥 Download All (ZIP)",
                data=zip_buffer,
                file_name="synthetic_signatures.zip",
                mime="application/zip"
            )
    
    # Footer
    st.markdown("---")
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        ### 📖 How to Use
        1. Select a model from the sidebar.
        2. Set the latent dimension (default: 100).
        3. Adjust generation settings in the main panel.
        4. Click 'Generate' to create signatures.
        5. Download results using the ZIP button.
        """
    )
    
    st.markdown(
        """
        ### ✍️ About the Project
        This tool leverages a **Vanilla Generative Adversarial Network (GAN)** trained on thousands of handwritten signatures.
        
        **Key Features:**
        - **Realistic Generation**: Produces diverse signature styles.
        - **Data Augmentation**: Useful for training verification models.
        - **Instant Preview**: View generated samples directly in the browser.
        - **Batch Export**: Download all generated images in a single ZIP file.
        """
    )


if __name__ == "__main__":
    main()
