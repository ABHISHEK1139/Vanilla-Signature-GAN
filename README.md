# Vanilla GAN Signature Generation

A deep learning framework for generating and validating synthetic handwritten signatures. It features a Vanilla GAN pipeline, a Siamese Network for verification, and a Streamlit UI for real-time generation. Includes a REST API, ablation tools, and automated preprocessing for a complete end-to-end signature synthesis workflow.

## 📂 Project Structure

- `src/`: Core source code
  - `vanilla_gan_model.py`: Combined GAN model class.
  - `generator_vanilla_gan.py`: Generator architecture.
  - `discriminator_vanilla_gan.py`: Discriminator architecture.
  - `train_vanilla_gan_signatures.py`: Training pipeline.
  - `evaluate_vanilla_gan_signatures.py`: Model evaluation tools.
  - `data_loader_signatures.py`: Data loading and augmentation.
  - `preprocess_signatures.py`: Script for cleaning and resizing signature images.
  - `app_vanilla_gan_signatures.py`: Streamlit web interface.
  - `api_vanilla_gan_signatures.py`: REST API for inference.
  - `signature_verifier_train.py` & `signature_verifier_eval.py`: Siamese network for signature verification.
  - `ablation_vanilla_gan_signatures.py`: Experiments with different hyperparameters.
- `checkpoints/`: Directory for saving trained models (includes `G_final.pth`).
- `generated_samples/`: Example output images.
- `requirements.txt`: Python dependencies.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Preprocess your dataset:
   ```bash
   python src/preprocess_signatures.py --input path/to/raw/data --output data/processed
   ```

3. Train the model:
   ```bash
   python src/train_vanilla_gan_signatures.py --data_dir data/processed
   ```

## Evaluation

Run the evaluation script to generate sample grids and compute metrics:
```bash
python src/evaluate_vanilla_gan_signatures.py --model_path checkpoints/G_final.pth --data_dir data/processed
```

## Running the Web App

Launch the Streamlit interface for interactive signature generation:
```bash
streamlit run src/app_vanilla_gan_signatures.py
```

## API Usage

Start the Flask API:
```bash
python src/api_vanilla_gan_signatures.py
```
You can then send POST requests to `/generate` to receive synthetic signatures in base64 format.
