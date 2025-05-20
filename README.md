
# Latent Diffusion Model with Shared and Modality-Specific Representations

This project implements a framework for training Latent Diffusion Models (LDM) using disentangled representations. Specifically, we separate input data into **shared (c)** and **modality-specific (uv)** components, and model the generation process in the latent space using VAEs and Diffusion Models.

## Folder Structure

```
.
├── vae_uv/               # VAE for modality-specific (uv) representation
│   └── vae_train.py      # Training script for VAE
│
├── diffusion_uv/         # Diffusion model operating on uv and c parts
│   └── diffusion_train.py# Training script for latent diffusion
│
├── utils/                # Utility functions (optional)
├── data/                 # Dataset files (if any)
└── README.md             # This file
```

## Representation Disentanglement

- **uv**: Modality-specific representation extracted from each modality.
- **c**: Shared or common representation across modalities.
- These representations are first learned using a VAE and then used for conditional or unconditional diffusion model training.

## 1. VAE Training for Modality-Specific Representation

The VAE is trained to encode and reconstruct the **uv** (modality-specific) latent variables.

### Usage

```bash
cd vae_uv
python vae_train.py
```

Make sure to modify `vae_train.py` to point to your dataset and specify hyperparameters if needed.

## 2. Diffusion Training in Latent Space

The diffusion model operates on the latent space defined by `uv` and optionally conditioned on `c`.

### Usage

```bash
cd diffusion_uv
python diffusion_train.py
```

Ensure that:
- The pretrained VAE model is properly loaded to encode the input into latent space.
- The `c` (shared part) is extracted and passed as a condition (if applicable).

## Dependencies

- Python 3.10.16
- Other common libraries: numpy, tqdm, torchvision, etc.

You can install them via:

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` is not yet prepared, consider adding it for clarity.)*

## Notes

- Make sure the output of `vae_train.py` (typically latent representations) is saved and used in `diffusion_train.py`.
- You can customize the encoder-decoder architecture in `vae_uv`, or diffusion configuration in `diffusion_uv`.

## Acknowledgment
The code is based on https://github.com/mueller-franzes/medfusion