# SHG Cornea Pix2Pix

AI-driven image interpretation of backward-SHG images of the cornea using a Pix2Pix conditional GAN.

## Overview

Second Harmonic Generation (SHG) microscopy enables label-free visualisation of collagen microstructure in the cornea. Forward-detected SHG signals clearly reveal collagen fibre organisation, but are not accessible in vivo, since the detector must be placed behind the tissue. Backward-detected SHG is clinically accessible but difficult to interpret due to complex optical scattering effects.

This project investigates whether a Pix2Pix conditional generative adversarial network (cGAN) can learn to predict forward-SHG images directly from backward-SHG images, using a paired dataset of corneal images collected under varying mechanical pressure. This is a first step toward validating backward-SHG as a viable clinical imaging modality for non-invasive corneal assessment.

## Method

- **Architecture**: U-Net generator + patch-based discriminator (Pix2Pix)
- **Loss**: Adversarial loss (BCEWithLogitsLoss) + weighted L1 pixel loss (lambda = 100)
- **Training**: Adam optimiser (lr = 2e-4, betas = 0.5/0.999), batch size 16
- **Hardware**: MacBook Pro (Apple M4 Pro, 24GB unified memory), PyTorch MPS backend

## Contents

- `dataset.py` — dataset loading class for paired backward/forward images
- `model.py` — generator and discriminator architectures
- `train.py` — training script
- `test.py` — test-set evaluation
- `1_preprocessing.ipynb` — data preparation walkthrough
- `2_training_and_evaluation.ipynb` — training and evaluation walkthrough

## Results

_To be added once a stable, well-tuned model is available._

## Acknowledgements

Supervised by Dr Abby Wilson. Built on paired SHG imaging data collected by the research group.
