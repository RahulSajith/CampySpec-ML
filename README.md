# CampySpec-ML: Uncertainty-Aware Deep Learning for Campylobacter Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

&gt; **1D-CNN with Monte Carlo Dropout and Attention Mechanisms for Spectral-Based Pathogen Detection**

## Overview

This project implements an uncertainty-aware deep learning approach for detecting *Campylobacter* from absorbance spectroscopy data. The model combines:

- **1D Convolutional Neural Networks** for spectral feature extraction
- **Monte Carlo Dropout** for epistemic uncertainty quantification
- **Attention Mechanisms** for interpretable wavelength importance
- **Batch Normalization** for stable training

### Key Innovation

Unlike traditional machine learning approaches (SVM: ~76%, RF: ~74%, LR: ~72%) that provide point predictions, our model provides **uncertainty-calibrated predictions** — critical for food safety applications where false negatives can have severe public health consequences.

## Research Context

This work is directly inspired by the research interests of **Dr. Xiaonan Lu's lab at McGill University**, which focuses on:

- Nano-biosensor and biophotonics for rapid food hazard detection
- *Campylobacter*-associated food safety
- Intelligent food safety and authentication using AI, big data, and IoT
- Application of deep machine learning to improve food chain safety

## Dataset

The project uses **synthetic spectral data** generated to replicate the characteristics of the MPN-Spectro-ML dataset (Zenodo: 10.5281/zenodo.10547727):

| Property | Value |
|----------|-------|
| Wavelength Range | 400-700 nm (visible spectrum) |
| Features | 400 wavelength points |
| Samples | 100 (55 positive, 45 negative) |
| Diagnostic Peaks | 540-542 nm, 575-576 nm |
| Task | Binary classification (Campylobacter+ / Campylobacter-) |

## Model Architecture
CampyCNN (1.69M parameters)
├── Conv1: 1 → 32 filters, k=7, MaxPool
├── Conv2: 32 → 64 filters, k=5, MaxPool
├── Conv3: 64 → 128 filters, k=3, MaxPool
├── Attention: 1D conv + Softmax (wavelength importance)
├── FC1: 6400 → 256, BatchNorm, ReLU
├── MC Dropout (p=0.3) ← Always active!
├── FC2: 256 → 64, BatchNorm, ReLU
├── MC Dropout (p=0.3)
└── FC3: 64 → 1 (sigmoid)



## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **100.00%** |
| ROC-AUC | **1.0000** |
| Mean Epistemic Uncertainty | **0.0449 ± 0.004** |
| Best Epoch | 1 |
| Training Time | ~2 minutes (CUDA) |

### Key Advantages Over Classical ML

1. **Uncertainty Quantification**: Monte Carlo Dropout provides epistemic uncertainty for each prediction, enabling reliable decision-making
2. **Interpretability**: Attention maps highlight diagnostically relevant wavelengths (540-542nm, 575-576nm)
3. **Deep Feature Learning**: Captures non-linear spectral patterns invisible to SVM/RF/LR

## Installation

```bash
# Clone repository
git clone https://github.com/RahulSajith/CampySpec-ML.git
cd CampySpec-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
