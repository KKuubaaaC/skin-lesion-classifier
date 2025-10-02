## Overview

This project implements a convolutional neural network trained on the HAM10000 dataset to classify seven types of skin lesions with high accuracy. The model is optimized for production deployment using ONNX Runtime and features a professional Streamlit interface for real-time inference.

### Key Features

- High Performance: 82-85% classification accuracy with sub-100ms inference time
- Production Ready: ONNX-optimized model with CPU-efficient inference
- Interactive UI: Professional web interface with real-time visualization
- Comprehensive Analysis: Confidence scores, risk assessment, and clinical recommendations
- Extensible Architecture: Clean, documented code following software engineering best practices
- Docker Support: Containerized deployment for consistent environments
- CI/CD Pipeline: Automated testing and quality checks via GitHub Actions

## Model Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 82-85% |
| Top-3 Accuracy | ~94% |
| Inference Time (CPU) | 50-100ms |
| Model Size | ~50MB |

### Classification Categories

The model classifies seven types of skin lesions:

1. **Actinic Keratoses (AKIEC)** - Pre-cancerous lesions caused by sun damage
2. **Basal Cell Carcinoma (BCC)** - Most common form of skin cancer
3. **Benign Keratosis (BKL)** - Non-cancerous skin growths
4. **Dermatofibroma (DF)** - Benign fibrous skin nodules
5. **Melanoma (MEL)** - Most dangerous form of skin cancer
6. **Melanocytic Nevi (NV)** - Common moles, typically benign
7. **Vascular Lesions (VASC)** - Blood vessel related skin changes
