# CycleGAN - Unpaired Image-to-Image Translation

CycleGAN is a type of Generative Adversarial Network (GAN) designed for image-to-image translation tasks without paired examples. This repository implements the CycleGAN architecture to learn transformations between two different image domains (e.g., horses ↔ zebras, summer ↔ winter, etc.).

## 📌 Features

- ✅ Unpaired image translation between two domains
- ✅ Cycle-consistency loss for preserving image content
- ✅ Identity loss for better color preservation
- ✅ Custom dataset support
- ✅ Training and testing scripts included

## 🧠 Architecture

CycleGAN uses two sets of:
- **Generators**: `G: X → Y`, `F: Y → X`
- **Discriminators**: `D_X`, `D_Y`

### Loss Functions:
- **Adversarial Loss**: Make translated images indistinguishable from real images.
- **Cycle Consistency Loss**: Ensure `x → y → x ≈ x` and `y → x → y ≈ y`.
- **Identity Loss**: Helps in color preservation by penalizing unnecessary changes.

## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/navyagupta01/CycleGan.git
cd CycleGan
