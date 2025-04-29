# ARM64 Autoencoder Anomaly Detector

This repository contains an ARM64 (AArch64) NEON-optimized implementation of a simple autoencoder for unsupervised anomaly detection. The model is defined and run in two fully-connected layers (encoder/decoder) in hand-written assembly, with a C driver that:

1. Initializes toy weights and input  
2. Runs the encoder (ReLU activation)  
3. Runs the decoder (linear)  
4. Computes reconstruction MSE  
5. Flags anomalies based on a threshold  

---

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Building](#building)  
- [Usage](#usage)  
- [Code Structure](#code-structure)  
- [Assembly Kernels](#assembly-kernels)  
- [License](#license)  

---

## Prerequisites

- **Clang/LLVM** targeting `arm64-apple-darwin` (or adjust for your platform)  
- **GNU Make** (optional)  
- AArch64-capable hardware or emulator (e.g., Apple Silicon Mac, QEMU)  

---

## Building

```bash
# Assemble the NEON routines
clang -target arm64-apple-darwin -c autoenc.s -o autoenc.o

# Compile the C driver and link
clang -target arm64-apple-darwin -O3 main.c autoenc.o -o autoenc_test
