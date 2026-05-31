# 1.58-Bit Ternary LLM for OpenAI Parameter Golf

This repository contains a highly optimized, parameter-constrained language model submitted for the **OpenAI Model Craft Challenge: Parameter Golf**. The challenge requires training a model that minimizes validation loss (Bits Per Byte / BPB) on the FineWeb dataset, with a strict limitation: the final artifact (code + compressed weights) must strictly fit within a **16.0 MB** limit and train in under 10 minutes on an 8xH100 setup.

This implementation introduces a **1.58-bit (Ternary) quantization architecture**. This extreme compression paradigm allows scaling the parameter count to massive dimensions (~60M+ parameters) while remaining under the 16.0 MB file size limit, optimizing the $L(N)$ trade-off far beyond what is possible with standard 16-bit precision.

## Technical Approach & Architecture

To maximize the number of parameters N within the constrained byte limit, the standard 16-bit linear layers were replaced with a custom Ternary architecture inspired by BitNet b1.58.

### 1. Extreme Quantization & Compression
* **Ternary Matrices:** The weights in the Attention (`c_q`, `c_k`, `c_v`, `proj`) and MLP (`fc`, `proj`) layers are constrained to `{-1, 0, 1}`.
* **8-bit Embeddings:** Token embeddings are quantized using a min-max `int8` approach.
* **Entropy-Optimized Packing (Byte-Shuffling):** Before applying Brotli/LZMA compression, the raw quantized bytes are pre-processed using a custom stride-based byte-shuffling algorithm. This aligns identical discrete states in memory, artificially lowering the Shannon entropy and allowing the compressor to pack significantly more parameters into the strict 16.0 MB boundary.

### 2. Quantization-Aware Training (QAT) & Structural Stability
The model does not rely on post-training compression; it natively trains in the quantized space from step zero.
* **Straight-Through Estimator (STE):** Implemented via `(w_quant * w_scale) + w_float - w_float.detach()`. This mathematically allows the gradient to bypass the non-differentiable rounding functions during the backward pass, successfully updating the underlying latent float weights.
* **Per-Token / Absmean Scaling:** To mitigate precision loss from ternary rounding, weights are dynamically scaled using absolute mean scaling, and activations are scaled per-token to handle mathematical outliers.
* **Parallel Residual Streams:** Standard sequential Transformer blocks compound quantization errors as the signal passes through mathematically lossy Attention and MLP layers in series. This model utilizes **Parallel Residuals** (Attention and MLP compute concurrently from the same input and add to the stream simultaneously), acting as a gradient highway that prevents signal degradation in the 1.58-bit forward/backward pass.
* **RMSNorm:** Used exclusively across the network to maintain stable variance distributions, which is critical for preventing gradient vanishing in highly quantized networks.

### 3. Optimizer Engineering (Ternary-Specific AdamW)
Training 1.58-bit models requires a departure from standard continuous-weight optimization. The model utilizes a highly customized **AdamW** setup tailored for discrete latent clusters:
* **Zero Weight Decay on Matrices:** Weight decay for all ternary matrix parameters is strictly set to `0.0`. Traditional weight decay pulls latent float weights toward zero, causing the ternary rounding function to permanently collapse weights into the `0` state (creating dead neurons).
* **Decoupled Learning Rates:** Matrix learning rates are tuned and decoupled from scalar parameters (norms, biases) to ensure stable convergence without overshooting the ternary boundaries.
