# AWQ & Quantization Demo for Neural Networks

This repository demonstrates various quantization strategies, including **Activation-Aware Weight Quantization (AWQ)**, for compressing neural networks. The code is based on a simple MLP trained on MNIST, but the ideas generalize to larger models and other domains.

---

## Why Quantization is Important

Quantization is essential for deploying neural networks on resource-constrained devices (e.g., mobile phones, embedded systems). By reducing the bitwidth of weights (e.g., from 32-bit float to 4-bit int), quantization:

- **Reduces memory footprint** (smaller model files, less RAM needed)
- **Speeds up inference** (faster computation, lower latency)
- **Lowers power consumption**
- **Enables real-time, on-device AI**

However, naive quantization can degrade accuracy. Advanced methods like AWQ aim to minimize this loss while maximizing compression.

---

## What is AWQ? (Activation-Aware Weight Quantization)

**AWQ** is a quantization strategy that takes into account both the weights and the activations of the network. The main idea, introduced in [AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), is:

- **Protect the most important channels**: Channels with the largest activations (or other importance metrics) are "protected" by scaling their weights before quantization.
- **Per-channel scaling**: Each channel can have its own scaling factor, allowing finer control and less quantization error where it matters most.
- **Flexible importance metrics**: The code supports various metrics to select important channels (activation, weight norm, variance, gradient, etc.).

This approach enables aggressive quantization (e.g., 4 bits) with minimal loss in model accuracy.

---

## Key Code and Explanation

Below is a complete example of the code, including training, evaluation, quantization, and model saving/loading. The quantization supports multiple "importance" metrics for channel protection.

