# Mechanistic Interpretability Research

This repository contains implementations for mechanistic interpretability research, focusing on understanding the internal representations and circuits of transformer language models.

## Components

### Sparse Autoencoder (`sparse-autoencoder/`)

Implementation of sparse autoencoders for analyzing transformer activations. Features include:

- Training sparse autoencoders on MLP layer activations
- 8x expansion factor with L1 sparsity penalty
- Token-aligned activation extraction and analysis
- Neuron patching and intervention experiments
- Comprehensive training monitoring and visualization

See `sparse-autoencoder/README.md` for detailed documentation.

### Circuit Tracer (`circuit-tracer/`)

Safety-focused circuit discovery and analysis framework. Features include:

- Automated circuit pattern mining across model layers
- Safety benchmark evaluation (deception, harmful content, manipulation, power-seeking)
- Contrasting example analysis for safety feature identification
- Interactive dashboard for circuit visualization
- Graph-based circuit representation and analysis

See `circuit-tracer/README.md` for detailed documentation.

## Overview

This repository provides tools for two complementary approaches to mechanistic interpretability:

1. **Sparse Autoencoders**: Decompose model activations into interpretable sparse features
2. **Circuit Tracing**: Identify and analyze computational circuits responsible for specific behaviors

Both approaches aim to understand how language models process information and make decisions, with particular focus on safety-relevant behaviors. 