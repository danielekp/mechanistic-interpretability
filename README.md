# Mechanistic Interpretability Research

This repository contains implementations for mechanistic interpretability research, focusing on understanding the internal representations and circuits of transformer language models.

![Mondrian and mech interp](./mondrian-mechinter.png)

## Components

### Sparse Autoencoder (`sparse-autoencoder/`)

Implementation of sparse autoencoders for analyzing transformer activations. Features include:

- Training sparse autoencoders on MLP layer activations
- 8x expansion factor with L1 sparsity penalty
- Token-aligned activation extraction and analysis
- Neuron patching and intervention experiments
- Comprehensive training monitoring and visualization

See `sparse-autoencoder/README.md` for detailed documentation.

### Circuit Tracer Playground (`circuit-tracer-playground/`)

Safety-focused circuit discovery and analysis framework using ![circuit-tracer](https://github.com/safety-research/circuit-tracer) library. Features include:

- Automated circuit pattern mining across model layers
- Safety benchmark evaluation (deception, harmful content, manipulation, power-seeking)
- Contrasting example analysis for safety feature identification
- Interactive dashboard for circuit visualization
- Graph-based circuit representation and analysis

See `circuit-tracer-playground/README.md` for detailed documentation.
