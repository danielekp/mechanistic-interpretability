# Sparse Autoencoder for Mechanistic Interpretability

This directory contains implementations for training sparse autoencoders (SAEs) on transformer activations and using them for neuron patching experiments. The primary focus is on analyzing the internal representations of language models through sparse feature decomposition.

## Core Components

### SAE Training (`sae_training.ipynb`)

The main notebook implements a complete pipeline for training sparse autoencoders on transformer activations. The training process follows these key steps:

#### Model and Layer Configuration
- **Target Model**: Google Gemma-3-1B-IT (configurable to other models)
- **Target Layer**: MLP down projection layers (configurable layer index)
- **Activation Dimension**: 1152 (varies by model architecture)
- **Layer Selection**: Focuses on `model.layers.{layer_idx}.mlp.down_proj` for MLP output activations

#### Dataset Processing
- **Dataset**: WikiText-103-raw-v1 (3M activations by default)
- **Sequence Length**: 128 tokens maximum
- **Token Alignment**: Precise alignment between activations and tokens using attention masks
- **Normalization**: Unit norm normalization with mean centering
- **Filtering**: Removes BOS tokens and padding tokens to ensure clean activation-token pairs

#### SAE Architecture
- **Expansion Factor**: 8x (9216 hidden dimensions for 1152 input)
- **Architecture**: Simple encoder-decoder with ReLU activation
- **Initialization**: Zero initialization for decoder bias
- **Loss Function**: MSE reconstruction + L1 sparsity penalty

#### Training Configuration
- **Batch Size**: 512
- **Learning Rate**: 3e-3 with cosine decay scheduling
- **Epochs**: 300
- **L1 Coefficient**: 3e-4 (sparsity penalty strength)
- **Learning Rate Schedule**: 
  - 500 step warmup
  - Cosine decay to 50% of initial LR
  - Final minimum LR maintained

#### Training Process
The training implements a sophisticated learning rate schedule with warmup and decay phases. The loss function balances reconstruction accuracy (MSE) against sparsity (L1 penalty), encouraging the SAE to learn sparse, interpretable features.

Key monitoring metrics include:
- Total loss, MSE loss, L1 loss per epoch
- L0 norm (average active features per sample)
- Feature activity statistics
- Dead feature analysis

#### Analysis and Visualization
- **Feature Strength Analysis**: Maximum activation analysis across features
- **Token Context Analysis**: Top activating examples for each feature
- **Visualization**: 2D heatmaps and 3D bar plots of feature strengths
- **Report Generation**: PDF reports with training curves and feature analysis

#### Model Persistence
- Saves SAE weights and activation mean for later use
- TensorBoard logging for training metrics
- Comprehensive experiment reports

### Neuron Patching (`patch_neurons.ipynb`)

The patching notebook enables intervention experiments on trained SAEs by replacing MLP layers in the original model.

#### SAE Integration
- **Wrapper Class**: `SAEIntervenableMLP` provides a drop-in replacement for MLP layers
- **Normalization Consistency**: Applies the same normalization used during training
- **Patching Interface**: Flexible patching functions for feature manipulation

#### Patching Operations
Three primary patching functions are implemented:
- **Ablation**: Set specific features to zero
- **Amplification**: Scale feature activations by a factor
- **Value Setting**: Set features to specific values

#### Model Replacement
- **Layer Replacement**: Replaces target MLP layer with SAE wrapper
- **Sequential Integration**: Maintains original MLP followed by SAE intervention
- **Device Management**: Ensures proper device placement

#### Experimental Workflow
1. Load trained SAE checkpoint
2. Replace target layer in model
3. Apply patching functions
4. Generate text with modified model
5. Compare outputs between original and patched models

## Training Decisions and Rationale

### Architecture Choices
- **8x Expansion**: Balances feature richness with computational efficiency
- **ReLU Activation**: Standard choice for sparse coding, encourages feature specialization
- **L1 Penalty**: Proven effective for inducing sparsity in autoencoders

### Data Processing Decisions
- **Unit Norm**: Ensures consistent scale across activations
- **Token Alignment**: Critical for interpretability - each feature corresponds to specific tokens
- **Large Dataset**: 3M activations provides sufficient coverage of model behavior

### Training Strategy
- **Warmup Schedule**: Prevents early training instability
- **Cosine Decay**: Smooth learning rate reduction maintains training stability
- **L1 Coefficient Tuning**: 3e-4 provides good sparsity without excessive reconstruction loss

### Evaluation Approach
- **Multiple Metrics**: Tracks both reconstruction quality and sparsity
- **Feature Analysis**: Identifies dead features and activation patterns
- **Token-Level Analysis**: Links features to specific linguistic patterns

## Usage

1. **Training**: Run `sae_training.ipynb` with desired configuration
2. **Patching**: Use `patch_neurons.ipynb` with trained SAE checkpoints
3. **Analysis**: Review generated reports and TensorBoard logs

## File Structure

- `sae_training.ipynb`: Complete SAE training pipeline
- `patch_neurons.ipynb`: Neuron patching and intervention experiments
- `secret_tokens.py`: API access tokens (not included in repository)
- `runs/`: Training logs and model checkpoints
- `*.pth`: Saved SAE models with weights and normalization parameters

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- Datasets
- TensorBoard
- Matplotlib
- NumPy

The implementation prioritizes interpretability and experimental flexibility, enabling detailed analysis of transformer internal representations through sparse feature decomposition. 