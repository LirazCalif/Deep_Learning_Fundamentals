# Deep Learning Fundamentals

PyTorch implementations of core deep learning components, 
including backpropagation, custom optimizers, CNNs with
residual connections, a VAE, an RNN, and a Transformer encoder with
sliding window attention.

**Tech Stack:** Python · PyTorch · NumPy

## What I Built

Built on top of a framework that provided base class interfaces, data
loading utilities, and function signatures. All mathematical logic,
architectural decisions, and algorithm implementations are my own.

**Backpropagation engine** (`layers.py`)
Forward and backward passes for LeakyReLU, ReLU, Linear,
CrossEntropyLoss, and Dropout from scratch. Each layer caches
intermediate values during forward for use in backprop — no
torch.autograd used.

**Optimizers** (`optimizers.py`)
SGD with L2 regularization, Momentum SGD with velocity accumulation,
and RMSProp with per-parameter adaptive rates. No torch.optim.

**MLP** (`mlp.py`)
General-purpose multi-layer perceptron with configurable depth, width,
and activation functions, including support for nn.Module instances as
activations.

**CNN & ResNet** (`cnn.py`)
Configurable CNN with [(Conv→Act)×P→Pool]×(N/P) architecture.
ResidualBlock with 1×1 projection shortcuts, ResidualBottleneckBlock
(1×1→3×3→1×1), and ResNet with grouped residual blocks and pooling.
Extended with YourCNN — a custom architecture adding grouped
convolutions (ResNeXt-style, cardinality=32), bottleneck blocks with
adaptive inner dimensions, AdaptiveAvgPool2d for input-size
independence, and a classifier MLP with Dropout(0.4)+LayerNorm between
layers.

**VAE** (`autoencoder.py`)
Reparameterization trick with learned μ and log σ² projections. VAE
loss combining reconstruction error and closed-form KL divergence with
diagonal covariance.

**Character-level RNN** (`charnn.py`)
One-hot encoding/decoding, sequence batching with next-character
labels, GRU and LSTM cell forward passes, temperature sampling.

**Transformer encoder with sliding window attention** (`transformer.py`)
Each token attends only to a local window of ±w/2 neighbors.
Implemented efficiently using unfold to extract sliding windows,
torch.einsum for batched local dot products, and scatter_add to
reconstruct the full attention matrix. Handles multi-head, boundary
masking, and padding masks. Full encoder stack with sinusoidal
positional encoding, LayerNorm, GELU FFN, and CLS classification head.

---

## Project Structure

```
src/
├── layers.py         # Backprop engine: ReLU, Linear, CrossEntropy, Dropout
├── optimizers.py     # SGD, Momentum SGD, RMSProp
├── mlp.py            # General-purpose MLP
├── cnn.py            # CNN, ResNet, ResidualBlock, YourCNN (grouped + bottleneck)
├── training.py       # Trainer loop with early stopping and checkpointing
├── classifier.py     # ArgMax and Binary classifiers with threshold selection
├── experiments.py    # CNN architecture sweep experiments on CIFAR-10
├── autoencoder.py    # EncoderCNN, DecoderCNN, VAE, VAE loss
├── charnn.py         # Char-level RNN: encoding, batching, GRU/LSTM, sampling
└── transformer.py    # Sliding window attention, MultiHeadAttention, Encoder
results/
└── *.json            # CNN architecture sweep results (depth, width, kernel size)
```

## Running

Install dependencies:

```bash
pip install torch torchvision numpy
```

Run CNN architecture experiments on CIFAR-10:

```bash
python src/experiments.py --model resnet --depth 8 --width 64
```
