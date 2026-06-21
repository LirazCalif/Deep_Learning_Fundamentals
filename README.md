# Deep Learning Fundamentals

PyTorch implementations of core deep learning components built from
scratch, including backpropagation, custom optimizers, CNNs with
residual connections, a VAE, an RNN, and a Transformer encoder with
sliding window attention.

**Tech Stack:** Python · PyTorch · NumPy

## What I Built

Built on top of a provided course framework that handled data loading,
experiment running, and result logging. All model architectures,
training logic, and algorithms were implemented from scratch.

---

### Backpropagation Engine (`layers.py`)

A custom automatic differentiation system using abstract `Layer`
classes with explicit `forward` and `backward` methods. Implemented:

- `LeakyReLU` / `ReLU` — forward and gradient w.r.t. input
- `Linear` — forward pass and gradients w.r.t. weights, biases,
  and input
- `CrossEntropyLoss` — numerically stable implementation with
  log-sum-exp trick
- `Dropout` — training/eval mode switching, inverted scaling

Each layer caches intermediate values in `grad_cache` during the
forward pass for use in backpropagation.

---

### Optimizers (`optimizers.py`)

From-scratch implementations (no `torch.optim`):

- **SGD** — gradient step with L2 regularization
- **Momentum SGD** — velocity accumulation with configurable β
- **RMSProp** — per-parameter adaptive learning rate with
  exponential moving average of squared gradients

---

### MLP (`mlp.py`)

A general-purpose multi-layer perceptron that accepts arbitrary depth,
width, and activation functions (including `nn.Module` instances for
custom activations like Dropout+LayerNorm sequences).

---

### CNN & ResNet (`cnn.py`)

- **CNN** — configurable conv-pool architecture:
  `[(Conv → Act) × P → Pool] × (N/P)` followed by an MLP classifier
- **ResidualBlock** — dimension-preserving skip connections with
  optional BatchNorm and Dropout; uses a 1×1 projection shortcut when
  input/output channels differ
- **ResidualBottleneckBlock** — 1×1 → 3×3 → 1×1 bottleneck structure
- **ResNet** — ResidualBlock groups with pooling between each group
- **YourCNN** — custom architecture extending ResNet with:
  - Grouped convolutions (ResNeXt-style cardinality=32)
  - Bottleneck blocks with adaptive inner dimensions
  - `AdaptiveAvgPool2d` at the end of the feature extractor for
    input-size independence
  - MLP classifier with Dropout(0.4) + LayerNorm between layers

---

### VAE (`autoencoder.py`)

A Variational Autoencoder for image generation:

- **EncoderCNN** — 4-layer CNN (5×5 convolutions, stride-2
  downsampling, BatchNorm, ReLU) projecting to a latent feature map
- **DecoderCNN** — mirror architecture using `ConvTranspose2d` with
  `output_padding=1` to exactly recover spatial dimensions; output
  scaled to [−1, 1] with Tanh
- **VAE** — reparameterization trick with learned μ and log σ²
  projections; KL divergence loss + reconstruction loss (closed-form
  with diagonal covariance)

---

### Character-Level RNN (`charnn.py`)

Character-level language model with:

- One-hot encoding / decoding with vocabulary maps
- Sequence batching with next-character labels
- GRU and LSTM cell implementations with hidden state management
- Text generation via temperature sampling

---

### Transformer Encoder with Sliding Window Attention (`transformer.py`)

A Transformer encoder for sequence classification, with a custom
attention mechanism:

**Sliding Window Attention** (from Longformer)
Each token attends only to a local window of ±w/2 neighbors instead
of the full sequence. Implemented efficiently using `unfold` to
extract sliding windows from padded key/value tensors, then
`torch.einsum` for batched local dot products. Handles:
- Multi-head attention (`[Batch, Heads, SeqLen, Dim]`)
- Out-of-bounds masking at sequence boundaries
- Padding token masking (zero out pad query rows)

**Full Encoder**
- Token embedding + sinusoidal positional encoding
- N × `EncoderLayer`: sliding-window multi-head attention →
  Add & Norm → position-wise FFN (GELU) → Add & Norm
- CLS-token classification head with a 2-layer MLP

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
