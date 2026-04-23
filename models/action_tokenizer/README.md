# FACT Action Tokenizer

VQ-VAE with flow matching for discretizing continuous action chunks, following the FACT paper architecture.

## Quick Start

### Installation

```bash
source .venv/bin/activate
cd /path/to/GTA-VLA
```

### Test the Implementation

```bash
# Test individual components
python -m models.action_tokenizer.mmdit_block
python -m models.action_tokenizer.codebook
python -m models.action_tokenizer.fact_encoder
python -m models.action_tokenizer.fact_decoder
python -m models.action_tokenizer.fact_tokenizer

# Or use the test script
python scripts/test_tokenizer.py
```

### Train the Tokenizer

```bash
# Train on libero dataset
python scripts/train_action_tokenizer.py \
    --data_path data/libero_mix_meta.json \
    --output_dir runnings/fact_tokenizer \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --iters 100000

# Monitor training with wandb
# Check runnings/fact_tokenizer/train.log for training progress
```

### Configuration

Edit `configs/action_tokenizer/fact_ee6d.json` to customize:
- Model architecture (codebook size, latent dimensions, layers)
- Training hyperparameters (learning rate, batch size, iterations)
- Data settings (action mode, control mode)

## Architecture

### Encoder
- Transformer-based encoder
- Compresses action chunks (B, 30, 20) → latents (B, 8, 256)
- Uses positional embedding + self-attention + temporal pooling

### Vector Quantization
- Codebook size: 1024 codes, dimension: 256
- EMA updates for stable training
- Commitment loss for encoder-codebook alignment

### Decoder (Flow Matching with MMDiT)
- MMDiT blocks with adaptive LayerNorm
- Self-attention + cross-attention to quantized latents
- Predicts velocity for linear flow matching
- Reconstruction: x_0 = x_t - t * v

## Usage in Code

```python
from models.action_tokenizer import FACTTokenizer

# Create tokenizer
tokenizer = FACTTokenizer(
    action_dim=20,
    num_actions=30,
    codebook_size=1024,
)

# Encode actions to discrete tokens
z_q, indices, vq_loss = tokenizer.encode(actions)  # actions: [B, 30, 20]
# indices: [B, 8] discrete tokens

# Decode tokens back to actions
reconstructed = tokenizer.decode(z_q, num_steps=10)  # [B, 30, 20]

# Or decode from indices directly
reconstructed = tokenizer.decode_indices(indices, num_steps=10)
```

## Training Metrics

The training script logs:
- `total_loss`: Combined flow matching + VQ loss
- `fm_loss`: Flow matching reconstruction loss
- `vq_loss`: VQ commitment + codebook loss
- `pos_loss`, `rot_loss`, `gripper_loss`: Per-component losses
- `recon_mse`: Overall reconstruction MSE
- `pos_mse`, `rot_mse`, `gripper_acc`: Per-component metrics
- `codebook_usage`: Percentage of codebook codes used

## Integration with VLA (Phase 2)

After training the tokenizer, integrate it into the VLA model:

1. Load trained tokenizer checkpoint
2. Replace continuous action head with discrete token prediction
3. Use autoregressive decoding for action generation

See `vla-integration` todo for implementation details.

## Troubleshooting

### Segmentation Fault
If you encounter segfaults during testing, it's likely due to TensorFlow/PyTorch conflicts in the environment. Try:
```bash
export TF_CPP_MIN_LOG_LEVEL=3
# Or test individual modules instead of the full tokenizer
```

### CUDA Out of Memory
Reduce batch size or latent dimensions in the config file.

### Low Codebook Usage
- Increase commitment_cost (default: 0.25)
- Use more training data
- Increase codebook size


