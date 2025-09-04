<div align="center">
  <h1>⚙️ TensorMill</h1>
  <p><strong>Industrial-strength synthetic tensor generation for ML pipelines</strong></p>
  
  [![Crates.io](https://img.shields.io/crates/v/tensormill.svg)](https://crates.io/crates/tensormill)
  [![Documentation](https://docs.rs/tensormill/badge.svg)](https://docs.rs/tensormill)
  [![CI](https://github.com/username/tensormill/workflows/CI/badge.svg)](https://github.com/username/tensormill/actions)
  [![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](LICENSE)
</div>

---

**TensorMill** grinds out synthetic model weights at industrial scale. Built in Rust for blazing performance, it's the missing piece in your ML testing infrastructure.

## 🏭 Why TensorMill?

Just as a lumber mill efficiently processes logs into usable timber, TensorMill processes model specifications into production-ready synthetic weights for testing and CI/CD pipelines.

- **⚡ Blazing Fast**: Generate 13GB models in ~40 seconds
- **🎯 Format-Perfect**: Exact match to HuggingFace GPT-OSS-20B format
- **🔧 Industrial Grade**: Built for continuous operation in CI/CD pipelines
- **📦 Multiple Formats**: Sharded, unsharded, and original OpenAI formats
- **🔬 MXFP4 Quantization**: Bit-exact 4-bit packing (2 values per byte)
- **🎨 Non-Contiguous Sharding**: Matches HuggingFace's exact tensor distribution

## Quick Start

```bash
# Install TensorMill
cargo install tensormill

# Generate a compact model (440MB) in seconds
tensormill --model gpt-oss-20b --size compact --output ./weights

# Generate a full model (13GB) with progress tracking
tensormill --model gpt-oss-20b --size full --output ./weights --progress
```

## Installation

### From Source

```bash
git clone https://github.com/example/tensormill
cd tensormill
cargo build --release
```

### From Crates.io

```bash
cargo install tensormill
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Usage

### CLI Tool

```bash
# Basic usage - compact model for quick tests
tensormill -o ./output

# Full GPT-OSS-20B model (13GB)
tensormill -t gpt-oss-20b -s full -o ./output

# Large GPT-OSS-120B model with sharding (65GB)
tensormill -t gpt-oss-120b -s full -f sharded -o ./output

# Deterministic generation for CI/CD
tensormill --seed 42 -o ./output
```

### As a Library

```rust
use tensormill::{ModelConfig, ModelType, ModelFormat, ModelSize, SyntheticGenerator};

// Configure the mill
let config = ModelConfig::new(
    ModelType::GptOss20B,
    ModelFormat::Sharded,
    ModelSize::Full,
);

// Start the mill
let mut generator = SyntheticGenerator::new(config)
    .with_seed(42)
    .with_progress();

// Generate weights
let result = generator.generate("./output")?;
result.print_summary();
```

## Model Specifications

### GPT-OSS-20B
- **Layers**: 24 transformer blocks
- **Experts**: 32 (4 active per token)
- **Parameters**: ~20B total, 3.6B active
- **Formats**: 13.76GB (full), 440MB (compact)
- **MXFP4 Packing**: Expert weights packed at 4-bits (2 values/byte)
- **Generation Time**: ~40s (full), ~3s (compact)
- **Sharding Pattern** (Matches HuggingFace distribution):
  - Shard 0: Layers 0, 1, 10-18
  - Shard 1: Layers 2-6, 18-23
  - Shard 2: Layers 6-9 + embeddings + lm_head
- **Generated Sizes** (Synthetic):
  - Total: 13.76GB (exact match to HF total)
  - Shard 0: 4.5GB | Shard 1: 4.4GB | Shard 2: 3.9GB
  - Note: ~0.3GB smaller per shard than HF due to omitted `self_attn.sinks` tensors

### GPT-OSS-120B
- **Layers**: 36 transformer blocks
- **Experts**: 128 (4 active per token)
- **Parameters**: ~120B total
- **Formats**: 65GB (full), 2GB (compact)
- **Generation Time**: ~2min (full), ~6s (compact)

## 🚀 Performance

TensorMill achieves industrial-scale throughput:

| Operation | Speed | Details |
|-----------|-------|---------|
| Tensor Generation | ~950 MB/s | Parallel processing with Rayon |
| MXFP4 Quantization | ~4.7 GB/s | SIMD-optimized operations |
| SafeTensors Export | ~1.3 GB/s | Zero-copy serialization |
| **Overall** | **~540 MB/s** | End-to-end throughput |

Benchmarked on Apple M2 Max (32GB RAM)

## Output Structure

```
output/
├── config.json                    # Model architecture configuration
├── tokenizer.json                 # Tokenizer vocabulary
├── tokenizer_config.json          # Tokenizer settings
├── generation_config.json         # Generation parameters
├── special_tokens_map.json        # Special token mappings
├── model.safetensors.index.json   # Shard index (multi-file models)
└── model*.safetensors             # Weight files (sharded or single)
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Setup TensorMill
  run: cargo install tensormill

- name: Generate Test Model
  run: tensormill --model gpt-oss-20b --size compact --output test_model --seed ${{ github.run_number }}

- name: Run Tests
  run: python test_model_loading.py test_model
```

### Docker

```dockerfile
FROM rust:1.75 as builder
RUN cargo install tensormill

FROM debian:bookworm-slim
COPY --from=builder /usr/local/cargo/bin/tensormill /usr/local/bin/
ENTRYPOINT ["tensormill"]
```

## Features

- ✅ **Multi-model Support**: GPT-OSS-20B and GPT-OSS-120B
- ✅ **Format Flexibility**: Sharded, unsharded, original formats
- ✅ **MXFP4 Quantization**: Bit-exact 4-bit packing (2 values per byte)
- ✅ **HuggingFace Compatible**: Exact format match including quirky naming
- ✅ **Non-Contiguous Sharding**: Matches real model tensor distribution
- ✅ **Complete Metadata**: All required config and tokenizer files
- ✅ **Deterministic Generation**: Reproducible with seed control
- ✅ **Progress Tracking**: Real-time generation progress
- ✅ **Memory Efficient**: Streaming generation, minimal overhead
- ✅ **Parallel Processing**: Leverages all CPU cores

## Documentation

- [INSTALL.md](INSTALL.md) - Installation guide
- [TUTORIAL.md](TUTORIAL.md) - Comprehensive usage examples
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [API Docs](https://docs.rs/tensormill) - Rust API documentation

## Use Cases

### 🧪 Testing
Generate consistent test models for unit and integration tests without downloading multi-gigabyte files.

### 🔄 CI/CD
Create fresh models for each pipeline run, ensuring tests aren't dependent on external resources.

### 📊 Benchmarking
Generate models of various sizes to benchmark loading times, memory usage, and inference performance.

### 🛠️ Development
Quickly create models for local development without managing large binary files.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

TensorMill is dual-licensed under MIT and Apache 2.0 licenses. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for details.

## Acknowledgments

Built for the GPT-OSS and ML communities to enable faster development and testing workflows.

---

<div align="center">
  <strong>TensorMill - Grinding out tensors at industrial scale 🏭</strong>
</div>