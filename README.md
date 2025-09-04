<div align="center">
  <h1>⚙️ TensorMill</h1>
  <p><strong>Industrial-strength synthetic tensor generation for ML pipelines</strong></p>
  
  [![Crates.io](https://img.shields.io/crates/v/tensormill.svg)](https://crates.io/crates/tensormill)
  [![Documentation](https://docs.rs/tensormill/badge.svg)](https://docs.rs/tensormill)
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
tensormill --model-type gpt-oss-20b --size compact --output ./weights

# Generate a full model (13GB) with progress tracking
tensormill --model-type gpt-oss-20b --size full --output ./weights --progress
```

## 📚 Documentation

### Installation
**TLDR**: Install with `cargo install tensormill` or build from source with `cargo build --release`

For detailed installation instructions, dependency management, and platform-specific notes, see [docs/INSTALL.md](docs/INSTALL.md).

### Tutorial
**TLDR**: Use CLI for quick generation, library API for integration, supports compact/full models with deterministic seeds.

For comprehensive usage examples, advanced configurations, and integration patterns, see [docs/TUTORIAL.md](docs/TUTORIAL.md).

### Architecture
**TLDR**: Modular Rust design with parallel tensor generation, MXFP4 packing, and streaming SafeTensors export achieving ~540 MB/s throughput.

For technical details, module descriptions, and performance characteristics, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
- **Parameters**: ~20B total, 3.6B active
- **Format**: 13.76GB (full), 440MB (compact)
- **Generation**: ~40s (full), ~3s (compact)
- **MXFP4**: Expert weights packed at 4-bits (2 values/byte)

### GPT-OSS-120B
- **Parameters**: ~120B total
- **Format**: 65GB (full), 2GB (compact)
- **Generation**: ~2min (full), ~6s (compact)

## Features

- ✅ **Multi-model Support**: GPT-OSS-20B and GPT-OSS-120B
- ✅ **Format Flexibility**: Sharded, unsharded, original formats
- ✅ **MXFP4 Quantization**: Bit-exact 4-bit packing
- ✅ **HuggingFace Compatible**: Exact format match
- ✅ **Complete Metadata**: All required config and tokenizer files
- ✅ **Deterministic**: Reproducible with seed control
- ✅ **Parallel Processing**: Leverages all CPU cores

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

TensorMill is dual-licensed under MIT and Apache 2.0 licenses.

---

<div align="center">
  <strong>TensorMill - Grinding out tensors at industrial scale 🏭</strong>
</div>
