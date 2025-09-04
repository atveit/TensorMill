# Installation Guide

This guide covers installation of TensorMill for GPT-OSS on various platforms.

## Prerequisites

### Required
- **Rust**: Version 1.70 or higher
- **Git**: For cloning the repository
- **C Compiler**: For building native dependencies

### Recommended
- **16GB RAM**: For generating full models
- **15GB disk space**: For full GPT-OSS-20B output
- **75GB disk space**: For full GPT-OSS-120B output

## Quick Install

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/example/tensormill
cd tensormill

# Build and install
cargo install --path .

# Verify installation
tensormill --version
```

### From Crates.io

```bash
# Once published to crates.io
cargo install tensormill

# Verify installation
tensormill --version
```

## Platform-Specific Instructions

### macOS

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install build tools
xcode-select --install

# Clone and build
git clone https://github.com/example/tensormill
cd tensormill
cargo build --release

# Add to PATH (optional)
echo 'export PATH="$PATH:'"$(pwd)"'/target/release"' >> ~/.zshrc
source ~/.zshrc
```

### Linux (Ubuntu/Debian)

```bash
# Install prerequisites
sudo apt update
sudo apt install -y build-essential git curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Clone and build
git clone https://github.com/example/tensormill
cd tensormill
cargo build --release

# Install system-wide (optional)
sudo cp target/release/tensormill /usr/local/bin/
```

### Linux (RHEL/CentOS/Fedora)

```bash
# Install prerequisites
sudo dnf groupinstall "Development Tools"
sudo dnf install git

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Clone and build
git clone https://github.com/example/tensormill
cd tensormill
cargo build --release
```

### Windows

```powershell
# Install Rust (download from https://rustup.rs)
# Run the installer and follow prompts

# Install Git for Windows from https://git-scm.com/download/win

# Clone and build
git clone https://github.com/example/tensormill
cd tensormill
cargo build --release

# Add to PATH
$env:Path += ";$(pwd)\target\release"
```

## Build Options

### Release Build (Recommended)

```bash
cargo build --release
```

### Debug Build

```bash
cargo build
```

### With Python Bindings

```bash
cargo build --release --features python
```

### Optimized for Your CPU

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Docker Installation

### Using Pre-built Image

```bash
docker pull ghcr.io/example/tensormill:latest
docker run -v $(pwd)/output:/output tensormill -o /output
```

### Building Docker Image

```dockerfile
# Dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/tensormill /usr/local/bin/
ENTRYPOINT ["tensormill"]
```

```bash
# Build and run
docker build -t tensormill .
docker run -v $(pwd)/output:/output tensormill -o /output
```

## Python Library Installation

### Via pip (once published)

```bash
pip install tensormill-for-gpt-oss
```

### From Source with Python Bindings

```bash
# Install maturin
pip install maturin

# Clone repository
git clone https://github.com/example/tensormill
cd tensormill

# Build and install Python module
maturin develop --release --features python

# Test installation
python -c "import tensormill; print(tensormill.__version__)"
```

## Verification

### CLI Verification

```bash
# Check version
tensormill --version

# Run help
tensormill --help

# Generate test model
tensormill -t gpt-oss-20b -s compact -o /tmp/test-model

# Verify output
ls -la /tmp/test-model/
```

### Library Verification

```rust
// test.rs
use tensormill::{ModelConfig, ModelType, ModelFormat, ModelSize};

fn main() {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Unsharded,
        ModelSize::Compact,
    );
    println!("Config created: {:?}", config);
}
```

```bash
rustc test.rs -L target/release/deps
./test
```

## Troubleshooting

### Common Issues

#### 1. Rust Not Found
```bash
# Error: rustc not found
# Solution: Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### 2. Out of Memory
```bash
# Error: memory allocation failed
# Solution: Use compact model size
tensormill -s compact -o ./output
```

#### 3. Permission Denied
```bash
# Error: permission denied
# Solution: Use user directory or sudo
tensormill -o ~/output
# or
sudo tensormill -o /opt/models
```

#### 4. Linking Errors on Linux
```bash
# Error: linking with cc failed
# Solution: Install build essentials
sudo apt install build-essential  # Ubuntu/Debian
sudo dnf groupinstall "Development Tools"  # Fedora/RHEL
```

#### 5. Slow Compilation
```bash
# Use parallel compilation
cargo build --release -j 4  # Use 4 cores
```

## Performance Optimization

### Compile-time Optimizations

```toml
# Add to Cargo.toml for maximum performance
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

### Runtime Optimizations

```bash
# Use all available CPU cores
RAYON_NUM_THREADS=$(nproc) tensormill -o ./output

# Set specific thread count
RAYON_NUM_THREADS=8 tensormill -o ./output
```

## Uninstallation

### Cargo Installation

```bash
cargo uninstall tensormill
```

### Manual Installation

```bash
# Remove binary
rm /usr/local/bin/tensormill
# or
rm ~/.cargo/bin/tensormill

# Remove source
rm -rf /path/to/tensormill
```

### Python Module

```bash
pip uninstall tensormill
```

## Next Steps

- Read the [TUTORIAL.md](TUTORIAL.md) for usage examples
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- See the [README.md](README.md) for quick start guide