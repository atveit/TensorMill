# TensorMill Tutorial

This tutorial covers common use cases and advanced features of TensorMill for GPT-OSS.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [CI/CD Integration](#cicd-integration)
3. [Testing Workflows](#testing-workflows)
4. [Advanced Usage](#advanced-usage)
5. [Python Integration](#python-integration)
6. [Performance Tuning](#performance-tuning)
7. [Custom Configurations](#custom-configurations)

## Basic Usage

### Generate Your First Model

```bash
# Simplest usage - compact GPT-OSS-20B
tensormill -o my_first_model

# Check the output
ls -la my_first_model/
```

Expected output:
```
my_first_model/
├── config.json                 # 2.1 KB
├── generation_config.json      # 145 B
├── model.safetensors          # ~440 MB
├── special_tokens_map.json    # 551 B
├── tokenizer.json             # 1.2 KB
└── tokenizer_config.json      # 487 B
```

### Generate Different Model Sizes

```bash
# Compact model for quick tests (~440MB)
tensormill -t gpt-oss-20b -s compact -o compact_model

# Full model for realistic testing (~13GB)
tensormill -t gpt-oss-20b -s full -o full_model

# Large model for stress testing (~65GB)
tensormill -t gpt-oss-120b -s full -o large_model
```

### Control Output Format

```bash
# Single file (unsharded)
tensormill -f unsharded -o single_file_model

# Multiple files (sharded) - HuggingFace format
tensormill -f sharded -o sharded_model

# OpenAI original format
tensormill -f original -o openai_format_model
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Model Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Install TensorMill
      run: |
        git clone https://github.com/example/tensormill
        cd tensormill
        cargo install --path .
        
    - name: Generate Test Model
      run: |
        tensormill -t gpt-oss-20b -s compact -o test_model --seed 42
        
    - name: Run Model Loading Tests
      run: |
        python tests/test_model_loading.py test_model
        
    - name: Cache Model (optional)
      uses: actions/cache@v3
      with:
        path: test_model
        key: test-model-${{ hashFiles('**/Cargo.lock') }}
```

### GitLab CI

```yaml
stages:
  - build
  - test

variables:
  MODEL_PATH: "test_model"

generate-model:
  stage: build
  script:
    - cargo install tensormill
    - tensormill -s compact -o $MODEL_PATH --seed 42
  artifacts:
    paths:
      - $MODEL_PATH/
    expire_in: 1 day

test-model:
  stage: test
  dependencies:
    - generate-model
  script:
    - python -m pytest tests/model_tests.py --model-path $MODEL_PATH
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    environment {
        MODEL_DIR = 'synthetic_model'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'cargo install tensormill'
            }
        }
        
        stage('Generate Model') {
            steps {
                sh "tensormill -t gpt-oss-20b -s compact -o ${MODEL_DIR} --seed ${BUILD_NUMBER}"
            }
        }
        
        stage('Test') {
            steps {
                sh "python test_suite.py ${MODEL_DIR}"
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
```

## Testing Workflows

### Unit Testing with Synthetic Models

```python
# test_model_loading.py
import unittest
from pathlib import Path
from safetensors import safe_open
import json

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generate model once for all tests"""
        import subprocess
        subprocess.run([
            "tensormill", "-t", "gpt-oss-20b", 
            "-s", "compact", "-o", "test_model",
            "--seed", "12345"
        ], check=True)
        cls.model_path = Path("test_model")
    
    def test_config_exists(self):
        config_path = self.model_path / "config.json"
        self.assertTrue(config_path.exists())
        
        with open(config_path) as f:
            config = json.load(f)
            self.assertEqual(config["num_hidden_layers"], 24)
            self.assertEqual(config["num_local_experts"], 32)
    
    def test_weights_loadable(self):
        weights_path = self.model_path / "model.safetensors"
        self.assertTrue(weights_path.exists())
        
        with safe_open(weights_path, framework="pt") as f:
            # Check key tensors exist
            self.assertIn("model.embed_tokens.weight", f.keys())
            self.assertIn("lm_head.weight", f.keys())
            
            # Check tensor shapes
            embed = f.get_tensor("model.embed_tokens.weight")
            self.assertEqual(embed.shape[1], 2880)  # hidden_size
    
    def test_tokenizer_files(self):
        required_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]
        for filename in required_files:
            path = self.model_path / filename
            self.assertTrue(path.exists(), f"{filename} not found")

if __name__ == "__main__":
    unittest.main()
```

### Integration Testing

```bash
#!/bin/bash
# integration_test.sh

set -e

echo "🧪 Running integration tests..."

# Test 1: Compact model generation
echo "Test 1: Compact model"
tensormill -t gpt-oss-20b -s compact -o test1 --seed 100
python -c "
from safetensors import safe_open
with safe_open('test1/model.safetensors', 'pt') as f:
    assert len(f.keys()) > 0
    print('✅ Compact model test passed')
"

# Test 2: Sharded model generation
echo "Test 2: Sharded model"
tensormill -t gpt-oss-20b -s full -f sharded -o test2 --seed 200
count=$(ls test2/model-*.safetensors 2>/dev/null | wc -l)
if [ "$count" -eq 3 ]; then
    echo "✅ Sharded model test passed (3 shards found)"
else
    echo "❌ Sharded model test failed (expected 3 shards, found $count)"
    exit 1
fi

# Test 3: Deterministic generation
echo "Test 3: Deterministic generation"
tensormill -s compact -o test3a --seed 42
tensormill -s compact -o test3b --seed 42
if cmp -s test3a/model.safetensors test3b/model.safetensors; then
    echo "✅ Deterministic generation test passed"
else
    echo "❌ Files differ despite same seed"
    exit 1
fi

echo "🎉 All integration tests passed!"
```

## Advanced Usage

### Custom Seeds for Reproducibility

```bash
# Use build number as seed in CI
tensormill --seed $BUILD_NUMBER -o model_v$BUILD_NUMBER

# Use git commit hash as seed
SEED=$(git rev-parse HEAD | head -c 8 | od -An -tu4 | tr -d ' ')
tensormill --seed $SEED -o model_$(git rev-parse --short HEAD)
```

### Batch Generation

```bash
#!/bin/bash
# generate_test_suite.sh

# Generate models of different sizes
for size in compact full; do
    for format in sharded unsharded; do
        output="models/${size}_${format}"
        echo "Generating $output..."
        tensormill -s $size -f $format -o $output --progress
    done
done
```

### Parallel Generation

```bash
# Generate multiple models in parallel
tensormill -t gpt-oss-20b -s compact -o model1 --seed 1 &
tensormill -t gpt-oss-20b -s compact -o model2 --seed 2 &
tensormill -t gpt-oss-20b -s compact -o model3 --seed 3 &
wait
echo "All models generated"
```

## Python Integration

### Using as a Python Library

```python
# generate_model.py
import subprocess
import json
from pathlib import Path

def generate_synthetic_model(
    model_type="gpt-oss-20b",
    size="compact",
    format="unsharded",
    output_dir="./model",
    seed=42
):
    """Generate synthetic GPT-OSS model"""
    
    cmd = [
        "tensormill",
        "-t", model_type,
        "-s", size,
        "-f", format,
        "-o", output_dir,
        "--seed", str(seed)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Generation failed: {result.stderr}")
    
    # Load and return config
    config_path = Path(output_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    return {
        "path": output_dir,
        "config": config,
        "files": list(Path(output_dir).glob("*.safetensors"))
    }

# Example usage
model_info = generate_synthetic_model(
    size="compact",
    output_dir="./my_model",
    seed=12345
)

print(f"Model generated at: {model_info['path']}")
print(f"Number of weight files: {len(model_info['files'])}")
print(f"Model layers: {model_info['config']['num_hidden_layers']}")
```

### PyTorch Integration

```python
import torch
from safetensors.torch import load_file
from pathlib import Path

def load_synthetic_model(model_dir):
    """Load synthetic model weights into PyTorch"""
    
    model_dir = Path(model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    # Find weight files
    weight_files = sorted(model_dir.glob("model*.safetensors"))
    
    # Load weights
    state_dict = {}
    for weight_file in weight_files:
        weights = load_file(weight_file)
        state_dict.update(weights)
    
    return state_dict, config

# Generate and load
subprocess.run(["tensormill", "-o", "test_model"], check=True)
state_dict, config = load_synthetic_model("test_model")

print(f"Loaded {len(state_dict)} tensors")
print(f"Total parameters: {sum(t.numel() for t in state_dict.values()):,}")
```

## Performance Tuning

### Optimize Generation Speed

```bash
# Use all CPU cores
export RAYON_NUM_THREADS=$(nproc)
tensormill -s full -o model

# Limit to specific cores
export RAYON_NUM_THREADS=8
tensormill -s full -o model

# Disable progress bar for faster generation
tensormill -s full -o model  # No -p flag
```

### Memory Management

```bash
# Monitor memory during generation
/usr/bin/time -v tensormill -t gpt-oss-120b -s full -o large_model

# Use compact models for limited memory
if [ $(free -g | awk '/^Mem:/{print $2}') -lt 16 ]; then
    echo "Using compact model due to limited memory"
    tensormill -s compact -o model
else
    tensormill -s full -o model
fi
```

## Custom Configurations

### Create Custom Test Scenarios

```bash
#!/bin/bash
# test_scenarios.sh

# Scenario 1: Minimal model for quick tests
tensormill -t gpt-oss-20b -s compact -f unsharded \
    -o scenarios/minimal --seed 1000

# Scenario 2: Realistic sharded model
tensormill -t gpt-oss-20b -s full -f sharded \
    -o scenarios/realistic --seed 2000

# Scenario 3: Stress test with large model
tensormill -t gpt-oss-120b -s full -f sharded \
    -o scenarios/stress --seed 3000

# Scenario 4: OpenAI format for compatibility
tensormill -t gpt-oss-20b -s full -f original \
    -o scenarios/openai --seed 4000
```

### Validation Script

```python
# validate_model.py
import sys
import json
from pathlib import Path
from safetensors import safe_open

def validate_synthetic_model(model_dir):
    """Validate synthetic model structure"""
    
    model_dir = Path(model_dir)
    errors = []
    
    # Check required files
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    for file in required_files:
        if not (model_dir / file).exists():
            errors.append(f"Missing {file}")
    
    # Load and validate config
    try:
        with open(model_dir / "config.json") as f:
            config = json.load(f)
            
        # Validate config fields
        required_fields = [
            "model_type", "vocab_size", "hidden_size",
            "num_hidden_layers", "num_attention_heads"
        ]
        
        for field in required_fields:
            if field not in config:
                errors.append(f"Config missing {field}")
                
    except Exception as e:
        errors.append(f"Config error: {e}")
    
    # Check weight files
    weight_files = list(model_dir.glob("*.safetensors"))
    if not weight_files:
        errors.append("No weight files found")
    
    # Validate weights
    for weight_file in weight_files:
        try:
            with safe_open(weight_file, framework="pt") as f:
                if len(f.keys()) == 0:
                    errors.append(f"{weight_file.name} is empty")
        except Exception as e:
            errors.append(f"Weight file error in {weight_file.name}: {e}")
    
    # Report results
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Model validation passed")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_model.py MODEL_DIR")
        sys.exit(1)
    
    success = validate_synthetic_model(sys.argv[1])
    sys.exit(0 if success else 1)
```

## Best Practices

1. **Always use seeds in CI/CD** for reproducible tests
2. **Start with compact models** for development
3. **Use sharded format** for models >5GB
4. **Cache generated models** in CI to save time
5. **Validate generated models** before use
6. **Monitor memory usage** for large models
7. **Use parallel generation** for multiple models

## Troubleshooting

See [INSTALL.md](INSTALL.md#troubleshooting) for common issues and solutions.

## Next Steps

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Read the [README.md](README.md) for quick reference
- Explore the source code for customization options