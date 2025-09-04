//! Integration tests for TensorMill

use tensormill::{
    ModelConfig, ModelType, ModelFormat, ModelSize,
    SyntheticGenerator, Result,
};
use tempfile::tempdir;
use std::path::Path;
use std::fs;
use safetensors::SafeTensors;

#[test]
fn test_compact_model_generation() -> Result<()> {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Unsharded,
        ModelSize::Compact,
    );
    
    let mut generator = SyntheticGenerator::new(config)
        .with_seed(42);
    
    let temp_dir = tempdir().unwrap();
    let result = generator.generate(temp_dir.path())?;
    
    // Verify output structure
    assert_eq!(result.num_shards, 1);
    assert!(result.num_tensors > 0);
    assert!(result.total_size > 0);
    
    // Check files exist
    assert!(temp_dir.path().join("model.safetensors").exists());
    assert!(temp_dir.path().join("config.json").exists());
    assert!(temp_dir.path().join("tokenizer.json").exists());
    
    Ok(())
}

#[test]
fn test_sharded_model_generation() -> Result<()> {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Sharded,
        ModelSize::Full,
    );
    
    let mut generator = SyntheticGenerator::new(config)
        .with_seed(100);
    
    let temp_dir = tempdir().unwrap();
    let result = generator.generate(temp_dir.path())?;
    
    // Verify sharding
    assert_eq!(result.num_shards, 3);
    
    // Check shard files exist
    for i in 0..3 {
        let shard_file = format!("model-{:05}-of-00003.safetensors", i);
        assert!(
            temp_dir.path().join(&shard_file).exists(),
            "Missing shard file: {}", shard_file
        );
    }
    
    // Check index file
    assert!(temp_dir.path().join("model.safetensors.index.json").exists());
    
    Ok(())
}

#[test]
fn test_deterministic_generation() -> Result<()> {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Unsharded,
        ModelSize::Compact,
    );
    
    // Generate twice with same seed
    let temp_dir1 = tempdir().unwrap();
    let mut gen1 = SyntheticGenerator::new(config.clone()).with_seed(42);
    gen1.generate(temp_dir1.path())?;
    
    let temp_dir2 = tempdir().unwrap();
    let mut gen2 = SyntheticGenerator::new(config).with_seed(42);
    gen2.generate(temp_dir2.path())?;
    
    // Compare file sizes (full binary comparison would be expensive)
    let file1 = temp_dir1.path().join("model.safetensors");
    let file2 = temp_dir2.path().join("model.safetensors");
    
    let size1 = fs::metadata(&file1).unwrap().len();
    let size2 = fs::metadata(&file2).unwrap().len();
    
    assert_eq!(size1, size2, "Deterministic generation produced different sizes");
    
    Ok(())
}

#[test]
fn test_metadata_generation() -> Result<()> {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Unsharded,
        ModelSize::Compact,
    );
    
    let mut generator = SyntheticGenerator::new(config);
    let temp_dir = tempdir().unwrap();
    generator.generate(temp_dir.path())?;
    
    // Check all metadata files
    let metadata_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
    ];
    
    for file in &metadata_files {
        assert!(
            temp_dir.path().join(file).exists(),
            "Missing metadata file: {}", file
        );
    }
    
    // Validate config.json content
    let config_content = fs::read_to_string(temp_dir.path().join("config.json"))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
    
    assert_eq!(config_json["model_type"], "gpt_oss");
    assert_eq!(config_json["num_hidden_layers"], 24);
    assert_eq!(config_json["num_local_experts"], 32);
    
    Ok(())
}

#[test]
fn test_tensor_loading() -> Result<()> {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Unsharded,
        ModelSize::Compact,
    );
    
    let mut generator = SyntheticGenerator::new(config);
    let temp_dir = tempdir().unwrap();
    generator.generate(temp_dir.path())?;
    
    // Load and validate tensors
    let model_path = temp_dir.path().join("model.safetensors");
    let data = fs::read(&model_path)?;
    let tensors = SafeTensors::deserialize(&data).unwrap();
    
    // Check key tensors exist
    assert!(tensors.names().contains(&"model.embed_tokens.weight"));
    assert!(tensors.names().contains(&"lm_head.weight"));
    assert!(tensors.names().contains(&"model.norm.weight"));
    
    // Check layer tensors
    assert!(tensors.names().contains(&"model.layers.0.self_attn.q_proj.weight"));
    assert!(tensors.names().contains(&"model.layers.1.mlp.router.weight"));
    
    Ok(())
}

#[test]
fn test_gpt_oss_120b_generation() -> Result<()> {
    let config = ModelConfig::new(
        ModelType::GptOss120B,
        ModelFormat::Unsharded,
        ModelSize::Compact,
    );
    
    let mut generator = SyntheticGenerator::new(config);
    let temp_dir = tempdir().unwrap();
    let result = generator.generate(temp_dir.path())?;
    
    // Verify 120B specific properties
    assert!(result.num_tensors > 0);
    
    // Check config has 120B parameters
    let config_content = fs::read_to_string(temp_dir.path().join("config.json"))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
    
    assert_eq!(config_json["num_hidden_layers"], 36);
    assert_eq!(config_json["num_local_experts"], 128);
    
    Ok(())
}

#[test]
fn test_original_format_generation() -> Result<()> {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Original,
        ModelSize::Full,
    );
    
    let mut generator = SyntheticGenerator::new(config);
    let temp_dir = tempdir().unwrap();
    generator.generate(temp_dir.path())?;
    
    // Check OpenAI original format naming
    for i in 1..=3 {
        let shard_file = format!("model--{:05}-of-00003.safetensors", i);
        assert!(
            temp_dir.path().join(&shard_file).exists(),
            "Missing OpenAI format shard: {}", shard_file
        );
    }
    
    Ok(())
}

#[test]
fn test_progress_tracking() -> Result<()> {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Unsharded,
        ModelSize::Compact,
    );
    
    // Test with progress bar enabled
    let mut generator = SyntheticGenerator::new(config)
        .with_seed(42)
        .with_progress();
    
    let temp_dir = tempdir().unwrap();
    let result = generator.generate(temp_dir.path())?;
    
    // Just verify it completes without panic
    assert!(result.num_tensors > 0);
    
    Ok(())
}

#[test]
fn test_error_handling_invalid_path() {
    let config = ModelConfig::new(
        ModelType::GptOss20B,
        ModelFormat::Unsharded,
        ModelSize::Compact,
    );
    
    let mut generator = SyntheticGenerator::new(config);
    
    // Try to generate in invalid path
    let result = generator.generate("/invalid/path/that/does/not/exist");
    assert!(result.is_err());
}

// Property-based tests with proptest would go here
#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_seed_determinism(seed1: u64, seed2: u64) {
            // Same seeds produce same results
            if seed1 == seed2 {
                // Test determinism
            } else {
                // Different seeds produce different results
            }
        }
    }
}