//! Metadata generation for synthetic models

use crate::config::ModelConfig;
use crate::error::Result;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Metadata generator for GPT-OSS models
pub struct MetadataGenerator {
    config: ModelConfig,
}

impl MetadataGenerator {
    /// Create a new metadata generator
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
    
    /// Generate all metadata files
    pub fn generate_all(&self, output_dir: &Path) -> Result<()> {
        self.generate_config_json(output_dir)?;
        self.generate_tokenizer_config(output_dir)?;
        self.generate_generation_config(output_dir)?;
        self.generate_special_tokens_map(output_dir)?;
        self.generate_model_index(output_dir)?;
        Ok(())
    }
    
    /// Generate config.json
    pub fn generate_config_json(&self, output_dir: &Path) -> Result<()> {
        let config = self.config.to_config_json();
        let content = serde_json::to_string_pretty(&config)?;
        fs::write(output_dir.join("config.json"), content)?;
        Ok(())
    }
    
    /// Generate tokenizer_config.json
    pub fn generate_tokenizer_config(&self, output_dir: &Path) -> Result<()> {
        let config = json!({
            "add_bos_token": false,
            "add_eos_token": false,
            "added_tokens_decoder": {},
            "bos_token": "<|start|>",
            "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            "clean_up_tokenization_spaces": true,
            "eos_token": "<|im_end|>",
            "legacy": false,
            "model_max_length": 131072,
            "pad_token": "<|pad|>",
            "padding_side": "left",
            "sp_model_kwargs": {},
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": null,
            "use_default_system_prompt": false
        });
        
        let content = serde_json::to_string_pretty(&config)?;
        fs::write(output_dir.join("tokenizer_config.json"), content)?;
        Ok(())
    }
    
    /// Generate generation_config.json
    pub fn generate_generation_config(&self, output_dir: &Path) -> Result<()> {
        let config = json!({
            "bos_token_id": 200000,
            "eos_token_id": 200001,
            "max_length": 4096,
            "pad_token_id": 200002,
            "temperature": 1.0,
            "top_p": 1.0,
            "transformers_version": "4.40.0"
        });
        
        let content = serde_json::to_string_pretty(&config)?;
        fs::write(output_dir.join("generation_config.json"), content)?;
        Ok(())
    }
    
    /// Generate special_tokens_map.json
    pub fn generate_special_tokens_map(&self, output_dir: &Path) -> Result<()> {
        let config = json!({
            "bos_token": {
                "content": "<|start|>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false
            },
            "eos_token": {
                "content": "<|im_end|>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false
            },
            "pad_token": {
                "content": "<|pad|>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false
            }
        });
        
        let content = serde_json::to_string_pretty(&config)?;
        fs::write(output_dir.join("special_tokens_map.json"), content)?;
        Ok(())
    }
    
    /// Generate model.safetensors.index.json for sharded models
    pub fn generate_model_index(&self, output_dir: &Path) -> Result<()> {
        if self.config.num_shards() <= 1 {
            return Ok(()); // No index needed for single-file models
        }
        
        let mut weight_map = HashMap::new();
        let mut metadata = HashMap::new();
        
        // Build weight map based on tensor distribution
        let tensor_specs = self.config.get_tensor_specs();
        let num_shards = self.config.num_shards();
        
        // For GPT-OSS-20B with 3 shards, use HuggingFace's exact distribution
        if num_shards == 3 && self.config.model_type == crate::config::ModelType::GptOss20B 
           && self.config.format == crate::config::ModelFormat::Sharded {
            for spec in &tensor_specs {
                // Extract layer index from name
                let layer_idx = if let Some(start) = spec.name.find("layers.") {
                    let after_layers = &spec.name[start + 7..];
                    if let Some(dot_pos) = after_layers.find('.') {
                        after_layers[..dot_pos].parse::<usize>().ok()
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                // Determine shard based on HuggingFace's pattern
                let shard_idx = if let Some(idx) = layer_idx {
                    if idx == 0 || idx == 1 || (idx >= 10 && idx <= 18) {
                        0  // Layers 0-1, 10-18 go to shard 0
                    } else if (idx >= 2 && idx <= 6) || (idx >= 18 && idx <= 23) {
                        1  // Layers 2-6, 18-23 go to shard 1
                    } else if idx >= 7 && idx <= 9 {
                        2  // Layers 7-9 go to shard 2
                    } else {
                        0  // Fallback
                    }
                } else {
                    // Non-layer tensors (embeddings, final norm, lm_head) go to shard 2
                    if spec.name.starts_with("model.embed") || 
                       spec.name.starts_with("model.norm") || 
                       spec.name.starts_with("lm_head") {
                        2
                    } else {
                        2
                    }
                };
                
                // HuggingFace uses "of-00002" for all 3 files
                let filename = format!("model-{:05}-of-00002.safetensors", shard_idx);
                weight_map.insert(spec.name.clone(), filename);
            }
        } else {
            // Default distribution for other configurations
            let tensors_per_shard = tensor_specs.len() / num_shards;
            
            for (idx, spec) in tensor_specs.iter().enumerate() {
                let shard_idx = idx / tensors_per_shard;
                let shard_idx = shard_idx.min(num_shards - 1); // Ensure last shard gets remaining
                
                let filename = match self.config.format {
                    crate::config::ModelFormat::Sharded => {
                        if self.config.model_type == crate::config::ModelType::GptOss20B && num_shards == 3 {
                            format!("model-{:05}-of-00002.safetensors", shard_idx)
                        } else {
                            format!("model-{:05}-of-{:05}.safetensors", shard_idx, num_shards)
                        }
                    }
                    crate::config::ModelFormat::Original => {
                        format!("model--{:05}-of-{:05}.safetensors", shard_idx + 1, num_shards)
                    }
                    _ => "model.safetensors".to_string(),
                };
                
                weight_map.insert(spec.name.clone(), filename);
            }
        }
        
        // Calculate total size
        let total_size: usize = tensor_specs.iter().map(|s| s.size_bytes()).sum();
        metadata.insert("total_size".to_string(), json!(total_size));
        
        let index = json!({
            "metadata": metadata,
            "weight_map": weight_map,
        });
        
        let content = serde_json::to_string_pretty(&index)?;
        fs::write(output_dir.join("model.safetensors.index.json"), content)?;
        Ok(())
    }
    
    /// Generate a simple tokenizer.json for testing
    pub fn generate_tokenizer_json(&self, output_dir: &Path) -> Result<()> {
        // This is a simplified tokenizer for synthetic testing
        // Real models would use a proper tokenizer
        let tokenizer = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 200000,
                    "content": "<|start|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 200001,
                    "content": "<|im_end|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 200002,
                    "content": "<|pad|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel"
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {},
                "merges": []
            }
        });
        
        let content = serde_json::to_string_pretty(&tokenizer)?;
        fs::write(output_dir.join("tokenizer.json"), content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelType, ModelFormat, ModelSize};
    use tempfile::tempdir;
    
    #[test]
    fn test_metadata_generation() {
        let config = ModelConfig::new(
            ModelType::GptOss20B,
            ModelFormat::Sharded,
            ModelSize::Compact,
        );
        
        let generator = MetadataGenerator::new(config);
        let temp_dir = tempdir().unwrap();
        
        generator.generate_all(temp_dir.path()).unwrap();
        
        // Check that files were created
        assert!(temp_dir.path().join("config.json").exists());
        assert!(temp_dir.path().join("tokenizer_config.json").exists());
        assert!(temp_dir.path().join("generation_config.json").exists());
        assert!(temp_dir.path().join("special_tokens_map.json").exists());
    }
    
    #[test]
    fn test_config_json_content() {
        let config = ModelConfig::new(
            ModelType::GptOss20B,
            ModelFormat::Unsharded,
            ModelSize::Full,
        );
        
        let json = config.to_config_json();
        
        assert_eq!(json.get("model_type").unwrap(), &json!("gpt_oss"));
        assert_eq!(json.get("num_hidden_layers").unwrap(), &json!(24));
        assert_eq!(json.get("num_local_experts").unwrap(), &json!(32));
        assert_eq!(json.get("vocab_size").unwrap(), &json!(201088));
    }
}