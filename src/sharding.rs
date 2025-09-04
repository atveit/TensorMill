//! Sharding strategy for distributing tensors across files

use crate::config::ModelConfig;
use crate::tensor::TensorData;
use crate::error::Result;
use std::collections::HashMap;

/// Sharding strategy for model distribution
#[derive(Debug, Clone)]
pub struct ShardingStrategy {
    config: ModelConfig,
    max_shard_size: usize,
}

impl ShardingStrategy {
    /// Create a new sharding strategy
    pub fn new(config: ModelConfig) -> Self {
        // Set max shard size based on model
        let max_shard_size = match config.model_type {
            crate::config::ModelType::GptOss20B => 5 * 1024 * 1024 * 1024, // 5GB
            crate::config::ModelType::GptOss120B => 5 * 1024 * 1024 * 1024, // 5GB
        };
        
        Self {
            config,
            max_shard_size,
        }
    }
    
    /// Distribute tensors across shards
    pub fn distribute_tensors(
        &self,
        tensors: Vec<TensorData>,
    ) -> Result<Vec<ShardData>> {
        let num_shards = self.config.num_shards();
        
        if num_shards == 1 {
            // Single shard - all tensors go together
            return Ok(vec![ShardData {
                shard_index: 0,
                tensors,
                filename: self.get_shard_filename(0),
            }]);
        }
        
        // Distribute based on layer for better load balancing
        let mut shards: Vec<ShardData> = (0..num_shards)
            .map(|idx| ShardData {
                shard_index: idx,
                tensors: Vec::new(),
                filename: self.get_shard_filename(idx),
            })
            .collect();
        
        // Group tensors by layer
        let mut layer_tensors: HashMap<Option<usize>, Vec<TensorData>> = HashMap::new();
        
        for tensor in tensors {
            let layer_idx = extract_layer_index(&tensor.spec.name);
            layer_tensors.entry(layer_idx).or_default().push(tensor);
        }
        
        // For GPT-OSS-20B with 3 shards, match HuggingFace's exact distribution:
        // Shard 0: layers 0, 1, 10-18 (non-contiguous) -> 4.79 GB
        // Shard 1: layers 2-6, 18-23 (non-contiguous) -> 4.8 GB  
        // Shard 2: layers 6-9 + embeddings + lm_head -> 4.17 GB
        // Note: Layers 6 and 18 are split across shards in the real model
        
        if num_shards == 3 && self.config.model_type == crate::config::ModelType::GptOss20B {
            // HuggingFace's exact non-contiguous distribution
            for (layer_idx, tensors) in layer_tensors.drain() {
                if let Some(idx) = layer_idx {
                    // Determine shard based on HuggingFace's pattern
                    let shard_indices = if idx == 0 || idx == 1 {
                        vec![0]  // Layers 0-1 go to shard 0
                    } else if idx >= 10 && idx <= 17 {
                        vec![0]  // Layers 10-17 go to shard 0
                    } else if idx == 18 {
                        vec![0, 1]  // Layer 18 is split between shards 0 and 1
                    } else if idx >= 2 && idx <= 5 {
                        vec![1]  // Layers 2-5 go to shard 1
                    } else if idx == 6 {
                        vec![1, 2]  // Layer 6 is split between shards 1 and 2
                    } else if idx >= 19 && idx <= 23 {
                        vec![1]  // Layers 19-23 go to shard 1
                    } else if idx >= 7 && idx <= 9 {
                        vec![2]  // Layers 7-9 go to shard 2
                    } else {
                        vec![0]  // Fallback (shouldn't happen for 24 layers)
                    };
                    
                    // For layers that need to be split across shards
                    if shard_indices.len() > 1 {
                        // Split tensors roughly evenly across target shards
                        let chunk_size = (tensors.len() + shard_indices.len() - 1) / shard_indices.len();
                        for (i, chunk) in tensors.chunks(chunk_size).enumerate() {
                            if i < shard_indices.len() {
                                shards[shard_indices[i]].tensors.extend_from_slice(chunk);
                            }
                        }
                    } else {
                        // Single shard, add all tensors
                        shards[shard_indices[0]].tensors.extend(tensors);
                    }
                } else {
                    // Non-layer tensors (embeddings, final norm, lm_head)
                    for tensor in tensors {
                        if tensor.spec.name.starts_with("model.embed") || tensor.spec.name.starts_with("model.norm") {
                            shards[2].tensors.push(tensor);  // Embeddings and norm go to shard 2
                        } else if tensor.spec.name.starts_with("lm_head") {
                            shards[2].tensors.push(tensor);  // LM head goes to shard 2
                        } else {
                            shards[2].tensors.push(tensor);  // Other non-layer tensors to shard 2
                        }
                    }
                }
            }
        } else {
            // Default distribution for other configurations
            
            // Distribute non-layer tensors (embeddings, final norm, lm_head) to first and last shards
            if let Some(non_layer) = layer_tensors.remove(&None) {
                for tensor in non_layer {
                    if tensor.spec.name.starts_with("model.embed") {
                        shards[0].tensors.push(tensor);
                    } else {
                        shards[num_shards - 1].tensors.push(tensor);
                    }
                }
            }
            
            // Distribute layer tensors evenly
            let mut layer_indices: Vec<_> = layer_tensors.keys().copied().collect();
            layer_indices.sort_by_key(|k| k.unwrap_or(0));
            
            for (idx, layer_idx) in layer_indices.into_iter().enumerate() {
                if let Some(tensors) = layer_tensors.remove(&layer_idx) {
                    let shard_idx = (idx * num_shards) / self.config.num_layers;
                    let shard_idx = shard_idx.min(num_shards - 1);
                    shards[shard_idx].tensors.extend(tensors);
                }
            }
        }
        
        // Validate shard sizes
        for shard in &shards {
            let size = shard.total_size();
            if size > self.max_shard_size {
                tracing::warn!(
                    "Shard {} exceeds max size: {} > {}",
                    shard.shard_index,
                    size,
                    self.max_shard_size
                );
            }
        }
        
        Ok(shards)
    }
    
    /// Get the filename for a shard
    fn get_shard_filename(&self, shard_index: usize) -> String {
        let num_shards = self.config.num_shards();
        
        match self.config.format {
            crate::config::ModelFormat::Sharded => {
                // HuggingFace has a quirk where GPT-OSS-20B uses "of-00002" for all 3 files
                // This appears to be a bug in their naming but we match it for compatibility
                if self.config.model_type == crate::config::ModelType::GptOss20B && num_shards == 3 {
                    // Use "of-00002" for all 3 files to match HuggingFace exactly
                    format!("model-{:05}-of-00002.safetensors", shard_index)
                } else {
                    format!("model-{:05}-of-{:05}.safetensors", shard_index, num_shards)
                }
            }
            crate::config::ModelFormat::Original => {
                // Original format uses single file named "model.safetensors"
                if num_shards == 1 {
                    "model.safetensors".to_string()
                } else {
                    // Fallback for future multi-shard original format
                    format!("model--{:05}-of-{:05}.safetensors", shard_index + 1, num_shards)
                }
            }
            crate::config::ModelFormat::Unsharded => "model.safetensors".to_string(),
        }
    }
}

/// Data for a single shard
#[derive(Debug, Clone)]
pub struct ShardData {
    pub shard_index: usize,
    pub tensors: Vec<TensorData>,
    pub filename: String,
}

impl ShardData {
    /// Calculate total size of tensors in this shard
    pub fn total_size(&self) -> usize {
        self.tensors.iter().map(|t| t.size_bytes()).sum()
    }
    
    /// Get tensor count
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

/// Extract layer index from tensor name
fn extract_layer_index(name: &str) -> Option<usize> {
    if let Some(start) = name.find("layers.") {
        let after_layers = &name[start + 7..];
        if let Some(dot_pos) = after_layers.find('.') {
            after_layers[..dot_pos].parse().ok()
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelType, ModelFormat, ModelSize, TensorSpec, DType};
    
    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("model.layers.0.self_attn.q_proj.weight"), Some(0));
        assert_eq!(extract_layer_index("model.layers.23.mlp.router.weight"), Some(23));
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }
    
    #[test]
    fn test_single_shard_distribution() {
        let config = ModelConfig::new(
            ModelType::GptOss20B,
            ModelFormat::Unsharded,
            ModelSize::Compact,
        );
        
        let strategy = ShardingStrategy::new(config);
        
        // Create some test tensors
        let tensors = vec![
            TensorData::new(
                TensorSpec {
                    name: "test1".to_string(),
                    shape: vec![100],
                    dtype: DType::Float32,
                    use_mxfp4: false,
                },
                vec![0; 400],
            ),
            TensorData::new(
                TensorSpec {
                    name: "test2".to_string(),
                    shape: vec![100],
                    dtype: DType::Float32,
                    use_mxfp4: false,
                },
                vec![0; 400],
            ),
        ];
        
        let shards = strategy.distribute_tensors(tensors).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].tensor_count(), 2);
        assert_eq!(shards[0].filename, "model.safetensors");
    }
    
    #[test]
    fn test_multi_shard_distribution() {
        let config = ModelConfig::new(
            ModelType::GptOss20B,
            ModelFormat::Sharded,
            ModelSize::Full,
        );
        
        let strategy = ShardingStrategy::new(config);
        
        // Create layer tensors
        let mut tensors = vec![];
        for i in 0..24 {
            tensors.push(TensorData::new(
                TensorSpec {
                    name: format!("model.layers.{}.self_attn.q_proj.weight", i),
                    shape: vec![100],
                    dtype: DType::Float32,
                    use_mxfp4: false,
                },
                vec![0; 400],
            ));
        }
        
        // Add non-layer tensors
        tensors.push(TensorData::new(
            TensorSpec {
                name: "model.embed_tokens.weight".to_string(),
                shape: vec![100],
                dtype: DType::Float32,
                use_mxfp4: false,
            },
            vec![0; 400],
        ));
        
        tensors.push(TensorData::new(
            TensorSpec {
                name: "lm_head.weight".to_string(),
                shape: vec![100],
                dtype: DType::Float32,
                use_mxfp4: false,
            },
            vec![0; 400],
        ));
        
        let shards = strategy.distribute_tensors(tensors).unwrap();
        assert_eq!(shards.len(), 3);
        
        // Check that all shards have tensors
        for shard in &shards {
            assert!(shard.tensor_count() > 0);
        }
    }
}