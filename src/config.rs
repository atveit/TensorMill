//! Configuration module for GPT-OSS model specifications

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported model types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// GPT-OSS 20B model (24 layers, 32 experts)
    GptOss20B,
    /// GPT-OSS 120B model (36 layers, 128 experts)
    GptOss120B,
}

impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GptOss20B => "gpt-oss-20b",
            Self::GptOss120B => "gpt-oss-120b",
        }
    }
}

/// Output format for generated weights
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelFormat {
    /// HuggingFace multi-file format (model-00000-of-NNNNN.safetensors)
    Sharded,
    /// Single file format (model.safetensors)
    Unsharded,
    /// OpenAI original format (model--00001-of-NNNNN.safetensors)
    Original,
}

/// Model size variant
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelSize {
    /// Compact model for quick CI/CD tests (~440MB for 20B)
    Compact,
    /// Full-size model matching production (~13GB for 20B, ~65GB for 120B)
    Full,
}

/// Data type for tensors
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    MXFP4,
    Int8,
    UInt8,
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::MXFP4 => "mxfp4",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
        }
    }
}

/// Tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Whether this tensor should be quantized with MXFP4
    pub use_mxfp4: bool,
}

impl TensorSpec {
    /// Calculate the number of elements in this tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Calculate the size in bytes for this tensor
    pub fn size_bytes(&self) -> usize {
        let element_size = match self.dtype {
            DType::Float32 => 4,
            DType::Float16 | DType::BFloat16 => 2,
            DType::MXFP4 => 1, // 4-bit packed into bytes
            DType::Int8 | DType::UInt8 => 1,
        };
        
        // For MXFP4 blocks tensors, the shape already accounts for packing
        // (last dimension is halved), so just use regular size calculation
        self.num_elements() * element_size
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub format: ModelFormat,
    pub size: ModelSize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(model_type: ModelType, format: ModelFormat, size: ModelSize) -> Self {
        match model_type {
            ModelType::GptOss20B => Self {
                model_type,
                format,
                size,
                num_layers: 24,
                num_experts: 32,
                vocab_size: 201_088,
                hidden_size: 2880,
                intermediate_size: 2880,
                num_heads: 64,
                num_kv_heads: 8,
                head_dim: 64, // Actual head dimension for attention
                max_position_embeddings: 4096,
                rope_theta: 500_000.0,
            },
            ModelType::GptOss120B => Self {
                model_type,
                format,
                size,
                num_layers: 36,
                num_experts: 128,
                vocab_size: 201_088,
                hidden_size: 2880,
                intermediate_size: 2880,
                num_heads: 64,
                num_kv_heads: 8,
                head_dim: 64,
                max_position_embeddings: 4096,
                rope_theta: 500_000.0,
            },
        }
    }

    /// Get tensor specifications for this model
    pub fn get_tensor_specs(&self) -> Vec<TensorSpec> {
        let mut specs = Vec::new();
        
        // Adjust for compact vs full size
        let (vocab_size, num_layers, num_experts) = match self.size {
            ModelSize::Compact => (8_000, 2, 4), // Minimal for testing
            ModelSize::Full => (self.vocab_size, self.num_layers, self.num_experts),
        };
        
        // Embedding layer
        specs.push(TensorSpec {
            name: "model.embed_tokens.weight".to_string(),
            shape: vec![vocab_size, self.hidden_size],
            dtype: DType::BFloat16,
            use_mxfp4: false,
        });
        
        // Transformer layers
        for layer_idx in 0..num_layers {
            let prefix = format!("model.layers.{}", layer_idx);
            
            // Layer normalization weights
            specs.push(TensorSpec {
                name: format!("{}.input_layernorm.weight", prefix),
                shape: vec![self.hidden_size],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            
            specs.push(TensorSpec {
                name: format!("{}.post_attention_layernorm.weight", prefix),
                shape: vec![self.hidden_size],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            
            // Self-attention projections
            let q_dim = self.num_heads * self.head_dim;
            let kv_dim = self.num_kv_heads * self.head_dim;
            
            // Query projection
            specs.push(TensorSpec {
                name: format!("{}.self_attn.q_proj.weight", prefix),
                shape: vec![q_dim, self.hidden_size],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            specs.push(TensorSpec {
                name: format!("{}.self_attn.q_proj.bias", prefix),
                shape: vec![q_dim],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            
            // Key projection
            specs.push(TensorSpec {
                name: format!("{}.self_attn.k_proj.weight", prefix),
                shape: vec![kv_dim, self.hidden_size],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            specs.push(TensorSpec {
                name: format!("{}.self_attn.k_proj.bias", prefix),
                shape: vec![kv_dim],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            
            // Value projection
            specs.push(TensorSpec {
                name: format!("{}.self_attn.v_proj.weight", prefix),
                shape: vec![kv_dim, self.hidden_size],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            specs.push(TensorSpec {
                name: format!("{}.self_attn.v_proj.bias", prefix),
                shape: vec![kv_dim],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            
            // Output projection
            specs.push(TensorSpec {
                name: format!("{}.self_attn.o_proj.weight", prefix),
                shape: vec![self.hidden_size, q_dim],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            specs.push(TensorSpec {
                name: format!("{}.self_attn.o_proj.bias", prefix),
                shape: vec![self.hidden_size],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            
            // MoE router
            specs.push(TensorSpec {
                name: format!("{}.mlp.router.weight", prefix),
                shape: vec![num_experts, self.hidden_size],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            specs.push(TensorSpec {
                name: format!("{}.mlp.router.bias", prefix),
                shape: vec![num_experts],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            
            // MoE experts (MXFP4 quantized)
            // MXFP4 blocks are stored with half the last dimension (packed 2 4-bit values per byte)
            specs.push(TensorSpec {
                name: format!("{}.mlp.experts.gate_up_proj_blocks", prefix),
                shape: vec![num_experts, self.intermediate_size * 2, self.hidden_size / 2],
                dtype: DType::UInt8,
                use_mxfp4: true,
            });
            specs.push(TensorSpec {
                name: format!("{}.mlp.experts.gate_up_proj_scales", prefix),
                shape: vec![num_experts, (self.intermediate_size * 2 * self.hidden_size + 31) / 32],
                dtype: DType::Int8,
                use_mxfp4: true,
            });
            specs.push(TensorSpec {
                name: format!("{}.mlp.experts.gate_up_proj_bias", prefix),
                shape: vec![num_experts, self.intermediate_size * 2],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
            
            specs.push(TensorSpec {
                name: format!("{}.mlp.experts.down_proj_blocks", prefix),
                shape: vec![num_experts, self.hidden_size, self.intermediate_size / 2],
                dtype: DType::UInt8,
                use_mxfp4: true,
            });
            specs.push(TensorSpec {
                name: format!("{}.mlp.experts.down_proj_scales", prefix),
                shape: vec![num_experts, (self.hidden_size * self.intermediate_size + 31) / 32],
                dtype: DType::Int8,
                use_mxfp4: true,
            });
            specs.push(TensorSpec {
                name: format!("{}.mlp.experts.down_proj_bias", prefix),
                shape: vec![num_experts, self.hidden_size],
                dtype: DType::BFloat16,
                use_mxfp4: false,
            });
        }
        
        // Final layer norm
        specs.push(TensorSpec {
            name: "model.norm.weight".to_string(),
            shape: vec![self.hidden_size],
            dtype: DType::BFloat16,
            use_mxfp4: false,
        });
        
        // Language model head
        specs.push(TensorSpec {
            name: "lm_head.weight".to_string(),
            shape: vec![vocab_size, self.hidden_size],
            dtype: DType::BFloat16,
            use_mxfp4: false,
        });
        
        specs
    }
    
    /// Get the number of shards for this configuration
    pub fn num_shards(&self) -> usize {
        match (self.model_type, self.format, self.size) {
            // Compact models use single shard
            (_, _, ModelSize::Compact) => 1,
            // Full models use format-specific sharding
            (ModelType::GptOss20B, ModelFormat::Sharded, ModelSize::Full) => 3,  // HuggingFace uses 3 shards
            (ModelType::GptOss20B, ModelFormat::Original, ModelSize::Full) => 1,  // Original is single file
            (ModelType::GptOss20B, ModelFormat::Unsharded, ModelSize::Full) => 1,
            (ModelType::GptOss120B, ModelFormat::Sharded, ModelSize::Full) => 14,
            (ModelType::GptOss120B, ModelFormat::Original, ModelSize::Full) => 1,  // Original is single file
            (ModelType::GptOss120B, ModelFormat::Unsharded, ModelSize::Full) => 1,
        }
    }
    
    /// Generate the config.json content for this model
    pub fn to_config_json(&self) -> HashMap<String, serde_json::Value> {
        use serde_json::json;
        
        let mut config = HashMap::new();
        
        config.insert("model_type".to_string(), json!("gpt_oss"));
        config.insert("vocab_size".to_string(), json!(self.vocab_size));
        config.insert("hidden_size".to_string(), json!(self.hidden_size));
        config.insert("intermediate_size".to_string(), json!(self.intermediate_size));
        config.insert("num_hidden_layers".to_string(), json!(self.num_layers));
        config.insert("num_attention_heads".to_string(), json!(self.num_heads));
        config.insert("num_key_value_heads".to_string(), json!(self.num_kv_heads));
        config.insert("head_dim".to_string(), json!(self.head_dim));
        config.insert("max_position_embeddings".to_string(), json!(self.max_position_embeddings));
        config.insert("rope_theta".to_string(), json!(self.rope_theta));
        config.insert("num_local_experts".to_string(), json!(self.num_experts));
        config.insert("num_experts_per_tok".to_string(), json!(4));
        config.insert("use_cache".to_string(), json!(true));
        config.insert("tie_word_embeddings".to_string(), json!(false));
        config.insert("architectures".to_string(), json!(["GPTOSSForCausalLM"]));
        
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_config_20b() {
        let config = ModelConfig::new(ModelType::GptOss20B, ModelFormat::Sharded, ModelSize::Full);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_experts, 32);
        assert_eq!(config.vocab_size, 201_088);
    }
    
    #[test]
    fn test_model_config_120b() {
        let config = ModelConfig::new(ModelType::GptOss120B, ModelFormat::Sharded, ModelSize::Full);
        assert_eq!(config.num_layers, 36);
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.vocab_size, 201_088);
    }
    
    #[test]
    fn test_tensor_spec_size_calculation() {
        let spec = TensorSpec {
            name: "test".to_string(),
            shape: vec![100, 200],
            dtype: DType::BFloat16,
            use_mxfp4: false,
        };
        assert_eq!(spec.num_elements(), 20_000);
        assert_eq!(spec.size_bytes(), 40_000); // 20k * 2 bytes
    }
    
    #[test]
    fn test_num_shards() {
        let config = ModelConfig::new(ModelType::GptOss20B, ModelFormat::Sharded, ModelSize::Full);
        assert_eq!(config.num_shards(), 3);
        
        let config = ModelConfig::new(ModelType::GptOss120B, ModelFormat::Sharded, ModelSize::Full);
        assert_eq!(config.num_shards(), 14);
        
        let config = ModelConfig::new(ModelType::GptOss20B, ModelFormat::Unsharded, ModelSize::Full);
        assert_eq!(config.num_shards(), 1);
    }
}