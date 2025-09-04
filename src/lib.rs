//! TensorMill - Industrial-strength synthetic tensor generation
//! 
//! This crate provides blazing-fast generation of synthetic model weights
//! for GPT-OSS models (20B and 120B variants), built for ML testing pipelines.
//! 
//! # Features
//! 
//! - **Multi-model support**: GPT-OSS-20B and GPT-OSS-120B configurations
//! - **Format flexibility**: Sharded, unsharded, and original OpenAI formats
//! - **MXFP4 quantization**: Accurate 4-bit quantization for MoE experts
//! - **High performance**: Parallel processing with Rayon
//! - **Metadata generation**: Complete config.json and tokenizer files
//! - **Memory efficient**: Streaming generation with minimal overhead
//! 
//! # Example
//! 
//! ```rust
//! use tensormill::{ModelConfig, ModelType, ModelFormat, ModelSize};
//! use tensormill::generator::SyntheticGenerator;
//! 
//! let config = ModelConfig::new(
//!     ModelType::GptOss20B,
//!     ModelFormat::Sharded,
//!     ModelSize::Full
//! );
//! 
//! let mut generator = SyntheticGenerator::new(config);
//! let result = generator.generate("./output")?;
//! ```

pub mod config;
pub mod tensor;
pub mod generator;
pub mod mxfp4;
pub mod metadata;
pub mod sharding;
pub mod error;

pub use config::{ModelConfig, ModelType, ModelFormat, ModelSize, TensorSpec};
pub use generator::{SyntheticGenerator, GenerationResult};
pub use mxfp4::MXFP4Quantizer;
pub use metadata::MetadataGenerator;
pub use sharding::ShardingStrategy;
pub use error::{TensorMillError, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Re-export commonly used types
pub mod prelude {
    pub use crate::config::{ModelConfig, ModelType, ModelFormat, ModelSize};
    pub use crate::generator::SyntheticGenerator;
    pub use crate::error::Result;
}