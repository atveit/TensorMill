//! Error handling for synthetic weight generation

use std::io;
use thiserror::Error;

/// Result type alias for the library
pub type Result<T> = std::result::Result<T, TensorMillError>;

/// Main error type for TensorMill operations
#[derive(Error, Debug)]
pub enum TensorMillError {
    /// I/O operation failed
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    /// Tensor generation error
    #[error("Tensor generation error: {0}")]
    TensorGeneration(String),
    
    /// MXFP4 quantization error
    #[error("MXFP4 quantization error: {0}")]
    MXFP4Error(String),
    
    /// Sharding error
    #[error("Sharding error: {0}")]
    ShardingError(String),
    
    /// Metadata generation error
    #[error("Metadata generation error: {0}")]
    MetadataError(String),
    
    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    /// Generic error with context
    #[error("{context}: {source}")]
    WithContext {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl TensorMillError {
    /// Create an error with additional context
    pub fn with_context<E>(context: impl Into<String>, error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::WithContext {
            context: context.into(),
            source: Box::new(error),
        }
    }
    
    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::ValidationError(msg.into())
    }
    
    /// Create a tensor generation error
    pub fn tensor_generation(msg: impl Into<String>) -> Self {
        Self::TensorGeneration(msg.into())
    }
    
    /// Create an invalid config error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }
}

impl From<serde_json::Error> for TensorMillError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<safetensors::SafeTensorError> for TensorMillError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        Self::Serialization(format!("SafeTensor error: {:?}", err))
    }
}