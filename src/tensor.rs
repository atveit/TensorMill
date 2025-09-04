//! Tensor generation and manipulation

use crate::config::{DType, TensorSpec};
use crate::error::Result;
use half::f16;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::sync::Arc;

/// Generated tensor data
#[derive(Debug, Clone)]
pub struct TensorData {
    pub spec: TensorSpec,
    pub data: Arc<Vec<u8>>,
}

impl TensorData {
    /// Create a new tensor with the given specification and data
    pub fn new(spec: TensorSpec, data: Vec<u8>) -> Self {
        Self {
            spec,
            data: Arc::new(data),
        }
    }
    
    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
    
    /// Validate the tensor data
    pub fn validate(&self) -> Result<()> {
        let expected_size = self.spec.size_bytes();
        let actual_size = self.data.len();
        
        if actual_size != expected_size {
            return Err(crate::error::TensorMillError::validation(
                format!(
                    "Tensor '{}' size mismatch: expected {} bytes, got {} bytes",
                    self.spec.name, expected_size, actual_size
                )
            ));
        }
        
        Ok(())
    }
}

/// Tensor generator trait
pub trait TensorGenerator: Send + Sync {
    /// Generate a tensor with the given specification
    fn generate(&mut self, spec: &TensorSpec) -> Result<TensorData>;
    
    /// Set the random seed
    fn set_seed(&mut self, seed: u64);
}

/// Simple random tensor generator
pub struct RandomTensorGenerator {
    rng: StdRng,
}

impl RandomTensorGenerator {
    /// Create a new random tensor generator
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { rng }
    }
    
    /// Generate BFloat16 data
    fn generate_bfloat16(&mut self, num_elements: usize) -> Vec<u8> {
        let normal = Normal::new(0.0f32, 0.02).unwrap();
        let mut data = Vec::with_capacity(num_elements * 2);
        
        for _ in 0..num_elements {
            let value: f32 = normal.sample(&mut self.rng);
            // Convert to bfloat16 (truncate mantissa)
            let bits = value.to_bits();
            let bf16_bits = (bits >> 16) as u16;
            data.extend_from_slice(&bf16_bits.to_le_bytes());
        }
        
        data
    }
    
    /// Generate Float16 data
    fn generate_float16(&mut self, num_elements: usize) -> Vec<u8> {
        let normal = Normal::new(0.0f32, 0.02).unwrap();
        let mut data = Vec::with_capacity(num_elements * 2);
        
        for _ in 0..num_elements {
            let value: f32 = normal.sample(&mut self.rng);
            let f16_value = f16::from_f32(value);
            data.extend_from_slice(&f16_value.to_le_bytes());
        }
        
        data
    }
    
    /// Generate Float32 data
    fn generate_float32(&mut self, num_elements: usize) -> Vec<u8> {
        let normal = Normal::new(0.0f32, 0.02).unwrap();
        let mut data = Vec::with_capacity(num_elements * 4);
        
        for _ in 0..num_elements {
            let value: f32 = normal.sample(&mut self.rng);
            data.extend_from_slice(&value.to_le_bytes());
        }
        
        data
    }
    
    /// Generate Int8 data
    fn generate_int8(&mut self, num_elements: usize) -> Vec<u8> {
        let uniform = Uniform::new_inclusive(-127i8, 127);
        let mut data = Vec::with_capacity(num_elements);
        
        for _ in 0..num_elements {
            let value: i8 = uniform.sample(&mut self.rng);
            data.push(value as u8);
        }
        
        data
    }
    
    /// Generate UInt8 data
    fn generate_uint8(&mut self, num_elements: usize) -> Vec<u8> {
        let uniform = Uniform::new(0u8, 255);
        let mut data = Vec::with_capacity(num_elements);
        
        for _ in 0..num_elements {
            let value: u8 = uniform.sample(&mut self.rng);
            data.push(value);
        }
        
        data
    }
}

impl TensorGenerator for RandomTensorGenerator {
    fn generate(&mut self, spec: &TensorSpec) -> Result<TensorData> {
        let num_elements = spec.num_elements();
        
        let data = if spec.use_mxfp4 {
            // For MXFP4 tensors, generate appropriate quantized data
            // This will be handled by the MXFP4 module
            match spec.dtype {
                DType::UInt8 => self.generate_uint8(num_elements),
                DType::Int8 => self.generate_int8(num_elements),
                _ => {
                    return Err(crate::error::TensorMillError::tensor_generation(
                        format!("Invalid dtype for MXFP4 tensor: {:?}", spec.dtype)
                    ))
                }
            }
        } else {
            match spec.dtype {
                DType::Float32 => self.generate_float32(num_elements),
                DType::Float16 => self.generate_float16(num_elements),
                DType::BFloat16 => self.generate_bfloat16(num_elements),
                DType::Int8 => self.generate_int8(num_elements),
                DType::UInt8 => self.generate_uint8(num_elements),
                DType::MXFP4 => {
                    return Err(crate::error::TensorMillError::tensor_generation(
                        "MXFP4 dtype requires use_mxfp4 flag".to_string()
                    ))
                }
            }
        };
        
        let tensor = TensorData::new(spec.clone(), data);
        tensor.validate()?;
        Ok(tensor)
    }
    
    fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}

/// Deterministic tensor generator for testing
pub struct DeterministicTensorGenerator {
    value: f32,
}

impl DeterministicTensorGenerator {
    /// Create a new deterministic generator that fills all tensors with a specific value
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

impl TensorGenerator for DeterministicTensorGenerator {
    fn generate(&mut self, spec: &TensorSpec) -> Result<TensorData> {
        let num_elements = spec.num_elements();
        
        let data = match spec.dtype {
            DType::Float32 => {
                let mut data = Vec::with_capacity(num_elements * 4);
                for _ in 0..num_elements {
                    data.extend_from_slice(&self.value.to_le_bytes());
                }
                data
            }
            DType::BFloat16 => {
                let bits = self.value.to_bits();
                let bf16_bits = (bits >> 16) as u16;
                let mut data = Vec::with_capacity(num_elements * 2);
                for _ in 0..num_elements {
                    data.extend_from_slice(&bf16_bits.to_le_bytes());
                }
                data
            }
            DType::Float16 => {
                let f16_value = f16::from_f32(self.value);
                let mut data = Vec::with_capacity(num_elements * 2);
                for _ in 0..num_elements {
                    data.extend_from_slice(&f16_value.to_le_bytes());
                }
                data
            }
            DType::Int8 => vec![self.value as i8 as u8; num_elements],
            DType::UInt8 => vec![self.value as u8; num_elements],
            DType::MXFP4 => {
                return Err(crate::error::TensorMillError::tensor_generation(
                    "Deterministic MXFP4 generation not implemented".to_string()
                ))
            }
        };
        
        Ok(TensorData::new(spec.clone(), data))
    }
    
    fn set_seed(&mut self, _seed: u64) {
        // Deterministic generator doesn't use seeds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_tensor_generator() {
        let mut gen = RandomTensorGenerator::new(Some(42));
        let spec = TensorSpec {
            name: "test".to_string(),
            shape: vec![10, 20],
            dtype: DType::BFloat16,
            use_mxfp4: false,
        };
        
        let tensor = gen.generate(&spec).unwrap();
        assert_eq!(tensor.spec.name, "test");
        assert_eq!(tensor.data.len(), 400); // 200 elements * 2 bytes
    }
    
    #[test]
    fn test_deterministic_tensor_generator() {
        let mut gen = DeterministicTensorGenerator::new(1.0);
        let spec = TensorSpec {
            name: "test".to_string(),
            shape: vec![5, 5],
            dtype: DType::Float32,
            use_mxfp4: false,
        };
        
        let tensor = gen.generate(&spec).unwrap();
        assert_eq!(tensor.data.len(), 100); // 25 elements * 4 bytes
        
        // Verify all values are 1.0
        for i in (0..tensor.data.len()).step_by(4) {
            let bytes = [
                tensor.data[i],
                tensor.data[i + 1],
                tensor.data[i + 2],
                tensor.data[i + 3],
            ];
            let value = f32::from_le_bytes(bytes);
            assert_eq!(value, 1.0);
        }
    }
    
    #[test]
    fn test_tensor_validation() {
        let spec = TensorSpec {
            name: "test".to_string(),
            shape: vec![10],
            dtype: DType::Float32,
            use_mxfp4: false,
        };
        
        // Correct size
        let tensor = TensorData::new(spec.clone(), vec![0; 40]);
        assert!(tensor.validate().is_ok());
        
        // Incorrect size
        let tensor = TensorData::new(spec, vec![0; 20]);
        assert!(tensor.validate().is_err());
    }
}