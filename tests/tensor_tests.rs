//! Unit tests for tensor generation

#[cfg(test)]
mod tests {
    use tensormill::tensor::{
        TensorData, TensorGenerator, RandomTensorGenerator, DeterministicTensorGenerator,
    };
    use tensormill::config::{TensorSpec, DType};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_tensor_size_calculation() {
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
    fn test_mxfp4_tensor_size() {
        let spec = TensorSpec {
            name: "test.blocks".to_string(),
            shape: vec![32, 100, 100],
            dtype: DType::UInt8,
            use_mxfp4: true,
        };
        
        let num_elements = spec.num_elements();
        assert_eq!(num_elements, 320_000);
        
        // MXFP4 uses 4 bits per element + scales
        let expected_size = (num_elements / 2) + (num_elements / 32);
        assert_eq!(spec.size_bytes(), expected_size);
    }
    
    #[test]
    fn test_random_tensor_generation() {
        let mut gen = RandomTensorGenerator::new(Some(42));
        
        let spec = TensorSpec {
            name: "test".to_string(),
            shape: vec![10, 10],
            dtype: DType::Float32,
            use_mxfp4: false,
        };
        
        let tensor = gen.generate(&spec).unwrap();
        
        assert_eq!(tensor.spec.name, "test");
        assert_eq!(tensor.data.len(), 400); // 100 * 4 bytes
        
        // Verify it's not all zeros
        let all_zeros = tensor.data.iter().all(|&b| b == 0);
        assert!(!all_zeros);
    }
    
    #[test]
    fn test_deterministic_tensor_generation() {
        let mut gen = DeterministicTensorGenerator::new(1.5);
        
        let spec = TensorSpec {
            name: "test".to_string(),
            shape: vec![5, 5],
            dtype: DType::Float32,
            use_mxfp4: false,
        };
        
        let tensor = gen.generate(&spec).unwrap();
        
        // Check all values are 1.5
        for i in (0..tensor.data.len()).step_by(4) {
            let bytes = [
                tensor.data[i],
                tensor.data[i + 1],
                tensor.data[i + 2],
                tensor.data[i + 3],
            ];
            let value = f32::from_le_bytes(bytes);
            assert_relative_eq!(value, 1.5, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_bfloat16_generation() {
        let mut gen = RandomTensorGenerator::new(Some(100));
        
        let spec = TensorSpec {
            name: "bf16_test".to_string(),
            shape: vec![50],
            dtype: DType::BFloat16,
            use_mxfp4: false,
        };
        
        let tensor = gen.generate(&spec).unwrap();
        assert_eq!(tensor.data.len(), 100); // 50 * 2 bytes
        
        // Verify BFloat16 format (can be decoded)
        for i in (0..tensor.data.len()).step_by(2) {
            let bf16_bits = u16::from_le_bytes([tensor.data[i], tensor.data[i + 1]]);
            // BFloat16 to float32 conversion
            let f32_bits = (bf16_bits as u32) << 16;
            let _value = f32::from_bits(f32_bits);
            // Just verify it doesn't panic
        }
    }
    
    #[test]
    fn test_int8_generation() {
        let mut gen = RandomTensorGenerator::new(Some(200));
        
        let spec = TensorSpec {
            name: "int8_test".to_string(),
            shape: vec![100],
            dtype: DType::Int8,
            use_mxfp4: false,
        };
        
        let tensor = gen.generate(&spec).unwrap();
        assert_eq!(tensor.data.len(), 100);
        
        // Check values are in int8 range
        for &byte in &*tensor.data {
            let value = byte as i8;
            assert!(value >= -128 && value <= 127);
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
        let correct_tensor = TensorData::new(spec.clone(), vec![0; 40]);
        assert!(correct_tensor.validate().is_ok());
        
        // Incorrect size
        let incorrect_tensor = TensorData::new(spec, vec![0; 30]);
        assert!(incorrect_tensor.validate().is_err());
    }
    
    #[test]
    fn test_seed_determinism() {
        let spec = TensorSpec {
            name: "test".to_string(),
            shape: vec![100],
            dtype: DType::Float32,
            use_mxfp4: false,
        };
        
        // Same seed should produce same data
        let mut gen1 = RandomTensorGenerator::new(Some(42));
        let tensor1 = gen1.generate(&spec).unwrap();
        
        let mut gen2 = RandomTensorGenerator::new(Some(42));
        let tensor2 = gen2.generate(&spec).unwrap();
        
        assert_eq!(tensor1.data, tensor2.data);
        
        // Different seeds should produce different data
        let mut gen3 = RandomTensorGenerator::new(Some(43));
        let tensor3 = gen3.generate(&spec).unwrap();
        
        assert_ne!(tensor1.data, tensor3.data);
    }
    
    #[test]
    fn test_large_tensor_generation() {
        let mut gen = RandomTensorGenerator::new(Some(300));
        
        let spec = TensorSpec {
            name: "large".to_string(),
            shape: vec![1000, 1000],
            dtype: DType::BFloat16,
            use_mxfp4: false,
        };
        
        let tensor = gen.generate(&spec).unwrap();
        assert_eq!(tensor.data.len(), 2_000_000); // 1M * 2 bytes
    }
}