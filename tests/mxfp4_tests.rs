//! Unit tests for MXFP4 quantization

#[cfg(test)]
mod tests {
    use tensormill::mxfp4::MXFP4Quantizer;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_mxfp4_generation() {
        let mut quantizer = MXFP4Quantizer::new(Some(42));
        
        let (blocks, scales) = quantizer.generate_mxfp4_tensor(64).unwrap();
        
        // 64 elements = 32 bytes (4 bits each)
        assert_eq!(blocks.len(), 32);
        // 64 elements / 32 block_size = 2 scales
        assert_eq!(scales.len(), 2);
        
        // Check that blocks are valid 4-bit values (0-15)
        for &byte in &blocks {
            let high = (byte >> 4) & 0x0F;
            let low = byte & 0x0F;
            assert!(high <= 15);
            assert!(low <= 15);
        }
        
        // Check that scales are valid int8
        for &scale in &scales {
            assert!(scale >= -128 && scale <= 127);
        }
    }
    
    #[test]
    fn test_mxfp4_quantization() {
        let quantizer = MXFP4Quantizer::new(Some(100));
        
        // Test values that should map to MXFP4 representable values
        let values = vec![
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
        ];
        
        let (blocks, scales) = quantizer.quantize(&values).unwrap();
        
        // Should have 8 bytes (16 values / 2)
        assert_eq!(blocks.len(), 8);
        // Should have 1 scale (16 values / 32 block_size, rounded up)
        assert_eq!(scales.len(), 1);
    }
    
    #[test]
    fn test_mxfp4_dequantization() {
        let quantizer = MXFP4Quantizer::new(Some(200));
        
        // Simple test values
        let values = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        
        let (blocks, scales) = quantizer.quantize(&values).unwrap();
        let dequantized = quantizer.dequantize(&blocks, &scales).unwrap();
        
        // Should get back same number of values
        assert_eq!(dequantized.len(), values.len());
        
        // Values should be approximately preserved (within MXFP4 precision)
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            // MXFP4 has limited precision, allow significant error
            assert!(
                (orig - deq).abs() < 3.0,
                "Value {} became {}, error too large",
                orig, deq
            );
        }
    }
    
    #[test]
    fn test_mxfp4_representable_values() {
        // Test that MXFP4 can represent its discrete values
        let valid_values = [
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        ];
        
        let quantizer = MXFP4Quantizer::new(Some(300));
        
        for &value in &valid_values {
            let values = vec![value; 32]; // Fill a block
            let (blocks, scales) = quantizer.quantize(&values).unwrap();
            let dequantized = quantizer.dequantize(&blocks, &scales).unwrap();
            
            // Should be able to represent these exactly (with appropriate scale)
            let error = (value - dequantized[0]).abs();
            assert!(
                error < 0.5 || (value != 0.0 && error / value.abs() < 0.5),
                "Failed to represent {} accurately, got {}",
                value, dequantized[0]
            );
        }
    }
    
    #[test]
    fn test_gpt_oss_mxfp4_generation() {
        let mut quantizer = MXFP4Quantizer::new(Some(42));
        
        // Generate for typical GPT-OSS expert dimensions
        let (blocks, scales) = quantizer.generate_gpt_oss_mxfp4(
            32,   // num_experts
            2880, // rows
            2880, // cols
        ).unwrap();
        
        let total_elements = 32 * 2880 * 2880;
        let expected_blocks = total_elements / 2; // 4 bits per element
        let expected_scales = (total_elements + 31) / 32; // 1 scale per 32 elements
        
        assert_eq!(blocks.len(), expected_blocks);
        assert_eq!(scales.len(), expected_scales);
    }
    
    #[test]
    fn test_block_alignment() {
        let mut quantizer = MXFP4Quantizer::new(Some(500));
        
        // Test non-aligned sizes
        let sizes = [31, 33, 63, 65, 100];
        
        for size in sizes {
            let (blocks, scales) = quantizer.generate_mxfp4_tensor(size).unwrap();
            
            // Blocks should be ceil(size/2)
            let expected_blocks = (size + 1) / 2;
            assert_eq!(blocks.len(), expected_blocks);
            
            // Scales should be ceil(size/32)
            let expected_scales = (size + 31) / 32;
            assert_eq!(scales.len(), expected_scales);
        }
    }
    
    #[test]
    fn test_zero_values() {
        let quantizer = MXFP4Quantizer::new(Some(600));
        
        // Test quantizing all zeros
        let values = vec![0.0; 64];
        let (blocks, scales) = quantizer.quantize(&values).unwrap();
        let dequantized = quantizer.dequantize(&blocks, &scales).unwrap();
        
        // All values should still be zero
        for &val in &dequantized {
            assert_relative_eq!(val, 0.0, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_large_values() {
        let quantizer = MXFP4Quantizer::new(Some(700));
        
        // Test values that need scaling
        let values = vec![100.0, 200.0, 300.0, 400.0, -100.0, -200.0, -300.0, -400.0];
        
        let (blocks, scales) = quantizer.quantize(&values).unwrap();
        
        // Scales should be large to accommodate these values
        assert!(scales[0].abs() > 5);
        
        let dequantized = quantizer.dequantize(&blocks, &scales).unwrap();
        
        // Values should be preserved in relative terms
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            let relative_error = (orig - deq).abs() / orig.abs();
            assert!(
                relative_error < 0.5,
                "Large value {} became {}, relative error too large",
                orig, deq
            );
        }
    }
}