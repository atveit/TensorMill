//! MXFP4 quantization for MoE expert weights

use crate::error::Result;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// MXFP4 quantizer for 4-bit microscaling format
pub struct MXFP4Quantizer {
    block_size: usize,
    rng: StdRng,
}

impl MXFP4Quantizer {
    /// Create a new MXFP4 quantizer
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            block_size: 32,
            rng,
        }
    }
    
    /// Valid MXFP4 values (4-bit representation)
    const MXFP4_VALUES: [f32; 16] = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, // Last is NaN representation
    ];
    
    /// Generate MXFP4 blocks for a given shape
    /// Returns (blocks, scales) where blocks are uint8 and scales are int8
    pub fn generate_mxfp4_tensor(
        &mut self,
        num_elements: usize,
    ) -> Result<(Vec<u8>, Vec<i8>)> {
        let num_blocks = (num_elements + self.block_size - 1) / self.block_size;
        
        // Generate blocks (4 bits per element, packed into uint8)
        let mut blocks = Vec::with_capacity(num_elements / 2);
        let mut scales = Vec::with_capacity(num_blocks);
        
        for _ in 0..num_blocks {
            // Generate a random scale for this block
            let scale = self.rng.gen_range(-127..=127) as i8;
            scales.push(scale);
            
            // Generate 32 4-bit values for this block
            for _ in 0..(self.block_size / 2) {
                // Pack two 4-bit values into one byte
                let val1 = self.rng.gen_range(0..15) as u8;
                let val2 = self.rng.gen_range(0..15) as u8;
                let packed = (val1 << 4) | val2;
                blocks.push(packed);
            }
        }
        
        Ok((blocks, scales))
    }
    
    /// Generate synthetic MXFP4 data matching GPT-OSS format
    pub fn generate_gpt_oss_mxfp4(
        &mut self,
        num_experts: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<u8>, Vec<i8>)> {
        let total_elements = num_experts * rows * cols;
        self.generate_mxfp4_tensor(total_elements)
    }
    
    /// Quantize float32 values to MXFP4 format
    pub fn quantize(&self, values: &[f32]) -> Result<(Vec<u8>, Vec<i8>)> {
        let num_blocks = (values.len() + self.block_size - 1) / self.block_size;
        let mut blocks = Vec::with_capacity(values.len() / 2);
        let mut scales = Vec::with_capacity(num_blocks);
        
        for block_idx in 0..num_blocks {
            let start = block_idx * self.block_size;
            let end = ((block_idx + 1) * self.block_size).min(values.len());
            let block_values = &values[start..end];
            
            // Find max absolute value for scaling
            let max_abs = block_values
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            
            // Calculate scale (power of 2)
            let scale_power = if max_abs > 0.0 {
                (max_abs.ln() / 2.0_f32.ln()).ceil() as i8
            } else {
                0
            };
            scales.push(scale_power);
            
            // Quantize values in this block
            let scale_factor = 2.0_f32.powi(scale_power as i32);
            for chunk in block_values.chunks(2) {
                let val1 = self.quantize_single(chunk[0] / scale_factor);
                let val2 = if chunk.len() > 1 {
                    self.quantize_single(chunk[1] / scale_factor)
                } else {
                    0
                };
                let packed = (val1 << 4) | val2;
                blocks.push(packed);
            }
        }
        
        Ok((blocks, scales))
    }
    
    /// Quantize a single value to 4-bit MXFP4
    fn quantize_single(&self, value: f32) -> u8 {
        // Find nearest MXFP4 value
        let mut best_idx = 0;
        let mut best_diff = f32::INFINITY;
        
        for (idx, &mxfp4_val) in Self::MXFP4_VALUES[..15].iter().enumerate() {
            let diff = (value - mxfp4_val).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = idx;
            }
        }
        
        best_idx as u8
    }
    
    /// Dequantize MXFP4 format back to float32
    pub fn dequantize(&self, blocks: &[u8], scales: &[i8]) -> Result<Vec<f32>> {
        let mut values = Vec::new();
        
        for (block_idx, &scale) in scales.iter().enumerate() {
            let scale_factor = 2.0_f32.powi(scale as i32);
            let block_start = block_idx * self.block_size / 2;
            let block_end = ((block_idx + 1) * self.block_size / 2).min(blocks.len());
            
            for &packed in &blocks[block_start..block_end] {
                let val1_idx = (packed >> 4) & 0x0F;
                let val2_idx = packed & 0x0F;
                
                let val1 = Self::MXFP4_VALUES[val1_idx as usize] * scale_factor;
                let val2 = Self::MXFP4_VALUES[val2_idx as usize] * scale_factor;
                
                values.push(val1);
                values.push(val2);
            }
        }
        
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mxfp4_generation() {
        let mut quantizer = MXFP4Quantizer::new(Some(42));
        let (blocks, scales) = quantizer.generate_mxfp4_tensor(64).unwrap();
        
        // Should have 32 bytes (64 elements / 2)
        assert_eq!(blocks.len(), 32);
        // Should have 2 scales (64 elements / 32 block_size)
        assert_eq!(scales.len(), 2);
    }
    
    #[test]
    fn test_mxfp4_quantization_roundtrip() {
        let quantizer = MXFP4Quantizer::new(Some(42));
        let values = vec![0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0]; // 8 values
        
        let (blocks, scales) = quantizer.quantize(&values).unwrap();
        let dequantized = quantizer.dequantize(&blocks, &scales).unwrap();
        
        // Check that we get the right number of values back
        assert_eq!(dequantized.len(), values.len());
        
        // Check that values are approximately preserved
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            // MXFP4 has limited precision, so we allow some error
            assert!((orig - deq).abs() < 2.0, "Value {} became {}", orig, deq);
        }
    }
    
    #[test]
    fn test_gpt_oss_mxfp4_generation() {
        let mut quantizer = MXFP4Quantizer::new(Some(42));
        let (blocks, scales) = quantizer.generate_gpt_oss_mxfp4(32, 2880, 2880).unwrap();
        
        let total_elements = 32 * 2880 * 2880;
        let expected_blocks = total_elements / 2;
        let expected_scales = (total_elements + 31) / 32;
        
        assert_eq!(blocks.len(), expected_blocks);
        assert_eq!(scales.len(), expected_scales);
    }
}