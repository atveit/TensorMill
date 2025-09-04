//! Main synthetic weight generator

use crate::config::{ModelConfig, DType};
use crate::error::{Result, TensorMillError};
use crate::metadata::MetadataGenerator;
use crate::mxfp4::MXFP4Quantizer;
use crate::sharding::{ShardingStrategy, ShardData};
use crate::tensor::{TensorData, RandomTensorGenerator, TensorGenerator};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use safetensors::{serialize, Dtype};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Result of synthetic generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub output_dir: PathBuf,
    pub num_tensors: usize,
    pub total_size: usize,
    pub num_shards: usize,
    pub generation_time: std::time::Duration,
    pub export_time: std::time::Duration,
}

impl GenerationResult {
    /// Print a summary of the generation
    pub fn print_summary(&self) {
        println!("✅ Generation Complete!");
        println!("  📁 Output: {}", self.output_dir.display());
        println!("  📊 Tensors: {}", self.num_tensors);
        println!("  💾 Size: {:.2} GB", self.total_size as f64 / 1e9);
        println!("  📦 Shards: {}", self.num_shards);
        println!("  ⏱️  Generation: {:.2}s", self.generation_time.as_secs_f64());
        println!("  ⏱️  Export: {:.2}s", self.export_time.as_secs_f64());
        println!(
            "  🚀 Total: {:.2}s",
            (self.generation_time + self.export_time).as_secs_f64()
        );
    }
}

/// Main synthetic weight generator
pub struct SyntheticGenerator {
    config: ModelConfig,
    seed: Option<u64>,
    progress_bar: Option<ProgressBar>,
}

impl SyntheticGenerator {
    /// Create a new synthetic generator
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            seed: Some(42),
            progress_bar: None,
        }
    }
    
    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Enable progress bar
    pub fn with_progress(mut self) -> Self {
        self.progress_bar = Some(ProgressBar::new(0));
        self
    }
    
    /// Generate synthetic weights and metadata
    pub fn generate(&mut self, output_dir: impl AsRef<Path>) -> Result<GenerationResult> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir)?;
        
        println!("🚀 Generating synthetic {} weights...", self.config.model_type.as_str());
        
        // Generate metadata first
        self.generate_metadata(output_dir)?;
        
        // Generate tensors
        let generation_start = Instant::now();
        let tensors = self.generate_tensors()?;
        let generation_time = generation_start.elapsed();
        
        println!(
            "✅ Generated {} tensors in {:.2}s",
            tensors.len(),
            generation_time.as_secs_f64()
        );
        
        // Export to safetensors format
        let export_start = Instant::now();
        let num_shards = self.export_tensors(output_dir, tensors)?;
        let export_time = export_start.elapsed();
        
        // Calculate total size
        let total_size = self.calculate_total_size(output_dir)?;
        
        let result = GenerationResult {
            output_dir: output_dir.to_path_buf(),
            num_tensors: self.config.get_tensor_specs().len(),
            total_size,
            num_shards,
            generation_time,
            export_time,
        };
        
        result.print_summary();
        Ok(result)
    }
    
    /// Generate metadata files
    fn generate_metadata(&self, output_dir: &Path) -> Result<()> {
        let generator = MetadataGenerator::new(self.config.clone());
        generator.generate_all(output_dir)?;
        generator.generate_tokenizer_json(output_dir)?;
        Ok(())
    }
    
    /// Generate all tensors
    fn generate_tensors(&mut self) -> Result<Vec<TensorData>> {
        let specs = self.config.get_tensor_specs();
        let total = specs.len();
        
        // Setup progress bar
        if let Some(pb) = &self.progress_bar {
            pb.set_length(total as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
        }
        
        // Generate tensors in parallel
        let seed = self.seed.unwrap_or(42);
        let progress_bar = self.progress_bar.clone();
        
        let tensors: Result<Vec<_>> = specs
            .into_par_iter()
            .enumerate()
            .map(|(idx, spec)| {
                // Create a generator with a unique seed for this tensor
                let tensor_seed = seed + idx as u64;
                let mut generator = RandomTensorGenerator::new(Some(tensor_seed));
                
                // Handle MXFP4 tensors specially
                let tensor = if spec.use_mxfp4 {
                    self.generate_mxfp4_tensor(&spec, tensor_seed)?
                } else {
                    generator.generate(&spec)?
                };
                
                // Update progress
                if let Some(pb) = &progress_bar {
                    pb.inc(1);
                    pb.set_message(format!("Generated {}", spec.name));
                }
                
                Ok(tensor)
            })
            .collect();
        
        if let Some(pb) = &self.progress_bar {
            pb.finish_with_message("✅ Tensor generation complete");
        }
        
        tensors
    }
    
    /// Generate MXFP4 quantized tensor
    fn generate_mxfp4_tensor(&self, spec: &crate::config::TensorSpec, seed: u64) -> Result<TensorData> {
        let mut quantizer = MXFP4Quantizer::new(Some(seed));
        
        // For blocks tensors, generate MXFP4 data
        if spec.name.contains("blocks") {
            // The shape is already halved (e.g., [32, 5760, 1440] instead of [32, 5760, 2880])
            // Each byte will contain 2 4-bit values, so we generate exactly spec.num_elements() bytes
            let num_bytes = spec.num_elements();
            let logical_elements = num_bytes * 2; // Each byte holds 2 4-bit values
            
            let (blocks, _scales) = quantizer.generate_mxfp4_tensor(logical_elements)?;
            // Truncate to exact size needed (in case of rounding)
            let mut packed_data = blocks;
            packed_data.truncate(num_bytes);
            
            Ok(TensorData::new(spec.clone(), packed_data))
        }
        // For scales tensors, generate scale values
        else if spec.name.contains("scales") {
            let num_scales = spec.num_elements();
            let (_blocks, scales) = quantizer.generate_mxfp4_tensor(num_scales * 32)?;
            let scale_bytes: Vec<u8> = scales.iter().take(num_scales).map(|&s| s as u8).collect();
            Ok(TensorData::new(spec.clone(), scale_bytes))
        } else {
            // Non-MXFP4 tensor marked with use_mxfp4=true (shouldn't happen)
            let mut generator = RandomTensorGenerator::new(Some(seed));
            generator.generate(spec)
        }
    }
    
    /// Export tensors to safetensors format
    fn export_tensors(&self, output_dir: &Path, tensors: Vec<TensorData>) -> Result<usize> {
        // Distribute tensors across shards
        let strategy = ShardingStrategy::new(self.config.clone());
        let shards = strategy.distribute_tensors(tensors)?;
        let num_shards = shards.len();
        
        println!("📦 Exporting to {} shard(s)...", num_shards);
        
        // Export each shard
        for shard in shards {
            self.export_shard(output_dir, shard)?;
        }
        
        Ok(num_shards)
    }
    
    /// Export a single shard
    fn export_shard(&self, output_dir: &Path, shard: ShardData) -> Result<()> {
        use safetensors::tensor::TensorView;
        
        let path = output_dir.join(&shard.filename);
        
        // Build tensors for serialization
        let tensors: Vec<(String, TensorView)> = shard.tensors
            .iter()
            .map(|tensor| {
                // Determine dtype for safetensors
                let dtype = match tensor.spec.dtype {
                    DType::Float32 => Dtype::F32,
                    DType::Float16 => Dtype::F16,
                    DType::BFloat16 => Dtype::BF16,
                    DType::Int8 => Dtype::I8,
                    DType::UInt8 => Dtype::U8,
                    DType::MXFP4 => {
                        // MXFP4 tensors are stored as uint8
                        if tensor.spec.name.contains("scales") {
                            Dtype::I8
                        } else {
                            Dtype::U8
                        }
                    }
                };
                
                // Use the tensor shape as-is (already adjusted for MXFP4 packing)
                let shape = tensor.spec.shape.clone();
                
                // Create TensorView
                let view = TensorView::new(
                    dtype,
                    shape,
                    &tensor.data,
                ).unwrap();
                
                (tensor.spec.name.clone(), view)
            })
            .collect();
        
        // Serialize and write
        let serialized = serialize(tensors, &None)
            .map_err(|e| TensorMillError::Serialization(format!("Failed to serialize: {:?}", e)))?;
        
        fs::write(&path, serialized)?;
        
        println!(
            "  ✅ Exported {} ({} tensors, {:.2} GB)",
            shard.filename,
            shard.tensor_count(),
            shard.total_size() as f64 / 1e9
        );
        
        Ok(())
    }
    
    /// Calculate total size of generated files
    fn calculate_total_size(&self, output_dir: &Path) -> Result<usize> {
        let mut total = 0;
        for entry in fs::read_dir(output_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("safetensors") {
                total += entry.metadata()?.len() as usize;
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelType, ModelFormat, ModelSize};
    use tempfile::tempdir;
    
    #[test]
    fn test_generation_compact() {
        let config = ModelConfig::new(
            ModelType::GptOss20B,
            ModelFormat::Unsharded,
            ModelSize::Compact,
        );
        
        let mut generator = SyntheticGenerator::new(config);
        let temp_dir = tempdir().unwrap();
        
        let result = generator.generate(temp_dir.path()).unwrap();
        
        assert!(result.num_tensors > 0);
        assert_eq!(result.num_shards, 1);
        assert!(temp_dir.path().join("model.safetensors").exists());
        assert!(temp_dir.path().join("config.json").exists());
    }
    
    #[test]
    fn test_metadata_generation() {
        let config = ModelConfig::new(
            ModelType::GptOss20B,
            ModelFormat::Sharded,
            ModelSize::Compact,
        );
        
        let generator = SyntheticGenerator::new(config);
        let temp_dir = tempdir().unwrap();
        
        generator.generate_metadata(temp_dir.path()).unwrap();
        
        assert!(temp_dir.path().join("config.json").exists());
        assert!(temp_dir.path().join("tokenizer_config.json").exists());
        assert!(temp_dir.path().join("generation_config.json").exists());
        assert!(temp_dir.path().join("special_tokens_map.json").exists());
        assert!(temp_dir.path().join("tokenizer.json").exists());
    }
}