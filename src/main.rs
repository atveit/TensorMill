//! TensorMill - Command-line interface for industrial tensor generation

use clap::{Parser, ValueEnum};
use tensormill::{
    ModelConfig, ModelType, ModelFormat, ModelSize,
    SyntheticGenerator, Result,
};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "tensormill",
    about = "Industrial-strength synthetic tensor generation for ML pipelines",
    version = env!("CARGO_PKG_VERSION"),
    author = "TensorMill Contributors"
)]
struct Cli {
    /// Model type to generate
    #[arg(short = 't', long, value_enum, default_value = "gpt-oss-20b")]
    model_type: CliModelType,
    
    /// Output format for the weights
    #[arg(short = 'f', long, value_enum, default_value = "sharded")]
    format: CliModelFormat,
    
    /// Model size variant
    #[arg(short = 's', long, value_enum, default_value = "compact")]
    size: CliModelSize,
    
    /// Output directory
    #[arg(short = 'o', long, default_value = "./synthetic_output")]
    output: PathBuf,
    
    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,
    
    /// Show progress bar
    #[arg(short = 'p', long)]
    progress: bool,
    
    /// Verbose output
    #[arg(short = 'v', long)]
    verbose: bool,
}

#[derive(Debug, ValueEnum, Clone, Copy)]
enum CliModelType {
    #[value(name = "gpt-oss-20b")]
    GptOss20B,
    #[value(name = "gpt-oss-120b")]
    GptOss120B,
}

#[derive(Debug, ValueEnum, Clone, Copy)]
enum CliModelFormat {
    Sharded,
    Unsharded,
    Original,
}

#[derive(Debug, ValueEnum, Clone, Copy)]
enum CliModelSize {
    Compact,
    Full,
}

impl From<CliModelType> for ModelType {
    fn from(cli: CliModelType) -> Self {
        match cli {
            CliModelType::GptOss20B => ModelType::GptOss20B,
            CliModelType::GptOss120B => ModelType::GptOss120B,
        }
    }
}

impl From<CliModelFormat> for ModelFormat {
    fn from(cli: CliModelFormat) -> Self {
        match cli {
            CliModelFormat::Sharded => ModelFormat::Sharded,
            CliModelFormat::Unsharded => ModelFormat::Unsharded,
            CliModelFormat::Original => ModelFormat::Original,
        }
    }
}

impl From<CliModelSize> for ModelSize {
    fn from(cli: CliModelSize) -> Self {
        match cli {
            CliModelSize::Compact => ModelSize::Compact,
            CliModelSize::Full => ModelSize::Full,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Setup logging
    let filter = if cli.verbose {
        EnvFilter::from_default_env()
            .add_directive("tensormill=debug".parse().unwrap())
    } else {
        EnvFilter::from_default_env()
            .add_directive("tensormill=info".parse().unwrap())
    };
    
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();
    
    // Print configuration
    println!("🔧 Configuration:");
    println!("  Model: {:?}", cli.model_type);
    println!("  Format: {:?}", cli.format);
    println!("  Size: {:?}", cli.size);
    println!("  Output: {}", cli.output.display());
    println!("  Seed: {}", cli.seed);
    println!();
    
    // Create model configuration
    let config = ModelConfig::new(
        cli.model_type.into(),
        cli.format.into(),
        cli.size.into(),
    );
    
    // Create generator
    let mut generator = SyntheticGenerator::new(config)
        .with_seed(cli.seed);
    
    if cli.progress {
        generator = generator.with_progress();
    }
    
    // Generate weights
    let result = generator.generate(&cli.output)?;
    
    // Print final summary
    println!();
    println!("🎉 Success! Synthetic weights generated at:");
    println!("   {}", result.output_dir.display());
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cli_parsing() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}