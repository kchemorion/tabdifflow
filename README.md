# TabularDiffFlow

A novel hybrid framework combining diffusion models and normalizing flows for high-quality synthetic tabular data generation.

## Overview

TabularDiffFlow is a state-of-the-art framework for generating synthetic tabular data with high statistical fidelity and machine learning utility. This framework combines:

- **Diffusion Models**: For capturing the global structure of the data
- **Normalizing Flows**: For preserving fine-grained details
- **Feature Attention**: For modeling complex dependencies between features

## Key Features

- **High-quality synthetic data generation** with superior statistical properties
- **Mixed data type support** (continuous and categorical)
- **GPU acceleration** for faster training and generation
- **Built-in benchmark system** comparing against SOTA models
- **Privacy-preserving** data synthesis with configurable privacy guarantees
- **Flexible API** for easy integration into existing workflows

## Installation

```bash
# Clone the repository
git clone https://github.com/kchemorion/tabdifflow.git
cd tabdifflow

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from tabular_diffflow import train_tabular_diffflow_model, generate_synthetic_data
import pandas as pd

# Load your tabular dataset
df = pd.read_csv('your_dataset.csv')

# Train TabularDiffFlow model
model = train_tabular_diffflow_model(
    df,
    n_epochs=100,
    batch_size=64,
    device='cuda'  # Use GPU if available
)

# Generate synthetic data
synthetic_df = generate_synthetic_data(
    model,
    n_samples=1000
)

# Save the synthetic data
synthetic_df.to_csv('synthetic_data.csv', index=False)
```

### Running Benchmarks

TabularDiffFlow includes a comprehensive benchmarking system that compares it against other state-of-the-art models:

```bash
python tabdifflow.py --data your_dataset.csv --target target_column --task classification
```

Options:
- `--data`: Path to CSV dataset
- `--target`: Target column name for ML utility evaluation
- `--task`: Task type ('classification' or 'regression')
- `--epochs`: Number of training epochs
- `--samples`: Number of synthetic samples to generate
- `--device`: Device to use ('cuda' or 'cpu')
- `--models`: Specific models to include in benchmark

### Advanced Usage

For more control over the generation process:

```python
synthetic_df = generate_synthetic_data(
    model,
    n_samples=1000,
    device='cuda',
    temp=0.8,           # Temperature (lower = more conservative samples)
    guidance_scale=1.5, # Guidance scale (higher = better quality but less diversity)
    batch_size=256      # For memory-efficient generation
)
```

## Model Architecture

TabularDiffFlow combines several innovative components:

1. **Adaptive Noise Schedule**: Learns an optimal noise schedule for the diffusion process
2. **Sparse Feature Attention**: Captures complex dependencies between features
3. **Normalizing Flow**: Preserves fine-grained details lost in pure diffusion approaches
4. **Hybrid Loss Function**: Balances diffusion and flow components for optimal learning

## Benchmark Results

TabularDiffFlow consistently outperforms existing models:

- **Higher statistical fidelity**: Better preservation of feature distributions and correlations
- **Improved ML utility**: Models trained on TabularDiffFlow's synthetic data achieve higher accuracy on real test data
- **Enhanced privacy guarantees**: Synthetic data provides strong privacy protections while maintaining utility

## Citation

If you use TabularDiffFlow in your research, please cite:
```
@article{tabdifflow2023,
  title={TabularDiffFlow: A Hybrid Diffusion-Flow Framework for Synthetic Tabular Data Generation},
  author={Chemor, Kevin},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.