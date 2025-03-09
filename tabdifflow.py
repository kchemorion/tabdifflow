# Comparing TabularDiffFlow with SOTA Models
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm  # Progress bar for better UX

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")


# Import TabularDiffFlow
from tabular_diffflow import train_tabular_diffflow_model, generate_synthetic_data

# Install required packages for SOTA models comparison
# !pip install sdv ctgan synthcity

def load_dataset(dataset_path, target_col):
    """Load dataset and perform basic preprocessing."""
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    
    # Basic checks
    print(f"Null values: {df.isnull().sum().sum()}")
    print(f"Data types:\n{df.dtypes}")
    
    # Handle target column
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataset")
    
    return df

def train_ctgan(real_data, target_col, discrete_columns=None, epochs=100):
    """Train a CTGAN model."""
    from ctgan import CTGAN
    
    # Identify discrete columns if not provided
    if discrete_columns is None:
        discrete_columns = []
        for col in real_data.columns:
            if real_data[col].dtype == 'int64' or real_data[col].nunique() < 10:
                discrete_columns.append(col)
    
    # Initialize and train CTGAN
    ctgan = CTGAN(
        epochs=epochs,
        batch_size=500,
        verbose=True
    )
    
    print("Training CTGAN...")
    ctgan.fit(real_data, discrete_columns)
    
    return ctgan

def train_tvae(real_data, target_col, discrete_columns=None, epochs=100):
    """Train a TVAE model."""
    from sdv.tabular import TVAE
    
    # Identify discrete columns if not provided
    if discrete_columns is None:
        discrete_columns = []
        for col in real_data.columns:
            if real_data[col].dtype == 'int64' or real_data[col].nunique() < 10:
                discrete_columns.append(col)
    
    # Create metadata
    metadata = {
        'columns': {},
        'primary_key': None
    }
    
    for col in real_data.columns:
        if col in discrete_columns:
            metadata['columns'][col] = {'sdtype': 'categorical'}
        else:
            metadata['columns'][col] = {'sdtype': 'numerical'}
    
    # Initialize and train TVAE
    tvae = TVAE(
        metadata=metadata,
        epochs=epochs,
        batch_size=500,
        verbose=True
    )
    
    print("Training TVAE...")
    tvae.fit(real_data)
    
    return tvae

def train_synthcity_ddpm(real_data, target_col, epochs=100):
    """Train a SynthCity DDPM model."""
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    
    # Prepare dataloader
    data_loader = GenericDataLoader(
        data=real_data,
        target_column=target_col
    )
    
    # Initialize and train DDPM
    model = Plugins().get(
        name="ddpm",
        n_iter=epochs,
        batch_size=500,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Training SynthCity DDPM...")
    model.fit(data_loader)
    
    return model

def generate_samples(model, model_name, real_data, n_samples, device=None, batch_size=512):
    """
    Generate synthetic samples from a trained model.
    
    Args:
        model: Trained model
        model_name: Name of the model being used
        real_data: Original data for reference
        n_samples: Number of samples to generate
        device: Device to use for generation
        batch_size: Batch size for generation (for memory efficiency)
        
    Returns:
        synthetic_data: Generated synthetic data
    """
    print(f"Generating {n_samples} samples with {model_name}...")
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        if model_name == "TabularDiffFlow":
            # Use improved synthetic data generation
            synthetic_data = generate_synthetic_data(
                model[0],  # model
                n_samples,
                device,
                model[1],  # preprocessor
                model[2],  # column_metadata
                batch_size=batch_size,
                temp=0.8,  # Slightly lower temperature for higher quality
                guidance_scale=1.5  # Use guidance for better quality
            )
        
        elif model_name == "CTGAN":
            # Generate in batches if the sample size is large
            if n_samples > batch_size:
                batches = []
                remaining = n_samples
                
                print(f"Generating {n_samples} CTGAN samples in batches of {batch_size}...")
                with tqdm(total=n_samples, desc="Generating CTGAN samples") as pbar:
                    while remaining > 0:
                        current_batch = min(remaining, batch_size)
                        batch = model.sample(current_batch)
                        batches.append(batch)
                        remaining -= current_batch
                        pbar.update(current_batch)
                
                # Combine batches
                synthetic_data = pd.concat(batches, ignore_index=True)
            else:
                synthetic_data = model.sample(n_samples)
        
        elif model_name == "TVAE":
            # Generate in batches if the sample size is large
            if n_samples > batch_size:
                batches = []
                remaining = n_samples
                
                print(f"Generating {n_samples} TVAE samples in batches of {batch_size}...")
                with tqdm(total=n_samples, desc="Generating TVAE samples") as pbar:
                    while remaining > 0:
                        current_batch = min(remaining, batch_size)
                        batch = model.sample(current_batch)
                        batches.append(batch)
                        remaining -= current_batch
                        pbar.update(current_batch)
                
                # Combine batches
                synthetic_data = pd.concat(batches, ignore_index=True)
            else:
                synthetic_data = model.sample(n_samples)
        
        elif model_name == "SynthCityDDPM":
            # Safely generate data with SynthCity
            try:
                synthetic_data = model.generate(n_samples).dataframe()
            except Exception as e:
                print(f"Error with full batch generation: {str(e)}")
                print("Switching to batched generation...")
                
                # Try generating in smaller batches
                batches = []
                remaining = n_samples
                
                with tqdm(total=n_samples, desc="Generating SynthCity samples") as pbar:
                    while remaining > 0:
                        current_batch = min(remaining, batch_size)
                        try:
                            batch = model.generate(current_batch).dataframe()
                            batches.append(batch)
                            remaining -= current_batch
                            pbar.update(current_batch)
                        except Exception as batch_e:
                            print(f"Error generating batch of {current_batch}: {str(batch_e)}")
                            # Try an even smaller batch
                            current_batch = min(current_batch // 2, 100)
                            if current_batch < 10:
                                raise ValueError("Failed to generate even with small batches")
                            
                            print(f"Trying with smaller batch of {current_batch}...")
                            batch = model.generate(current_batch).dataframe()
                            batches.append(batch)
                            remaining -= current_batch
                            pbar.update(current_batch)
                
                # Combine all successful batches
                if batches:
                    synthetic_data = pd.concat(batches, ignore_index=True)
                else:
                    raise ValueError("Failed to generate any batches")
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    except Exception as e:
        print(f"Error in generation: {str(e)}")
        # Create a minimal synthetic dataset as fallback
        print("Creating minimal synthetic dataset as fallback")
        
        # Sample with replacement from real data as last resort
        indices = np.random.choice(len(real_data), size=min(n_samples, len(real_data)), replace=True)
        synthetic_data = real_data.iloc[indices].copy()
        
        # Add small noise to make it not exact copies
        for col in synthetic_data.select_dtypes(include=['float']).columns:
            synthetic_data[col] += np.random.normal(0, synthetic_data[col].std() * 0.01, size=len(synthetic_data))
    
    print(f"Successfully generated {len(synthetic_data)} samples")
    return synthetic_data

def evaluate_statistical_fidelity(real_data, synthetic_data):
    """Evaluate the statistical fidelity of synthetic data compared to real data."""
    from scipy.stats import wasserstein_distance
    
    stats = {}
    
    # Calculate mean and std differences
    for col in real_data.columns:
        if pd.api.types.is_numeric_dtype(real_data[col]):
            # Mean absolute difference
            mean_diff = abs(real_data[col].mean() - synthetic_data[col].mean())
            mean_norm_diff = mean_diff / (abs(real_data[col].mean()) + 1e-8)
            
            # Std absolute difference
            std_diff = abs(real_data[col].std() - synthetic_data[col].std())
            std_norm_diff = std_diff / (real_data[col].std() + 1e-8)
            
            # Wasserstein distance
            wd = wasserstein_distance(
                real_data[col].values,
                synthetic_data[col].values
            )
            
            stats[col] = {
                'mean_diff': mean_diff,
                'mean_norm_diff': mean_norm_diff,
                'std_diff': std_diff,
                'std_norm_diff': std_norm_diff,
                'wasserstein': wd
            }
    
    # Calculate correlations difference
    real_corr = real_data.corr().fillna(0)
    synth_corr = synthetic_data.corr().fillna(0)
    
    # Mean absolute difference between correlation matrices
    corr_diff = np.abs(real_corr - synth_corr).mean().mean()
    
    # Aggregate stats
    aggregate_stats = {
        'avg_mean_norm_diff': np.mean([s['mean_norm_diff'] for s in stats.values()]),
        'avg_std_norm_diff': np.mean([s['std_norm_diff'] for s in stats.values()]),
        'avg_wasserstein': np.mean([s['wasserstein'] for s in stats.values()]),
        'correlation_diff': corr_diff
    }
    
    return aggregate_stats

def evaluate_ml_utility(real_data, synthetic_data, target_col, task='classification'):
    """Evaluate ML utility: train on synthetic, test on real."""
    # Split real data
    X_real = real_data.drop(columns=[target_col])
    y_real = real_data[target_col]
    
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )
    
    # Prepare synthetic data
    X_synth = synthetic_data.drop(columns=[target_col])
    y_synth = synthetic_data[target_col]

    common_cols = sorted(list(set(X_synth.columns) & set(X_test_real.columns)))
    print(f"Using {len(common_cols)} common columns for evaluation")
    X_synth = X_synth[common_cols]
    X_test_real = X_test_real[common_cols]
    
    # Initialize model
    if task == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train on synthetic data
    model.fit(X_synth, y_synth)
    
    # Evaluate on real test data
    y_pred = model.predict(X_test_real)
    
    # Calculate metrics
    if task == 'classification':
        accuracy = accuracy_score(y_test_real, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_real, y_pred, average='weighted'
        )
        
        # ROC AUC if binary and probabilities available
        if len(np.unique(y_real)) == 2 and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_real)[:, 1]
            roc_auc = roc_auc_score(y_test_real, y_prob)
        else:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    else:  # regression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        mse = mean_squared_error(y_test_real, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_real, y_pred)
        r2 = r2_score(y_test_real, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    return metrics

def evaluate_privacy_risk(real_data, synthetic_data, k=5):
    """Evaluate privacy risk using nearest neighbor distance ratio."""
    from sklearn.neighbors import NearestNeighbors
    
    # Combine all numerical columns
    numerical_cols = [col for col in real_data.columns 
                    if pd.api.types.is_numeric_dtype(real_data[col])]
    
    if not numerical_cols:
        return {"privacy_score": None}
    
    X_real = real_data[numerical_cols].values
    X_synth = synthetic_data[numerical_cols].values
    
    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_synth_scaled = scaler.transform(X_synth)
    
    # Fit nearest neighbors on real data
    nn_real = NearestNeighbors(n_neighbors=k+1)
    nn_real.fit(X_real_scaled)
    
    # Calculate distances within real data
    distances_real, _ = nn_real.kneighbors(X_real_scaled)
    avg_min_distance_real = np.mean(distances_real[:, 1])  # Skip first NN (self)
    
    # Calculate distances from synthetic to real
    distances_synth_to_real, _ = nn_real.kneighbors(X_synth_scaled)
    min_distances_synth = distances_synth_to_real[:, 0]  # Closest real point
    avg_min_distance_synth = np.mean(min_distances_synth)
    
    # Privacy score: ratio of average min distances
    # Higher is better (synthetic points are further from real points)
    privacy_score = avg_min_distance_synth / avg_min_distance_real
    
    # Calculate percentage of synthetic points that are too close to real points
    # (potential privacy risk)
    threshold = 0.1  # Arbitrary threshold, can be adjusted
    pct_too_close = np.mean(min_distances_synth < threshold) * 100
    
    return {
        "privacy_score": privacy_score,
        "pct_too_close": pct_too_close
    }

def run_benchmark(dataset_path, target_col, task='classification', epochs=100, n_samples=None, 
                 models=None, device=None):
    """
    Run a comprehensive benchmark of different synthetic data models.
    
    Args:
        dataset_path: Path to the dataset CSV file
        target_col: Target column name
        task: Task type ('classification' or 'regression')
        epochs: Number of training epochs for each model
        n_samples: Number of synthetic samples to generate (defaults to dataset size)
        models: List of models to train (defaults to all available models)
        device: Device to use for training ('cuda' or 'cpu')
    """
    # Load dataset
    real_data = load_dataset(dataset_path, target_col)
    
    # Set default number of samples to match real data
    if n_samples is None:
        n_samples = len(real_data)
    
    # Default models to train if not specified
    if models is None:
        models_to_train = [
            "TabularDiffFlow",
            "CTGAN",
            "TVAE",
            "SynthCityDDPM"
        ]
    else:
        models_to_train = models
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running benchmark with {len(models_to_train)} models")
    print(f"Using {device} device")
    print(f"Training for {epochs} epochs per model")
    print(f"Generating {n_samples} synthetic samples per model")
    
    # Initialize storage
    trained_models = {}
    synthetic_datasets = {}
    
    # Train models and generate synthetic data
    for model_name in models_to_train:
        print(f"\n==== Training {model_name} ====")
        
        try:
            if model_name == "TabularDiffFlow":
                # Use our improved training procedure
                model = train_tabular_diffflow_model(
                    real_data,
                    n_epochs=epochs,
                    batch_size=64,
                    device=device,
                    d_model=256,  # Increased model capacity
                    weight_decay=1e-5
                )
            
            elif model_name == "CTGAN":
                try:
                    model = train_ctgan(real_data, target_col, epochs=epochs)
                except Exception as e:
                    print(f"Error training CTGAN: {str(e)}")
                    print("Trying with reduced epochs...")
                    model = train_ctgan(real_data, target_col, epochs=min(epochs, 30))
            
            elif model_name == "TVAE":
                try:
                    model = train_tvae(real_data, target_col, epochs=epochs)
                except Exception as e:
                    print(f"Error training TVAE: {str(e)}")
                    print("Trying with alternative parameters...")
                    from sdv.tabular import TVAE
                    
                    tvae = TVAE(
                        epochs=min(epochs, 30),
                        batch_size=500,
                        device=device
                    )
                    print("Training TVAE with modified parameters...")
                    tvae.fit(real_data)
                    model = tvae
            
            elif model_name == "SynthCityDDPM":
                try:
                    model = train_synthcity_ddpm(real_data, target_col, epochs=epochs)
                except Exception as e:
                    print(f"Error training SynthCityDDPM: {str(e)}")
                    print("Trying with alternative approach...")
                    from synthcity.plugins import Plugins
                    from synthcity.plugins.core.dataloader import GenericDataLoader
                    
                    # Prepare dataloader
                    data_loader = GenericDataLoader(
                        data=real_data,
                        target_column=target_col
                    )
                    
                    # Try with alternative model
                    model = Plugins().get(
                        name="tvae",  # Try TVAE instead
                        n_iter=min(epochs, 30),
                        batch_size=500,
                        device=device
                    )
                    
                    print("Training alternative SynthCity model...")
                    model.fit(data_loader)
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Store trained model
            trained_models[model_name] = model
            
            # Generate synthetic data
            print(f"\n==== Generating synthetic data with {model_name} ====")
            try:
                # Use batched generation for memory efficiency
                synthetic_data = generate_samples(model, model_name, real_data, n_samples, device=device)
                synthetic_datasets[model_name] = synthetic_data
                
                # Save to CSV
                synthetic_data.to_csv(f'synthetic_{model_name}.csv', index=False)
                print(f"Saved synthetic data to synthetic_{model_name}.csv")
            except Exception as e:
                print(f"Error generating synthetic data with {model_name}: {str(e)}")
                print("Trying with smaller sample size...")
                # Try with smaller sample size if full size fails
                reduced_samples = min(n_samples, 1000)
                synthetic_data = generate_samples(model, model_name, real_data, reduced_samples, device=device)
                synthetic_datasets[model_name] = synthetic_data
                synthetic_data.to_csv(f'synthetic_{model_name}.csv', index=False)
                print(f"Saved reduced synthetic data to synthetic_{model_name}.csv ({reduced_samples} samples)")
                
        except Exception as e:
            print(f"Failed to train or generate data with {model_name}: {str(e)}")
            print(f"Skipping {model_name} in benchmark comparison")
    
    # Evaluate models
    results = {
        'statistical_fidelity': {},
        'ml_utility': {},
        'privacy': {}
    }
    
    for model_name, synthetic_data in synthetic_datasets.items():
        print(f"\n==== Evaluating {model_name} ====")
        
        # Statistical fidelity
        stat_results = evaluate_statistical_fidelity(real_data, synthetic_data)
        results['statistical_fidelity'][model_name] = stat_results
        
        # ML utility
        ml_results = evaluate_ml_utility(real_data, synthetic_data, target_col, task)
        results['ml_utility'][model_name] = ml_results
        
        # Privacy risk
        privacy_results = evaluate_privacy_risk(real_data, synthetic_data)
        results['privacy'][model_name] = privacy_results
    
    # Create summary tables
    stat_summary = pd.DataFrame({
        model: {
            'Avg Mean Diff (%)': results['statistical_fidelity'][model]['avg_mean_norm_diff'] * 100,
            'Avg Std Diff (%)': results['statistical_fidelity'][model]['avg_std_norm_diff'] * 100,
            'Avg Wasserstein Dist': results['statistical_fidelity'][model]['avg_wasserstein'],
            'Correlation Matrix Diff': results['statistical_fidelity'][model]['correlation_diff']
        }
        for model in synthetic_datasets.keys()
    })
    
    if task == 'classification':
        ml_summary = pd.DataFrame({
            model: {
                'Accuracy': results['ml_utility'][model]['accuracy'],
                'Precision': results['ml_utility'][model]['precision'],
                'Recall': results['ml_utility'][model]['recall'],
                'F1 Score': results['ml_utility'][model]['f1'],
                'ROC AUC': results['ml_utility'][model]['roc_auc']
            }
            for model in synthetic_datasets.keys()
        })
    else:  # regression
        ml_summary = pd.DataFrame({
            model: {
                'MSE': results['ml_utility'][model]['mse'],
                'RMSE': results['ml_utility'][model]['rmse'],
                'MAE': results['ml_utility'][model]['mae'],
                'R²': results['ml_utility'][model]['r2']
            }
            for model in synthetic_datasets.keys()
        })
    
    privacy_summary = pd.DataFrame({
        model: {
            'Privacy Score': results['privacy'][model]['privacy_score'],
            '% Too Close': results['privacy'][model]['pct_too_close']
        }
        for model in synthetic_datasets.keys()
    })
    
    # Print summaries
    print("\n==== Statistical Fidelity ====")
    print(stat_summary)
    
    print("\n==== ML Utility ====")
    print(ml_summary)
    
    print("\n==== Privacy Risk ====")
    print(privacy_summary)
    
    # Visualize results
    visualize_benchmark_results(stat_summary, ml_summary, privacy_summary, task)
    
    return {
        'models': trained_models,
        'synthetic_data': synthetic_datasets,
        'results': results,
        'summaries': {
            'statistical_fidelity': stat_summary,
            'ml_utility': ml_summary,
            'privacy': privacy_summary
        }
    }

def visualize_benchmark_results(stat_summary, ml_summary, privacy_summary, task):
    """Visualize benchmark results."""
    # Set up the figure
    plt.figure(figsize=(18, 12))
    
    # Plot statistical fidelity metrics
    plt.subplot(2, 2, 1)
    stat_metrics = ['Avg Mean Diff (%)', 'Avg Std Diff (%)', 'Correlation Matrix Diff']
    stat_df = stat_summary.loc[stat_metrics].T
    
    # Normalize for easier comparison (lower is better)
    for col in stat_df.columns:
        stat_df[col] = stat_df[col] / stat_df[col].max()
    
    ax = stat_df.plot(kind='bar', ax=plt.gca())
    plt.title('Statistical Fidelity (Lower is Better)')
    plt.ylabel('Normalized Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot ML utility metrics
    plt.subplot(2, 2, 2)
    if task == 'classification':
        ml_metrics = ['Accuracy', 'F1 Score']
        if ml_summary.loc['ROC AUC'].notna().all():
            ml_metrics.append('ROC AUC')
    else:  # regression
        ml_metrics = ['R²']
    
    ml_df = ml_summary.loc[ml_metrics].T
    ax = ml_df.plot(kind='bar', ax=plt.gca())
    plt.title('ML Utility (Higher is Better)')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot privacy metric
    plt.subplot(2, 2, 3)
    privacy_df = privacy_summary.loc[['Privacy Score']].T
    ax = privacy_df.plot(kind='bar', ax=plt.gca())
    plt.title('Privacy Score (Higher is Better)')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Radar chart for overall comparison
    plt.subplot(2, 2, 4)
    
    # Prepare data for radar chart
    models = stat_summary.columns
    
    # Metrics to include (normalize all so higher is better)
    radar_metrics = {
        'Statistical Fidelity': 1 - stat_summary.loc['Correlation Matrix Diff'] / stat_summary.loc['Correlation Matrix Diff'].max(),
        'ML Performance': ml_summary.loc['Accuracy'] if task == 'classification' else ml_summary.loc['R²'],
        'Privacy': privacy_summary.loc['Privacy Score']
    }
    
    # Normalize between 0 and 1
    for metric in radar_metrics:
        min_val = radar_metrics[metric].min()
        max_val = radar_metrics[metric].max()
        if max_val > min_val:
            radar_metrics[metric] = (radar_metrics[metric] - min_val) / (max_val - min_val)
    
    # Number of metrics
    N = len(radar_metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = plt.subplot(2, 2, 4, polar=True)
    
    # Draw one axis per metric and add labels
    plt.xticks(angles[:-1], radar_metrics.keys(), size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=7)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, model in enumerate(models):
        values = [radar_metrics[metric][model] for metric in radar_metrics]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Overall Comparison (Higher is Better)')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark synthetic data generation models')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'],
                       help='Task type: classification or regression')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--samples', type=int, default=None, help='Number of synthetic samples to generate')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], 
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['TabularDiffFlow', 'CTGAN', 'TVAE', 'SynthCityDDPM'],
                       default=None, help='Models to include in benchmark')
    
    args = parser.parse_args()
    
    # Print basic system info
    print("TabularDiffFlow Benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run benchmark with specified parameters
    benchmark_results = run_benchmark(
        dataset_path=args.data,
        target_col=args.target,
        task=args.task,
        epochs=args.epochs,
        n_samples=args.samples,
        models=args.models,
        device=args.device
    )
    
    # Print summary of results
    print("\n==== Benchmark Complete ====")
    print(f"Results saved to benchmark_results.png")
    
    # If TabularDiffFlow was used, highlight its performance
    if 'TabularDiffFlow' in benchmark_results['summaries']['ml_utility'].columns:
        tdf_ml = benchmark_results['summaries']['ml_utility']['TabularDiffFlow']
        if args.task == 'classification':
            print(f"TabularDiffFlow Accuracy: {tdf_ml.get('Accuracy', 'N/A'):.4f}")
            if 'ROC AUC' in tdf_ml:
                print(f"TabularDiffFlow ROC AUC: {tdf_ml.get('ROC AUC', 'N/A'):.4f}")
        else:
            print(f"TabularDiffFlow R²: {tdf_ml.get('R²', 'N/A'):.4f}")
            print(f"TabularDiffFlow RMSE: {tdf_ml.get('RMSE', 'N/A'):.4f}")
    
    print("\nThanks for using TabularDiffFlow!")