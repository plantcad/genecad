import xarray as xr
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

def load_and_prepare_data(ds_path, sample_size=100, window_size=5, seed=42):
    """
    Load the dataset and prepare it for training by sampling around the middle token.
    
    Parameters:
    -----------
    ds_path : str
        Path to the zarr dataset
    sample_size : int
        Number of samples to use
    window_size : int
        Number of tokens to include on each side of the middle token
    seed : int
        Random seed for reproducible sampling
    
    Returns:
    --------
    X : np.ndarray
        Embeddings array of shape (n_samples, n_features)
    y : np.ndarray
        Input IDs array of shape (n_samples,)
    """
    # Initialize random state
    rs = np.random.RandomState(seed)
    
    # Load dataset
    ds = xr.open_zarr(ds_path)
    
    # Calculate middle point and slices
    midpoint = ds.sizes["sequence"] // 2
    sequence_slice = slice(midpoint - window_size, midpoint + window_size)
    
    # Randomly sample indices
    sample_indices = rs.choice(ds.sizes["sample"], size=sample_size, replace=False)
    
    # Prepare training dataset
    training_dataset = (
        ds.isel(sample=sample_indices, sequence=sequence_slice)[["embeddings", "input_ids"]]
        .compute()
        .stack(example=["sample", "sequence"])
        .transpose("example", "hidden_state")
    )

    print(f"Training dataset:\n{training_dataset}")
    
    return training_dataset.embeddings.values, training_dataset.input_ids.values

def train_and_evaluate(X, y, seed=42):
    """
    Train a logistic regression model and evaluate its accuracy per class.
    
    Parameters:
    -----------
    X : np.ndarray
        Embeddings array of shape (n_samples, n_features)
    y : np.ndarray
        Input IDs array of shape (n_samples,)
    seed : int
        Random seed for model training
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing accuracy metrics per class
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Train the model
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)
    
    # Get predictions on test set
    y_pred = model.predict(X_test)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert report to DataFrame
    results = pd.DataFrame({
        k: v for k, v in report.items() 
        if k not in ["accuracy"]
    }).transpose()
    
    return results

def main():
    # Set random seed for reproducibility
    seed = 42
    window_size = 5
    
    # Example usage
    ds_path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/prep/embedding_dataset/train.0.zarr"
    
    # Generate performance vs sample size
    sample_sizes = [1, 10, 100, 1000]
    all_results = []
    
    for size in sample_sizes:
        print(f"\nTraining with sample_size={size}")
        # Load and prepare data
        X, y = load_and_prepare_data(ds_path, sample_size=size, window_size=window_size, seed=seed)
        
        # Train and evaluate
        results = train_and_evaluate(X, y, seed=seed)
        
        # Add sample size column
        results['num_examples'] = size * window_size * 2
        
        # Store results
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, axis=0)
    print("\nLearning curves:")
    print("===============")
    print(combined_results.to_string())

    print("\nF1-score by sample size:")
    print("===============")
    print(
        combined_results.set_index("num_examples", append=True)
        ["f1-score"].unstack()
    )
    
if __name__ == "__main__":
    main()
