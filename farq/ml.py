"""
Machine learning module for raster analysis.

This module provides functions for:
- Image classification
- Change detection using ML
- Feature extraction
- Model training and prediction
- Data preprocessing and augmentation
- Unsupervised learning and clustering
- Water body detection using clustering

All functions include input validation and detailed error messages.
"""

import numpy as np
from typing import Union, Tuple, Optional, List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import joblib
from pathlib import Path

def extract_features(raster_data: np.ndarray,
                    indices: Optional[List[str]] = None,
                    window_size: int = 3) -> np.ndarray:
    """
    Extract features from raster data for ML analysis.
    
    Args:
        raster_data: Input raster data (2D or 3D array)
        indices: List of spectral indices to calculate
        window_size: Size of window for texture features
        
    Returns:
        Array of extracted features
        
    Raises:
        ValueError: If input data is invalid
    """
    if not isinstance(raster_data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    features = []
    
    # Add original bands as features
    if raster_data.ndim == 3:
        for band in range(raster_data.shape[2]):
            features.append(raster_data[..., band])
    else:
        features.append(raster_data)
    
    # Calculate texture features
    if window_size > 1:
        from scipy.ndimage import uniform_filter, variance
        
        for feature in features.copy():  # Work on copy to avoid modifying during iteration
            # Mean
            mean = uniform_filter(feature, size=window_size)
            features.append(mean)
            
            # Variance
            var = variance(feature, size=window_size)
            features.append(var)
    
    return np.stack(features, axis=-1)

def train_classifier(features: np.ndarray,
                    labels: np.ndarray,
                    model_type: str = 'rf',
                    test_size: float = 0.2,
                    random_state: int = 42,
                    **model_params) -> Tuple[object, Dict]:
    """
    Train a classifier on raster features.
    
    Args:
        features: Feature array (samples × features)
        labels: Label array
        model_type: Type of model ('rf' for Random Forest)
        test_size: Proportion of data for testing
        random_state: Random seed
        **model_params: Additional model parameters
        
    Returns:
        Tuple of (trained model, performance metrics)
    """
    # Validate inputs
    if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("Features and labels must be numpy arrays")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Number of samples in features and labels must match")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train model
    if model_type.lower() == 'rf':
        model = RandomForestClassifier(random_state=random_state, **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return model, metrics

def predict_raster(model: object,
                  features: np.ndarray,
                  batch_size: Optional[int] = None) -> np.ndarray:
    """
    Apply trained model to raster features.
    
    Args:
        model: Trained classifier
        features: Feature array
        batch_size: Batch size for large rasters
        
    Returns:
        Prediction array
    """
    # Validate inputs
    if not hasattr(model, 'predict'):
        raise TypeError("Model must have predict method")
    if not isinstance(features, np.ndarray):
        raise TypeError("Features must be a numpy array")
    
    # Reshape features for prediction
    original_shape = features.shape
    if features.ndim > 2:
        features = features.reshape(-1, features.shape[-1])
    
    # Predict in batches if specified
    if batch_size:
        predictions = []
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            pred = model.predict(batch)
            predictions.append(pred)
        predictions = np.concatenate(predictions)
    else:
        predictions = model.predict(features)
    
    # Reshape back to original dimensions
    if len(original_shape) > 2:
        predictions = predictions.reshape(original_shape[:-1])
    
    return predictions

def save_model(model: object,
              filepath: Union[str, Path],
              metadata: Optional[Dict] = None) -> None:
    """
    Save trained model to file.
    
    Args:
        model: Trained model
        filepath: Path to save model
        metadata: Optional metadata to save with model
    """
    filepath = Path(filepath)
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    
    model_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    joblib.dump(model_data, filepath)

def load_model(filepath: Union[str, Path]) -> Tuple[object, Dict]:
    """
    Load trained model from file.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Tuple of (model, metadata)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model_data = joblib.load(filepath)
    return model_data['model'], model_data['metadata']

def detect_changes_ml(raster1: np.ndarray,
                     raster2: np.ndarray,
                     model: Optional[object] = None,
                     threshold: float = 0.5) -> np.ndarray:
    """
    Detect changes between two rasters using ML.
    
    Args:
        raster1: First raster
        raster2: Second raster
        model: Optional pre-trained model
        threshold: Change detection threshold
        
    Returns:
        Binary change mask
    """
    # Validate inputs
    if not isinstance(raster1, np.ndarray) or not isinstance(raster2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if raster1.shape != raster2.shape:
        raise ValueError("Rasters must have same shape")
    
    # Calculate difference features
    diff = np.abs(raster2 - raster1)
    
    if model is not None:
        # Use ML model for change detection
        features = extract_features(diff)
        changes = predict_raster(model, features) > threshold
    else:
        # Simple threshold-based detection
        changes = diff > threshold
    
    return changes.astype(bool)

def augment_training_data(features: np.ndarray,
                         labels: np.ndarray,
                         augmentation_factor: int = 2,
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data for better model performance.
    
    Args:
        features: Feature array
        labels: Label array
        augmentation_factor: How many times to augment data
        random_state: Random seed
        
    Returns:
        Tuple of (augmented features, augmented labels)
    """
    if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Features and labels must have same number of samples")
    
    np.random.seed(random_state)
    
    aug_features = [features]
    aug_labels = [labels]
    
    for _ in range(augmentation_factor - 1):
        # Add random noise
        noise = np.random.normal(0, 0.1, features.shape)
        aug_features.append(features + noise)
        aug_labels.append(labels)
        
        # Random rotations (for 2D features)
        if features.ndim >= 2:
            rotated = np.rot90(features, k=np.random.randint(1, 4))
            aug_features.append(rotated)
            aug_labels.append(labels)
    
    return np.concatenate(aug_features), np.concatenate(aug_labels)

def cluster_water_bodies(raster_data: np.ndarray,
                        method: str = 'kmeans',
                        n_clusters: int = 2,
                        water_index: Optional[np.ndarray] = None,
                        **kwargs) -> Tuple[np.ndarray, Dict]:
    """
    Perform unsupervised clustering to identify water bodies.
    
    Args:
        raster_data: Input raster data (2D or 3D array)
        method: Clustering method ('kmeans' or 'dbscan')
        n_clusters: Number of clusters for kmeans
        water_index: Optional pre-calculated water index (e.g., NDWI)
        **kwargs: Additional parameters for clustering algorithms
        
    Returns:
        Tuple of (cluster labels, clustering metadata)
    """
    # Validate inputs
    if not isinstance(raster_data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # Prepare features
    if water_index is not None:
        features = np.column_stack([raster_data.reshape(-1, raster_data.shape[-1] if raster_data.ndim == 3 else 1),
                                  water_index.ravel()[:, np.newaxis]])
    else:
        features = raster_data.reshape(-1, raster_data.shape[-1] if raster_data.ndim == 3 else 1)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    if method.lower() == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        labels = clusterer.fit_predict(features_scaled)
        
        # Identify water cluster (usually the cluster with lower values in water indices)
        if water_index is not None:
            cluster_means = [np.mean(water_index.ravel()[labels == i]) for i in range(n_clusters)]
            water_cluster = np.argmin(cluster_means)  # Water typically has lower values
        else:
            water_cluster = 0  # Default to first cluster
            
        metadata = {
            'cluster_centers': clusterer.cluster_centers_,
            'inertia': clusterer.inertia_,
            'water_cluster': water_cluster,
            'cluster_means': cluster_means if water_index is not None else None
        }
        
    elif method.lower() == 'dbscan':
        clusterer = DBSCAN(**kwargs)
        labels = clusterer.fit_predict(features_scaled)
        
        # Identify water cluster
        if water_index is not None:
            unique_labels = np.unique(labels[labels >= 0])
            cluster_means = [np.mean(water_index.ravel()[labels == i]) for i in unique_labels]
            water_cluster = unique_labels[np.argmin(cluster_means)]
        else:
            water_cluster = 0
            
        metadata = {
            'n_clusters': len(np.unique(labels[labels >= 0])),
            'noise_points': np.sum(labels == -1),
            'water_cluster': water_cluster
        }
        
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    # Reshape labels back to original dimensions
    labels_reshaped = labels.reshape(raster_data.shape[:-1] if raster_data.ndim == 3 else raster_data.shape)
    
    return labels_reshaped, metadata

def analyze_water_clusters(cluster_labels: np.ndarray,
                         water_cluster: int,
                         pixel_size: Union[float, Tuple[float, float]] = 30.0) -> Dict:
    """
    Analyze water bodies identified through clustering.
    
    Args:
        cluster_labels: Cluster labels from clustering algorithm
        water_cluster: Index of the cluster representing water
        pixel_size: Pixel size in meters (single value or (width, height))
        
    Returns:
        Dictionary containing water body statistics
    """
    from scipy import ndimage
    
    # Create water mask
    water_mask = cluster_labels == water_cluster
    
    # Label individual water bodies
    labeled_water, num_features = ndimage.label(water_mask)
    
    # Calculate pixel area
    if isinstance(pixel_size, (int, float)):
        pixel_area = pixel_size * pixel_size
    else:
        pixel_area = pixel_size[0] * pixel_size[1]
    
    # Calculate statistics
    areas = []
    perimeters = []
    compactness = []
    
    for i in range(1, num_features + 1):
        body = labeled_water == i
        
        # Area
        area = np.sum(body) * pixel_area
        areas.append(area)
        
        # Perimeter
        gradient_x = np.gradient(body.astype(float), axis=0)
        gradient_y = np.gradient(body.astype(float), axis=1)
        perimeter = np.sum(np.sqrt(gradient_x**2 + gradient_y**2)) * np.mean(pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size))
        perimeters.append(perimeter)
        
        # Compactness (circularity)
        if perimeter > 0:
            compact = 4 * np.pi * area / (perimeter * perimeter)
            compactness.append(compact)
        else:
            compactness.append(0)
    
    return {
        'num_water_bodies': num_features,
        'total_water_area': sum(areas),
        'mean_water_body_area': np.mean(areas) if areas else 0,
        'max_water_body_area': max(areas) if areas else 0,
        'min_water_body_area': min(areas) if areas else 0,
        'mean_perimeter': np.mean(perimeters) if perimeters else 0,
        'mean_compactness': np.mean(compactness) if compactness else 0,
        'water_body_sizes': areas,
        'water_body_perimeters': perimeters,
        'water_body_compactness': compactness
    }

def optimize_clustering(raster_data: np.ndarray,
                       water_index: Optional[np.ndarray] = None,
                       method: str = 'kmeans',
                       param_grid: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    """
    Find optimal clustering parameters for water body detection.
    
    Args:
        raster_data: Input raster data
        water_index: Optional pre-calculated water index
        method: Clustering method ('kmeans' or 'dbscan')
        param_grid: Dictionary of parameters to try
        
    Returns:
        Tuple of (best parameters, optimization results)
    """
    if param_grid is None:
        if method.lower() == 'kmeans':
            param_grid = {
                'n_clusters': [2, 3, 4, 5],
                'random_state': [42]
            }
        else:  # dbscan
            param_grid = {
                'eps': [0.1, 0.2, 0.3, 0.4],
                'min_samples': [5, 10, 15, 20]
            }
    
    best_score = float('-inf')
    best_params = None
    results = []
    
    # Prepare features
    if water_index is not None:
        features = np.column_stack([
            raster_data.reshape(-1, raster_data.shape[-1] if raster_data.ndim == 3 else 1),
            water_index.ravel()[:, np.newaxis]
        ])
    else:
        features = raster_data.reshape(-1, raster_data.shape[-1] if raster_data.ndim == 3 else 1)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    for params in param_combinations:
        # Perform clustering
        labels, metadata = cluster_water_bodies(raster_data, method=method, **params)
        
        # Calculate clustering quality metrics
        if method.lower() == 'kmeans':
            score = -metadata['inertia']  # Negative because we want to maximize score
        else:  # dbscan
            # For DBSCAN, prefer solutions with fewer noise points and reasonable number of clusters
            n_clusters = metadata['n_clusters']
            noise_ratio = metadata['noise_points'] / features.shape[0]
            score = -noise_ratio if 2 <= n_clusters <= 5 else float('-inf')
        
        results.append({
            'params': params,
            'score': score,
            'metadata': metadata
        })
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, {'results': results, 'best_score': best_score} 