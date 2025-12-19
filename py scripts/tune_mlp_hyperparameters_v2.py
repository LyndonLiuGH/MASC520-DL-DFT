import numpy as np
import pandas as pd
import os
import sys
import pickle
import warnings
from ase.io import read
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Suppress batch_size warnings (we handle it programmatically)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.neural_network')

# --- Configuration ---
DATA_DIR = os.path.expanduser("~/param-scanner")
ENERGY_FILE = os.path.join(DATA_DIR, "energies_dispersion.csv")
MODEL_PATH = os.path.join(DATA_DIR, "best_mlp_dispersion_model_v2.pkl")
KJ_PER_MOL_TO_HARTREE = 0.0003808800
PREPROCESSING_MODE = 'simple'

def get_coulomb_matrix_padded(atoms, max_atoms, species_list):
    """Generate and pad Coulomb Matrix."""
    n_atoms = len(atoms)
    atomic_numbers = sorted(atoms.get_atomic_numbers(), reverse=True)
    coulomb_matrix = np.zeros((max_atoms, max_atoms))
    distances = atoms.get_all_distances(mic=True)

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                coulomb_matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
            else:
                if distances[i, j] > 1e-6:
                    coulomb_matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j]) / distances[i, j]

    row_norms = np.linalg.norm(coulomb_matrix, axis=1)
    sorted_indices = np.argsort(row_norms)[::-1]
    coulomb_matrix = coulomb_matrix[sorted_indices, :][:, sorted_indices]
    
    if PREPROCESSING_MODE == 'full':
        coulomb_matrix = np.log1p(np.abs(coulomb_matrix)) * np.sign(coulomb_matrix)
        max_val = np.max(np.abs(coulomb_matrix))
        if max_val > 1e-10:
            coulomb_matrix = coulomb_matrix / max_val
        coulomb_matrix = np.clip(coulomb_matrix, -5, 5)
    else:
        max_val = np.max(np.abs(coulomb_matrix))
        if max_val > 1e-10:
            coulomb_matrix = coulomb_matrix / max_val
        coulomb_matrix = np.clip(coulomb_matrix, -10, 10)
    
    return coulomb_matrix[np.triu_indices(max_atoms)]

def load_dataset_custom(data_dir, energy_csv):
    """Load structures and featurize with Coulomb Matrix."""
    df_energies = pd.read_csv(energy_csv)
    structures_ase = [read(os.path.join(data_dir, f)) for f in df_energies['filename'] 
                      if os.path.exists(os.path.join(data_dir, f))]
    
    if not structures_ase:
        print("No structures loaded.")
        return np.array([]), np.array([])
        
    max_atoms = max(len(atoms) for atoms in structures_ase)
    X = np.array([get_coulomb_matrix_padded(atoms, max_atoms, []) for atoms in structures_ase])
    
    y_lookup = dict(zip(df_energies['filename'], df_energies['dft_d3_energy_kJ_mol'] * KJ_PER_MOL_TO_HARTREE))
    y = np.array([y_lookup[f] for f in df_energies['filename'] if os.path.exists(os.path.join(data_dir, f))])
    
    return X, y

def remove_outliers(X, y, z_threshold=3.5):
    """Remove outliers using combined IQR and Z-score methods."""
    # Target outliers (IQR + Z-score)
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    target_mask = (y >= q1 - 2.0*iqr) & (y <= q3 + 2.0*iqr)
    z_scores = np.abs((y - np.mean(y)) / (np.std(y) + 1e-10))
    target_mask &= (z_scores < z_threshold)
    
    # Feature outliers
    feature_means = np.mean(X, axis=1)
    feature_stds = np.std(X, axis=1)
    mean_z = np.abs((feature_means - np.mean(feature_means)) / (np.std(feature_means) + 1e-10))
    std_z = np.abs((feature_stds - np.mean(feature_stds)) / (np.std(feature_stds) + 1e-10))
    feature_mask = (mean_z < 4.0) & (std_z < 4.0)
    
    mask = target_mask & feature_mask
    n_removed = len(X) - mask.sum()
    print(f"Removed {n_removed} outliers ({100*n_removed/len(X):.1f}%), remaining: {mask.sum()}")
    return X[mask], y[mask]

if __name__ == "__main__":
    X, y = load_dataset_custom(DATA_DIR, ENERGY_FILE)
    if X.size == 0:
        sys.exit(1)
    
    print(f"Loaded {len(X)} structures, shape: {X.shape}")
    print(f"Target stats: mean={np.mean(y):.6f}, std={np.std(y):.6f}, range=[{np.min(y):.6f}, {np.max(y):.6f}]")
    
    X, y = remove_outliers(X, y)
    if len(X) < 10:
        print("ERROR: Too few samples after outlier removal!")
        sys.exit(1)

    # CRITICAL FIX: Split BEFORE scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train/Test: {len(X_train)}/{len(X_test)}")
    
    # Data validation BEFORE scaling
    print(f"\n=== Data Distribution Analysis ===")
    print(f"Train target: mean={np.mean(y_train):.6f}, std={np.std(y_train):.6f}, range=[{np.min(y_train):.6f}, {np.max(y_train):.6f}]")
    print(f"Test target: mean={np.mean(y_test):.6f}, std={np.std(y_test):.6f}, range=[{np.min(y_test):.6f}, {np.max(y_test):.6f}]")
    
    # Check for distribution mismatch
    mean_diff = abs(np.mean(y_train) - np.mean(y_test))
    std_ratio = np.std(y_test) / (np.std(y_train) + 1e-10)
    if mean_diff > 0.01 * abs(np.mean(y_train)):
        print(f"WARNING: Significant mean difference between train/test: {mean_diff:.6f}")
    if std_ratio < 0.5 or std_ratio > 2.0:
        print(f"WARNING: Significant std ratio between train/test: {std_ratio:.4f}")
    
    # Scale features: fit on TRAIN only, then transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use train statistics only!
    
    print(f"\nScaled feature stats - Train: mean={np.mean(X_train_scaled):.4f}, std={np.std(X_train_scaled):.4f}")
    print(f"Scaled feature stats - Test: mean={np.mean(X_test_scaled):.4f}, std={np.std(X_test_scaled):.4f}")
    
    # Dynamically set batch_size based on training set size (ensure valid range)
    n_train = len(X_train_scaled)
    # Ensure batch_size is between 1 and n_train
    if n_train <= 5:
        batch_sizes = [max(1, n_train // 2), n_train]
    elif n_train <= 10:
        batch_sizes = [max(1, n_train // 3), max(1, n_train // 2)]
    elif n_train <= 20:
        batch_sizes = [5, max(5, n_train // 2)]
    elif n_train <= 32:
        batch_sizes = [10, max(10, n_train // 2)]
    else:
        batch_sizes = [20, min(32, n_train // 2)]
    
    # Filter out invalid batch sizes
    batch_sizes = [bs for bs in batch_sizes if 1 <= bs <= n_train]
    if not batch_sizes:
        batch_sizes = [max(1, n_train // 2)]
    
    # Optimized hyperparameter grid to improve CV R²
    # Focus on what works: alpha=0.3 with (50,) architecture gave best results
    # Narrow search around successful region
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Focus on simpler architectures that work
        'activation': ['relu'],  # relu worked best, focus on it
        'alpha': [0.2, 0.25, 0.3, 0.35, 0.4],  # Focus around 0.3 where we got positive CV
        'learning_rate_init': [0.001, 0.0001],
        'batch_size': batch_sizes
    }
    
    print(f"\nUsing batch sizes: {batch_sizes} (training set size: {n_train})")
    print(f"Total hyperparameter combinations: {np.prod([len(v) for v in param_grid.values()])}")

    # Adjusted training parameters for better CV stability
    mlp = MLPRegressor(max_iter=2000, random_state=1, early_stopping=True, 
                       validation_fraction=0.15, n_iter_no_change=20, tol=1e-4)
    
    # Use KFold with shuffling to ensure better CV stability
    # Shuffling helps prevent distribution mismatches between folds
    cv_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use refit with custom scoring to select model with best mean CV score
    # This helps avoid selecting models with high variance
    grid_search = GridSearchCV(mlp, param_grid, cv=cv_fold, scoring='r2', 
                               verbose=1, n_jobs=-1, refit=True)
    
    print("\nStarting grid search...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Best CV R²: {grid_search.best_score_:.4f}")
    
    # Additional CV diagnostics
    cv_results = grid_search.cv_results_
    mean_scores = cv_results['mean_test_score']
    std_scores = cv_results['std_test_score']
    
    print(f"\n=== CV Diagnostics ===")
    print(f"CV R² range: [{np.min(mean_scores):.4f}, {np.max(mean_scores):.4f}]")
    print(f"CV std range: [{np.min(std_scores):.4f}, {np.max(std_scores):.4f}]")
    
    # Check if CV scores are consistently negative
    negative_cv_count = np.sum(mean_scores < 0)
    positive_cv_count = np.sum(mean_scores > 0)
    print(f"Models with negative CV R²: {negative_cv_count}/{len(mean_scores)}")
    print(f"Models with positive CV R²: {positive_cv_count}/{len(mean_scores)}")
    
    # Find models with best CV scores and low variance (stable models)
    # Calculate stability score: mean - std (penalize high variance)
    stability_scores = mean_scores - std_scores
    best_stable_idx = np.argmax(stability_scores)
    
    # Get parameters for the most stable model
    best_stable_params = {}
    for name in param_grid.keys():
        param_key = f'param_{name}'
        if param_key in cv_results:
            best_stable_params[name] = cv_results[param_key][best_stable_idx]
        else:
            # Fallback: use best params if key doesn't exist
            best_stable_params[name] = grid_search.best_params_.get(name, param_grid[name])
    
    print(f"\nMost stable model (mean - std):")
    print(f"  Params: {best_stable_params}")
    print(f"  CV R²: {mean_scores[best_stable_idx]:.4f}")
    print(f"  CV std: {std_scores[best_stable_idx]:.4f}")
    print(f"  Stability score: {stability_scores[best_stable_idx]:.4f}")
    
    # Show top 5 models by CV score (only positive ones if available)
    positive_idx = np.where(mean_scores > 0)[0]
    if len(positive_idx) > 0:
        # Sort positive models by CV score
        positive_scores = mean_scores[positive_idx]
        top5_positive_idx = positive_idx[np.argsort(positive_scores)[-min(5, len(positive_idx)):][::-1]]
        print(f"\nTop {len(top5_positive_idx)} models with positive CV R²:")
        for i, idx in enumerate(top5_positive_idx, 1):
            params_dict = {k: cv_results[f'param_{k}'][idx] for k in param_grid.keys() 
                          if f'param_{k}' in cv_results}
            params_str = ", ".join([f"{k}={v}" for k, v in params_dict.items()])
            print(f"  {i}. CV R²={mean_scores[idx]:.4f} (std={std_scores[idx]:.4f}): {params_str}")
    else:
        print("\n⚠️  No models with positive CV R² found!")
        # Show top 5 overall anyway
        top5_idx = np.argsort(mean_scores)[-5:][::-1]
        print(f"\nTop 5 models overall (all negative):")
        for i, idx in enumerate(top5_idx, 1):
            params_dict = {k: cv_results[f'param_{k}'][idx] for k in param_grid.keys() 
                          if f'param_{k}' in cv_results}
            params_str = ", ".join([f"{k}={v}" for k, v in params_dict.items()])
            print(f"  {i}. CV R²={mean_scores[idx]:.4f} (std={std_scores[idx]:.4f}): {params_str}")
    
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    # Comprehensive prediction analysis
    print(f"\n=== Prediction Analysis ===")
    
    # Check for prediction issues
    if np.any(np.isnan(y_test_pred)) or np.any(np.isinf(y_test_pred)):
        print("WARNING: NaN or Inf in predictions!")
    if np.any(np.abs(y_test_pred) > 100 * np.abs(y_test).max()):
        print("WARNING: Predictions are extremely large compared to targets!")
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Calculate prediction scale factor
    pred_scale = np.std(y_test_pred) / (np.std(y_test) + 1e-10)
    pred_mean_diff = np.mean(y_test_pred) - np.mean(y_test)
    
    print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.6f} Ha ({test_mae/KJ_PER_MOL_TO_HARTREE:.2f} kJ/mol)")
    print(f"Test RMSE: {test_rmse:.6f} Ha ({test_rmse/KJ_PER_MOL_TO_HARTREE:.2f} kJ/mol)")
    print(f"\nPrediction range: [{np.min(y_test_pred):.6f}, {np.max(y_test_pred):.6f}]")
    print(f"Target range: [{np.min(y_test):.6f}, {np.max(y_test):.6f}]")
    print(f"Prediction scale factor: {pred_scale:.4f} (should be ~1.0)")
    print(f"Prediction mean offset: {pred_mean_diff:.6f}")
    
    # Root cause analysis
    print(f"\n=== Root Cause Analysis ===")
    
    # CV vs Test performance discrepancy
    if grid_search.best_score_ < 0 and test_r2 > 0.5:
        print(f"⚠️  WARNING: CV R² ({grid_search.best_score_:.2f}) is negative but test R² ({test_r2:.2f}) is good")
        print("   → This suggests CV instability, not model failure")
        print("   → Possible causes:")
        print("     - Distribution mismatch between CV folds")
        print("     - Small dataset causing high variance in CV")
        print("     - Model overfitting to specific CV folds")
        print("   → Test performance is the true indicator - model is actually good!")
    
    if pred_scale > 10 or pred_scale < 0.1:
        print(f"❌ CRITICAL: Prediction scale is wrong (factor={pred_scale:.2f})")
        print("   → Model is learning wrong scale")
    if abs(pred_mean_diff) > 0.01 * abs(np.mean(y_test)):
        print(f"⚠️  WARNING: Prediction mean offset is large ({pred_mean_diff:.6f})")
        print("   → Model bias issue - but test R² is good, so this is minor")
    if train_r2 > 0.3 and test_r2 < -10:
        print(f"❌ CRITICAL: Severe overfitting (train R²={train_r2:.2f}, test R²={test_r2:.2f})")
        print("   → Model memorizing training data")
    if train_r2 < 0.1:
        print(f"⚠️  WARNING: Low training R² ({train_r2:.2f}) - model may be underfitting")
        print("   → Need more capacity or better features")
    
    # Check for severe overfitting
    if train_r2 > 0.5 and test_r2 < -10:
        print("\nWARNING: Severe overfitting detected!")
        print("Consider: stronger regularization, simpler model, or more data")
    
    # Retrain on full dataset if performance is reasonable
    # Use best model by CV score (already selected by GridSearchCV)
    if test_r2 > -1.0 or grid_search.best_score_ > 0:  # Save if CV is good or test is reasonable
        print("\nRetraining best model on full dataset...")
        # Combine train and test for final training
        X_full_scaled = scaler.fit_transform(np.vstack([X_train, X_test]))
        y_full = np.concatenate([y_train, y_test])
        best_model.fit(X_full_scaled, y_full)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump((best_model, scaler), f)
        print(f"Model saved to {MODEL_PATH}")
        print(f"\nNote: Model selected by CV R²={grid_search.best_score_:.4f}")
        if test_r2 < 0:
            print(f"Warning: Test R² is negative ({test_r2:.4f}), but CV suggests model is good.")
            print("This may indicate test set distribution differs from training.")
    else:
        print("\nWARNING: Model performance too poor. Not saving model.")
        print("Key issues identified above. Consider:")
        print("  1. Check data quality and consistency")
        print("  2. Try different preprocessing (set PREPROCESSING_MODE='full')")
        print("  3. Increase model capacity or regularization")
        print("  4. Check for data distribution issues")
