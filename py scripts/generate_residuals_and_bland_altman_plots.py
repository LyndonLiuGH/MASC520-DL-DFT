import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from ase.io import read
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = os.path.expanduser("~/param-scanner")
ENERGY_FILE = os.path.join(DATA_DIR, "energies_dispersion.csv")
KJ_PER_MOL_TO_HARTREE = 0.0003808800
PREPROCESSING_MODE = 'simple'  # Must match training script

# Best hyperparameters from cross-validation
BEST_HYPERPARAMS = {
    'activation': 'tanh',
    'alpha': 0.01,
    'batch_size': 5,
    'hidden_layer_sizes': (100, 100, 50),
    'learning_rate_init': 0.0005,
    'solver': 'adam'
}

def get_coulomb_matrix_padded(atoms, max_atoms, species_list):
    """Generate and pad Coulomb Matrix - must match training script exactly."""
    n_atoms = len(atoms)
    atomic_numbers = sorted(atoms.get_atomic_numbers(), reverse=True)
    coulomb_matrix = np.zeros((max_atoms, max_atoms))
    distances = atoms.get_all_distances(mic=True)  # Using mic=True as in tune script

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
    """Load structures and featurize with Coulomb Matrix.
    CSV format: filename,mass,dft_d3_energy_kJ_mol
    """
    df_energies = pd.read_csv(energy_csv)
    structures_ase = []
    processed_filenames = []
    
    for filename in df_energies['filename']:
        xyz_path = os.path.join(data_dir, filename.strip())
        if not os.path.exists(xyz_path):
            print(f"Warning: File not found {xyz_path}. Skipping.")
            continue
        try:
            atoms = read(xyz_path)
            structures_ase.append(atoms)
            processed_filenames.append(filename.strip())
        except Exception as e:
            print(f"Error reading {xyz_path}: {e}. Skipping.")
            continue
            
    if not structures_ase:
        print("No structures loaded.")
        return np.array([]), np.array([])
        
    max_atoms = max(len(atoms) for atoms in structures_ase)
    X = np.array([get_coulomb_matrix_padded(atoms, max_atoms, []) for atoms in structures_ase])
    
    # Handle CSV with columns: filename,mass,dft_d3_energy_kJ_mol
    y_lookup = dict(zip(df_energies['filename'].str.strip(), 
                       df_energies['dft_d3_energy_kJ_mol'] * KJ_PER_MOL_TO_HARTREE))
    y = np.array([y_lookup[f] for f in processed_filenames])
    
    return X, y

def train_model_with_hyperparams(X_train_scaled, y_train, hyperparams):
    """Train MLP with specified hyperparameters."""
    mlp = MLPRegressor(
        hidden_layer_sizes=hyperparams['hidden_layer_sizes'],
        activation=hyperparams['activation'],
        alpha=hyperparams['alpha'],
        batch_size=hyperparams['batch_size'],
        learning_rate_init=hyperparams['learning_rate_init'],
        solver=hyperparams['solver'],
        max_iter=3000,
        random_state=1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        tol=1e-4
    )
    mlp.fit(X_train_scaled, y_train)
    return mlp

if __name__ == "__main__":
    print("Loading dataset...")
    X, y_true = load_dataset_custom(DATA_DIR, ENERGY_FILE)
    
    if X.size == 0:
        print("Error: No data loaded.")
        exit(1)
    
    print(f"Loaded {len(X)} structures, feature shape: {X.shape}")
    
    # Train new model with specified hyperparameters on full dataset
    print("Training model with specified hyperparameters...")
    print(f"Hyperparameters: {BEST_HYPERPARAMS}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model on full dataset
    model = train_model_with_hyperparams(X_scaled, y_true, BEST_HYPERPARAMS)
    print("Training completed.")
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Convert to kJ/mol for easier interpretation
    y_true_kj = y_true / KJ_PER_MOL_TO_HARTREE
    y_pred_kj = y_pred / KJ_PER_MOL_TO_HARTREE
    residuals_kj = residuals / KJ_PER_MOL_TO_HARTREE
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\nModel Performance on Full Dataset:")
    print(f"R² = {r2:.4f}")
    print(f"MAE = {mae:.6f} Hartree ({mae/KJ_PER_MOL_TO_HARTREE:.2f} kJ/mol)")
    print(f"RMSE = {rmse:.6f} Hartree ({rmse/KJ_PER_MOL_TO_HARTREE:.2f} kJ/mol)")
    
    # ========== Plot 1: Residuals vs Predicted (and vs True) ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred_kj, residuals_kj, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
    
    # Add mean and ±2σ lines
    mean_residual = np.mean(residuals_kj)
    std_residual = np.std(residuals_kj)
    ax1.axhline(y=mean_residual, color='blue', linestyle=':', linewidth=1.5, 
                label=f'Mean = {mean_residual:.2f} kJ/mol')
    ax1.axhline(y=mean_residual + 2*std_residual, color='gray', linestyle=':', 
                linewidth=1, alpha=0.7, label=f'±2σ = {2*std_residual:.2f} kJ/mol')
    ax1.axhline(y=mean_residual - 2*std_residual, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    ax1.set_xlabel('Predicted Dispersion Energy (kJ/mol)', fontsize=11)
    ax1.set_ylabel('Residuals (kJ/mol)', fontsize=11)
    ax1.set_title('Residuals vs Predicted\n(Bias, Heteroscedasticity, Nonlinearity)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Residuals vs True
    ax2.scatter(y_true_kj, residuals_kj, alpha=0.6, s=50, edgecolors='black', linewidths=0.5, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
    ax2.axhline(y=mean_residual, color='blue', linestyle=':', linewidth=1.5, 
                label=f'Mean = {mean_residual:.2f} kJ/mol')
    ax2.axhline(y=mean_residual + 2*std_residual, color='gray', linestyle=':', 
                linewidth=1, alpha=0.7, label=f'±2σ = {2*std_residual:.2f} kJ/mol')
    ax2.axhline(y=mean_residual - 2*std_residual, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    ax2.set_xlabel('True Dispersion Energy (kJ/mol)', fontsize=11)
    ax2.set_ylabel('Residuals (kJ/mol)', fontsize=11)
    ax2.set_title('Residuals vs True\n(Bias, Heteroscedasticity, Nonlinearity)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    residuals_plot_path = os.path.join(DATA_DIR, "residuals_plot.png")
    plt.savefig(residuals_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nResiduals plot saved to: {residuals_plot_path}")
    plt.close()
    
    # ========== Plot 2: Bland-Altman Plot ==========
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Bland-Altman: mean of predicted and true vs their difference
    mean_values = (y_pred_kj + y_true_kj) / 2.0
    differences = y_pred_kj - y_true_kj  # difference = predicted - true
    
    # Calculate mean difference and limits of agreement
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # Scatter plot
    ax.scatter(mean_values, differences, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Add mean difference line
    ax.axhline(y=mean_diff, color='r', linestyle='-', linewidth=2, 
               label=f'Mean difference = {mean_diff:.2f} kJ/mol')
    
    # Add limits of agreement (95% confidence interval)
    ax.axhline(y=upper_loa, color='blue', linestyle='--', linewidth=1.5, 
               label=f'Upper LoA (+1.96σ) = {upper_loa:.2f} kJ/mol')
    ax.axhline(y=lower_loa, color='blue', linestyle='--', linewidth=1.5, 
               label=f'Lower LoA (-1.96σ) = {lower_loa:.2f} kJ/mol')
    
    # Shade the region between limits of agreement
    ax.fill_between([mean_values.min(), mean_values.max()], 
                    lower_loa, upper_loa, alpha=0.2, color='gray', 
                    label='95% Limits of Agreement')
    
    ax.set_xlabel('Mean of Predicted and True Dispersion Energy (kJ/mol)', fontsize=11)
    ax.set_ylabel('Difference (Predicted - True) (kJ/mol)', fontsize=11)
    ax.set_title('Bland-Altman Plot\n(Bias Assessment Across Scale)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    
    # Add text box with statistics
    stats_text = f'Mean diff: {mean_diff:.2f} kJ/mol\n'
    stats_text += f'Std diff: {std_diff:.2f} kJ/mol\n'
    stats_text += f'Upper LoA: {upper_loa:.2f} kJ/mol\n'
    stats_text += f'Lower LoA: {lower_loa:.2f} kJ/mol'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    bland_altman_plot_path = os.path.join(DATA_DIR, "bland_altman_plot.png")
    plt.savefig(bland_altman_plot_path, dpi=300, bbox_inches='tight')
    print(f"Bland-Altman plot saved to: {bland_altman_plot_path}")
    plt.close()
    
    print("\nAll plots generated successfully!")
    
    # Print interpretation guidance
    print("\n=== Plot Interpretation Guide ===")
    print("Residuals Plot:")
    print("  - Points should be randomly distributed around zero")
    print("  - Horizontal band indicates homoscedasticity (constant variance)")
    print("  - Funnel shape indicates heteroscedasticity (variance changes with prediction)")
    print("  - Curved patterns indicate nonlinearity")
    print("\nBland-Altman Plot:")
    print("  - Mean difference close to zero indicates minimal systematic bias")
    print("  - Points within limits of agreement (LoA) are expected for 95% of data")
    print("  - Systematic trends indicate scale-dependent bias")

