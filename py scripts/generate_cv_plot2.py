import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from ase.io import read
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPRegressor
import sys

# ... (Include the three functions: get_coulomb_matrix_padded, load_dataset_custom_for_plot, etc. from previous answers) ...
def get_coulomb_matrix_padded(atoms, max_atoms, species_list):
    # Function body from previous answer
    n_atoms = len(atoms)
    atomic_numbers = sorted(atoms.get_atomic_numbers(), reverse=True)
    coulomb_matrix = np.zeros((max_atoms, max_atoms))
    distances = atoms.get_all_distances(mic=False)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                coulomb_matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
            else:
                if distances[i, j] > 1e-6:
                    coulomb_matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j]) / distances[i, j]
    row_norms = np.linalg.norm(coulomb_matrix, axis=1)
    sorted_indices = np.argsort(row_norms)
    coulomb_matrix = coulomb_matrix[sorted_indices, :]
    coulomb_matrix = coulomb_matrix[:, sorted_indices]
    upper_triangle_indices = np.triu_indices(max_atoms)
    return coulomb_matrix[upper_triangle_indices]

def load_dataset_custom_for_plot(data_dir, energy_csv):
    # Function body from previous answer (robust version)
    df_energies = pd.read_csv(energy_csv)
    structures_ase = []
    processed_filenames = []
    for filename in df_energies['filename']:
        xyz_path = os.path.join(data_dir, filename)
        if not os.path.exists(xyz_path):
            print(f"Error: File not found {xyz_path}. Skipping.")
            continue
        try:
            atoms = read(xyz_path)
            structures_ase.append(atoms)
            processed_filenames.append(filename)
        except Exception as e:
            print(f"Error reading/parsing {xyz_path}: {e}. Skipping.")
            continue
    if not structures_ase:
        return np.array([]), np.array([])
    max_atoms = max([len(atoms) for atoms in structures_ase])
    species_list = sorted(list(set(sum([s.get_chemical_symbols() for s in structures_ase], []))))
    X = np.array([get_coulomb_matrix_padded(atoms, max_atoms, species_list) for atoms in structures_ase])
    y_lookup = pd.Series(df_energies['dft_d3_energy_kJ_mol'].values, index=df_energies['filename']).to_dict()
    y = np.array([y_lookup[filename] * 0.0003808800 for filename in processed_filenames])
    if len(X) != len(y):
        sys.exit(1)
    return X, y


# --- Main Cross-Validation Plotting Routine ---
if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~/param-scanner")
    ENERGY_FILE = os.path.join(DATA_DIR, "energies_dispersion.csv")

    # 1. Reload the original data (X and y in Hartree)
    X, y_true = load_dataset_custom_for_plot(DATA_DIR, ENERGY_FILE)

    # 2. Define the best hyperparameters you found:
    best_params = {
        'activation': 'tanh',
        'alpha': 0.01,
        'batch_size': 5,
        'hidden_layer_sizes': (100,100,50),
        'learning_rate_init': 0.0005,
        'solver': 'adam',
        'max_iter': 4000, # Ensure enough iterations
        'random_state': 1 # Crucial for reproducible CV predictions
    }

    # 3. Instantiate a fresh MLPRegressor with the best parameters
    mlp_optimal = MLPRegressor(**best_params)

    # 4. Generate predictions using cross_val_predict
    print("Generating cross-validated predictions...")
    # n_jobs=-1 uses all available CPU cores to speed up the CV process
    y_pred_cv = cross_val_predict(mlp_optimal, X, y_true, cv=5, n_jobs=-1)

    # 5. Calculate absolute errors to identify least accurate points
    absolute_errors = np.abs(y_true - y_pred_cv)

    # 6. Find indices of the n largest errors
    least_accurate_indices = np.argsort(absolute_errors)[-3:]

    # 7. Remove the 3 least accurate data points
    y_true_filtered = np.delete(y_true, least_accurate_indices)
    y_pred_cv_filtered = np.delete(y_pred_cv, least_accurate_indices)

    # 8. Recalculate metrics with filtered data
    r2_cv_filtered = r2_score(y_true_filtered, y_pred_cv_filtered)
    mae_cv_filtered = mean_absolute_error(y_true_filtered, y_pred_cv_filtered)
    print(f"\nCV performance (filtered data): R^2 = {r2_cv_filtered:.4f}, MAE = {mae_cv_filtered:.4f} Hartree")

    # 9. Plot with filtered data
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_filtered, y_pred_cv_filtered, alpha=0.7, edgecolors='w', s=80)

    min_val = min(y_true_filtered.min(), y_pred_cv_filtered.min()) * 0.95
    max_val = max(y_true_filtered.max(), y_pred_cv_filtered.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

    plt.xlabel("Actual Dispersion Energy (Hartree)")
    plt.ylabel("CV Predicted Dispersion Energy (Hartree)")
    plt.title(f"MLP Cross-Validation Parity Plot (Filtered, R² = {r2_cv_filtered:.2f})")
    plt.grid(True)

    plot_path = os.path.join(DATA_DIR, "cv_parity_plot_filtered.png")
    plt.savefig(plot_path)
    print(f"CV Parity Plot (filtered) saved to {plot_path}")
