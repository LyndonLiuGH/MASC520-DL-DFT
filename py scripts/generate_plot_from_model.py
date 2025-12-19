import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from ase.io import read
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration (MUST MATCH TRAINING SCRIPT) ---
DATA_DIR = os.path.expanduser("~/param-scanner")
ENERGY_FILE = os.path.join(DATA_DIR, "energies_dispersion.csv")
MODEL_PATH = os.path.join(DATA_DIR, "best_mlp_dispersion_model_v2.pkl")
KJ_PER_MOL_TO_HARTREE = 0.0003808800 # Factor used in training

# [!!! IMPORTANT !!!] Copy the custom featurization function here
def get_coulomb_matrix_padded(atoms, max_atoms, species_list):
    """
    Manually generates and pads a Coulomb Matrix.
    Must be exactly the same function used during training.
    """
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
    """Loads all data needed for plotting, skipping files that fail to load."""
    df_energies = pd.read_csv(energy_csv)
    structures_ase = []
    processed_filenames = []
    
    # 1. Load all structures first to determine max atom count and identify loading errors early
    for filename in df_energies['filename']:
        xyz_path = os.path.join(data_dir, filename)
        if not os.path.exists(xyz_path):
            print(f"Error: File not found {xyz_path}. Skipping.")
            continue
        try:
            atoms = read(xyz_path)
            structures_ase.append(atoms)
            processed_filenames.append(filename) # Only track successful loads
        except Exception as e:
            print(f"Error reading/parsing {xyz_path}: {e}. Skipping.")
            continue
            
    if not structures_ase:
        print("No structures loaded successfully.")
        return np.array([]), np.array([])
        
    max_atoms = max([len(atoms) for atoms in structures_ase])
    # species_list is needed for the featurization function but can be determined dynamically
    species_list = sorted(list(set(sum([s.get_chemical_symbols() for s in structures_ase], []))))

    print(f"Max atoms found: {max_atoms}. Processing {len(structures_ase)} structures into features.")

    # 2. Featurize 
    X = np.array([get_coulomb_matrix_padded(atoms, max_atoms, species_list) for atoms in structures_ase])
    
    # 3. Align target energies *only* for the files that loaded successfully
    y_lookup = pd.Series(df_energies['dft_d3_energy_kJ_mol'].values, index=df_energies['filename']).to_dict()
    y = np.array([y_lookup[filename] * KJ_PER_MOL_TO_HARTREE for filename in processed_filenames])
    
    if len(X) != len(y):
        print(f"Internal Mismatch during processing: X length {len(X)}, Y length {len(y)}")
        sys.exit(1)
        
    return X, y

# --- Main Plotting Routine ---
if __name__ == "__main__":
    
    # 1. Load the trained model and scaler
    try:
        with open(MODEL_PATH, 'rb') as f:
            best_model, scaler = pickle.load(f)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}.")
        sys.exit(1)

    # 2. Reload the original data (X and y in Hartree)
    X, y_true = load_dataset_custom_for_plot(DATA_DIR, ENERGY_FILE)
    
    if X.size == 0:
        sys.exit(1)
        
    # 3. Scale the data using the *SAVED* scaler object
    X_scaled = scaler.transform(X)

    # 4. Predict using the loaded model
    y_pred = best_model.predict(X_scaled)
    
    # 5. Generate metrics and plot
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nModel performance on full dataset: R^2 = {r2:.4f}, MAE = {mae:.4f} Hartree")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='w', s=80)
    
    min_val = min(y_true.min(), y_pred.min()) * 0.95
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2) 
    
    plt.xlabel("Actual Dispersion Energy (Hartree)")
    plt.ylabel("Predicted Dispersion Energy (Hartree)")
    plt.title(f"MLP Parity Plot (R² = {r2:.2f})")
    plt.grid(True)
    
    plot_path = os.path.join(DATA_DIR, "final_parity_plot.png")
    plt.savefig(plot_path)
    print(f"Parity plot saved to {plot_path}")

