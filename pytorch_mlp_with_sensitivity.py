import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ase.io import read
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = os.path.expanduser("~/param-scanner")
ENERGY_FILE = os.path.join(DATA_DIR, "energies_dispersion.csv")
KJ_PER_MOL_TO_HARTREE = 0.0003808800
PREPROCESSING_MODE = 'simple'

# Hyperparameters
HIDDEN_SIZES = [100, 100, 50]
LEARNING_RATE = 0.0005
BATCH_SIZE = 5
N_EPOCHS = 1000
RANDOM_STATE = 42

def get_coulomb_matrix_padded(atoms, max_atoms, species_list):
    """Generate and pad Coulomb Matrix - same as training script."""
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

def load_dataset_with_mass(data_dir, energy_csv):
    """Load structures, extract Coulomb Matrix features, and include molecular mass."""
    df_energies = pd.read_csv(energy_csv)
    structures_ase = []
    processed_filenames = []
    masses = []
    
    for _, row in df_energies.iterrows():
        filename = row['filename'].strip()
        mass = row['mass']
        xyz_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(xyz_path):
            print(f"Warning: File not found {xyz_path}. Skipping.")
            continue
        try:
            atoms = read(xyz_path)
            structures_ase.append(atoms)
            processed_filenames.append(filename)
            masses.append(mass)
        except Exception as e:
            print(f"Error reading {xyz_path}: {e}. Skipping.")
            continue
            
    if not structures_ase:
        print("No structures loaded.")
        return np.array([]), np.array([]), np.array([])
        
    max_atoms = max(len(atoms) for atoms in structures_ase)
    
    # Extract Coulomb Matrix features
    X_cm = np.array([get_coulomb_matrix_padded(atoms, max_atoms, []) for atoms in structures_ase])
    
    # Get masses as numpy array
    masses_array = np.array(masses).reshape(-1, 1)  # Shape: (n_samples, 1)
    
    # Concatenate Coulomb Matrix features with mass
    X = np.hstack([X_cm, masses_array])  # Shape: (n_samples, n_cm_features + 1)
    
    # Get target energies
    y_lookup = dict(zip(df_energies['filename'].str.strip(), 
                       df_energies['dft_d3_energy_kJ_mol'] * KJ_PER_MOL_TO_HARTREE))
    y = np.array([y_lookup[f] for f in processed_filenames])
    
    return X, y, masses_array, len(X_cm[0])  # Return also mass array and CM feature size for analysis

# PyTorch Dataset
class MolecularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation='tanh'):
        super(FeedforwardNN, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def compute_gradient_importance(model, X_tensor, y_tensor, cm_feature_size):
    """Compute gradient-based feature importance.
    Returns mean absolute gradients for each feature.
    """
    model.eval()
    X_tensor.requires_grad_(True)
    
    # Forward pass
    y_pred = model(X_tensor)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=X_tensor,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=False,
        retain_graph=False
    )[0]
    
    # Mean absolute gradient per feature
    mean_abs_gradients = torch.abs(gradients).mean(dim=0).detach().numpy()
    
    # Separate Coulomb Matrix features and mass feature
    cm_gradients = mean_abs_gradients[:cm_feature_size]
    mass_gradient = mean_abs_gradients[cm_feature_size]
    
    return mean_abs_gradients, cm_gradients, mass_gradient

def integrated_gradients(model, X_baseline, X_input, n_steps=50):
    """Compute Integrated Gradients for a single sample.
    X_baseline: baseline input (e.g., zero or mean)
    X_input: input to analyze
    """
    model.eval()
    
    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, n_steps)
    gradients_sum = torch.zeros_like(X_input)
    
    for alpha in alphas:
        x_interp = X_baseline + alpha * (X_input - X_baseline)
        x_interp.requires_grad_(True)
        
        y_pred = model(x_interp)
        
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_interp,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=False,
            retain_graph=False
        )[0]
        
        gradients_sum += gradients * (X_input - X_baseline)
    
    integrated_grads = gradients_sum / n_steps
    return integrated_grads

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    print("Loading dataset...")
    X, y, masses, cm_feature_size = load_dataset_with_mass(DATA_DIR, ENERGY_FILE)
    
    if X.size == 0:
        print("Error: No data loaded.")
        exit(1)
    
    print(f"Loaded {len(X)} structures")
    print(f"Coulomb Matrix features: {cm_feature_size}")
    print(f"Total features (CM + mass): {X.shape[1]}")
    print(f"Molecular mass range: [{masses.min():.2f}, {masses.max():.2f}]")
    
    # Scale features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create dataset and dataloader
    dataset = MolecularDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = FeedforwardNN(X_scaled.shape[1], HIDDEN_SIZES, activation='tanh')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\nTraining model for {N_EPOCHS} epochs...")
    model.train()
    for epoch in range(N_EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {avg_loss:.6f}")
    
    print("Training completed.\n")
    
    # Make predictions on full dataset
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        y_pred_scaled = model(X_tensor).numpy().flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Convert back to original units for evaluation
    y_true_hartree = y
    y_pred_hartree = y_pred
    
    # Calculate metrics
    r2 = r2_score(y_true_hartree, y_pred_hartree)
    mae = mean_absolute_error(y_true_hartree, y_pred_hartree)
    
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.6f} Hartree ({mae/KJ_PER_MOL_TO_HARTREE:.2f} kJ/mol)")
    
    # ========== Sensitivity Analysis ==========
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS - Gradient-Based Feature Importance")
    print("=" * 60)
    
    # Convert to tensors for gradient computation
    X_tensor_grad = torch.FloatTensor(X_scaled)
    y_tensor_grad = torch.FloatTensor(y_scaled.reshape(-1, 1))
    
    # Compute gradients
    mean_abs_grads, cm_grads, mass_grad = compute_gradient_importance(
        model, X_tensor_grad, y_tensor_grad, cm_feature_size
    )
    
    print(f"\nMolecular mass gradient (absolute mean): {mass_grad:.6f}")
    print(f"Coulomb matrix features - gradient statistics:")
    print(f"  Mean: {cm_grads.mean():.6f}")
    print(f"  Std: {cm_grads.std():.6f}")
    print(f"  Max: {cm_grads.max():.6f}")
    print(f"  Min: {cm_grads.min():.6f}")
    
    # Rank features by importance
    print(f"\nTop 10 most important Coulomb matrix features (by gradient):")
    top_indices = np.argsort(cm_grads)[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Feature {idx}: {cm_grads[idx]:.6f}")
    
    # Compare mass importance to CM features
    print(f"\nMolecular mass importance vs Coulomb matrix features:")
    print(f"  Molecular mass gradient: {mass_grad:.6f}")
    print(f"  Molecular mass percentile rank: {100 * (cm_grads < mass_grad).mean():.1f}%")
    print(f"  (Higher percentile = more important)")
    
    # ========== Integrated Gradients Example ==========
    print("\n" + "=" * 60)
    print("INTEGRATED GRADIENTS (Example on First Sample)")
    print("=" * 60)
    
    # Use zero baseline
    X_baseline = torch.zeros(1, X_scaled.shape[1])
    X_sample = torch.FloatTensor(X_scaled[0:1])
    
    ig_attrs = integrated_gradients(model, X_baseline, X_sample, n_steps=50)
    ig_attrs_np = ig_attrs.detach().numpy().flatten()
    
    cm_ig = ig_attrs_np[:cm_feature_size]
    mass_ig = ig_attrs_np[cm_feature_size]
    
    print(f"Molecular mass integrated gradient: {mass_ig:.6f}")
    print(f"Coulomb matrix integrated gradients - mean: {cm_ig.mean():.6f}")
    print(f"Coulomb matrix integrated gradients - max: {cm_ig.max():.6f}")
    print(f"Coulomb matrix integrated gradients - min: {cm_ig.min():.6f}")
    
    # ========== Visualization ==========
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Plot 1: Feature Importance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Molecular mass vs CM feature importance (gradient-based)
    bar_labels = ['Molecular mass', 'CM features\n(mean)']
    bar_colors = ['orange', 'cyan']
    bar_values = [mass_grad, cm_grads.mean()]
    ax1.bar(bar_labels, bar_values, color=bar_colors, alpha=0.75)
    ax1.set_ylabel('Mean absolute gradient', fontsize=11)
    ax1.set_title('Feature importance: molecular mass vs Coulomb matrix\n(gradient-based)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for idx, val in enumerate(bar_values):
        ax1.text(idx, val, f'{val:.3f}', ha='center', va='bottom', fontsize=10, color='black')
    
    # Distribution of CM feature gradients
    ax2.hist(cm_grads, bins=30, alpha=0.7, color='olive', edgecolor='black', label='CM feature gradients')
    ax2.axvline(mass_grad, color='firebrick', linestyle='--', linewidth=2, 
                label=f'Molecular mass gradient = {mass_grad:.4f}')
    ax2.set_xlabel('Mean absolute gradient', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Coulomb matrix feature gradients', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    importance_plot_path = os.path.join(DATA_DIR, "feature_importance_gradients.png")
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {importance_plot_path}")
    plt.close()
    
    # Plot 2: Parity plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    y_true_kj = y_true_hartree / KJ_PER_MOL_TO_HARTREE
    y_pred_kj = y_pred_hartree / KJ_PER_MOL_TO_HARTREE
    
    # Find indices of two points with least true dispersion energy
    sorted_indices = np.argsort(y_true_kj)
    two_lowest_indices = sorted_indices[:2]
    
    # Create color array: limegreen for most points, red for two lowest
    point_colors = ['limegreen'] * len(y_true_kj)
    for idx in two_lowest_indices:
        point_colors[idx] = 'red'
    
    # Plot all points with appropriate colors
    for i, (true_val, pred_val) in enumerate(zip(y_true_kj, y_pred_kj)):
        ax.scatter(true_val, pred_val, alpha=0.6, s=50, edgecolors='black', 
                  linewidths=0.5, color=point_colors[i])
    
    min_val = min(y_true_kj.min(), y_pred_kj.min()) * 0.95
    max_val = max(y_true_kj.max(), y_pred_kj.max()) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'purple', linestyle='--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('True Dispersion Energy (kJ/mol)', fontsize=11)
    ax.set_ylabel('Predicted Dispersion Energy (kJ/mol)', fontsize=11)
    ax.set_title(f'Parity Plot (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    parity_plot_path = os.path.join(DATA_DIR, "pytorch_parity_plot.png")
    plt.savefig(parity_plot_path, dpi=300, bbox_inches='tight')
    print(f"Parity plot saved to: {parity_plot_path}")
    plt.close()
    
    print("\nAll analyses completed successfully!")

