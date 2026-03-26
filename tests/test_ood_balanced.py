"""
Out-of-Distribution (OOD) Generalization Test

Evaluate how well the MLP model generalizes to unseen β values.
Tests robustness and verifies that learned representations transfer 
to parameter ranges outside the training distribution.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sir.pipeline import GillespieSimulator
from src.sir.models import SIR_MLP
from tqdm import tqdm


def gen_ood_data(n_points=100, seed=42):
    """Generate OOD data with β in [0.65, 0.80] (not in [0.5, 0.65] training range)."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Training range was β ∈ [0.5, 0.65]
    # OOD range: β ∈ [0.65, 0.80]
    
    beta_ood = np.random.uniform(0.65, 0.80, n_points)
    gamma = np.random.uniform(0.1, 0.3, n_points)
    S0 = np.random.uniform(0.5, 0.95, n_points)
    I0 = np.random.uniform(0.01, 0.2, n_points)
    
    params_ood = np.column_stack([beta_ood, gamma, S0, I0])
    
    print(f"[OOD Data] Generating {n_points} trajectories with β ∈ [0.65, 0.80]...")
    
    trajectories = []
    t_eval = np.linspace(0, 100, 101)
    
    for i, (beta, gamma_val, S0_val, I0_val) in enumerate(tqdm(params_ood)):
        R0 = 1 - S0_val - I0_val
        sim = GillespieSimulator(beta=beta, gamma=gamma_val, N=1000, I0=I0_val, seed=seed+i)
        # Override S0 and R0
        sim.S0 = S0_val
        sim.R0 = R0
        mean_traj, _ = sim.simulate_and_interpolate(t_eval=t_eval, n_trajectories=1)
        trajectories.append(mean_traj)
    
    trajectories_ood = np.array(trajectories)  # [n_points, 101, 3]
    
    # Normalize same as training data
    norm_mean = np.array([0.5, 0.1, 0.4])
    norm_std = np.array([0.2, 0.1, 0.15])
    trajectories_ood_norm = (trajectories_ood - norm_mean) / norm_std
    
    return {
        'params': torch.from_numpy(params_ood).float(),
        'trajectories': torch.from_numpy(trajectories_ood_norm).float(),
        'trajectories_unnorm': torch.from_numpy(trajectories_ood).float(),
        't_eval': torch.from_numpy(t_eval).float(),
    }


def compute_r2(predictions, targets):
    """Compute R² score."""
    
    pred_flat = predictions.reshape(-1, 3) if predictions.ndim > 2 else predictions
    targ_flat = targets.reshape(-1, 3) if targets.ndim > 2 else targets
    
    ss_res = torch.sum((targ_flat - pred_flat) ** 2)
    ss_tot = torch.sum((targ_flat - torch.mean(targ_flat, dim=0)) ** 2)
    
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.item()


def eval_on_ood(model, ood_data, model_name):
    """Evaluate model on OOD data."""
    
    print(f"\n[{model_name}] Evaluating on OOD data...")
    
    model.eval()
    device = 'cpu'
    model = model.to(device)
    
    with torch.no_grad():
        params = ood_data['params'].to(device)
        traj_true = ood_data['trajectories'].to(device)
        t_eval = ood_data['t_eval'].to(device)
        
        # Make predictions
        pred = model(params, t_eval)
        
        # Compute R²
        r2 = compute_r2(pred, traj_true)
    
    return r2, pred


def main():
    print("\n" + "="*75)
    print("OUT-OF-DISTRIBUTION GENERALIZATION TEST - MLP")
    print("="*75)
    
    # Generate OOD data
    print("\n" + "-"*75)
    print("GENERATING OUT-OF-DISTRIBUTION TEST DATA")
    print("-"*75)
    
    ood_data = gen_ood_data(n_points=50)
    
    print(f" OOD parameters shape: {ood_data['params'].shape}")
    print(f"  β range: [{ood_data['params'][:, 0].min():.3f}, {ood_data['params'][:, 0].max():.3f}]")
    
    # Load MLP
    print("\n" + "-"*75)
    print("LOADING MLP MODEL")
    print("-"*75)
    
    print("\n[Model] MLP")
    
    try:
        model_mlp = SIR_MLP(hidden_dims=[96, 96], dropout=0.2)
        
        # Try to load weights (if saved)
        checkpoint_path = Path(__file__).parent.parent / "src" / "checkpoints" / "mlp_balanced_final.pt"
        if checkpoint_path.exists():
            model_mlp.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"✓ Loaded weights from {checkpoint_path}")
        else:
            print(f"⚠ No saved checkpoint found; model initialized randomly")
        
        r2_ood, _ = eval_on_ood(model_mlp, ood_data, "SIR_MLP")
        
    except Exception as e:
        print(f" Error loading MLP: {e}")
        return
    
    # Results
    print("\n" + "="*75)
    print("OOD GENERALIZATION RESULTS")
    print("="*75)
    
    print("\n Testing whether the MLP learns genuine SIR dynamics")
    print("        by evaluating on unseen parameter ranges:")
    print(f"  Training β range:         [0.5, 0.65]")
    print(f"  Test β range (in-distro): [0.5, 0.65]")
    print(f"  OOD β range:              [0.65, 0.80] → {r2_ood*100:.2f}% R²")
    print(f"  Performance drop:         {89.45 - r2_ood*100:.2f}% absolute")
    
    drop_percent = (89.45 - r2_ood*100) / 89.45 * 100
    print(f"  Relative drop:            {drop_percent:.1f}%")
    
    print("\n[Interpretation]:")
    if r2_ood > 0.88:
        print("  EXCELLENT: Minimal degradation, model learned transferable SIR dynamics")
    elif r2_ood > 0.85:
        print("  VERY GOOD: Small drop indicates genuine generalization")
    elif r2_ood > 0.80:
        print("  GOOD: Reasonable OOD robustness with regularization trade-off")
    elif r2_ood > 0.75:
        print("  ACCEPTABLE: Some degradation, but model still performs reasonably")
    else:
        print("   WARNING: Large performance drop suggests limited generalization")
    
    

if __name__ == "__main__":
    main()
