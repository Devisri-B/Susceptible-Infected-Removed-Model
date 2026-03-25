"""
Utility functions for SIR model pipeline.
"""

import numpy as np
import torch
import random
from pathlib import Path


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_parameter_grid(config, n_points):
    """
    Sample parameter grid uniformly from ranges.
    
    Returns:
        params: [n_points, 4] – (β, γ, N, I₀) samples
    """
    beta = np.random.uniform(*config["beta_range"], n_points)
    gamma = np.random.uniform(*config["gamma_range"], n_points)
    N = np.random.randint(config["N_range"][0], config["N_range"][1], n_points)
    I0 = np.random.randint(config["I0_range"][0], config["I0_range"][1], n_points)
    
    params = np.column_stack([beta, gamma, N, I0])
    return params.astype(np.float32)


def normalize_trajectory(traj, N):
    """
    Normalize compartments to [0, 1] by dividing by population N.
    
    Args:
        traj: [n_time_steps, 3] – (S, I, R) counts
        N: Population size
    
    Returns:
        norm_traj: [n_time_steps, 3] – normalized (s, i, r)
    """
    return traj.astype(np.float32) / N.astype(np.float32)





def split_dataset(n_samples, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset indices into train/val/test.
    Ensures parameter sets don't leak across splits.
    """
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return train_idx, val_idx, test_idx


def ground_truth_trajectory(beta, gamma, N, I0, t_eval):
    """
    Solve ground truth SIR ODEs using scipy for comparison.
    
    Returns:
        trajectory: [len(t_eval), 3] – (S, I, R) normalized
    """
    from scipy.integrate import odeint
    
    S0 = N - I0
    R0 = 0
    y0 = [S0, I0, R0]
    
    def sir_odes(y, t, beta, gamma, N):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    traj = odeint(sir_odes, y0, t_eval, args=(beta, gamma, N))
    return (traj / N).astype(np.float32)  # Normalize


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def is_stopped(self):
        return self.early_stop


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    print(" Random seed set")
    
    from config import STAGE1_CONFIG
    params = sample_parameter_grid(STAGE1_CONFIG, 10)
    print(f" Sampled params shape: {params.shape}")
    print(f"  β ∈ [{params[:, 0].min():.3f}, {params[:, 0].max():.3f}]")
    print(f"  γ ∈ [{params[:, 1].min():.3f}, {params[:, 1].max():.3f}]")
