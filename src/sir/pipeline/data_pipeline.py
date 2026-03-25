"""
Stage 2: Data Pipeline and PyTorch Dataset

This stage:
1. Loads stochastic simulations from Stage 1
2. Normalizes/standardizes data
3. Creates PyTorch Dataset and DataLoader
4. Splits into train/val/test ensuring no parameter leakage
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle

from ..config import STAGE2_CONFIG, DATA_DIR, GENERAL_CONFIG
from .stochastic_sim import load_stage1_data
from ..utils import set_seed, split_dataset, normalize_trajectory


class SIRDataset(Dataset):
    """PyTorch Dataset for SIR mean trajectories."""
    
    def __init__(self, params, trajectories, t_eval, normalize=True):
        """
        Args:
            params: [n_samples, 4] – (β, γ, N, I₀)
            trajectories: [n_samples, n_time_steps, 3] – (s, i, r) normalized
            t_eval: [n_time_steps] – time grid
            normalize: Whether to apply z-score standardization per compartment
        """
        self.params = torch.from_numpy(params).float()
        self.trajectories = torch.from_numpy(trajectories).float()
        self.t_eval = torch.from_numpy(t_eval).float()
        
        # Compute normalization statistics on trajectories
        if normalize:
            self.traj_mean = self.trajectories.mean(dim=(0, 1), keepdim=True)
            self.traj_std = self.trajectories.std(dim=(0, 1), keepdim=True)
            self.traj_std = torch.clamp(self.traj_std, min=1e-8)  # Avoid division by zero
            self.trajectories = (self.trajectories - self.traj_mean) / self.traj_std
        else:
            self.traj_mean = None
            self.traj_std = None
        
        # Normalize parameters to [-1, 1] range for stability
        self.param_min = self.params.min(dim=0)[0]
        self.param_max = self.params.max(dim=0)[0]
        param_range = self.param_max - self.param_min
        param_range = torch.clamp(param_range, min=1e-8)
        self.params = 2 * (self.params - self.param_min) / param_range - 1
    
    def __len__(self):
        return len(self.params)
    
    def __getitem__(self, idx):
        """
        Returns:
            params: [4] – (β, γ, N, I₀) normalized
            trajectory: [n_time_steps, 3] – (s, i, r) normalized
            time_grid: [n_time_steps]
        """
        return {
            'params': self.params[idx],
            'trajectory': self.trajectories[idx],
            'time_grid': self.t_eval,
        }


def create_sir_loaders(params, mean_trajectories, t_eval, config=None):
    """
    Create train/val/test DataLoaders with no parameter leakage.
    
    Args:
        params: [n_samples, 4]
        mean_trajectories: [n_samples, n_time_steps, 3]
        t_eval: [n_time_steps]
        config: STAGE2_CONFIG
    
    Returns:
        loaders: dict with 'train', 'val', 'test' DataLoaders
        dataset: dict with 'train', 'val', 'test' Datasets
    """
    
    config = config or STAGE2_CONFIG
    set_seed(GENERAL_CONFIG["seed"])
    
    # Split dataset (no parameter leakage)
    train_idx, val_idx, test_idx = split_dataset(
        len(params),
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
    )
    
    # Create datasets
    datasets = {}
    for split, indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        datasets[split] = SIRDataset(
            params[indices],
            mean_trajectories[indices],
            t_eval,
            normalize=config["normalize"],
        )
    
    # Create dataloaders
    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        ),
    }
    
    return loaders, datasets


def run_stage2_pipeline(params=None, mean_traj=None, t_eval=None, verbose=True):
    """
    Run Stage 2: Prepare data pipeline.
    
    Returns:
        loaders: dict with 'train', 'val', 'test' DataLoaders
        datasets: dict with 'train', 'val', 'test' Datasets
    """
    
    # Load Stage 1 data if not provided
    if params is None:
        if verbose:
            print("[Stage 2] Loading Stage 1 data...")
        params, mean_traj, _, t_eval = load_stage1_data()
    
    if verbose:
        print(f"[Stage 2] Creating PyTorch datasets...")
        print(f"  Total samples: {len(params)}")
        print(f"  Train/Val/Test split: {STAGE2_CONFIG['train_ratio']:.0%} / {STAGE2_CONFIG['val_ratio']:.0%} / {STAGE2_CONFIG['test_ratio']:.0%}")
    
    loaders, datasets = create_sir_loaders(params, mean_traj, t_eval)
    
    if verbose:
        print(f" DataLoaders created:")
        for split in ['train', 'val', 'test']:
            print(f"  {split.upper()}: {len(datasets[split])} samples, batch_size={STAGE2_CONFIG['batch_size']}")
        
        # Verify a batch
        batch = next(iter(loaders['train']))
        print(f"\nBatch shapes:")
        print(f"  params: {batch['params'].shape}")
        print(f"  trajectory: {batch['trajectory'].shape}")
        print(f"  time_grid: {batch['time_grid'].shape}")
    
    return loaders, datasets

def save_stage2_data(loaders, datasets, output_dir=None):
    """Save Stage 2 DataLoaders and Datasets."""
    
    output_dir = Path(output_dir or DATA_DIR)
    output_dir.mkdir(exist_ok=True)
    
    stage2_file = output_dir / "stage2_dataloaders.pkl"
    with open(stage2_file, 'wb') as f:
        pickle.dump({
            'loaders': loaders,
            'datasets': datasets,
        }, f)
    
    print(f" Saved Stage 2 data to {stage2_file}")
    return stage2_file

def load_stage2_data(input_dir=None):
    """Load Stage 2 DataLoaders."""
    
    input_dir = Path(input_dir or DATA_DIR)
    stage2_file = input_dir / "stage2_dataloaders.pkl"
    
    if not stage2_file.exists():
        raise FileNotFoundError(f"Stage 2 data not found at {stage2_file}")
    
    with open(stage2_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f" Loaded Stage 2 data from {stage2_file}")
    return data['loaders'], data['datasets']



if __name__ == "__main__":
    print("="*60)
    print("STAGE 2: DATA PIPELINE")
    print("="*60)
    
    loaders, datasets = run_stage2_pipeline(verbose=True)
    save_stage2_data(loaders, datasets)
    print("\n Stage 2 complete!")
