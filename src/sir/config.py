"""
Configuration for the SIR epidemic model learning pipeline.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  
DATA_DIR = PROJECT_ROOT / "src" / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "src" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "src" / "results"


for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# ============ STAGE 1: STOCHASTIC SIMULATION ============
STAGE1_CONFIG = {
    # Parameter grid: β, γ, N, I₀
    "beta_range": (0.3, 0.8),      # Transmission rate
    "gamma_range": (0.1, 0.3),     # Recovery rate
    "N_range": (1000, 5000),       # Population size
    "I0_range": (10, 50),          # Initial infected
    
    # Number of parameter combinations to sample
    "n_param_points": 2000,  # Mid-scale: 2000 parameter points
    
    # Trajectories per parameter point (for mean estimation)
    "n_trajectories": 50,
    
    # Time grid for simulation
    "t_max": 100.0,
    "n_time_steps": 101,  # 0, 1, 2, ..., 100
}

# ============ STAGE 2: DATA PIPELINE ============
STAGE2_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    
    "batch_size": 32,
    "num_workers": 0, 
    
    "normalize": True,  # Normalize to [0, 1]
    "standardize": False,  # Add z-score standardization
}

# ============ STAGE 3: NEURAL ODE TRAINING ============
STAGE3_CONFIG = {
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    
    "n_epochs": 50,
    "early_stopping_patience": 10,
    
    # ODE solver settings
    "solver": "dopri5",
    "rtol": 1e-4,
    "atol": 1e-5,
    "adjoint_rtol": 1e-4,
    "adjoint_atol": 1e-5,
    
    "device": "cpu",  # Change to "cuda" if GPU available
    "checkpoint_every": 5,
}

# ============ STAGE 4: SYMBOLIC RECOVERY ============
STAGE4_CONFIG = {
    # PySR configuration
    "populations": 30,
    "generations": 100,
    "population_size": 50,
    "max_complexity": 20,
    
    # Custom operators for epidemiology
    "binary_operators": ["plus", "sub", "mult", "div"],
    "unary_operators": ["exp", "log"],
    
    # Sampling strategy
    "n_derivative_points": 5000,  # Evaluate derivatives at many points
    
    "timeout_in_seconds": 3600,  # Max 1 hour for symbolic regression
    "verbosity": 1,
}

# ============ STAGE 5: EVALUATION ============
STAGE5_CONFIG = {
    "metrics": ["r2", "mse", "mae"],
    "plot_samples": 10,  # Plot 10 example trajectories
}

# ============ GENERAL ============
GENERAL_CONFIG = {
    "seed": 42,
    "verbose": True,
}
