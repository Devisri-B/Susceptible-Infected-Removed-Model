"""
Hyperparameter configurations for Stage 3 (MLP training)
"""

# ============ BALANCED MLP ============
MLP_BALANCED_CONFIG = {
    'hidden_dims': [96, 96],    # Reduced but not too small
    'learning_rate': 1e-3,
    'weight_decay': 0.01,       # Moderate regularization
    'dropout': 0.2,             # Light regularization
    'n_epochs': 100,
    'batch_size': 32,
    'early_stopping_patience': 8,
}

# ============ EXPLORATION CONFIGS ============
STAGE3_CONSERVATIVE_CONFIG = {
    'hidden_dim': 40,
    'learning_rate': 1e-3,
    'weight_decay': 0.02,       # Stronger than balanced
    'dropout': 0.2,
    'n_epochs': 100,
    'batch_size': 32,
    'gradient_clip': 1.0,
    'early_stopping_patience': 5,  # Stricter
}

STAGE3_PERMISSIVE_CONFIG = {
    'hidden_dim': 56,           # Closer to original 64
    'learning_rate': 1e-3,
    'weight_decay': 0.005,      # Lighter than balanced
    'dropout': 0.1,             # Lighter dropout
    'n_epochs': 100,
    'batch_size': 32,
    'gradient_clip': 1.0,
    'early_stopping_patience': 10,  # More permissive
}
