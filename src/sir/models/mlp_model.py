"""
Train a simple MLP to predict SIR trajectories pointwise: (β, γ, N, I₀, t) → (S, I, R)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from ..config import STAGE3_CONFIG, CHECKPOINT_DIR
from ..utils import EarlyStopping


class SIR_MLP(nn.Module):
    """
    The MLP learns to predict the SIR state at any given time point.
    
    Input:  [β, γ, N, I₀, t]  (5-dimensional)
    Output: [S, I, R]         (3-dimensional)
    
    This is a regression problem: map (4 params + 1 time) → 3 compartments
    """
    
    def __init__(self, hidden_dims=[128, 128, 128], dropout=0.1):
        super().__init__()
        
        layers = []
        in_dim = 5  # β, γ, N, I₀, t
        
        # Hidden layers with batch norm and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, 3))  # S, I, R
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, params, t_eval):
        """
        Args:
            params: [batch, 4] – (β, γ, N, I₀) normalized
            t_eval: [time_steps] or [batch, time_steps]
        
        Returns:
            trajectory: [batch, time_steps, 3] – (S, I, R)
        """
        batch_size = params.shape[0]
        
        # Ensure t_eval is 1D
        if t_eval.ndim > 1:
            t_eval = t_eval[0]
        n_time_steps = len(t_eval)
        
        # Expand params and t_eval to create input matrix
        # [batch, time_steps, 5]
        params_expanded = params.unsqueeze(1).expand(batch_size, n_time_steps, 4)
        t_expanded = t_eval.unsqueeze(-1).unsqueeze(0).expand(batch_size, n_time_steps, 1)
        
        # Concatenate: [batch, time_steps, 5]
        x = torch.cat([params_expanded, t_expanded], dim=2)
        
        # Reshape to [batch * time_steps, 5] for network
        x_flat = x.reshape(batch_size * n_time_steps, 5)
        
        # Forward pass
        output = self.network(x_flat)  # [batch * time_steps, 3]
        
        # Reshape back: [batch, time_steps, 3]
        trajectory = output.reshape(batch_size, n_time_steps, 3)
        
        return trajectory


class MLPTrainer:
    """Trainer for MLP model."""
    
    def __init__(self, model, device, config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config or STAGE3_CONFIG
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.early_stopping = EarlyStopping(patience=self.config["early_stopping_patience"])
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)
            time_grid = batch['time_grid'].to(self.device)
            
            # Forward pass
            pred_traj = self.model(params, time_grid)  # [batch, time, 3]
            
            # MSE loss
            loss = torch.mean((pred_traj - trajectory) ** 2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(params)
        
        return total_loss / len(train_loader.dataset)
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        for batch in val_loader:
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)
            time_grid = batch['time_grid'].to(self.device)
            
            pred_traj = self.model(params, time_grid)
            loss = torch.mean((pred_traj - trajectory) ** 2)
            total_loss += loss.item() * len(params)
        
        return total_loss / len(val_loader.dataset)
    
    def train(self, train_loader, val_loader, checkpoint_dir=None):
        """Train MLP model."""
        
        checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
        checkpoint_dir.mkdir(exist_ok=True)
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_epoch = 0
        
        print(f"Training MLP on {self.device.upper()}...")
        
        for epoch in range(1, self.config["n_epochs"] + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if self.early_stopping.is_stopped():
                print(f"Early stopping at epoch {epoch}")
                break
        
        history['best_epoch'] = best_epoch
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        
        return history


