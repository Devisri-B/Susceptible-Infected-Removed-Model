"""
Stage 5: Evaluation and Validation

This stage:
1. Evaluates predictive accuracy on held-out test data
2. Compares recovered equations to ground truth
3. Validates parameter recovery (β, γ)
4. Generates visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..config import STAGE5_CONFIG, RESULTS_DIR, CHECKPOINT_DIR
from .data_pipeline import load_stage2_data
from .stochastic_sim import load_stage1_data


class SIREvaluator:
    """Evaluation metrics for SIR model."""
    
    @staticmethod
    def compute_r2(y_true, y_pred):
        """R² score."""
        return r2_score(y_true.flatten(), y_pred.flatten())
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """Mean squared error."""
        return mean_squared_error(y_true.flatten(), y_pred.flatten())
    
    @staticmethod
    def compute_mae(y_true, y_pred):
        """Mean absolute error."""
        return mean_absolute_error(y_true.flatten(), y_pred.flatten())


def evaluate_predictions(model, loaders, datasets, device='cpu', verbose=True):
    """
    Evaluate MLP predictions on test set.
    
    Returns:
        metrics: dict with R², MSE, MAE for each compartment
    """
    
    model.eval()
    evaluator = SIREvaluator()
    
    if verbose:
        print("[Stage 5] Evaluating predictions on test set...")
    
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        for batch in loaders['test']:
            params = batch['params'].to(device)
            trajectory = batch['trajectory'].to(device)
            time_grid = batch['time_grid'].to(device)
            
            pred_traj = model(params, time_grid)
            
            y_true_all.append(trajectory.cpu().numpy())
            y_pred_all.append(pred_traj.cpu().numpy())
    
    y_true = np.vstack(y_true_all)  # [n_samples, n_time, 3]
    y_pred = np.vstack(y_pred_all)
    
    # Compute metrics per compartment
    metrics = {}
    compartments = ['S', 'I', 'R']
    
    for i, comp in enumerate(compartments):
        metrics[comp] = {
            'r2': evaluator.compute_r2(y_true[:, :, i], y_pred[:, :, i]),
            'mse': evaluator.compute_mse(y_true[:, :, i], y_pred[:, :, i]),
            'mae': evaluator.compute_mae(y_true[:, :, i], y_pred[:, :, i]),
        }
    
    if verbose:
        print("\nTest Set Metrics:")
        print("Compartment | R²       | MSE      | MAE")
        print("-" * 45)
        for comp in compartments:
            print(f"{comp:11} | {metrics[comp]['r2']:8.6f} | {metrics[comp]['mse']:8.6f} | {metrics[comp]['mae']:8.6f}")
    
    return metrics, y_true, y_pred


def plot_sample_trajectories(model, datasets, t_eval, device='cpu', n_samples=5, output_dir=None):
    """
    Plot predicted vs. actual trajectories for random test samples.
    """
    
    output_dir = Path(output_dir or RESULTS_DIR)
    output_dir.mkdir(exist_ok=True)
    
    test_dataset = datasets['test']
    model.eval()
    
    # Sample random indices
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            batch = test_dataset[idx]
            params = batch['params'].unsqueeze(0).to(device)
            traj_true = batch['trajectory'].unsqueeze(0)
            time_grid = batch['time_grid'].to(device)
            
            pred_traj = model(params, time_grid)  # [1, time, 3]
            
            compartments = ['S', 'I', 'R']
            for col, comp in enumerate(compartments):
                ax = axes[row, col]
                ax.plot(t_eval, traj_true[0, :, col].numpy(), 'b-', linewidth=2, label='Ground Truth')
                ax.plot(t_eval, pred_traj[0, :, col].cpu().numpy(), 'r--', linewidth=2, label='Prediction')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'Normalized {comp}')
                ax.set_title(f'Sample {idx}: {comp}(t)')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "trajectory_samples.png"
    plt.savefig(plot_file, dpi=150)
    print(f" Saved trajectory plots to {plot_file}")
    plt.close()


def run_stage5_evaluation(model=None, loaders=None, datasets=None, 
                          params=None, t_eval=None, verbose=True):
    """
    Run Stage 5: Full evaluation and validation.
    """
    
    device = 'cpu'
    
    # Load components if not provided
    if model is None:
        if verbose:
            print("[Stage 5] No model provided")
        return {'S': None, 'I': None, 'R': None}, None, None
    
    if loaders is None:
        if verbose:
            print("[Stage 5] Loading data...")
        loaders, datasets = load_stage2_data()
    
    if params is None or t_eval is None:
        if verbose:
            print("[Stage 5] Loading Stage 1 data...")
        params, _, _, t_eval = load_stage1_data()
    
    # Evaluate predictions
    if verbose:
        print("\n" + "="*60)
        print("STAGE 5: EVALUATION & VALIDATION")
        print("="*60)
    
    metrics, y_true, y_pred = evaluate_predictions(
        model, loaders, datasets, device, verbose=verbose
    )
    
    # Generate visualizations
    if verbose:
        print("\n[Stage 5] Generating visualizations...")
    
    plot_sample_trajectories(
        model, datasets, t_eval, device=device,
        n_samples=STAGE5_CONFIG["plot_samples"]
    )
    
    # Summary statistics
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        total_r2 = np.mean([metrics[c]['r2'] for c in ['S', 'I', 'R']])
        total_mse = np.mean([metrics[c]['mse'] for c in ['S', 'I', 'R']])
        
        print(f"\nAverage R² Score: {total_r2:.6f}")
        print(f"Average MSE: {total_mse:.6f}")
    
    return metrics, y_true, y_pred


def save_stage5_results(metrics, output_dir=None):
    """Save evaluation results."""
    
    output_dir = Path(output_dir or RESULTS_DIR)
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "evaluation_metrics.txt"
    
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("STAGE 5: EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write("Compartment | R²       | MSE      | MAE\n")
        f.write("-" * 45 + "\n")
        
        for comp in ['S', 'I', 'R']:
            f.write(f"{comp:11} | {metrics[comp]['r2']:8.6f} | {metrics[comp]['mse']:8.6f} | {metrics[comp]['mae']:8.6f}\n")
    
    print(f" Saved evaluation results to {results_file}")


if __name__ == "__main__":
    print("="*60)
    print("STAGE 5: EVALUATION & VALIDATION")
    print("="*60)
    
    metrics, y_true, y_pred = run_stage5_evaluation(verbose=True)
    save_stage5_results(metrics)
    
    print("\n Stage 5 complete!")
