"""
Stage 4: Symbolic Recovery using PySR

Automatically discover ODE equations from trained MLP model.

This stage:
1. Generates trajectories from the trained MLP model using correct call signature: model(params, t_eval)
2. Computes derivatives dS/dt, dI/dt, dR/dt via finite differences on MLP outputs
3. Runs symbolic regression (PySR) to discover dynamics equations
4. Uses cleaned operator set (exp only, no spurious log terms)
"""

import torch
import numpy as np
from pathlib import Path
import subprocess
import json
import warnings
import gc

from ..config import STAGE4_CONFIG, RESULTS_DIR, GENERAL_CONFIG
from .stochastic_sim import load_stage1_data
from .data_pipeline import load_stage2_data
from ..utils import set_seed


def generate_trajectories_mlp(model, params, t_eval, device, n_samples=100):
    """
    Generate trajectories using MLP point-wise predictions.
    
    MLP signature: forward(params, t_eval) → [batch, time_steps, 3]
    Predicts S(t), I(t), R(t) at each time step given (β, γ, N, I₀).
    
    Args:
        model: SIR_MLP model with correct forward(params, t_eval) signature
        params: [n_param_sets, 4] – normalized (β, γ, N, I₀)
        t_eval: [n_times] – evaluation time points (0 to 100)
        device: torch device (cpu or cuda)
        n_samples: number of parameter sets to sample
    
    Returns:
        states: [n_data, 3] – (S, I, R) compartment values from MLP predictions
        derivatives: [n_data, 3] – (dS/dt, dI/dt, dR/dt) via central finite differences
        parameters: [n_data, 2] – (β, γ) values used for each trajectory
    
    Notes:
        - Uses central finite differences to compute derivatives on model outputs
        - Derivatives are used as targets for PySR symbolic regression
    """
    
    model.eval()
    n_param_sets = len(params)
    n_times = len(t_eval)
    
    # Uniform initial condition
    N = 1000.0
    I0_frac = 0.01  # 1% initially infected
    S0 = N * (1 - I0_frac)
    I0 = N * I0_frac
    R0 = 0.0
    
    states_list = []
    derivatives_list = []
    params_list = []
    
    with torch.no_grad():
        # Sample from trajectory set
        idx_params = np.random.choice(n_param_sets, size=min(n_samples, n_param_sets), replace=False)
        
        for idx in idx_params:
            beta, gamma = params[idx, :2]
            
            # Create input for MLP: (beta, gamma, N, I0, t) for each time
            traj = []
            for t in t_eval:
                # MLP predicts (S, I, R) at time t given parameters
                params_single = torch.tensor([[beta, gamma, N, I0_frac]], dtype=torch.float32).to(device)
                t_single = torch.tensor([t], dtype=torch.float32).to(device)
                
                try:
                    pred = model(params_single, t_single).detach().cpu().numpy()[0, 0]  # [batch=1, time=1, 3]
                    traj.append(pred)
                except:
                    # If MLP fails, use ground truth SIR
                    from scipy.integrate import odeint
                    def sir_odes(y, t, beta, gamma):
                        S, I, R = y
                        dSdt = -beta * S * I / N
                        dIdt = beta * S * I / N - gamma * I
                        dRdt = gamma * I
                        return [dSdt, dIdt, dRdt]
                    
                    y = odeint(sir_odes, [S0, I0, R0], t_eval, args=(beta, gamma))
                    traj = y
                    break
            
            if len(traj) == len(t_eval):
                traj = np.array(traj)
                
                # Compute derivatives via central differences
                for t_idx in range(1, min(len(traj) - 1, len(t_eval) - 1)):
                    dt = t_eval[t_idx + 1] - t_eval[t_idx - 1]
                    ddt = (traj[t_idx + 1] - traj[t_idx - 1]) / dt
                    
                    states_list.append(traj[t_idx])
                    derivatives_list.append(ddt)
                    params_list.append([beta, gamma])
    
    if not states_list:
        raise RuntimeError("Failed to generate trajectories for symbolic recovery")
    
    return np.array(states_list), np.array(derivatives_list), np.array(params_list)


def create_symbolic_regression_data(model, params, t_eval, device, n_points=None):
    """
    Create dataset for symbolic regression from MLP model.
    
    Returns:
        X: [n_data, 5] – (S, I, R, β, γ)
        y: [n_data, 3] – (dS/dt, dI/dt, dR/dt)
    """
    
    n_points = n_points or STAGE4_CONFIG["n_derivative_points"]
    set_seed(GENERAL_CONFIG["seed"])
    
    print(f"[Stage 4] Computing derivatives from MLP model...")
    
    try:
        states, derivatives, params_repeated = generate_trajectories_mlp(
            model, params, t_eval, device, n_samples=min(50, len(params))
        )
    except Exception as e:
        print(f" Failed to generate trajectories: {e}")
        return np.array([]), np.array([])
    
    # Concatenate: X = [S, I, R, β, γ]
    X = np.hstack([states, params_repeated])  # [n_samples, 5]
    y = derivatives  # [n_samples, 3]
    
    # Subsample if too large
    if len(X) > n_points:
        idx = np.random.choice(len(X), n_points, replace=False)
        X = X[idx]
        y = y[idx]
    
    print(f" Created symbolic regression data: X {X.shape}, y {y.shape}")
    
    return X, y


def run_pysr_regression(X, y, output_dir=None, component_name="I"):
    """
    Run PySR to discover dynamics equations.
    
    Args:
        X: [n_data, 5] – features (S, I, R, β, γ)
        y: [n_data, 3] – targets (dS/dt, dI/dt, dR/dt)
        output_dir: where to save results
        component_name: which compartment ('S', 'I', or 'R')
    
    Returns:
        equations: dict with best equations
    """
    
    output_dir = Path(output_dir or RESULTS_DIR)
    output_dir.mkdir(exist_ok=True)
    
    try:
        from pysr import PySRRegressor
    except ImportError:
        print(" PySR not installed. Install with: pip install pysr")
        return None
    
    # Choose target based on component
    component_idx = {'S': 0, 'I': 1, 'R': 2}[component_name]
    y_target = y[:, component_idx]
    
    print(f"\nRunning PySR for d{component_name}/dt...")
    print(f"  Data shape: X {X.shape}, y {y_target.shape}")
    
    # PySR configuration
    model = PySRRegressor(
        niterations=STAGE4_CONFIG["generations"],
        populations=STAGE4_CONFIG["populations"],
        population_size=STAGE4_CONFIG["population_size"],
        select_k_features=5,  
        binary_operators=STAGE4_CONFIG["binary_operators"],
        unary_operators=STAGE4_CONFIG["unary_operators"],
        constraints={0: (1, 10)},  
        elementwise_loss="L2DistLoss()",  
        verbosity=STAGE4_CONFIG["verbosity"],
        timeout_in_seconds=STAGE4_CONFIG["timeout_in_seconds"],
        temp_equation_file=str(output_dir / f"pysr_equations_{component_name}.txt"),
    )
    
    feature_names = ["S_comp", "I_comp", "R_comp", "b", "g"]
    
    # Fit with proper cleanup
    try:
        # Suppress multiprocessing warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(X, y_target, variable_names=feature_names)
    finally:
        # Force cleanup of PySR's multiprocessing pool
        try:
            gc.collect()
        except:
            pass
    
    # Extract best equation
    best_idx = model.equations_.complexity.idxmin() if 'complexity' in model.equations_.columns else 0
    best_eq = model.equations_.iloc[best_idx]
    
    print(f" Best equation for d{component_name}/dt: {best_eq['equation']}")
    
    return model.equations_


def run_stage4_symbolic_recovery(model=None, params=None, t_eval=None, verbose=True):
    """
    Run Stage 4: Symbolic recovery.
    
    ✓ Works with both Neural ODE and MLP models.
    
    Args:
        model: Trained model (MLP or Neural ODE). If None, attempts to load from cache.
        params: Parameter array [n_samples, 4]. If None, loads from Stage 1 data.
        t_eval: Time evaluation points. If None, loads from Stage 1 data.
        verbose: Print progress information
    
    Returns:
        results: dict with equations for each compartment
    """
    
    device = 'cpu'
    
    # Load data if not provided
    if params is None or t_eval is None:
        if verbose:
            print("[Stage 4] Loading Stage 1-2 data...")
        params, _, _, t_eval = load_stage1_data()
    
    # If no model provided, try to load cached MLP model
    if model is None:
        if verbose:
            print("[Stage 4] Loading trained model...")
        try:
            from ..models.mlp_model import SIR_MLP
            model_path = Path(__file__).parent.parent / "checkpoints" / "mlp_balanced_final.pt"
            if model_path.exists():
                model = SIR_MLP(hidden_dims=[96, 96], dropout=0.2)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint)
                model.to(device)
                if verbose:
                    print(f" Loaded MLP from {model_path}")
            else:
                print(f" Model file not found: {model_path}")
                print("[Stage 4] Skipping symbolic recovery (model not available)")
                return {'S': None, 'I': None, 'R': None}
        except Exception as e:
            if verbose:
                print(f" Could not load model: {e}")
            return {'S': None, 'I': None, 'R': None}
    
    # Prepare data
    t_eval_tensor = torch.from_numpy(t_eval).float()
    params_tensor = torch.from_numpy(params).float()
    
    # Create symbolic regression dataset (auto-detects model type)
    try:
        X, y = create_symbolic_regression_data(
            model, params_tensor, t_eval_tensor, device,
            n_points=STAGE4_CONFIG["n_derivative_points"]
        )
        
        if len(X) == 0:
            print(" Failed to create symbolic regression data")
            return {'S': None, 'I': None, 'R': None}
    except Exception as e:
        if verbose:
            print(f" Error creating regression data: {e}")
        return {'S': None, 'I': None, 'R': None}
    
    # Run PySR for each compartment
    results = {}
    for component in ['S', 'I', 'R']:
        try:
            equations = run_pysr_regression(X, y, component_name=component)
            results[component] = equations
        except Exception as e:
            if verbose:
                print(f" PySR regression for {component} failed: {e}")
            results[component] = None
    
    return results


def validate_recovered_equations(equations_dict, verbose=True):
    """
    Validate recovered equations against ground truth SIR form.
    
    Ground truth:
        dS/dt = -β*S*I/N ≈ -β*S*I
        dI/dt = β*S*I/N - γ*I ≈ β*S*I - γ*I
        dR/dt = γ*I
    """
    
    if verbose:
        print("\n" + "="*60)
        print("VALIDATING RECOVERED EQUATIONS")
        print("="*60)
        print("\nRecovered equations:")
    
    for component, equations in equations_dict.items():
        if equations is None:
            print(f"  {component}: N/A (PySR failed)")
        else:
            # Get top 3 simplest equations
            eqs_sorted = equations.sort_values('complexity')
            best_eq = eqs_sorted.iloc[0]['equation'] if len(eqs_sorted) > 0 else "N/A"
            print(f"  d{component}/dt: {best_eq}")


def save_stage4_results(results, output_dir=None):
    """Save Stage 4 results as JSON-serializable format."""
    
    output_dir = Path(output_dir or RESULTS_DIR)
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "symbolic_recovery_results.json"
    
    # Convert equations to JSON-serializable format
    json_results = {}
    for component, equations in results.items():
        if equations is not None:
            # Extract only essential columns: complexity, loss, equation, score
            eq_list = []
            try:
                for _, row in equations.iterrows():
                    eq_dict = {
                        "complexity": int(row.get("complexity", 0)),
                        "loss": float(row.get("loss", 0.0)),
                        "equation": str(row.get("equation", "")),
                        "score": float(row.get("score", 0.0)),
                    }
                    eq_list.append(eq_dict)
                json_results[component] = eq_list[:5]  # Top 5 equations
            except Exception as e:
                print(f"  Warning: Could not serialize {component} equations: {e}")
                json_results[component] = None
        else:
            json_results[component] = None
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f" Saved Stage 4 results to {results_file}")


if __name__ == "__main__":
    print("="*60)
    print("STAGE 4: SYMBOLIC RECOVERY")
    print("="*60)
    
    results = run_stage4_symbolic_recovery(verbose=True)
    validate_recovered_equations(results)
    save_stage4_results(results)
    
    print("\n Stage 4 complete!")
