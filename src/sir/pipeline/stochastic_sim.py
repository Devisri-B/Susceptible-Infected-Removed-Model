"""
Stage 1: Stochastic SIR Simulation using Gillespie Algorithm

This stage generates synthetic epidemics for a grid of SIR parameters.
For each parameter point (β, γ, N, I₀), multiple stochastic trajectories are run,
and their mean is computed to serve as the regression target for Stage 3.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

from ..config import STAGE1_CONFIG, DATA_DIR, GENERAL_CONFIG
from ..utils import set_seed, sample_parameter_grid, normalize_trajectory


class GillespieSimulator:
    """Stochastic SIR simulation using Gillespie algorithm."""
    
    def __init__(self, beta, gamma, N, I0, seed=None):
        """
        Args:
            beta: Transmission rate
            gamma: Recovery rate
            N: Population size
            I0: Initial infected count
            seed: Random seed for reproducibility
        """
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.I0 = I0
        self.S0 = N - I0
        self.R0 = 0
        
        if seed is not None:
            np.random.seed(seed)
    
    def simulate(self, t_max, max_steps=10000):
        """
        Run one stochastic trajectory using Gillespie algorithm.
        
        Args:
            t_max: Maximum simulation time
            max_steps: Maximum number of reaction events
        
        Returns:
            times: [n_events] – event times
            trajectory: [n_events, 3] – (S, I, R) at each event
        """
        S, I, R = self.S0, self.I0, self.R0
        t = 0.0
        times = [t]
        trajectory = [[S, I, R]]
        
        for _ in range(max_steps):
            if I == 0:  # Epidemic over
                break
            
            # Propensity functions (reaction rates)
            a_infection = self.beta * S * I / self.N
            a_recovery = self.gamma * I
            a_total = a_infection + a_recovery
            
            # Time to next event: exponential distribution
            if a_total > 0:
                tau = np.random.exponential(1.0 / a_total)
            else:
                break
            
            t += tau
            if t > t_max:
                break
            
            # Which reaction occurs?
            event = np.random.uniform(0, a_total)
            if event < a_infection:
                # Infection
                S -= 1
                I += 1
            else:
                # Recovery
                I -= 1
                R += 1
            
            times.append(t)
            trajectory.append([S, I, R])
        
        return np.array(times), np.array(trajectory)
    
    def simulate_and_interpolate(self, t_eval, n_trajectories=100):
        """
        Run multiple stochastic trajectories and interpolate to fixed time grid.
        
        Args:
            t_eval: [n_time_steps] – fixed time grid
            n_trajectories: Number of stochastic runs
        
        Returns:
            mean_traj: [n_time_steps, 3] – mean (S, I, R) at each time
            std_traj: [n_time_steps, 3] – standard deviation
        """
        trajectories = []
        
        for i in range(n_trajectories):
            times, traj = self.simulate(t_eval[-1])
            
            # Interpolate to fixed time grid
            interp_traj = np.zeros((len(t_eval), 3))
            for j, t in enumerate(t_eval):
                # Find closest event time ≤ t
                idx = np.searchsorted(times, t, side='right') - 1
                idx = np.clip(idx, 0, len(traj) - 1)
                interp_traj[j] = traj[idx]
            
            trajectories.append(interp_traj)
        
        trajectories = np.array(trajectories)  # [n_traj, n_time_steps, 3]
        mean_traj = trajectories.mean(axis=0)
        std_traj = trajectories.std(axis=0)
        
        return mean_traj, std_traj


def run_stage1_simulation(n_param_points=None, n_trajectories=None, t_max=None, 
                          n_time_steps=None, verbose=True):
    """
    Run Stage 1: Generate synthetic epidemics across parameter grid.
    
    Returns:
        params: [n_param_points, 4] – (β, γ, N, I₀)
        mean_trajectories: [n_param_points, n_time_steps, 3] – mean (s, i, r)
        std_trajectories: [n_param_points, n_time_steps, 3] – std (s, i, r)
        t_eval: [n_time_steps] – time grid
    """
    
    config = STAGE1_CONFIG
    n_param_points = n_param_points or config["n_param_points"]
    n_trajectories = n_trajectories or config["n_trajectories"]
    t_max = t_max or config["t_max"]
    n_time_steps = n_time_steps or config["n_time_steps"]
    
    set_seed(GENERAL_CONFIG["seed"])
    
    # Time grid
    t_eval = np.linspace(0, t_max, n_time_steps)
    
    # Sample parameters
    if verbose:
        print(f"[Stage 1] Sampling {n_param_points} parameter points...")
    params = sample_parameter_grid(config, n_param_points)
    
    # Run simulations
    mean_trajectories = []
    std_trajectories = []
    
    if verbose:
        pbar = tqdm(total=n_param_points, desc="Simulating epidemics")
    
    for i, (beta, gamma, N, I0) in enumerate(params):
        simulator = GillespieSimulator(beta, gamma, int(N), int(I0), seed=i)
        mean_traj, std_traj = simulator.simulate_and_interpolate(t_eval, n_trajectories)
        
        # Normalize to [0, 1]
        mean_traj = normalize_trajectory(mean_traj, N)
        std_traj = normalize_trajectory(std_traj, N)
        
        mean_trajectories.append(mean_traj)
        std_trajectories.append(std_traj)
        
        if verbose:
            pbar.update(1)
    
    if verbose:
        pbar.close()
    
    mean_trajectories = np.array(mean_trajectories)
    std_trajectories = np.array(std_trajectories)
    
    if verbose:
        print(f" Generated {n_param_points} trajectories")
        print(f"  Shape: {mean_trajectories.shape} – (params, time_steps, compartments)")
        print(f"  Time range: [0, {t_max}]")
        print(f"  Mean (s, i, r) ranges:")
        for comp_name in ['S', 'I', 'R']:
            comp_idx = {'S': 0, 'I': 1, 'R': 2}[comp_name]
            print(f"    {comp_name}: [{mean_trajectories[:,:,comp_idx].min():.4f}, {mean_trajectories[:,:,comp_idx].max():.4f}]")
    
    return params, mean_trajectories, std_trajectories, t_eval


def save_stage1_data(params, mean_traj, std_traj, t_eval, output_dir=None):
    """Save Stage 1 results to disk."""
    
    output_dir = Path(output_dir or DATA_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Save as pickle for efficiency
    stage1_file = output_dir / "stage1_simulations.pkl"
    with open(stage1_file, 'wb') as f:
        pickle.dump({
            'params': params,
            'mean_trajectories': mean_traj,
            'std_trajectories': std_traj,
            't_eval': t_eval,
        }, f)
    
    print(f" Saved Stage 1 data to {stage1_file}")
    
    return stage1_file


def load_stage1_data(input_dir=None):
    """Load Stage 1 results from disk."""
    
    input_dir = Path(input_dir or DATA_DIR)
    stage1_file = input_dir / "stage1_simulations.pkl"
    
    if not stage1_file.exists():
        raise FileNotFoundError(f"Stage 1 data not found at {stage1_file}")
    
    with open(stage1_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f" Loaded Stage 1 data from {stage1_file}")
    return data['params'], data['mean_trajectories'], data['std_trajectories'], data['t_eval']
