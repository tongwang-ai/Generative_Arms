#!/usr/bin/env python3
"""
Run Random Baseline From Pre-Generated Data

This script runs the random baseline simulation using pre-generated simulation data,
similar to run_simulation_from_data.py but using random action generation instead
of intelligent optimization.

Input folder layout (created by generate_simulation_data.py):
  <data_dir>/
    data_config.json
    initialization/action_bank/action_bank.json
    iteration_1/users/users.json
    ...

This script:
  - Loads the initial action bank from <data_dir>
  - For each iteration i, loads users from <data_dir>/iteration_i/users/users.json
  - Runs the company simulation with LinUCB strategy
  - Uses random baseline algorithm (generates K random actions per iteration)
  - Saves all results under results/random_baseline_from_data_YYYYMMDD_HHMMSS
"""

import os
import json
import argparse
import shutil
import time
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm
import pandas as pd

import numpy as np

from src.simulation.workflow.company_simulator import CompanySimulator
from src.util.action_embedder import EmbeddedAction, OpenAIActionEmbedder
from src.algorithm.workflow.random_baseline_algorithm import RandomBaselineAlgorithm


def convert_to_embedded_action(action_data: Dict[str, Any]) -> EmbeddedAction:
    """Convert action dictionary to EmbeddedAction object."""
    return EmbeddedAction(
        action_id=action_data['action_id'],
        text=action_data['text'],
        embedding=np.array(action_data['embedding']),
        category=action_data.get('category', 'generated'),
        metadata=action_data.get('metadata', {})
    )


def load_data_config(data_dir: str) -> Dict[str, Any]:
    """Load configuration from data directory."""
    cfg_file = os.path.join(data_dir, 'data_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file, 'r') as f:
            return json.load(f)
    return {}


def run_random_baseline_from_data(data_dir: str,
                                 iterations: int,
                                 results_dir: str,
                                 openai_api_key: str = None,
                                 algorithm_config: Dict[str, Any] = None,
                                 company_strategy: str = "linucb",
                                 strategy_config: Dict[str, Any] = None,
                                 ground_truth_type: str = "mixture_of_experts",
                                 ground_truth_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run random baseline simulation from pre-generated data.
    """
    print("="*80)
    print("RANDOM BASELINE FROM DATA: Company + Random Algorithm")
    print("="*80)
    print(f"🎲 Using random baseline with {algorithm_config.get('action_bank_size', 20)} actions per iteration")
    print(f"📁 Data source: {data_dir}")
    print(f"🕐 Simulation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize components
    print(f"\n1. Initializing Components...")
    print(f"   Company strategy: {company_strategy}")
    print(f"   Ground truth type: {ground_truth_type}")
    print(f"   Algorithm type: RANDOM BASELINE")
    
    # Initialize company simulator
    company_sim = CompanySimulator(
        results_dir=results_dir,
        openai_api_key=openai_api_key,
        strategy_type=company_strategy,
        strategy_config=strategy_config,
        ground_truth_type=ground_truth_type,
        ground_truth_config=ground_truth_config,
        random_seed=42
    )
    
    # Initialize random baseline algorithm
    algorithm = RandomBaselineAlgorithm(
        results_dir=results_dir,
        algorithm_config=algorithm_config
    )
    
    # Load initial action bank from data
    init_action_bank_file = os.path.join(data_dir, 'initialization', 'action_bank', 'action_bank.json')
    if not os.path.exists(init_action_bank_file):
        raise FileNotFoundError(f"Initial action bank not found: {init_action_bank_file}")
    
    embedder = OpenAIActionEmbedder(api_key=openai_api_key)
    initial_bank = embedder.load_embedded_actions(init_action_bank_file)
    company_sim.current_action_bank = initial_bank
    
    print(f"   📋 Loaded {len(initial_bank)} initial actions from data")
    
    # Create initialization folder in results directory
    init_results_dir = os.path.join(results_dir, "initialization")
    os.makedirs(init_results_dir, exist_ok=True)
    init_action_bank_results_dir = os.path.join(init_results_dir, "action_bank")
    os.makedirs(init_action_bank_results_dir, exist_ok=True)
    
    # Copy action bank to results initialization folder
    dest_action_bank_file = os.path.join(init_action_bank_results_dir, "action_bank.json")
    shutil.copy2(init_action_bank_file, dest_action_bank_file)
    
    # Load data config to get user count
    data_cfg = load_data_config(data_dir)
    n_users_per_iteration = int(data_cfg.get('n_users_per_iteration', 1000))
    company_sim.n_users = n_users_per_iteration
    
    # Create initialization summary
    init_summary = {
        'timestamp': datetime.now().isoformat(),
        'n_users_per_iteration': n_users_per_iteration,
        'n_initial_actions': len(initial_bank),
        'initial_action_bank_size': len(initial_bank),
        'embedding_model': 'loaded_from_data',
        'source_data_dir': os.path.abspath(data_dir),
        'loaded_from_data': True,
        'baseline_type': 'random_generation'
    }
    
    init_summary_file = os.path.join(init_results_dir, "initialization_summary.json")
    with open(init_summary_file, 'w') as f:
        json.dump(init_summary, f, indent=2)
    
    print(f"   ✅ Component initialization completed")
    
    # Run iterations using pre-generated data
    simulation_results = {
        'simulation_type': 'random_baseline_from_data',
        'source_data_dir': data_dir,
        'iterations': [],
        'final_summary': {}
    }
    
    print(f"\n2. Running {iterations} iterations with Random Baseline from Data...")
    
    with tqdm(total=iterations, desc="🎲 Random Baseline from Data", unit="iteration") as pbar:
        for iteration in range(1, iterations + 1):
            print(f"\n{'='*60}")
            print(f"RANDOM BASELINE FROM DATA - ITERATION {iteration}/{iterations}")
            print(f"{'='*60}")
            print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
            
            # Load users from the data folder
            users_file = os.path.join(data_dir, f"iteration_{iteration}", 'users', 'users.json')
            if not os.path.exists(users_file):
                raise FileNotFoundError(f"Users file not found: {users_file}")
            
            print(f"📥 Loading users from: {users_file}")
            loaded_users = company_sim.user_generator.load_users(users_file)
            print(f"   ✅ Loaded {len(loaded_users)} users from data")
            
            # Monkeypatch the user generator to return loaded users
            def _return_loaded_users(_n):
                return loaded_users
            company_sim.user_generator.generate_users = _return_loaded_users
            
            # Company phase: run iteration with loaded users
            print(f"\n--- Company Phase (Using Pre-Generated Users) ---")
            company_phase_start = time.time()
            company_results = company_sim.run_iteration(iteration)
            company_phase_time = time.time() - company_phase_start
            print(f"   ⏱️  Company phase completed in {company_phase_time:.2f}s")
            
            # Random baseline algorithm phase
            print(f"\n--- Random Baseline Algorithm Phase ---")
            algorithm_phase_start = time.time()
            algorithm_results = algorithm.process_iteration(iteration)
            algorithm_phase_time = time.time() - algorithm_phase_start
            print(f"   ⏱️  Random baseline phase completed in {algorithm_phase_time:.2f}s")
            
            # Integration phase
            print(f"\n--- Integration Phase ---")
            integration_start = time.time()
            new_action_bank_file = algorithm_results['files_created']['new_action_bank']
            with open(new_action_bank_file, 'r') as f:
                new_bank_data = json.load(f)
            
            new_action_bank = [convert_to_embedded_action(action_data) 
                              for action_data in new_bank_data['actions']]
            
            print(f"Random baseline returned {len(new_action_bank)} actions")
            
            if len(new_action_bank) == 0:
                print("   No new actions returned - keeping current action bank unchanged")
            else:
                company_sim.update_action_bank_preserve_ids(new_action_bank, iteration)
            
            integration_time = time.time() - integration_start
            print(f"   ⏱️  Integration phase completed in {integration_time:.2f}s")
            
            # Store iteration results
            iteration_results = {
                'iteration': iteration,
                'data_source': f"{data_dir}/iteration_{iteration}",
                'algorithm_type': 'random_baseline',
                'company_results': company_results,
                'algorithm_results': algorithm_results,
                'new_action_bank_size': len(new_action_bank),
                'users_loaded_from_data': len(loaded_users)
            }
            
            simulation_results['iterations'].append(iteration_results)
            
            print(f"\nRandom Baseline from Data Iteration {iteration} Complete:")
            print(f"  Users loaded from data: {len(loaded_users)}")
            print(f"  Company reward: {company_results['company_metrics']['avg_reward']:.4f}")
            print(f"  Random actions generated: {algorithm_config.get('action_bank_size', 20)}")
            print(f"  New action bank size: {len(new_action_bank)}")
            
            pbar.update(1)
    
    # Generate final summaries
    print(f"\n{'='*60}")
    print("GENERATING FINAL SUMMARIES")
    print(f"{'='*60}")
    
    company_summary = company_sim.get_simulation_summary()
    
    # Simple algorithm summary for random baseline
    algorithm_summary = {
        'algorithm_type': 'random_baseline_from_data',
        'source_data_dir': data_dir,
        'total_iterations_processed': iterations,
        'actions_generated_per_iteration': algorithm_config.get('action_bank_size', 20),
        'total_random_actions_generated': iterations * algorithm_config.get('action_bank_size', 20),
        'baseline_description': 'Randomly generates K actions per iteration using pre-generated user data'
    }
    
    simulation_results['final_summary'] = {
        'simulation_type': 'random_baseline_from_data',
        'source_data_dir': data_dir,
        'company_summary': company_summary,
        'algorithm_summary': algorithm_summary,
        'overall_performance': {
            'total_iterations': iterations,
            'final_avg_reward': company_summary.get('overall_avg_reward', 0),
            'reward_trend': company_summary.get('avg_reward_by_iteration', []),
            'baseline_type': 'random_generation_from_data',
            'final_exploration_rate': company_summary.get('final_exploration_rate', 0),
            'company_learned': company_summary.get('company_learned', False)
        }
    }
    
    # Save complete results
    final_results_file = os.path.join(results_dir, "random_baseline_from_data_results.json")
    with open(final_results_file, 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        json.dump(simulation_results, f, indent=2, default=convert_numpy)
    
    company_sim.save_final_results()
    
    print(f"\n{'='*80}")
    print("RANDOM BASELINE FROM DATA COMPLETE!")
    print(f"{'='*80}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_dir}")
    print(f"Complete results: {final_results_file}")
    
    print(f"\nFinal Performance (Random Baseline from Data):")
    print(f"  Data source: {data_dir}")
    print(f"  Total users processed: {company_summary.get('total_users', 0)}")
    print(f"  Total observations: {company_summary.get('total_observations', 0)}")
    print(f"  Final avg reward: {company_summary.get('overall_avg_reward', 0):.4f}")
    print(f"  Total random actions generated: {iterations * algorithm_config.get('action_bank_size', 20)}")
    
    return simulation_results


def main():
    """Main function with command line interface for random baseline from data."""
    parser = argparse.ArgumentParser(description="Run Random Baseline from Pre-Generated Data")
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to folder created by generate_simulation_data.py')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Root results folder (default: results)')
    
    # Optional data overrides
    parser.add_argument('--iterations', type=int, default=None,
                       help='Iterations to run (default: infer from data_config.json)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                       help='OpenAI API key (optional)')
    
    # Random baseline specific options
    parser.add_argument('--random_actions_per_iteration', type=int, default=20,
                       help='Number of random actions to generate per iteration (default: 20)')
    
    # Company strategy options
    parser.add_argument('--company_strategy', choices=['linucb', 'bootstrapped_dqn', 'legacy'],
                       default='linucb', help='Company contextual bandit strategy (default: linucb)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='LinUCB alpha parameter (default: 1.0)')
    parser.add_argument('--n_heads', type=int, default=10,
                       help='Bootstrapped DQN heads (default: 10)')
    
    args = parser.parse_args()
    
    # Load data configuration
    data_cfg = load_data_config(args.data_dir)
    
    # Infer iterations from data config if not provided
    iterations = args.iterations
    if iterations is None:
        iterations = int(data_cfg.get('n_iterations', 4))
    
    # Determine embedding dimension from saved action bank
    init_action_bank_file = os.path.join(args.data_dir, 'initialization', 'action_bank', 'action_bank.json')
    if not os.path.exists(init_action_bank_file):
        raise FileNotFoundError(f"Initial action bank not found: {init_action_bank_file}")
    with open(init_action_bank_file, 'r') as f:
        ab = json.load(f)
    action_dim = int(ab.get('embedding_dimension', 1536))
    
    # Configure random baseline algorithm
    algorithm_config = {
        'action_bank_size': args.random_actions_per_iteration,
        'action_pool_size': args.random_actions_per_iteration,  # Generate only what we need
        'random_seed': 42,
        'action_dim': action_dim,
        'user_dim': 8
    }
    
    # Configure company strategy
    strategy_config = {}
    if args.company_strategy == 'linucb':
        strategy_config = {'alpha': args.alpha, 'use_pca': True, 'pca_components': 128}
    elif args.company_strategy == 'bootstrapped_dqn':
        strategy_config = {'n_heads': args.n_heads}
    
    # Configure ground truth model from data config
    ground_truth_type = (data_cfg.get('ground_truth') or {}).get('type', 'mixture_of_experts')
    ground_truth_config = (data_cfg.get('ground_truth') or {}).get('config', {})
    ground_truth_config['action_dim'] = action_dim
    ground_truth_config['user_dim'] = 8
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Set it in the environment or pass --openai_api_key.")
    
    # Create unique timestamped results directory
    timestamp = datetime.now().strftime("from_data_%Y%m%d_%H%M%S")
    unique_results_dir = os.path.join(args.results_dir, f"random_baseline_{timestamp}")
    
    print(f"📁 Random Baseline from Data results will be saved to: {unique_results_dir}")
    
    # Create results directory and save configuration
    os.makedirs(unique_results_dir, exist_ok=True)
    simulation_config = {
        "simulation_type": "random_baseline_from_data",
        "timestamp": timestamp,
        "source_data_dir": os.path.abspath(args.data_dir),
        "iterations": iterations,
        "random_actions_per_iteration": args.random_actions_per_iteration,
        "company_strategy": args.company_strategy,
        "strategy_config": strategy_config,
        "ground_truth_type": ground_truth_type,
        "ground_truth_config": ground_truth_config,
        "algorithm_config": algorithm_config,
        "data_config": data_cfg,
        "openai_api_key_provided": openai_api_key is not None
    }
    
    config_file = os.path.join(unique_results_dir, "random_baseline_from_data_config.json")
    with open(config_file, 'w') as f:
        json.dump(simulation_config, f, indent=2, default=str)
    
    print(f"💾 Random Baseline from Data configuration saved to: {config_file}")
    
    # Import time for timing
    import time
    
    # Run random baseline simulation from data
    results = run_random_baseline_from_data(
        data_dir=args.data_dir,
        iterations=iterations,
        results_dir=unique_results_dir,
        openai_api_key=openai_api_key,
        algorithm_config=algorithm_config,
        company_strategy=args.company_strategy,
        strategy_config=strategy_config,
        ground_truth_type=ground_truth_type,
        ground_truth_config=ground_truth_config
    )
    
    return results


if __name__ == "__main__":
    results = main()
