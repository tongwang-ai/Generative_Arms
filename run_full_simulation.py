#!/usr/bin/env python3
"""
Main script for running the complete simulation:
1. Company simulation (generates users, runs contextual bandit strategy)
2. Algorithm processing (our optimization algorithm)

This demonstrates the full workflow where:
- Company generates observation data each iteration
- Our algorithm processes the data and provides new action banks
- Company updates strategy and continues
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm

# Add paths to find modules
sys.path.append(os.path.dirname(__file__))

from simulation.company_simulator import CompanySimulator
from simulation.action_embedder import EmbeddedAction
from algorithm.optimization_algorithm import PersonalizedMarketingAlgorithm


def convert_to_embedded_action(action_data: Dict[str, Any]) -> EmbeddedAction:
    """Convert action dictionary to EmbeddedAction object."""
    import numpy as np
    
    return EmbeddedAction(
        action_id=action_data['action_id'],
        text=action_data['text'],
        embedding=np.array(action_data['embedding']),
        category=action_data.get('category', 'generated'),
        metadata=action_data.get('metadata', {})
    )


def run_full_simulation(n_iterations: int = 5,
                       n_users: int = 1000,
                       n_initial_actions: int = 30,
                       results_dir: str = "results",
                       openai_api_key: str = None,
                       algorithm_config: Dict[str, Any] = None,
                       company_strategy: str = "linucb",
                       strategy_config: Dict[str, Any] = None,
                       ground_truth_type: str = "mixture_of_experts",
                       ground_truth_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run the complete simulation with both company and algorithm components.
    
    Args:
        n_iterations: Number of iterations to run
        n_users: Number of users to generate
        n_initial_actions: Initial action bank size
        results_dir: Directory to store results
        openai_api_key: OpenAI API key for embeddings
        algorithm_config: Configuration for the algorithm
        company_strategy: 'linucb', 'bootstrapped_dqn', or 'legacy'
        strategy_config: Configuration for company strategy
        ground_truth_type: 'mixture_of_experts' or 'gmm'
        ground_truth_config: Configuration for ground truth model
        
    Returns:
        Complete simulation results
    """
    start_time = time.time()
    print("="*80)
    print("FULL SIMULATION: Company + Algorithm")
    print("="*80)
    print(f"🕐 Simulation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize both components
    print(f"\n1. Initializing Components...")
    print(f"   Company strategy: {company_strategy}")
    print(f"   Ground truth type: {ground_truth_type}")
    
    init_start = time.time()
    
    # Determine action embedding dimension based on OpenAI model
    # Check for API key to determine if we'll use OpenAI embeddings
    import os
    if openai_api_key or os.getenv('OPENAI_API_KEY'):
        # Default to text-embedding-3-large (most recent model)
        action_dim = 3072
    else:
        # Fallback embeddings
        action_dim = 1536
    
    # Add embedding dimension info to configs
    if algorithm_config is None:
        algorithm_config = {}
    if ground_truth_config is None:
        ground_truth_config = {}
    
    algorithm_config['action_dim'] = action_dim
    algorithm_config['user_dim'] = 8  # Fixed user feature dimensions
    ground_truth_config['action_dim'] = action_dim
    ground_truth_config['user_dim'] = 8
    
    company_sim = CompanySimulator(
        results_dir=results_dir,
        openai_api_key=openai_api_key,
        strategy_type=company_strategy,
        strategy_config=strategy_config,
        ground_truth_type=ground_truth_type,
        ground_truth_config=ground_truth_config,
        random_seed=42
    )
    
    algorithm = PersonalizedMarketingAlgorithm(
        results_dir=results_dir,
        algorithm_config=algorithm_config
    )
    init_time = time.time() - init_start
    print(f"   ⏱️  Component initialization completed in {init_time:.2f}s")
    
    # Initialize simulation
    print("\n2. Company: Initializing simulation...")
    sim_init_start = time.time()
    init_results = company_sim.initialize_simulation(
        n_users=n_users,
        n_initial_actions=n_initial_actions
    )
    sim_init_time = time.time() - sim_init_start
    print(f"   ⏱️  Simulation initialization completed in {sim_init_time:.2f}s")
    
    # Run iterations
    simulation_results = {
        'initialization': init_results,
        'iterations': [],
        'final_summary': {}
    }
    
    print(f"\n3. Running {n_iterations} iterations...")
    iterations_start = time.time()
    
    # Use tqdm for overall progress
    with tqdm(total=n_iterations, desc="🔄 Overall Progress", unit="iteration") as pbar:
        for iteration in range(1, n_iterations + 1):
            iteration_start = time.time()
            
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration}/{n_iterations}")
            print(f"{'='*50}")
            print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
            
            # Company runs iteration (generates observation data)
            print(f"\n--- Company Phase ---")
            company_phase_start = time.time()
            company_results = company_sim.run_iteration(iteration)
            company_phase_time = time.time() - company_phase_start
            print(f"   ⏱️  Company phase completed in {company_phase_time:.2f}s")
            
            # Algorithm processes the data and generates new action bank
            print(f"\n--- Algorithm Phase ---")
            algorithm_phase_start = time.time()
            algorithm_results = algorithm.process_iteration(iteration)
            algorithm_phase_time = time.time() - algorithm_phase_start
            print(f"   ⏱️  Algorithm phase completed in {algorithm_phase_time:.2f}s")
            
            # Load the new action bank generated by algorithm
            integration_start = time.time()
            new_action_bank_file = algorithm_results['files_created']['new_action_bank']
            with open(new_action_bank_file, 'r') as f:
                new_bank_data = json.load(f)
            
            # Convert to EmbeddedAction objects for company
            new_action_bank = [convert_to_embedded_action(action_data) 
                              for action_data in new_bank_data['actions']]
            
            # Update company's action bank with algorithm's output (preserving LinUCB knowledge)
            print(f"\n--- Integration Phase ---")
            print(f"Algorithm returned {len(new_action_bank)} actions")
            
            if len(new_action_bank) == 0:
                print("   No new actions returned - keeping current action bank unchanged")
                print(f"   Current action bank remains: {len(company_sim.current_action_bank)} actions")
            else:
                company_sim.update_action_bank_preserve_ids(new_action_bank, iteration)
            
            integration_time = time.time() - integration_start
            print(f"   ⏱️  Integration phase completed in {integration_time:.2f}s")
            
            # Store iteration results
            iteration_results = {
                'iteration': iteration,
                'company_results': company_results,
                'algorithm_results': algorithm_results,
                'new_action_bank_size': len(new_action_bank)
            }
            
            simulation_results['iterations'].append(iteration_results)
            
            iteration_time = time.time() - iteration_start
            elapsed_total = time.time() - start_time
            avg_iteration_time = (time.time() - iterations_start) / iteration
            remaining_iterations = n_iterations - iteration
            eta_seconds = remaining_iterations * avg_iteration_time
            eta_formatted = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 60 else f"{int(eta_seconds)}s"
            
            print(f"\nIteration {iteration} Complete:")
            print(f"  Company reward: {company_results['company_metrics']['avg_reward']:.4f}")
            print(f"  Algorithm expected value: {algorithm_results['evaluation_results'].get('total_value', 0):.4f}")
            print(f"  New action bank size: {len(new_action_bank)}")
            print(f"  ⏱️  Iteration time: {iteration_time:.2f}s | Total elapsed: {elapsed_total/60:.1f}m | ETA: {eta_formatted}")
            
            # Update progress bar
            pbar.set_postfix({
                'Iter_Time': f'{iteration_time:.1f}s',
                'Avg_Time': f'{avg_iteration_time:.1f}s', 
                'ETA': eta_formatted
            })
            pbar.update(1)
    
    # Generate final summaries
    print(f"\n{'='*50}")
    print("GENERATING FINAL SUMMARIES")
    print(f"{'='*50}")
    
    summary_start = time.time()
    company_summary = company_sim.get_simulation_summary()
    algorithm_summary = algorithm.get_algorithm_summary()
    summary_time = time.time() - summary_start
    print(f"   ⏱️  Summary generation completed in {summary_time:.2f}s")
    
    simulation_results['final_summary'] = {
        'company_summary': company_summary,
        'algorithm_summary': algorithm_summary,
        'overall_performance': {
            'total_iterations': n_iterations,
            'final_avg_reward': company_summary.get('overall_avg_reward', 0),
            'reward_trend': company_summary.get('avg_reward_by_iteration', []),
            'algorithm_value_trend': algorithm_summary.get('performance_trend', []),
            'final_exploration_rate': company_summary.get('final_exploration_rate', 0),
            'company_learned': company_summary.get('company_learned', False)
        }
    }
    
    # Save complete results
    final_results_file = os.path.join(results_dir, "complete_simulation_results.json")
    with open(final_results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        json.dump(simulation_results, f, indent=2, default=convert_numpy)
    
    # Save individual component summaries
    company_sim.save_final_results()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE!")
    print(f"{'='*80}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total execution time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Results saved to: {results_dir}")
    print(f"Complete results: {final_results_file}")
    
    print(f"\nFinal Performance:")
    print(f"  Total users: {company_summary.get('total_users', 0)}")
    print(f"  Total observations: {company_summary.get('total_observations', 0)}")
    print(f"  Final avg reward: {company_summary.get('overall_avg_reward', 0):.4f}")
    print(f"  Company strategy learned: {company_summary.get('company_learned', False)}")
    print(f"  Algorithm iterations processed: {algorithm_summary.get('total_iterations_processed', 0)}")
    print(f"⏱️  Average time per iteration: {(time.time() - iterations_start) / n_iterations:.1f}s")
    
    return simulation_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Run complete simulation (Company + Algorithm)')
    
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations to run (default: 5)')
    parser.add_argument('--users', type=int, default=1000,
                       help='Number of users to generate (default: 1000)')
    parser.add_argument('--initial_actions', type=int, default=30,
                       help='Initial action bank size (default: 30)')
    parser.add_argument('--results_dir', default='results',
                       help='Directory to store results (default: results)')
    parser.add_argument('--openai_api_key',
                       help='OpenAI API key for embeddings (or set OPENAI_API_KEY env var)')
    parser.add_argument('--config_file',
                       help='JSON file with algorithm configuration')
    
    # Company strategy options
    parser.add_argument('--company_strategy', choices=['linucb', 'bootstrapped_dqn', 'legacy'],
                       default='linucb', help='Company contextual bandit strategy (default: linucb)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='LinUCB alpha parameter (default: 1.0)')
    parser.add_argument('--n_heads', type=int, default=10,
                       help='Bootstrapped DQN number of heads (default: 10)')
    
    # Ground truth model options
    parser.add_argument('--ground_truth_type', choices=['mixture_of_experts', 'gmm'],
                       default='mixture_of_experts', 
                       help='Ground truth model type (default: mixture_of_experts)')
    parser.add_argument('--gmm_components', type=int, default=5,
                       help='Number of GMM components (default: 5)')
    parser.add_argument('--density_scale_factor', type=float, default=500.0,
                       help='GMM density scale factor (default: 500.0)')
    parser.add_argument('--min_utility', type=float, default=0.1,
                       help='GMM minimum utility (default: 0.1)')
    parser.add_argument('--max_utility', type=float, default=0.9,
                       help='GMM maximum utility (default: 0.9)')
    
    # Algorithm configuration options
    parser.add_argument('--diversity_weight', type=float, default=0.15,
                       help='Diversity penalty weight (default: 0.15)')
    parser.add_argument('--action_pool_size', type=int, default=2000,
                       help='Action pool size for generation (default: 2000)')
    parser.add_argument('--action_bank_size', type=int, default=20,
                       help='New action bank size per iteration (default: 20)')
    
    # Reward model selection
    parser.add_argument('--reward_model_type', choices=['neural', 'lightgbm', 'gaussian_process', 'bayesian_neural'],
                       default='lightgbm', 
                       help='Type of reward model (default: lightgbm)')
    parser.add_argument('--bnn_mc_samples', type=int, default=30,
                       help='MC Dropout samples for bayesian_neural (default: 30)')
    
    # LightGBM model options
    parser.add_argument('--lgb_n_estimators', type=int, default=100,
                       help='LightGBM number of estimators (default: 100)')
    parser.add_argument('--lgb_learning_rate', type=float, default=0.1,
                       help='LightGBM learning rate (default: 0.1)')
    parser.add_argument('--lgb_num_leaves', type=int, default=31,
                       help='LightGBM number of leaves (default: 31)')
    parser.add_argument('--lgb_feature_fraction', type=float, default=0.8,
                       help='LightGBM feature fraction (default: 0.8)')
    parser.add_argument('--lgb_bagging_fraction', type=float, default=0.8,
                       help='LightGBM bagging fraction (default: 0.8)')
    
    # PCA options for action embeddings
    parser.add_argument('--use_pca', action='store_true', default=False,
                       help='Apply PCA to action embeddings before feeding to models')
    parser.add_argument('--pca_components', type=int, default=50,
                       help='Number of PCA components for action embeddings (default: 50)')
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    algorithm_config = {}
    if args.config_file:
        with open(args.config_file, 'r') as f:
            algorithm_config = json.load(f)
    
    # Override with command line arguments
    algorithm_config.update({
        'diversity_weight': args.diversity_weight,
        'action_pool_size': args.action_pool_size,
        'action_bank_size': args.action_bank_size,
        'reward_model_type': args.reward_model_type,
        'bnn_mc_samples': getattr(args, 'bnn_mc_samples', 30)
    })
    
    # Configure model-specific parameters
    if args.reward_model_type == 'lightgbm':
        # LightGBM configuration with sensible defaults
        lightgbm_config = {
            'n_estimators': getattr(args, 'lgb_n_estimators', 100),
            'learning_rate': getattr(args, 'lgb_learning_rate', 0.1),
            'num_leaves': getattr(args, 'lgb_num_leaves', 31),
            'feature_fraction': getattr(args, 'lgb_feature_fraction', 0.8),
            'bagging_fraction': getattr(args, 'lgb_bagging_fraction', 0.8)
        }
        algorithm_config['lightgbm_config'] = lightgbm_config
    
    # Add PCA configuration to all model types
    if args.use_pca:
        pca_config = {
            'use_pca': args.use_pca,
            'pca_components': args.pca_components
        }
        algorithm_config['pca_config'] = pca_config
    
    # Configure company strategy
    strategy_config = {}
    if args.company_strategy == 'linucb':
        strategy_config = {'alpha': args.alpha, 'use_pca': True, 'pca_components': 128}
    elif args.company_strategy == 'bootstrapped_dqn':
        strategy_config = {'n_heads': args.n_heads}
    
    # Configure ground truth model
    ground_truth_config = {}
    if args.ground_truth_type == 'gmm':
        ground_truth_config = {
            'n_components': args.gmm_components,
            'density_scale_factor': args.density_scale_factor,
            'min_utility': args.min_utility,
            'max_utility': args.max_utility
        }
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Warning: No OpenAI API key provided. Using fallback embeddings.")
        print("Set OPENAI_API_KEY environment variable or use --openai_api_key argument for better embeddings.")
    
    # Create unique timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_results_dir = os.path.join(args.results_dir, f"simulation_{timestamp}")
    
    print(f"📁 Results will be saved to: {unique_results_dir}")
    
    # Save simulation configuration for this run
    os.makedirs(unique_results_dir, exist_ok=True)
    simulation_config = {
        "timestamp": timestamp,
        "iterations": args.iterations,
        "users": args.users,
        "initial_actions": args.initial_actions,
        "company_strategy": args.company_strategy,
        "strategy_config": strategy_config,
        "ground_truth_type": args.ground_truth_type,
        "ground_truth_config": ground_truth_config,
        "reward_model_type": args.reward_model_type,
        "lightgbm_config": algorithm_config.get('lightgbm_config', {}),
        "algorithm_config": algorithm_config,
        "openai_api_key_provided": openai_api_key is not None
    }
    
    config_file = os.path.join(unique_results_dir, "simulation_config.json")
    with open(config_file, 'w') as f:
        json.dump(simulation_config, f, indent=2, default=str)
    
    print(f"💾 Configuration saved to: {config_file}")
    
    # Run simulation
    results = run_full_simulation(
        n_iterations=args.iterations,
        n_users=args.users,
        n_initial_actions=args.initial_actions,
        results_dir=unique_results_dir,
        openai_api_key=openai_api_key,
        algorithm_config=algorithm_config,
        company_strategy=args.company_strategy,
        strategy_config=strategy_config,
        ground_truth_type=args.ground_truth_type,
        ground_truth_config=ground_truth_config
    )
    
    return results


if __name__ == "__main__":
    results = main()
