#!/usr/bin/env python3
"""
Run the full simulation and algorithm from a pre-generated dataset.

Input folder layout (created by generate_simulation_data.py):
  <data_dir>/
    data_config.json
    initialization/action_bank/action_bank.json
    iteration_1/users/users.json
    ...

This script:
  - Loads the initial action bank from <data_dir>
  - For each iteration i, loads users from <data_dir>/iteration_i/users/users.json
  - Runs the company simulation (action selection, observations) and the algorithm
  - Saves all results under a new results directory (by default results/from_data_YYYYMMDD_HHMMSS)
"""

import os
import json
import argparse
import shutil
from datetime import datetime
from typing import Dict, Any

import numpy as np

from src.simulation.workflow.company_simulator import CompanySimulator
from src.util.action_embedder import EmbeddedAction, OpenAIActionEmbedder
from src.algorithm.workflow.optimization_algorithm import PersonalizedMarketingAlgorithm


def convert_to_embedded_action(action_data: Dict[str, Any]) -> EmbeddedAction:
    return EmbeddedAction(
        action_id=action_data['action_id'],
        text=action_data['text'],
        embedding=np.array(action_data['embedding']),
        category=action_data.get('category', 'generated'),
        metadata=action_data.get('metadata', {})
    )


def load_data_config(data_dir: str) -> Dict[str, Any]:
    cfg_file = os.path.join(data_dir, 'data_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file, 'r') as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run simulation+algorithm from a pre-generated data folder")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to a folder created by generate_simulation_data.py')
    parser.add_argument('--results_dir', type=str, default='results', help='Root results folder (default: results)')
    parser.add_argument('--iterations', type=int, default=None, help='Iterations to run (default: infer from data_config.json or folders)')
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key (optional)')

    # Algorithm configuration
    parser.add_argument('--diversity_weight', type=float, default=0.15, help='Diversity penalty weight (default: 0.15)')
    parser.add_argument('--action_pool_size', type=int, default=2000, help='Action pool size for generation (default: 2000)')
    parser.add_argument('--action_bank_size', type=int, default=20, help='New action bank size per iteration (default: 20)')
    # Action generation mix ratios
    parser.add_argument('--exploit_ratio', type=float, default=0.4, help='Fraction of exploit actions in pool (default: 0.4)')
    parser.add_argument('--explore_ratio', type=float, default=0.3, help='Fraction of explore actions in pool (default: 0.3)')
    parser.add_argument('--targeted_ratio', type=float, default=0.3, help='Fraction of targeted actions in pool (default: 0.3)')
    parser.add_argument('--reward_model_type', choices=['neural', 'lightgbm', 'gaussian_process', 'bayesian_neural', 'ft_transformer'], default='lightgbm', help='Reward model type')
    parser.add_argument('--bnn_mc_samples', type=int, default=30, help='MC Dropout samples for bayesian_neural (default: 30)')
    parser.add_argument('--task_type', choices=['binary', 'regression'], default='binary', help='User preference task type (default: binary)')
    parser.add_argument('--use_pca', action='store_true', default=False, help='Apply PCA to action embeddings before models')
    parser.add_argument('--pca_components', type=int, default=50, help='Number of PCA components for action embeddings (default: 50)')

    parser.add_argument('--use_segment', action='store_true', help='Operate on segment-level aggregates if available')
    parser.add_argument('--number_of_segments', type=int, default=None, help='Override number of segments when aggregating users')
    parser.add_argument('--users_per_segment_per_iteration', type=float, default=None,
                        help='Portion (0-1) or count of users targeted per segment each iteration')
    parser.add_argument('--use_static_user_base', action='store_true',
                        help='Reuse the same user cohort across iterations')

    # Company strategy
    parser.add_argument('--company_strategy', choices=['linucb', 'bootstrapped_dqn', 'legacy'], default='linucb', help='Company contextual bandit strategy (default: linucb)')
    parser.add_argument('--alpha', type=float, default=1.0, help='LinUCB alpha parameter (default: 1.0)')
    parser.add_argument('--n_heads', type=int, default=10, help='Bootstrapped DQN heads (default: 10)')

    args = parser.parse_args()

    data_cfg = load_data_config(args.data_dir)
    segment_options = data_cfg.get('segment_options', {})
    dataset_number_of_segments = segment_options.get('number_of_segments')
    dataset_users_per_segment = segment_options.get('users_per_segment_per_iteration')
    dataset_use_static = segment_options.get('use_static_user_base', False)

    # Infer iterations
    iterations = args.iterations
    if iterations is None:
        iterations = int(data_cfg.get('n_iterations', 4))

    use_segment = args.use_segment
    number_of_segments = args.number_of_segments if args.number_of_segments is not None else dataset_number_of_segments
    users_per_segment = args.users_per_segment_per_iteration if args.users_per_segment_per_iteration is not None else dataset_users_per_segment
    use_static_user_base = args.use_static_user_base or (use_segment and dataset_use_static)
    if use_segment:
        if number_of_segments is None:
            number_of_segments = 1024
    else:
        users_per_segment = None
        use_static_user_base = False

    # Determine embedding dimension from saved action bank
    init_action_bank_file = os.path.join(args.data_dir, 'initialization', 'action_bank', 'action_bank.json')
    if not os.path.exists(init_action_bank_file):
        raise FileNotFoundError(f"Initial action bank not found: {init_action_bank_file}")
    with open(init_action_bank_file, 'r') as f:
        ab = json.load(f)
    action_dim = int(ab.get('embedding_dimension', 1536))

    # Configure algorithm and ground truth
    algorithm_config: Dict[str, Any] = {
        'diversity_weight': args.diversity_weight,
        'action_pool_size': args.action_pool_size,
        'action_bank_size': args.action_bank_size,
        'reward_model_type': args.reward_model_type,
        'bnn_mc_samples': getattr(args, 'bnn_mc_samples', 30),
        'action_dim': action_dim,
        'user_dim': 30,
        'use_segment_data': use_segment,
        'segment_feature_dim': 30,
        'task_type': args.task_type
    }
    algorithm_config['action_strategy_mix'] = {
        'exploit': args.exploit_ratio,
        'explore': args.explore_ratio,
        'targeted': args.targeted_ratio
    }
    if args.use_pca:
        algorithm_config['pca_config'] = {'use_pca': True, 'pca_components': args.pca_components}

    # Detect user feature dimensionality from first iteration users, if present
    detected_user_dim = None
    first_users_file = os.path.join(args.data_dir, 'iteration_1', 'users', 'users.json')
    try:
        if os.path.exists(first_users_file):
            with open(first_users_file, 'r') as f:
                users_obj = json.load(f)
            if users_obj.get('users'):
                fv = users_obj['users'][0].get('feature_vector')
                if isinstance(fv, list):
                    detected_user_dim = int(len(fv))
    except Exception:
        detected_user_dim = None

    if detected_user_dim:
        algorithm_config['user_dim'] = detected_user_dim

    ground_truth_type = (data_cfg.get('ground_truth') or {}).get('type', 'mixture_of_experts')
    ground_truth_config = (data_cfg.get('ground_truth') or {}).get('config', {})
    ground_truth_config['action_dim'] = action_dim
    ground_truth_config['user_dim'] = detected_user_dim or algorithm_config['user_dim']

    # Build a fresh results directory for this run
    timestamp = datetime.now().strftime("from_data_%Y%m%d_%H%M%S")
    run_results_dir = os.path.join(args.results_dir, f"simulation_{timestamp}")
    os.makedirs(run_results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {run_results_dir}")

    # Save run configuration
    run_config = {
        'source_data_dir': os.path.abspath(args.data_dir),
        'iterations': iterations,
        'company_strategy': args.company_strategy,
        'algorithm_config': algorithm_config,
        'ground_truth_type': ground_truth_type,
        'ground_truth_config': ground_truth_config,
        'openai_api_key_provided': bool(args.openai_api_key),
        'use_segment_data': use_segment,
        'number_of_segments': number_of_segments,
        'users_per_segment_per_iteration': users_per_segment,
        'use_static_user_base': use_static_user_base,
        'task_type': args.task_type
    }
    with open(os.path.join(run_results_dir, 'run_from_data_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    # Initialize simulator and algorithm
    company_sim = CompanySimulator(
        results_dir=run_results_dir,
        openai_api_key=args.openai_api_key,
        strategy_type=args.company_strategy,
        strategy_config={'alpha': args.alpha, 'use_pca': True, 'pca_components': 128} if args.company_strategy == 'linucb' else {'n_heads': args.n_heads},
        ground_truth_type=ground_truth_type,
        ground_truth_config=ground_truth_config,
        random_seed=42,
        use_segment_data=use_segment,
        number_of_segments=number_of_segments,
        users_per_segment_per_iteration=users_per_segment,
        use_static_user_base=use_static_user_base
    )

    algorithm = PersonalizedMarketingAlgorithm(
        results_dir=run_results_dir,
        algorithm_config=algorithm_config
    )

    # Load initial action bank and set current bank
    embedder = OpenAIActionEmbedder(api_key=args.openai_api_key)
    initial_bank = embedder.load_embedded_actions(init_action_bank_file)
    company_sim.current_action_bank = initial_bank

    # Create initialization folder in results directory and copy action bank
    # This is needed for the algorithm's _load_current_action_bank() method
    init_results_dir = os.path.join(run_results_dir, "initialization")
    os.makedirs(init_results_dir, exist_ok=True)
    init_action_bank_results_dir = os.path.join(init_results_dir, "action_bank")
    os.makedirs(init_action_bank_results_dir, exist_ok=True)
    
    # Copy action bank to the results initialization folder
    dest_action_bank_file = os.path.join(init_action_bank_results_dir, "action_bank.json")
    shutil.copy2(init_action_bank_file, dest_action_bank_file)
    print(f"📋 Copied initial action bank to results: {dest_action_bank_file}")
    
    # Create initialization summary to match CompanySimulator.initialize_simulation()
    init_summary = {
        'timestamp': datetime.now().isoformat(),
        'n_users_per_iteration': int(data_cfg.get('n_users_per_iteration', 1000)),
        'n_initial_actions': len(initial_bank),
        'initial_action_bank_size': len(initial_bank),
        'embedding_model': 'loaded_from_data',
        'source_data_dir': os.path.abspath(args.data_dir),
        'loaded_from_data': True
    }
    
    init_summary_file = os.path.join(init_results_dir, "initialization_summary.json")
    with open(init_summary_file, 'w') as f:
        json.dump(init_summary, f, indent=2)

    # Ensure simulator knows the user count per iteration (for prints/stats)
    company_sim.n_users = int(data_cfg.get('n_users_per_iteration', 1000))

    # Run iterations: for each, load users from data_dir, then run the standard company+algorithm flow
    for iteration in range(1, iterations + 1):
        print(f"\n{'='*50}\nRUN FROM DATA - ITERATION {iteration}/{iterations}\n{'='*50}")

        # Load users from the data folder
        users_file = os.path.join(args.data_dir, f"iteration_{iteration}", 'users', 'users.json')
        if not os.path.exists(users_file):
            raise FileNotFoundError(f"Users file not found: {users_file}")

        loaded_users = company_sim.user_generator.load_users(users_file)

        # Monkeypatch generate_users to return loaded users, so CompanySimulator.run_iteration
        # will use these and still save to the new results dir
        def _return_loaded_users(_n):
            return loaded_users
        company_sim.user_generator.generate_users = _return_loaded_users  # type: ignore

        # Company phase: will save observations and action bank to run_results_dir/iteration_i
        company_results = company_sim.run_iteration(iteration)

        # Algorithm phase
        algorithm_results = algorithm.process_iteration(iteration)

        # Integration: update company's action bank with algorithm's output
        new_action_bank_file = algorithm_results['files_created']['new_action_bank']
        with open(new_action_bank_file, 'r') as f:
            new_bank_data = json.load(f)
        new_action_bank = [convert_to_embedded_action(a) for a in new_bank_data['actions']]

        if len(new_action_bank) == 0:
            print("No new actions returned - keeping current action bank unchanged")
        else:
            company_sim.update_action_bank_preserve_ids(new_action_bank, iteration)

    # Final summaries
    company_summary = company_sim.get_simulation_summary()
    algorithm_summary = algorithm.get_algorithm_summary()
    final = {
        'company_summary': company_summary,
        'algorithm_summary': algorithm_summary,
        'source_data_dir': os.path.abspath(args.data_dir)
    }
    with open(os.path.join(run_results_dir, 'complete_simulation_results.json'), 'w') as f:
        json.dump(final, f, indent=2)

    print("\nRun complete.")
    print(f"Results saved under: {run_results_dir}")


if __name__ == '__main__':
    main()
