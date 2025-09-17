#!/usr/bin/env python3
"""
Generate reproducible simulation data only.

This script creates:
- An initial action bank (embedded + saved)
- Users for a fixed number of iterations (default: 4)

Outputs are saved under: ./data/simulation_data/<unique_id>

Use this to freeze a dataset so you can compare different algorithms on the same data
by running the separate runner that reads from this folder.
"""

import os
import json
import argparse
import shutil
from datetime import datetime
from typing import Dict, Any

from src.simulation.workflow.company_simulator import CompanySimulator


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate simulation data only (initial action bank + users)")
    parser.add_argument('--users', type=int, default=1000, help='Users per iteration (default: 1000)')
    parser.add_argument('--iterations', type=int, default=4, help='Number of iterations to generate users for (default: 4)')
    parser.add_argument('--initial_actions', type=int, default=30, help='Initial action bank size (default: 30)')
    parser.add_argument('--output_root', type=str, default=os.path.join('data', 'simulation_data'), help='Root folder for generated data')
    parser.add_argument('--id', type=str, default=None, help='Optional unique id for this dataset (default: timestamp)')
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key (optional)')
    parser.add_argument('--number_of_segments', type=int, default=None, help='Number of segments to record (default: 1024)')
    parser.add_argument('--users_per_segment_per_iteration', type=float, default=None,
                        help='Portion or count of users targeted per segment when running simulations from data')
    parser.add_argument('--use_static_user_base', action='store_true',
                        help='Reuse the same user cohort across iterations in downstream simulations')

    # Ground truth model options
    parser.add_argument('--ground_truth_type', choices=['mixture_of_experts', 'gmm'], default='mixture_of_experts', help='Ground truth model type')
    parser.add_argument('--gmm_components', type=int, default=5, help='GMM components (if using gmm)')

    args = parser.parse_args()

    # Create target folder
    unique_id = args.id or datetime.now().strftime("data_%Y%m%d_%H%M%S")
    data_dir = os.path.join(args.output_root, unique_id)
    ensure_dir(data_dir)

    # Prepare ground truth config
    ground_truth_config: Dict[str, Any] = {}
    if args.ground_truth_type == 'gmm':
        ground_truth_config = {
            'n_components': args.gmm_components
        }
    ground_truth_config['user_dim'] = 30

    # Instantiate CompanySimulator to leverage initialization (embeddings, action bank saving)
    company_sim = CompanySimulator(
        results_dir=data_dir,
        openai_api_key=args.openai_api_key,
        strategy_type='linucb',
        strategy_config={'alpha': 1.0, 'use_pca': True, 'pca_components': 128},
        ground_truth_type=args.ground_truth_type,
        ground_truth_config=ground_truth_config,
        random_seed=42,
        batch_update_size=1,
        use_chatgpt_actions=True,
        performance_tracking_interval=5000,
        use_segment_data=True,
        number_of_segments=args.number_of_segments,
        users_per_segment_per_iteration=args.users_per_segment_per_iteration,
        use_static_user_base=args.use_static_user_base
    )

    # Initialize to create and save the initial action bank
    print("=== Generating initial action bank ===")
    company_sim.initialize_simulation(n_users=args.users, n_initial_actions=args.initial_actions)

    # Prepare base users if static population requested
    base_users_file = None
    base_segment_file = None
    if args.use_static_user_base:
        base_users_file = os.path.join(data_dir, 'initialization', 'users', 'users.json')
        base_segment_file = os.path.join(data_dir, 'initialization', 'users', 'segment_summary.json')
        if not os.path.exists(base_users_file):
            raise FileNotFoundError("Static user base requested but initialization/users/users.json is missing")

    # Generate users for each iteration and save only user data (no simulation, no observations)
    print(f"\n=== Preparing users for {args.iterations} iterations ===")
    for iteration in range(1, args.iterations + 1):
        iteration_dir = os.path.join(data_dir, f"iteration_{iteration}")
        users_dir = os.path.join(iteration_dir, "users")
        ensure_dir(users_dir)

        if args.use_static_user_base:
            print(f"- Iteration {iteration}: reusing static user cohort")
            users = company_sim.user_generator.load_users(base_users_file)
        else:
            print(f"- Iteration {iteration}: generating {args.users} users")
            users = company_sim.user_generator.generate_users(args.users)

        users_file = os.path.join(users_dir, "users.json")
        company_sim.user_generator.save_users(users, users_file)

        segment_summary = company_sim.user_generator.get_segment_summary(users)
        with open(os.path.join(users_dir, "user_segments.json"), 'w') as f:
            json.dump(segment_summary, f, indent=2)

    # Persist data config for reproducibility when running from data
    print("\n=== Saving data configuration ===")
    # Read embedding info from the saved action bank
    action_bank_meta = {}
    init_action_bank_file = os.path.join(data_dir, 'initialization', 'action_bank', 'action_bank.json')
    try:
        with open(init_action_bank_file, 'r') as f:
            ab_data = json.load(f)
            action_bank_meta = {
                'embedding_model': ab_data.get('embedding_model', 'unknown'),
                'embedding_dimension': ab_data.get('embedding_dimension', 0),
                'total_actions': ab_data.get('total_actions', 0)
            }
    except Exception as e:
        print(f"Warning: Could not read action bank meta: {e}")

    data_config = {
        'schema': 1,
        'generated_at': datetime.now().isoformat(),
        'n_iterations': int(args.iterations),
        'n_users_per_iteration': int(args.users),
        'n_initial_actions': int(args.initial_actions),
        'ground_truth': {
            'type': args.ground_truth_type,
            'config': ground_truth_config,
            'random_seed': 42
        },
        'action_embeddings': action_bank_meta,
        'segment_options': {
            'number_of_segments': args.number_of_segments or 1024,
            'users_per_segment_per_iteration': args.users_per_segment_per_iteration,
            'use_static_user_base': args.use_static_user_base
        },
        'notes': 'This folder contains only initial action bank and user cohorts per iteration.'
    }

    with open(os.path.join(data_dir, 'data_config.json'), 'w') as f:
        json.dump(data_config, f, indent=2)

    print(f"\nDone. Data saved under: {data_dir}")


if __name__ == '__main__':
    main()
