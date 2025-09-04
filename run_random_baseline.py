#!/usr/bin/env python3
"""
Random Baseline Simulation

This script runs the simulation with a random baseline algorithm instead of our optimization.
The baseline randomly generates and adds K actions to the current action bank, providing
a comparison point for our intelligent action generation approach.
"""

import os
import sys
import argparse
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm

# Add paths to find modules
sys.path.append(os.path.dirname(__file__))

from simulation.company_simulator import CompanySimulator
from simulation.action_embedder import EmbeddedAction
from algorithm.optimization_algorithm import PersonalizedMarketingAlgorithm


class RandomBaselineAlgorithm(PersonalizedMarketingAlgorithm):
    """
    Random baseline that generates K random actions instead of using optimization.
    Inherits from PersonalizedMarketingAlgorithm to reuse infrastructure but overrides
    the core generation logic.
    """
    
    def __init__(self, results_dir: str = "results", algorithm_config: Dict[str, Any] = None):
        # Initialize with minimal configuration since we won't use the reward model
        minimal_config = {
            'action_pool_size': algorithm_config.get('action_bank_size', 20),  # Generate only what we need
            'action_bank_size': algorithm_config.get('action_bank_size', 20),
            'random_seed': algorithm_config.get('random_seed', 42),
            'reward_model_type': 'lightgbm'  # Still need to initialize, but won't use
        }
        super().__init__(results_dir, minimal_config)
        self.is_baseline = True
        
    def _load_current_users(self, iteration: int):
        """Load current users - simplified for baseline."""
        users_dir = os.path.join(self.results_dir, f"iteration_{iteration}", "users")
        users_file = os.path.join(users_dir, "users.json")
        
        if not os.path.exists(users_file):
            raise FileNotFoundError(f"Users file not found: {users_file}")
            
        with open(users_file, 'r') as f:
            users_data = json.load(f)
            
        from simulation.user_generator import MeaningfulUser
        import numpy as np
        
        users = []
        for user_data in users_data['users']:
            # Extract from nested structure
            demographics = user_data['demographics']
            behavioral = user_data['behavioral_traits']
            
            user = MeaningfulUser(
                user_id=user_data['user_id'],
                age=demographics['age'],
                income_level=demographics['income_level'],
                tech_savviness=behavioral['tech_savviness'],
                price_sensitivity=behavioral['price_sensitivity'],
                brand_loyalty=behavioral['brand_loyalty'],
                social_influence=behavioral['social_influence'],
                urgency_response=behavioral['urgency_response'],
                quality_focus=behavioral['quality_focus'],
                feature_vector=np.array(user_data['feature_vector']),
                segment=user_data['segment']
            )
            users.append(user)
            
        return users
        
    def _load_current_action_bank(self, iteration: int):
        """Load current action bank - simplified for baseline."""
        # Check iteration-specific action bank first
        action_bank_dir = os.path.join(self.results_dir, f"iteration_{iteration}", "action_bank")
        action_bank_file = os.path.join(action_bank_dir, "action_bank.json")
        
        # If not found, try initialization
        if not os.path.exists(action_bank_file):
            action_bank_dir = os.path.join(self.results_dir, "initialization", "action_bank")
            action_bank_file = os.path.join(action_bank_dir, "action_bank.json")
            
        if not os.path.exists(action_bank_file):
            raise FileNotFoundError(f"Action bank file not found: {action_bank_file}")
            
        with open(action_bank_file, 'r') as f:
            action_bank_data = json.load(f)
            
        from simulation.action_embedder import EmbeddedAction
        import numpy as np
        
        actions = []
        for action_data in action_bank_data['actions']:
            action = EmbeddedAction(
                action_id=action_data['action_id'],
                text=action_data['text'],
                embedding=np.array(action_data['embedding']),
                category=action_data.get('category', 'unknown'),
                metadata=action_data.get('metadata', {})
            )
            actions.append(action)
            
        return [], actions  # Return empty previous_best, current_actions
        
    def _embedded_action_to_dict(self, action):
        """Convert EmbeddedAction to dictionary for JSON serialization."""
        return {
            'action_id': action.action_id,
            'text': action.text,
            'embedding': action.embedding.tolist() if hasattr(action.embedding, 'tolist') else action.embedding,
            'category': action.category,
            'metadata': action.metadata
        }
        
    def _generate_new_action_bank(self, users: List, previous_best: List[EmbeddedAction], 
                                current_action_bank: List[EmbeddedAction]) -> List[EmbeddedAction]:
        """
        Generate K random actions instead of using optimization.
        This is the key method that makes this a random baseline.
        """
        print(f"🎲 Random Baseline: Generating {self.config['action_bank_size']} random actions...")
        
        # Generate random actions using the action generator  
        raw_actions = self.action_generator.generate_action_pool(
            pool_size=self.config['action_bank_size'],
            previous_best=None,
            user_segments=None,
            embedding_dim=8  # Will be overridden by actual dimensions
        )
        
        print(f"   ✅ Generated {len(raw_actions)} raw actions, embedding...")
        
        # Convert to embedded actions using the embedder
        from simulation.action_embedder import OpenAIActionEmbedder
        import os
        
        embedder = OpenAIActionEmbedder(api_key=os.getenv('OPENAI_API_KEY'))
        embedded_actions = []
        
        for i, action in enumerate(raw_actions):
            # Some generated Action objects may not carry category/metadata
            category = getattr(action, 'category', 'generated')
            metadata = getattr(action, 'metadata', {})
            embedded_action = embedder.embed_single_action(
                action_id=f"random_{len(current_action_bank) + i:04d}",
                text=action.text,
                category=category,
                metadata=metadata
            )
            embedded_actions.append(embedded_action)
        
        print(f"   ✅ Embedded {len(embedded_actions)} random actions")
        
        # Return the combined action bank: current + new random actions
        combined_bank = current_action_bank + embedded_actions
        
        print(f"   📊 Total action bank size: {len(combined_bank)} actions")
        return combined_bank
    
    def process_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Process iteration with random baseline - skips model training and uses random generation.
        """
        process_start = time.time()
        
        print(f"🎲 Random Baseline Processing Iteration {iteration}")
        print("=" * 60)
        
        # Create iteration directory
        iteration_dir = os.path.join(self.results_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Load observation data (but don't train model)
        observations_file = os.path.join(iteration_dir, "observations", "observations.csv")
        
        if not os.path.exists(observations_file):
            raise FileNotFoundError(f"Observations file not found: {observations_file}")
            
        print(f"📊 Loading observation data from {observations_file}")
        observations_df = pd.read_csv(observations_file)
        print(f"   ⏱️  Loaded {len(observations_df)} observations")
        
        # Load users and current action bank (needed for generation)
        users = self._load_current_users(iteration)
        previous_best, current_action_bank = self._load_current_action_bank(iteration)
        
        print(f"📥 Loaded {len(users)} users and {len(current_action_bank)} current actions")
        
        # Skip model training - go directly to random action generation
        print("⏭️  Skipping model training (random baseline)")
        
        # Generate random action bank
        generation_start = time.time()
        new_action_bank = self._generate_new_action_bank(users, previous_best, current_action_bank)
        generation_time = time.time() - generation_start
        
        print(f"⏱️  Random generation completed in {generation_time:.2f}s")
        
        # Save the new action bank
        new_action_bank_dir = os.path.join(iteration_dir, "new_action_bank")
        os.makedirs(new_action_bank_dir, exist_ok=True)
        
        new_action_bank_file = os.path.join(new_action_bank_dir, "new_action_bank.json")
        action_bank_data = {
            'actions': [self._embedded_action_to_dict(action) for action in new_action_bank],
            'metadata': {
                'generation_method': 'random_baseline',
                'timestamp': datetime.now().isoformat(),
                'total_actions': len(new_action_bank),
                'generation_time_seconds': generation_time
            }
        }
        
        with open(new_action_bank_file, 'w') as f:
            json.dump(action_bank_data, f, indent=2, default=str)
        
        # Perform basic evaluation (without ground truth comparison)
        print("📋 Performing basic evaluation...")
        evaluation_results = {
            'total_actions': len(new_action_bank),
            'new_actions_added': self.config['action_bank_size'],
            'baseline_method': 'random_generation',
            'generation_time': generation_time,
            'evaluation_time': 0.0  # No complex evaluation for random baseline
        }
        
        total_time = time.time() - process_start
        
        # Create results summary
        results = {
            'iteration': iteration,
            'algorithm_type': 'random_baseline',
            'config': self.config,
            'processing_time': total_time,
            'data_loading': {
                'observations_count': len(observations_df),
                'users_count': len(users),
                'current_actions_count': len(current_action_bank)
            },
            'model_training': {
                'skipped': True,
                'reason': 'random_baseline'
            },
            'action_generation': {
                'method': 'random',
                'generation_time': generation_time,
                'actions_generated': len(new_action_bank)
            },
            'evaluation_results': evaluation_results,
            'files_created': {
                'new_action_bank': new_action_bank_file
            },
            'summary': {
                'total_time': total_time,
                'new_action_bank_size': len(new_action_bank),
                'baseline_type': 'random_generation'
            }
        }
        
        # Save iteration results
        results_file = os.path.join(iteration_dir, "random_baseline_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Random Baseline Iteration {iteration} Complete:")
        print(f"   🎲 Generated: {self.config['action_bank_size']} random actions")
        print(f"   📊 Total action bank: {len(new_action_bank)} actions")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📁 Results saved to: {results_file}")
        
        return results


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


def run_random_baseline_simulation(n_iterations: int = 5,
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
    Run the simulation with random baseline algorithm instead of optimization.
    """
    start_time = time.time()
    print("="*80)
    print("RANDOM BASELINE SIMULATION: Company + Random Algorithm")
    print("="*80)
    print(f"🎲 This baseline randomly generates {algorithm_config.get('action_bank_size', 20)} actions per iteration")
    print(f"🕐 Simulation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize components (same as regular simulation)
    print(f"\n1. Initializing Components...")
    print(f"   Company strategy: {company_strategy}")
    print(f"   Ground truth type: {ground_truth_type}")
    print(f"   Algorithm type: RANDOM BASELINE")
    
    init_start = time.time()
    
    # Configure dimensions
    import os
    if openai_api_key or os.getenv('OPENAI_API_KEY'):
        action_dim = 3072
    else:
        action_dim = 1536
    
    if algorithm_config is None:
        algorithm_config = {}
    if ground_truth_config is None:
        ground_truth_config = {}
    
    algorithm_config['action_dim'] = action_dim
    algorithm_config['user_dim'] = 8
    ground_truth_config['action_dim'] = action_dim
    ground_truth_config['user_dim'] = 8
    
    # Initialize company simulator (same as regular)
    company_sim = CompanySimulator(
        results_dir=results_dir,
        openai_api_key=openai_api_key,
        strategy_type=company_strategy,
        strategy_config=strategy_config,
        ground_truth_type=ground_truth_type,
        ground_truth_config=ground_truth_config,
        random_seed=42
    )
    
    # Initialize RANDOM baseline algorithm instead of optimization algorithm
    algorithm = RandomBaselineAlgorithm(
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
    
    # Run iterations (same structure as regular simulation)
    simulation_results = {
        'simulation_type': 'random_baseline',
        'initialization': init_results,
        'iterations': [],
        'final_summary': {}
    }
    
    print(f"\n3. Running {n_iterations} iterations with Random Baseline...")
    iterations_start = time.time()
    
    with tqdm(total=n_iterations, desc="🎲 Random Baseline Progress", unit="iteration") as pbar:
        for iteration in range(1, n_iterations + 1):
            iteration_start = time.time()
            
            print(f"\n{'='*50}")
            print(f"RANDOM BASELINE ITERATION {iteration}/{n_iterations}")
            print(f"{'='*50}")
            print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
            
            # Company runs iteration (same as regular)
            print(f"\n--- Company Phase ---")
            company_phase_start = time.time()
            company_results = company_sim.run_iteration(iteration)
            company_phase_time = time.time() - company_phase_start
            print(f"   ⏱️  Company phase completed in {company_phase_time:.2f}s")
            
            # Random baseline processes the data
            print(f"\n--- Random Baseline Algorithm Phase ---")
            algorithm_phase_start = time.time()
            algorithm_results = algorithm.process_iteration(iteration)
            algorithm_phase_time = time.time() - algorithm_phase_start
            print(f"   ⏱️  Random baseline phase completed in {algorithm_phase_time:.2f}s")
            
            # Integration phase (same as regular)
            integration_start = time.time()
            new_action_bank_file = algorithm_results['files_created']['new_action_bank']
            with open(new_action_bank_file, 'r') as f:
                new_bank_data = json.load(f)
            
            new_action_bank = [convert_to_embedded_action(action_data) 
                              for action_data in new_bank_data['actions']]
            
            print(f"\n--- Integration Phase ---")
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
                'algorithm_type': 'random_baseline',
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
            
            print(f"\nRandom Baseline Iteration {iteration} Complete:")
            print(f"  Company reward: {company_results['company_metrics']['avg_reward']:.4f}")
            print(f"  Random actions generated: {algorithm_config.get('action_bank_size', 20)}")
            print(f"  New action bank size: {len(new_action_bank)}")
            print(f"  ⏱️  Iteration time: {iteration_time:.2f}s | Total elapsed: {elapsed_total/60:.1f}m | ETA: {eta_formatted}")
            
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
    
    # Simple algorithm summary for random baseline
    algorithm_summary = {
        'algorithm_type': 'random_baseline',
        'total_iterations_processed': n_iterations,
        'actions_generated_per_iteration': algorithm_config.get('action_bank_size', 20),
        'total_random_actions_generated': n_iterations * algorithm_config.get('action_bank_size', 20),
        'baseline_description': 'Randomly generates K actions per iteration without optimization'
    }
    
    summary_time = time.time() - summary_start
    print(f"   ⏱️  Summary generation completed in {summary_time:.2f}s")
    
    simulation_results['final_summary'] = {
        'simulation_type': 'random_baseline',
        'company_summary': company_summary,
        'algorithm_summary': algorithm_summary,
        'overall_performance': {
            'total_iterations': n_iterations,
            'final_avg_reward': company_summary.get('overall_avg_reward', 0),
            'reward_trend': company_summary.get('avg_reward_by_iteration', []),
            'baseline_type': 'random_generation',
            'final_exploration_rate': company_summary.get('final_exploration_rate', 0),
            'company_learned': company_summary.get('company_learned', False)
        }
    }
    
    # Save complete results
    final_results_file = os.path.join(results_dir, "random_baseline_simulation_results.json")
    with open(final_results_file, 'w') as f:
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
    
    company_sim.save_final_results()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("RANDOM BASELINE SIMULATION COMPLETE!")
    print(f"{'='*80}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total execution time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Results saved to: {results_dir}")
    print(f"Complete results: {final_results_file}")
    
    print(f"\nFinal Performance (Random Baseline):")
    print(f"  Total users: {company_summary.get('total_users', 0)}")
    print(f"  Total observations: {company_summary.get('total_observations', 0)}")
    print(f"  Final avg reward: {company_summary.get('overall_avg_reward', 0):.4f}")
    print(f"  Total random actions generated: {n_iterations * algorithm_config.get('action_bank_size', 20)}")
    print(f"⏱️  Average time per iteration: {(time.time() - iterations_start) / n_iterations:.1f}s")
    
    return simulation_results


def main():
    """Main function with command line interface for random baseline."""
    parser = argparse.ArgumentParser(description='Run Random Baseline Simulation (Company + Random Algorithm)')
    
    # Same arguments as regular simulation
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
    
    # Company strategy options (same as regular)
    parser.add_argument('--company_strategy', choices=['linucb', 'bootstrapped_dqn', 'legacy'],
                       default='linucb', help='Company contextual bandit strategy (default: linucb)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='LinUCB alpha parameter (default: 1.0)')
    parser.add_argument('--n_heads', type=int, default=10,
                       help='Bootstrapped DQN number of heads (default: 10)')
    
    # Ground truth model options (same as regular)
    parser.add_argument('--ground_truth_type', choices=['mixture_of_experts', 'gmm'],
                       default='mixture_of_experts', 
                       help='Ground truth model type (default: mixture_of_experts)')
    parser.add_argument('--gmm_components', type=int, default=5,
                       help='Number of GMM components (default: 5)')
    
    # Random baseline specific options
    parser.add_argument('--random_actions_per_iteration', type=int, default=20,
                       help='Number of random actions to generate per iteration (default: 20)')
    
    args = parser.parse_args()
    
    # Configure random baseline algorithm
    algorithm_config = {
        'action_bank_size': args.random_actions_per_iteration,
        'action_pool_size': args.random_actions_per_iteration,  # Generate only what we need
        'random_seed': 42
    }
    
    # Configure company strategy (same as regular)
    strategy_config = {}
    if args.company_strategy == 'linucb':
        strategy_config = {'alpha': args.alpha, 'use_pca': True, 'pca_components': 128}
    elif args.company_strategy == 'bootstrapped_dqn':
        strategy_config = {'n_heads': args.n_heads}
    
    # Configure ground truth model (same as regular)
    ground_truth_config = {}
    if args.ground_truth_type == 'gmm':
        ground_truth_config = {'n_components': args.gmm_components}
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Warning: No OpenAI API key provided. Using fallback embeddings.")
    
    # Create unique timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_results_dir = os.path.join(args.results_dir, f"random_baseline_{timestamp}")
    
    print(f"📁 Random Baseline results will be saved to: {unique_results_dir}")
    
    # Save simulation configuration
    os.makedirs(unique_results_dir, exist_ok=True)
    simulation_config = {
        "simulation_type": "random_baseline",
        "timestamp": timestamp,
        "iterations": args.iterations,
        "users": args.users,
        "initial_actions": args.initial_actions,
        "random_actions_per_iteration": args.random_actions_per_iteration,
        "company_strategy": args.company_strategy,
        "strategy_config": strategy_config,
        "ground_truth_type": args.ground_truth_type,
        "ground_truth_config": ground_truth_config,
        "algorithm_config": algorithm_config,
        "openai_api_key_provided": openai_api_key is not None
    }
    
    config_file = os.path.join(unique_results_dir, "random_baseline_config.json")
    with open(config_file, 'w') as f:
        json.dump(simulation_config, f, indent=2, default=str)
    
    print(f"💾 Random Baseline configuration saved to: {config_file}")
    
    # Run random baseline simulation
    results = run_random_baseline_simulation(
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
