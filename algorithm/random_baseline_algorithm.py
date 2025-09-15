#!/usr/bin/env python3
"""
Random Baseline Algorithm

Implements a baseline algorithm that randomly generates K actions per iteration
and appends them to the current action bank, without training a reward model.

This class mirrors the behavior previously embedded in run_random_baseline.py,
but is now a first-class algorithm under the algorithm/ package, parallel to
optimization_algorithm.py.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd

from .optimization_algorithm import PersonalizedMarketingAlgorithm
from simulation.action_embedder import EmbeddedAction


class RandomBaselineAlgorithm(PersonalizedMarketingAlgorithm):
    """
    Random baseline that generates K random actions instead of using optimization.
    Inherits from PersonalizedMarketingAlgorithm to reuse infrastructure but overrides
    the core generation logic.
    """

    def __init__(self, results_dir: str = "results", algorithm_config: Dict[str, Any] = None):
        # Initialize with minimal configuration since we won't use the reward model
        algorithm_config = algorithm_config or {}
        minimal_config = {
            'action_pool_size': algorithm_config.get('action_bank_size', 20),  # Generate only what we need
            'action_bank_size': algorithm_config.get('action_bank_size', 20),
            'random_seed': algorithm_config.get('random_seed', 42),
            'reward_model_type': 'lightgbm'  # Still need to initialize, but won't use
        }
        super().__init__(results_dir, minimal_config)
        self.is_baseline = True

    def _load_current_users(self, iteration: int) -> List:
        """Load current users for the given iteration from results dir."""
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

    def _load_current_action_bank(self, iteration: int) -> Tuple[List[EmbeddedAction], List[EmbeddedAction]]:
        """Load current action bank for the iteration (or from initialization)."""
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

        import numpy as np

        actions: List[EmbeddedAction] = []
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

    def _embedded_action_to_dict(self, action: EmbeddedAction) -> Dict[str, Any]:
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
        Generate K random actions and append them to the current bank.
        """
        print(f"🎲 Random Baseline: Generating {self.config['action_bank_size']} random actions...")

        # Generate random actions using the action generator
        raw_actions = self.action_generator.generate_action_pool(
            pool_size=self.config['action_bank_size'],
            previous_best=None,
            user_segments=None
        )

        print(f"   ✅ Generated {len(raw_actions)} raw actions with embeddings, converting...")

        # Convert to EmbeddedAction without re-embedding (ActionGenerator already embedded)
        embedded_actions: List[EmbeddedAction] = []
        for i, action in enumerate(raw_actions):
            category = getattr(action, 'category', 'generated')
            metadata = getattr(action, 'metadata', {})
            embedded_actions.append(
                EmbeddedAction(
                    action_id=f"random_{len(current_action_bank) + i:04d}",
                    text=action.text,
                    embedding=action.embedding,
                    category=category,
                    metadata=metadata
                )
            )

        print(f"   ✅ Converted {len(embedded_actions)} random actions to EmbeddedAction")

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
