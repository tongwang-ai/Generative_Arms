import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import data structures
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.entities import Action, User
from simulation.user_generator import MeaningfulUser
from simulation.ground_truth import create_ground_truth_utility


@dataclass
class GroundTruthResult:
    """Results from ground truth evaluation."""
    action_id: str
    action_text: str
    expected_reward: float
    conversion_rate: float
    sample_size: int
    confidence_interval: Tuple[float, float]


class GroundTruthEvaluator:
    """
    Evaluates action banks on ground truth user data to measure algorithm performance.
    This shows whether our algorithm is generating better action banks over iterations.
    """
    
    def __init__(self, 
                 users_file: str = None, 
                 sample_size: int = 1000,
                 ground_truth_type: str = "mixture_of_experts",
                 ground_truth_config: Dict[str, Any] = None,
                 random_seed: int = 42):
        """
        Initialize ground truth evaluator.
        
        Args:
            users_file: Path to users JSON file (uses default if None)
            sample_size: Number of users to sample for evaluation
            ground_truth_type: Type of ground truth model ('mixture_of_experts' or 'gmm')
            ground_truth_config: Configuration for ground truth model
            random_seed: Random seed for reproducibility
        """
        self.sample_size = sample_size
        self.users_file = users_file
        self.users = None
        self.ground_truth_type = ground_truth_type
        self.random_seed = random_seed
        
        # Initialize ground truth model
        if ground_truth_config is None:
            ground_truth_config = {}
        self.conversion_model = create_ground_truth_utility(
            ground_truth_type=ground_truth_type,
            random_seed=random_seed,
            **ground_truth_config
        )
        
    def load_users(self, users_file: str = None) -> List[User]:
        """Load users from JSON file."""
        if users_file:
            self.users_file = users_file
        
        if not self.users_file:
            # Use default path - this should be passed from the calling algorithm
            # which has the correct timestamped results directory
            raise ValueError("Users file path not provided. Ground truth evaluator requires explicit users_file path.")
        
        if not os.path.exists(self.users_file):
            raise ValueError(f"Users file not found: {self.users_file}")
        
        print(f"Loading users from: {self.users_file}")
        with open(self.users_file, 'r') as f:
            data = json.load(f)
        
        users = []
        for user_data in data['users']:
            user = User(
                user_id=user_data['user_id'],
                features=np.array(user_data['feature_vector'])
            )
            users.append(user)
        
        self.users = users
        print(f"Loaded {len(users)} users for ground truth evaluation")
        return users
    
    def _simulate_user_action_conversion(self, user: User, action: Action) -> float:
        """
        Simulate the conversion probability for a user-action pair using ground truth.
        This uses the same ground truth model as the simulation.
        """
        # Convert to MeaningfulUser format for ground truth model
        meaningful_user = type('MeaningfulUser', (), {
            'user_id': user.user_id,
            'feature_vector': user.features,
            'segment': 'unknown'
        })()
        
        # Convert to EmbeddedAction format for ground truth model
        embedded_action = type('EmbeddedAction', (), {
            'action_id': action.action_id,
            'text': action.text,
            'embedding': action.embedding,
            'category': 'unknown'
        })()
        
        # Use the actual ground truth model (same as simulation)
        conversion_probability = self.conversion_model.calculate_utility(meaningful_user, embedded_action)
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.05)
        conversion_prob = np.clip(conversion_probability + noise, 0.05, 0.95)
        
        return float(conversion_prob)
    
    def evaluate_action_bank(self, action_bank: List[Action], 
                           iteration: int = None, users_file: str = None) -> Dict[str, Any]:
        """
        Evaluate an action bank on ground truth user data.
        
        Args:
            action_bank: List of actions to evaluate
            iteration: Optional iteration number for logging
            users_file: Optional path to users file
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.users:
            self.load_users(users_file)
        
        print(f"\n🔍 Ground Truth Evaluation (Iteration {iteration or 'N/A'})")
        print(f"   Evaluating {len(action_bank)} actions on {len(self.users)} users")
        
        # Sample users for evaluation to manage computation
        if len(self.users) > self.sample_size:
            sampled_users = np.random.choice(self.users, self.sample_size, replace=False)
        else:
            sampled_users = self.users
        
        action_results = []
        total_expected_reward = 0
        
        for action in action_bank:
            # Evaluate this action on all sampled users
            conversion_probs = []
            
            for user in sampled_users:
                conversion_prob = self._simulate_user_action_conversion(user, action)
                conversion_probs.append(conversion_prob)
            
            # Calculate statistics
            expected_reward = np.mean(conversion_probs)
            conversion_rate = expected_reward  # Same thing in this context
            std_dev = np.std(conversion_probs)
            
            # Calculate confidence interval (95%)
            margin_error = 1.96 * (std_dev / np.sqrt(len(conversion_probs)))
            ci_lower = max(0, expected_reward - margin_error)
            ci_upper = min(1, expected_reward + margin_error)
            
            result = GroundTruthResult(
                action_id=action.action_id,
                action_text=action.text,
                expected_reward=expected_reward,
                conversion_rate=conversion_rate,
                sample_size=len(sampled_users),
                confidence_interval=(ci_lower, ci_upper)
            )
            
            action_results.append(result)
            total_expected_reward += expected_reward
        
        # Sort by expected reward (descending)
        action_results.sort(key=lambda x: x.expected_reward, reverse=True)
        
        # Calculate overall statistics
        if len(action_bank) == 0:
            # Handle empty action bank case
            avg_expected_reward = 0.0
            best_action = None
            worst_action = None
        else:
            avg_expected_reward = total_expected_reward / len(action_bank)
            best_action = action_results[0] if action_results else None
            worst_action = action_results[-1] if action_results else None
        
        # Calculate diversity (spread of rewards)
        rewards = [r.expected_reward for r in action_results]
        reward_std = np.std(rewards) if rewards else 0.0
        
        evaluation_results = {
            'iteration': iteration,
            'total_actions_evaluated': len(action_bank),
            'users_sampled': len(sampled_users),
            'average_expected_reward': avg_expected_reward,
            'total_expected_value': total_expected_reward,
            'reward_standard_deviation': reward_std,
            'best_action': {
                'action_id': best_action.action_id,
                'text': best_action.action_text,
                'expected_reward': best_action.expected_reward,
                'confidence_interval': best_action.confidence_interval
            } if best_action else None,
            'worst_action': {
                'action_id': worst_action.action_id,
                'text': worst_action.action_text,
                'expected_reward': worst_action.expected_reward,
                'confidence_interval': worst_action.confidence_interval
            } if worst_action else None,
            'top_5_actions': [
                {
                    'action_id': r.action_id,
                    'text': r.action_text,
                    'expected_reward': r.expected_reward,
                    'confidence_interval': r.confidence_interval
                }
                for r in action_results[:5]
            ],
            'action_details': [
                {
                    'action_id': r.action_id,
                    'text': r.action_text,
                    'expected_reward': r.expected_reward,
                    'conversion_rate': r.conversion_rate,
                    'confidence_interval': r.confidence_interval
                }
                for r in action_results
            ]
        }
        
        # Print summary
        print(f"   📊 Average Expected Reward: {avg_expected_reward:.4f}")
        
        if best_action:
            print(f"   🏆 Best Action: '{best_action.action_text[:50]}...' ({best_action.expected_reward:.4f})")
        else:
            print(f"   🏆 Best Action: None (empty action bank)")
            
        if worst_action:
            print(f"   📉 Worst Action: '{worst_action.action_text[:50]}...' ({worst_action.expected_reward:.4f})")
        else:
            print(f"   📉 Worst Action: None (empty action bank)")
            
        print(f"   📏 Reward Diversity (std): {reward_std:.4f}")
        
        return evaluation_results
    
    def compare_iterations(self, results_dir: str, 
                          iterations: List[int] = None) -> Dict[str, Any]:
        """
        Compare ground truth performance across multiple iterations.
        
        Args:
            results_dir: Directory containing iteration results
            iterations: List of iterations to compare (auto-detect if None)
            
        Returns:
            Comparison results showing algorithm improvement over time
        """
        if iterations is None:
            # Auto-detect iterations
            iterations = []
            for item in os.listdir(results_dir):
                if item.startswith('iteration_') and os.path.isdir(os.path.join(results_dir, item)):
                    try:
                        iter_num = int(item.split('_')[1])
                        iterations.append(iter_num)
                    except ValueError:
                        continue
            iterations.sort()
        
        print(f"\n📈 Comparing Ground Truth Performance Across Iterations")
        print(f"   Iterations to compare: {iterations}")
        
        iteration_results = {}
        
        for iteration in iterations:
            # Load the new action bank from this iteration
            action_bank_file = os.path.join(
                results_dir, f"iteration_{iteration}", "new_action_bank", "new_action_bank.json"
            )
            
            if not os.path.exists(action_bank_file):
                print(f"   ⚠️  Action bank not found for iteration {iteration}")
                continue
            
            # Load actions
            with open(action_bank_file, 'r') as f:
                data = json.load(f)
            
            actions = []
            for action_data in data['actions']:
                action = Action(
                    action_id=action_data['action_id'],
                    text=action_data['text'],
                    embedding=np.array(action_data['embedding'])
                )
                actions.append(action)
            
            # Evaluate this action bank
            evaluation = self.evaluate_action_bank(actions, iteration)
            iteration_results[iteration] = evaluation
        
        # Calculate improvement metrics
        if len(iteration_results) > 1:
            sorted_iterations = sorted(iteration_results.keys())
            first_iter = sorted_iterations[0]
            last_iter = sorted_iterations[-1]
            
            first_reward = iteration_results[first_iter]['average_expected_reward']
            last_reward = iteration_results[last_iter]['average_expected_reward']
            
            improvement = last_reward - first_reward
            improvement_pct = (improvement / first_reward) * 100 if first_reward > 0 else 0
            
            comparison_results = {
                'iterations_compared': sorted_iterations,
                'iteration_results': iteration_results,
                'improvement_metrics': {
                    'absolute_improvement': improvement,
                    'percentage_improvement': improvement_pct,
                    'first_iteration_reward': first_reward,
                    'last_iteration_reward': last_reward,
                    'is_improving': improvement > 0
                },
                'reward_trend': [
                    iteration_results[i]['average_expected_reward'] 
                    for i in sorted_iterations
                ]
            }
            
            print(f"\n🎯 Algorithm Performance Summary:")
            print(f"   First Iteration Reward: {first_reward:.4f}")
            print(f"   Latest Iteration Reward: {last_reward:.4f}")
            print(f"   Absolute Improvement: {improvement:+.4f}")
            print(f"   Percentage Improvement: {improvement_pct:+.2f}%")
            print(f"   Algorithm is {'✅ IMPROVING' if improvement > 0 else '❌ NOT IMPROVING'}")
            
            return comparison_results
        
        else:
            return {
                'iterations_compared': list(iteration_results.keys()),
                'iteration_results': iteration_results,
                'note': 'Insufficient iterations for comparison'
            }
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                               output_file: str):
        """Save evaluation results to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   💾 Ground truth evaluation saved to: {output_file}")