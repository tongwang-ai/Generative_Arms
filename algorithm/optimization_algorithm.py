import numpy as np
import pandas as pd
import os
import json
import pickle
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import scipy.stats

# Import from the original src modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import LightGBMUserPreferenceModel, NeuralUserPreferenceModel, LinearUserPreferenceModel
from src.selection.action_generator import ActionGenerator
from src.selection.action_selector import ActionSelector
from src.data.entities import User, Action

# Import simulation components for data structures
from simulation.user_generator import MeaningfulUser
from simulation.action_embedder import EmbeddedAction

# Import ground truth evaluator
from .ground_truth_evaluator import GroundTruthEvaluator


class PersonalizedMarketingAlgorithm:
    """
    Our algorithm that processes company observation data and generates new action banks.
    This is the core algorithm that the company sends observation data to.
    """
    
    def __init__(self, 
                 results_dir: str = "results",
                 algorithm_config: Dict[str, Any] = None):
        """
        Initialize the optimization algorithm.
        
        Args:
            results_dir: Directory where iteration data is stored
            algorithm_config: Configuration for the algorithm
        """
        self.results_dir = results_dir
        
        # Default algorithm configuration
        default_config = {
            'diversity_weight': 0.15,
            'action_pool_size': 100,
            'action_bank_size': 20,
            'value_mode': 'direct_reward',
            'random_seed': 42,
            'reward_model_type': 'lightgbm',  # 'lightgbm' or 'simple'
            # PCA configuration
            'pca_config': {
                'use_pca': False,
                'pca_components': 50
            }
        }
        
        self.config = {**default_config, **(algorithm_config or {})}
        
        # Initialize algorithm components
        reward_model_type = self.config.get('reward_model_type', 'lightgbm')
        pca_config = self.config.get('pca_config', {})
        
        print(f"Initializing UserPreferenceModel with type: {reward_model_type}")
        
        if reward_model_type == 'lightgbm':
            self.reward_model = LightGBMUserPreferenceModel(
                diversity_weight=self.config['diversity_weight'],
                random_seed=self.config['random_seed'],
                use_pca=pca_config.get('use_pca', False),
                pca_components=pca_config.get('pca_components', 50)
            )
        elif reward_model_type == 'neural':
            self.reward_model = NeuralUserPreferenceModel(
                diversity_weight=self.config['diversity_weight'],
                random_seed=self.config['random_seed'],
                use_pca=pca_config.get('use_pca', False),
                pca_components=pca_config.get('pca_components', 50),
                hidden_dims=self.config.get('hidden_dims', [128, 64]),
                dropout_rate=self.config.get('dropout_rate', 0.3),
                learning_rate=self.config.get('learning_rate', 0.001)
            )
        elif reward_model_type == 'linear':
            self.reward_model = LinearUserPreferenceModel(
                diversity_weight=self.config['diversity_weight'],
                random_seed=self.config['random_seed'],
                use_pca=pca_config.get('use_pca', False),
                pca_components=pca_config.get('pca_components', 50)
            )
        else:
            raise ValueError(f"Unknown reward model type: {reward_model_type}. Choose from: lightgbm, neural, linear")
        
        self.action_generator = ActionGenerator(
            random_seed=self.config['random_seed'],
            use_llm=True  # Enable LLM-based action generation
        )
        
        self.action_selector = ActionSelector(
            value_mode=self.config['value_mode']
        )
        
        # Algorithm state
        self.training_history = []
        self.performance_history = []
        
        # Initialize ground truth evaluator with dimensions from config
        ground_truth_config = {
            'user_dim': self.config.get('user_dim', 8),
            'action_dim': self.config.get('action_dim', 3072)  # Default to text-embedding-3-large
        }
        self.ground_truth_evaluator = GroundTruthEvaluator(
            sample_size=500,
            ground_truth_config=ground_truth_config
        )
        
        print(f"Personalized Marketing Algorithm initialized")
        print(f"  Results directory: {results_dir}")
        print(f"  Configuration: {self.config}")
    
    def process_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Process observation data from the company and generate new action bank.
        
        Args:
            iteration: Iteration number to process
            
        Returns:
            Algorithm results including new action bank
        """
        process_start = time.time()
        print(f"\n=== Algorithm Processing Iteration {iteration} ===")
        
        iteration_dir = os.path.join(self.results_dir, f"iteration_{iteration}")
        
        if not os.path.exists(iteration_dir):
            raise ValueError(f"Iteration directory not found: {iteration_dir}")
        
        # 1. Load observation data from company
        observations_file = os.path.join(iteration_dir, "observations", "observations.csv")
        if not os.path.exists(observations_file):
            raise ValueError(f"Observations file not found: {observations_file}")
        
        load_start = time.time()
        print("1. Loading observation data from company...")
        observations_df = pd.read_csv(observations_file)
        
        # Convert string representations of arrays back to arrays
        if 'user_features' in observations_df.columns:
            observations_df['user_features'] = observations_df['user_features'].apply(
                lambda x: np.array([float(v) for v in x.split(',')]) if isinstance(x, str) else np.array(x)
            )
        if 'action_embedding' in observations_df.columns:
            observations_df['action_embedding'] = observations_df['action_embedding'].apply(
                lambda x: np.array([float(v) for v in x.split(',')]) if isinstance(x, str) else np.array(x)
            )
        
        print(f"   Loaded {len(observations_df)} observations")
        
        # 2. Load users data from current iteration
        users_file = os.path.join(self.results_dir, f"iteration_{iteration}", "users", "users.json")
        if not os.path.exists(users_file):
            raise ValueError(f"Users file not found: {users_file}")
        
        print("2. Loading user data...")
        users = self._load_users(users_file)
        print(f"   Loaded {len(users)} users")
        
        # 3. Load current action bank
        action_bank_file = os.path.join(iteration_dir, "action_bank", "action_bank.json")
        if not os.path.exists(action_bank_file):
            raise ValueError(f"Action bank file not found: {action_bank_file}")
        
        print("3. Loading current action bank...")
        current_actions = self._load_actions(action_bank_file)
        print(f"   Loaded {len(current_actions)} actions")
        
        # 4. Combine with historical data if available
        print("4. Combining with historical observation data...")
        all_observations = self._combine_historical_data(iteration)
        print(f"   Total historical observations: {len(all_observations)}")
        load_time = time.time() - load_start
        print(f"   ⏱️  Data loading completed in {load_time:.2f}s")
        
        # 5. Train models on observation data
        print("5. Training models on observation data...")
        training_start = time.time()
        training_results = self._train_models(all_observations, users, current_actions)
        training_time = time.time() - training_start
        print(f"   ⏱️  Model training completed in {training_time:.2f}s")
        
        # 5.1. Evaluate trained reward model on ground truth
        print("5.1. Evaluating reward model on ground truth...")
        eval_start = time.time()
        model_evaluation_results = self._evaluate_reward_model_on_ground_truth(users, current_actions)
        eval_time = time.time() - eval_start
        print(f"   ⏱️  Model evaluation completed in {eval_time:.2f}s")
        
        # 6. Generate new action bank
        print("6. Generating optimized action bank...")
        generation_start = time.time()
        
        # Get top performing actions from current bank
        current_performance = observations_df.groupby('action_id')['reward'].agg(['mean', 'count'])
        current_performance = current_performance[current_performance['count'] >= 3]  # Min 3 observations
        top_actions = current_performance.sort_values('mean', ascending=False).head(10)
        
        # Find corresponding action objects
        previous_best = []
        for action_id in top_actions.index:
            for action in current_actions:
                if action.action_id == action_id:
                    previous_best.append(action)
                    break
        
        # Load current action bank for this iteration (what the company is actually using)
        current_action_bank = self._load_current_action_bank(iteration)
        print(f"   Loaded current action bank with {len(current_action_bank)} actions")
        
        new_action_bank = self._generate_new_action_bank(users, previous_best, current_action_bank)
        generation_time = time.time() - generation_start
        print(f"   ⏱️  Action bank generation completed in {generation_time:.2f}s")
        
        # 7. Evaluate the new action bank
        print("7. Evaluating new action bank...")
        bank_eval_start = time.time()
        evaluation_results = self._evaluate_action_bank(new_action_bank, users)
        bank_eval_time = time.time() - bank_eval_start
        print(f"   ⏱️  Action bank evaluation completed in {bank_eval_time:.2f}s")
        
        # 8. Ground truth evaluation
        print("8. Ground truth evaluation...")
        gt_eval_start = time.time()
        # Pass the correct users file path to the evaluator
        users_file = os.path.join(self.results_dir, f"iteration_{iteration}", "users", "users.json")
        ground_truth_results = self.ground_truth_evaluator.evaluate_action_bank(
            new_action_bank, iteration, users_file=users_file
        )
        gt_eval_time = time.time() - gt_eval_start
        print(f"   ⏱️  Ground truth evaluation completed in {gt_eval_time:.2f}s")
        
        # 9. Save algorithm results
        print("9. Saving algorithm results...")
        save_start = time.time()
        algorithm_results = self._save_results(iteration, {
            'training_results': training_results,
            'model_evaluation_results': model_evaluation_results,
            'new_action_bank': new_action_bank,
            'evaluation_results': evaluation_results,
            'ground_truth_results': ground_truth_results,
            'previous_best_actions': previous_best,
            'observations_processed': len(all_observations)
        })
        save_time = time.time() - save_start
        print(f"   ⏱️  Results saving completed in {save_time:.2f}s")
        
        total_process_time = time.time() - process_start
        print(f"Algorithm processing complete for iteration {iteration}!")
        print(f"  Observations processed: {len(all_observations)}")
        print(f"  New action bank size: {len(new_action_bank)}")
        print(f"  Expected improvement: {evaluation_results.get('total_value', 0):.4f}")
        print(f"  Ground truth reward: {ground_truth_results.get('average_expected_reward', 0):.4f}")
        print(f"  ⏱️  Total algorithm processing time: {total_process_time:.2f}s")
        
        # Print model evaluation summary
        if model_evaluation_results:
            reward_eval = model_evaluation_results.get('reward_model_evaluation', {})
            
            if reward_eval.get('status') == 'evaluated':
                print(f"  Reward model ground truth correlation: {reward_eval.get('correlation', 0):.4f}")
                print(f"  Reward model ranking correlation: {reward_eval.get('rank_correlation', 0):.4f}")
        
        return algorithm_results
    
    def _load_users(self, users_file: str) -> List[User]:
        """Load users from JSON file and convert to User objects."""
        with open(users_file, 'r') as f:
            data = json.load(f)
        
        users = []
        for user_data in data['users']:
            user = User(
                user_id=user_data['user_id'],
                features=np.array(user_data['feature_vector'])
            )
            users.append(user)
        
        return users
    
    def _load_actions(self, action_bank_file: str) -> List[Action]:
        """Load actions from JSON file and convert to Action objects."""
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
        
        return actions
    
    def _load_current_action_bank(self, current_iteration: int) -> List[Action]:
        """Load the current action bank that the company is using for this iteration."""
        # For iteration 1, load from initialization folder
        if current_iteration == 1:
            action_bank_file = os.path.join(self.results_dir, "initialization", "action_bank", "action_bank.json")
        else:
            # For iteration > 1, load the company's current action bank from this iteration
            # (The company saves its current action bank during run_iteration)
            action_bank_file = os.path.join(self.results_dir, f"iteration_{current_iteration}", "action_bank", "action_bank.json")
        
        if os.path.exists(action_bank_file):
            return self._load_actions(action_bank_file)
        else:
            print(f"Warning: Action bank file not found: {action_bank_file}")
            return []
    
    def _load_cumulative_action_bank(self, current_iteration: int) -> List[Action]:
        """Load all action banks from previous iterations to create cumulative bank."""
        cumulative_actions = []
        seen_action_ids = set()
        
        # Load initial action bank from initialization folder
        initial_bank_file = os.path.join(self.results_dir, "initialization", "action_bank", "action_bank.json")
        if os.path.exists(initial_bank_file):
            initial_actions = self._load_actions(initial_bank_file)
            for action in initial_actions:
                if action.action_id not in seen_action_ids:
                    cumulative_actions.append(action)
                    seen_action_ids.add(action.action_id)
        
        # Load action banks from previous iterations (1 to current_iteration-1)
        for i in range(1, current_iteration):
            action_bank_file = os.path.join(self.results_dir, f"iteration_{i}", "new_action_bank", "new_action_bank.json")
            if os.path.exists(action_bank_file):
                iter_actions = self._load_actions(action_bank_file)
                for action in iter_actions:
                    if action.action_id not in seen_action_ids:
                        cumulative_actions.append(action)
                        seen_action_ids.add(action.action_id)
        
        return cumulative_actions
    
    def _combine_historical_data(self, current_iteration: int) -> pd.DataFrame:
        """Combine observation data from all previous iterations."""
        all_observations = []
        
        for i in range(1, current_iteration + 1):
            iter_obs_file = os.path.join(self.results_dir, f"iteration_{i}", 
                                       "observations", "observations.csv")
            if os.path.exists(iter_obs_file):
                iter_df = pd.read_csv(iter_obs_file)
                # Convert string representations of arrays back to arrays
                if 'user_features' in iter_df.columns:
                    iter_df['user_features'] = iter_df['user_features'].apply(
                        lambda x: np.array([float(v) for v in x.split(',')]) if isinstance(x, str) else np.array(x)
                    )
                if 'action_embedding' in iter_df.columns:
                    iter_df['action_embedding'] = iter_df['action_embedding'].apply(
                        lambda x: np.array([float(v) for v in x.split(',')]) if isinstance(x, str) else np.array(x)
                    )
                all_observations.append(iter_df)
        
        if all_observations:
            return pd.concat(all_observations, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _train_models(self, observations_df: pd.DataFrame, 
                     users: List[User], actions: List[Action]) -> Dict[str, Any]:
        """Train reward model on observation data."""
        training_results = {}
        
        if len(observations_df) == 0:
            print("   No observation data available for training")
            return training_results
        
        # Prepare data for training (rename columns to match expected format)
        training_data = observations_df.copy()
        if 'reward' in training_data.columns:
            training_data['outcome'] = training_data['reward']
        
        # Train reward model
        try:
            print("   Training reward model...")
            reward_metrics = self.reward_model.fit(training_data)
            training_results['reward_model'] = reward_metrics
            print(f"   Reward model trained - AUC: {reward_metrics.get('val_auc', 'N/A'):.4f}")
        except Exception as e:
            print(f"   Error training reward model: {e}")
            training_results['reward_model'] = {'error': str(e)}
        
        
        # Store training history
        self.training_history.append({
            'iteration': len(self.training_history) + 1,
            'training_data_size': len(observations_df),
            'results': training_results
        })
        
        return training_results
    
    def _evaluate_reward_model_on_ground_truth(self, users: List[User], 
                                             current_actions: List[Action]) -> Dict[str, Any]:
        """
        Evaluate the trained reward model on ground truth data.
        
        Args:
            users: List of users for evaluation
            current_actions: Current action bank
            
        Returns:
            Dictionary with reward model evaluation results
        """
        if not self.reward_model.is_trained:
            print("   Reward model not trained, skipping evaluation")
            return {'reward_model_evaluation': {'status': 'not_trained'}}
        
        print("   Evaluating reward model on ground truth...")
        reward_eval = self._evaluate_reward_model_ground_truth(users, current_actions)
        print(f"   Reward model correlation with ground truth: {reward_eval.get('correlation', 'N/A'):.4f}")
        
        return {'reward_model_evaluation': reward_eval}
    
    def _evaluate_reward_model_ground_truth(self, users: List[User], 
                                          actions: List[Action]) -> Dict[str, Any]:
        """Evaluate reward model predictions against ground truth conversion rates."""
        from simulation.ground_truth import create_ground_truth_utility
        
        # Initialize ground truth utility (same as used in simulation)
        # Determine action dimension from actual actions
        action_dim = len(actions[0].embedding) if actions else 3072  # Default to 3072 for text-embedding-3-large
        ground_truth = create_ground_truth_utility(
            ground_truth_type="mixture_of_experts",
            random_seed=self.config['random_seed'],
            user_dim=8,
            action_dim=action_dim
        )
        
        predicted_rewards = []
        true_rewards = []
        
        # Sample subset of user-action pairs for evaluation
        n_samples = min(500, len(users) * len(actions))  # Limit for computational efficiency
        
        for _ in range(n_samples):
            user = np.random.choice(users)
            action = np.random.choice(actions)
            
            # Get model prediction
            predicted_reward = self.reward_model.predict(user, action)
            predicted_rewards.append(predicted_reward)
            
            # Get ground truth utility (more accurate, less impacted by randomness)
            true_reward = ground_truth.calculate_utility(
                # Convert to MeaningfulUser format
                type('MeaningfulUser', (), {
                    'user_id': user.user_id,
                    'feature_vector': user.features,
                    'segment': 'unknown'
                })(),
                # Convert to EmbeddedAction format
                type('EmbeddedAction', (), {
                    'action_id': action.action_id,
                    'text': action.text,
                    'embedding': action.embedding,
                    'category': 'unknown'
                })()
            )
            true_rewards.append(true_reward)
        
        # Calculate correlation and other metrics
        predicted_rewards = np.array(predicted_rewards)
        true_rewards = np.array(true_rewards)
        
        correlation = np.corrcoef(predicted_rewards, true_rewards)[0, 1]
        mse = np.mean((predicted_rewards - true_rewards) ** 2)
        mae = np.mean(np.abs(predicted_rewards - true_rewards))
        
        # Calculate ranking correlation (Spearman)
        from scipy.stats import spearmanr
        rank_correlation, _ = spearmanr(predicted_rewards, true_rewards)
        
        return {
            'status': 'evaluated',
            'n_samples': n_samples,
            'correlation': float(correlation),
            'rank_correlation': float(rank_correlation),
            'mean_squared_error': float(mse),
            'mean_absolute_error': float(mae),
            'predicted_mean': float(np.mean(predicted_rewards)),
            'predicted_std': float(np.std(predicted_rewards)),
            'true_mean': float(np.mean(true_rewards)),
            'true_std': float(np.std(true_rewards))
        }
    
    # Policy-related methods removed - using direct reward-based assignment
    
    def _generate_new_action_bank(self, users: List[User], 
                                previous_best: List[Action], 
                                current_action_bank: List[Action]) -> List[Action]:
        """Generate new optimized action bank."""
        
        # If action_bank_size is 0, return empty list (no new actions)
        if self.config['action_bank_size'] == 0:
            print("   action_bank_size=0: Returning empty action bank (no changes)")
            return []
        
        if not self.reward_model.is_trained:
            raise RuntimeError("Reward model not trained")
        
        # Generate large candidate pool
        action_pool = self.action_generator.generate_action_pool(
            pool_size=self.config['action_pool_size'],
            previous_best=previous_best,
            embedding_dim=len(previous_best[0].embedding) if previous_best else 8
        )
        
        # Select optimal subset using our algorithm (start with current action bank)
        selection_result = self.action_selector.select_actions_with_evaluation(
            action_pool=action_pool,
            action_bank_size_per_iter=self.config['action_bank_size'],
            users=users,
            reward_model=self.reward_model,
            current_action_bank=current_action_bank
        )
        
        return selection_result['selected_actions']
    
    def _evaluate_action_bank(self, action_bank: List[Action], 
                            users: List[User]) -> Dict[str, Any]:
        """Evaluate the quality of the new action bank using direct reward-based assignment."""
        if not self.reward_model.is_trained:
            return {'total_value': 0, 'note': 'reward_model_not_trained'}
        
        evaluation = self.action_selector.evaluate_action_bank(
            action_bank, users, self.reward_model
        )
        
        return evaluation
    
    def _save_results(self, iteration: int, results: Dict[str, Any]) -> Dict[str, Any]:
        """Save algorithm results to files."""
        iteration_dir = os.path.join(self.results_dir, f"iteration_{iteration}")
        
        # Save new action bank
        new_action_bank_dir = os.path.join(iteration_dir, "new_action_bank")
        os.makedirs(new_action_bank_dir, exist_ok=True)
        
        # Convert Action objects to EmbeddedAction format for saving
        new_actions_data = []
        for action in results['new_action_bank']:
            action_dict = {
                'action_id': action.action_id,
                'text': action.text,
                'embedding': action.embedding.tolist(),
                'category': 'generated',  # Default category
                'metadata': {
                    'generated_by': 'optimization_algorithm',
                    'iteration': iteration,
                    'timestamp': datetime.now().isoformat()
                }
            }
            new_actions_data.append(action_dict)
        
        new_action_bank_file = os.path.join(new_action_bank_dir, "new_action_bank.json")
        with open(new_action_bank_file, 'w') as f:
            json.dump({
                'actions': new_actions_data,
                'total_actions': len(new_actions_data),
                'generation_config': self.config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save model checkpoint
        model_checkpoint_dir = os.path.join(iteration_dir, "model_checkpoint")
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        
        # Save reward model
        reward_model_file = os.path.join(model_checkpoint_dir, "reward_model.pkl")
        with open(reward_model_file, 'wb') as f:
            pickle.dump(self.reward_model, f)
        
        
        # Save algorithm state
        algorithm_state = {
            'config': self.config,
            'training_history': self.training_history,
            'performance_history': self.performance_history,
            'models_trained': {
                'reward_model': self.reward_model.is_trained
            }
        }
        
        algorithm_state_file = os.path.join(model_checkpoint_dir, "algorithm_state.json")
        with open(algorithm_state_file, 'w') as f:
            json.dump(algorithm_state, f, indent=2)
        
        # Save ground truth results if available
        ground_truth_file = None
        if 'ground_truth_results' in results:
            ground_truth_file = os.path.join(iteration_dir, "ground_truth_evaluation.json")
            self.ground_truth_evaluator.save_evaluation_results(
                results['ground_truth_results'], ground_truth_file
            )
        
        # Compile complete results
        complete_results = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'algorithm_config': self.config,
            'training_results': results['training_results'],
            'model_evaluation_results': results.get('model_evaluation_results', {}),
            'new_action_bank_size': len(results['new_action_bank']),
            'evaluation_results': results['evaluation_results'],
            'ground_truth_results': results.get('ground_truth_results', {}),
            'previous_best_count': len(results['previous_best_actions']),
            'observations_processed': results['observations_processed'],
            'files_created': {
                'new_action_bank': new_action_bank_file,
                'reward_model': reward_model_file,
                'algorithm_state': algorithm_state_file,
                'ground_truth_evaluation': ground_truth_file
            }
        }
        
        # Save complete results summary
        results_summary_file = os.path.join(iteration_dir, "algorithm_results.json")
        with open(results_summary_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        # Update performance history
        ground_truth_reward = results.get('ground_truth_results', {}).get('average_expected_reward', 0)
        self.performance_history.append({
            'iteration': iteration,
            'expected_value': results['evaluation_results'].get('total_value', 0),
            'ground_truth_reward': ground_truth_reward,
            'diversity': results['evaluation_results'].get('embedding_diversity', 0),
            'action_bank_size': len(results['new_action_bank'])
        })
        
        return complete_results
    
    def load_checkpoint(self, iteration: int) -> bool:
        """Load algorithm state from a checkpoint."""
        model_checkpoint_dir = os.path.join(self.results_dir, f"iteration_{iteration}", 
                                          "model_checkpoint")
        
        try:
            # Load reward model
            reward_model_file = os.path.join(model_checkpoint_dir, "reward_model.pkl")
            if os.path.exists(reward_model_file):
                with open(reward_model_file, 'rb') as f:
                    self.reward_model = pickle.load(f)
            
            
            # Load algorithm state
            algorithm_state_file = os.path.join(model_checkpoint_dir, "algorithm_state.json")
            if os.path.exists(algorithm_state_file):
                with open(algorithm_state_file, 'r') as f:
                    state = json.load(f)
                
                self.training_history = state.get('training_history', [])
                self.performance_history = state.get('performance_history', [])
            
            print(f"Algorithm checkpoint loaded from iteration {iteration}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def get_algorithm_summary(self) -> Dict[str, Any]:
        """Get summary of algorithm performance across iterations."""
        if not self.performance_history:
            return {'status': 'no_iterations_processed'}
        
        expected_values = [h['expected_value'] for h in self.performance_history]
        diversities = [h['diversity'] for h in self.performance_history]
        
        summary = {
            'total_iterations_processed': len(self.performance_history),
            'avg_expected_value': np.mean(expected_values),
            'best_expected_value': max(expected_values),
            'avg_diversity': np.mean(diversities),
            'performance_trend': expected_values,
            'diversity_trend': diversities,
            'models_trained': {
                'reward_model': self.reward_model.is_trained
            },
            'current_config': self.config
        }
        
        return summary