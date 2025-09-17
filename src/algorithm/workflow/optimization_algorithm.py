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

from src.algorithm.models import (
    LightGBMUserPreferenceModel,
    NeuralUserPreferenceModel,
    LinearUserPreferenceModel,
    GaussianProcessUserPreferenceModel,
    BayesianNeuralUserPreferenceModel,
    FTTransformerUserPreferenceModel,
)
from src.algorithm.action_selector import ActionSelector
from src.util.action_generator import ActionGenerator
from src.data.entities import User, Action

# Import shared data structures
from src.util.user_generator import MeaningfulUser
from src.util.action_embedder import EmbeddedAction

# Import ground truth evaluator
from ..evaluation.ground_truth_evaluator import GroundTruthEvaluator


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
            'use_segment_data': False,
            'segment_feature_dim': 30,
            # PCA configuration
            'pca_config': {
                'use_pca': False,
                'pca_components': 50
            }
        }
        
        self.config = {**default_config, **(algorithm_config or {})}
        
        self.task_type = self.config.get('task_type', 'binary')
        
        self.use_segment_data = self.config.get('use_segment_data', False)
        self.segment_feature_dim = self.config.get('segment_feature_dim', 30)
        if self.use_segment_data:
            self.config['user_dim'] = self.segment_feature_dim

        # Initialize algorithm components
        reward_model_type = self.config.get('reward_model_type', 'lightgbm')
        pca_config = self.config.get('pca_config', {})
        
        print(f"Initializing UserPreferenceModel with type: {reward_model_type}")
        
        if reward_model_type == 'lightgbm':
            self.reward_model = LightGBMUserPreferenceModel(
                diversity_weight=self.config['diversity_weight'],
                random_seed=self.config['random_seed'],
                use_pca=pca_config.get('use_pca', False),
                pca_components=pca_config.get('pca_components', 50),
                task_type=self.task_type,
                lightgbm_config=self.config.get('lightgbm_config')
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
        elif reward_model_type == 'gaussian_process':
            self.reward_model = GaussianProcessUserPreferenceModel(
                diversity_weight=self.config['diversity_weight'],
                random_seed=self.config['random_seed'],
                use_pca=pca_config.get('use_pca', False),
                pca_components=pca_config.get('pca_components', 50)
            )
        elif reward_model_type == 'bayesian_neural':
            self.reward_model = BayesianNeuralUserPreferenceModel(
                diversity_weight=self.config['diversity_weight'],
                random_seed=self.config['random_seed'],
                use_pca=pca_config.get('use_pca', False),
                pca_components=pca_config.get('pca_components', 50),
                hidden_dims=self.config.get('hidden_dims', [128, 64]),
                dropout_rate=self.config.get('dropout_rate', 0.3),
                learning_rate=self.config.get('learning_rate', 0.001),
                mc_samples=self.config.get('bnn_mc_samples', 30)
            )
        elif reward_model_type == 'ft_transformer':
            # New FT-Transformer based model for tabular user features
            self.reward_model = FTTransformerUserPreferenceModel(
                diversity_weight=self.config['diversity_weight'],
                random_seed=self.config['random_seed'],
                # FT-Transformer hyperparameters (with sensible defaults)
                fusion_type=self.config.get('fusion_type', 'concat'),  # 'concat' or 'attention'
                hidden_dim=self.config.get('hidden_dim', 128),
                n_heads=self.config.get('n_heads', 4),
                n_layers=self.config.get('n_layers', 2),
                dropout_rate=self.config.get('dropout_rate', 0.1),
                batch_size=self.config.get('batch_size', 128),
                epochs=self.config.get('epochs', 50),
                patience=self.config.get('patience', 8),
                learning_rate=self.config.get('learning_rate', 1e-3),
                weight_decay=self.config.get('weight_decay', 1e-4),
                # PCA settings for action embeddings (optional)
                use_pca=pca_config.get('use_pca', False),
                pca_components=pca_config.get('pca_components', 256)
            )
        else:
            raise ValueError(
                f"Unknown reward model type: {reward_model_type}. Choose from: lightgbm, neural, linear, gaussian_process, bayesian_neural, ft_transformer"
            )

        if self.task_type != 'binary' and reward_model_type != 'lightgbm':
            raise ValueError("Regression task is currently supported only with the LightGBM reward model")
        
        # Choose embedding model to match configured action_dim to keep dimensions consistent
        action_dim_cfg = int(self.config.get('action_dim', 3072))
        if action_dim_cfg == 3072:
            embedder_model = 'text-embedding-3-large'
        elif action_dim_cfg == 1536:
            # Prefer text-embedding-3-small for 1536-dim
            embedder_model = 'text-embedding-3-small'
        else:
            embedder_model = 'text-embedding-3-large'

        self.action_generator = ActionGenerator(
            random_seed=self.config['random_seed'],
            use_llm=True,  # Enable LLM-based action generation
            embedder_model=embedder_model
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
        
        use_segment = self.use_segment_data
        segment_data_dir = os.path.join(iteration_dir, "segment_data")

        # 1. Load observation data from company
        load_start = time.time()
        if use_segment:
            observations_file = os.path.join(segment_data_dir, "segment_action_observations.csv")
            if not os.path.exists(observations_file):
                raise ValueError(f"Segment observations file not found: {observations_file}")

            print("1. Loading segment-level observation data...")
            observations_df = pd.read_csv(observations_file)
            if 'feature_vector' in observations_df.columns:
                observations_df['segment_features'] = observations_df['feature_vector'].apply(self._parse_array)
            if 'action_embedding' in observations_df.columns:
                observations_df['action_embedding_array'] = observations_df['action_embedding'].apply(self._parse_array)
            observations_df['reward'] = observations_df.get('conversion_rate', observations_df.get('reward', 0.0))
            observations_df['sample_weight'] = observations_df.get('count', observations_df.get('targeted_count', 1.0))
            effective_obs = float(observations_df['sample_weight'].sum()) if len(observations_df) else 0.0
            print(f"   Loaded {len(observations_df)} segment-action aggregates (effective interactions: {effective_obs:.0f})")
        else:
            observations_file = os.path.join(iteration_dir, "observations", "observations.csv")
            if not os.path.exists(observations_file):
                raise ValueError(f"Observations file not found: {observations_file}")

            print("1. Loading user-level observation data...")
            observations_df = pd.read_csv(observations_file)
            if 'user_features' in observations_df.columns:
                observations_df['user_features'] = observations_df['user_features'].apply(self._parse_array)
            if 'action_embedding' in observations_df.columns:
                observations_df['action_embedding'] = observations_df['action_embedding'].apply(self._parse_array)
            observations_df['reward'] = observations_df['reward'].astype(float)
            observations_df['sample_weight'] = 1.0
            print(f"   Loaded {len(observations_df)} observations")

        # 2. Load users data from current iteration (user-level for evaluation)
        users_file = os.path.join(iteration_dir, "users", "users.json")
        if not os.path.exists(users_file):
            raise ValueError(f"Users file not found: {users_file}")

        print("2. Loading user data for evaluation...")
        evaluation_users = self._load_users(users_file)
        print(f"   Loaded {len(evaluation_users)} users for evaluation")

        if use_segment:
            segment_users_file = os.path.join(segment_data_dir, "segment_users.json")
            if not os.path.exists(segment_users_file):
                raise ValueError(f"Segment users file not found: {segment_users_file}")
            print("   Loading segment prototypes for selection...")
            selection_users = self._load_segment_users(segment_users_file)
            total_segment_weight = sum(user.weight for user in selection_users)
            print(f"   Loaded {len(selection_users)} segments (total weight {total_segment_weight:.0f})")
        else:
            selection_users = evaluation_users

        # 3. Load current action bank
        action_bank_file = os.path.join(iteration_dir, "action_bank", "action_bank.json")
        if not os.path.exists(action_bank_file):
            raise ValueError(f"Action bank file not found: {action_bank_file}")

        print("3. Loading current action bank...")
        current_actions = self._load_actions(action_bank_file)
        print(f"   Loaded {len(current_actions)} actions")
        load_time = time.time() - load_start
        print(f"   ⏱️  Data loading completed in {load_time:.2f}s")

        # 4. Combine with historical data if available
        print("4. Combining with historical observation data...")
        all_observations = self._combine_historical_data(iteration)
        if use_segment and 'count' in all_observations.columns:
            hist_weight = float(all_observations['count'].sum()) if len(all_observations) else 0.0
        elif use_segment and 'sample_weight' in all_observations.columns:
            hist_weight = float(all_observations['sample_weight'].sum()) if len(all_observations) else 0.0
        else:
            hist_weight = float(len(all_observations))
        print(f"   Total historical records: {len(all_observations)} (effective weight {hist_weight:.0f})")

        action_lookup = {action.action_id: action for action in current_actions}
        training_df = self._prepare_training_dataframe(all_observations, action_lookup)
        training_weight = float(training_df['sample_weight'].sum()) if 'sample_weight' in training_df.columns else float(len(training_df))
        print(f"   Prepared {len(training_df)} training rows (effective weight {training_weight:.0f})")

        # 5. Train models on observation data
        print("5. Training models on observation data...")
        training_start = time.time()
        training_results = self._train_models(training_df)
        training_time = time.time() - training_start
        print(f"   ⏱️  Model training completed in {training_time:.2f}s")
        
        # 5.1. Evaluate trained reward model on ground truth
        print("5.1. Evaluating reward model on ground truth...")
        eval_start = time.time()
        model_evaluation_results = self._evaluate_reward_model_on_ground_truth(evaluation_users, current_actions)
        eval_time = time.time() - eval_start
        print(f"   ⏱️  Model evaluation completed in {eval_time:.2f}s")
        
        # 6. Generate new action bank
        print("6. Generating optimized action bank...")
        generation_start = time.time()

        # Get top performing actions from current data
        if use_segment:
            perf_df = observations_df.copy()
            perf_df['weighted_reward'] = perf_df['reward'] * perf_df['sample_weight']
            current_performance = perf_df.groupby('action_id').agg(
                {'weighted_reward': 'sum', 'sample_weight': 'sum'}
            )
            current_performance = current_performance[current_performance['sample_weight'] > 0]
            current_performance['mean'] = current_performance['weighted_reward'] / current_performance['sample_weight']
            top_actions = current_performance.sort_values('mean', ascending=False).head(10)
        else:
            current_performance = observations_df.groupby('action_id')['reward'].agg(['mean', 'count'])
            current_performance = current_performance[current_performance['count'] >= 3]
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

        selection_result = self._generate_new_action_bank(selection_users, previous_best, current_action_bank)
        new_action_bank = selection_result['selected_actions']
        generation_time = time.time() - generation_start
        print(f"   ⏱️  Action bank generation completed in {generation_time:.2f}s")

        # 7. Evaluate the new action bank
        print("7. Evaluating new action bank...")
        bank_eval_start = time.time()
        # Use selection's evaluation if provided, otherwise compute
        evaluation_results = selection_result.get('evaluation') or self._evaluate_action_bank(new_action_bank, selection_users)
        bank_eval_time = time.time() - bank_eval_start
        print(f"   ⏱️  Action bank evaluation completed in {bank_eval_time:.2f}s")
        
        # 8. Ground truth evaluation
        print("8. Ground truth evaluation...")
        gt_eval_start = time.time()
        # Pass the correct users file path to the evaluator
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
            'observations_processed': len(training_df),
            'training_weight': training_weight,
            'selection_weight': float(sum(u.weight for u in selection_users)),
            'selection_uncertainty': selection_result.get('uncertainty')
        })
        save_time = time.time() - save_start
        print(f"   ⏱️  Results saving completed in {save_time:.2f}s")
        
        total_process_time = time.time() - process_start
        selection_weight_total = float(sum(u.weight for u in selection_users))
        print(f"Algorithm processing complete for iteration {iteration}!")
        print(f"  Training rows: {len(training_df)} (effective weight {training_weight:.0f})")
        print(f"  Selection population weight: {selection_weight_total:.0f}")
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
    
    def _parse_array(self, value: Any) -> np.ndarray:
        """Utility to convert serialized vectors into numpy arrays."""
        if isinstance(value, np.ndarray):
            return value.astype(float)
        if isinstance(value, list):
            return np.array(value, dtype=float)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return np.array([], dtype=float)
            parts = [p for p in value.split(',') if p]
            return np.array([float(p) for p in parts], dtype=float)
        if value is None:
            return np.array([], dtype=float)
        return np.array([float(value)], dtype=float)

    def _load_users(self, users_file: str) -> List[User]:
        """Load users from JSON file and convert to User objects."""
        with open(users_file, 'r') as f:
            data = json.load(f)
        
        users = []
        for user_data in data['users']:
            user = User(
                user_id=user_data['user_id'],
                features=self._parse_array(user_data.get('feature_vector', [])),
                weight=float(user_data.get('weight', 1.0)),
                metadata={
                    'segment': user_data.get('segment'),
                    'source': 'user_level'
                }
            )
            users.append(user)
        
        return users

    def _load_segment_users(self, segment_users_file: str) -> List[User]:
        """Load segment-level users from JSON file."""
        with open(segment_users_file, 'r') as f:
            data = json.load(f)

        segments = []
        for segment_data in data.get('segments', []):
            segment_id = segment_data.get('segment_id', f"segment_{len(segments)}")
            features = self._parse_array(segment_data.get('feature_vector', []))
            weight = float(segment_data.get('weight', segment_data.get('targeted_count', 1.0)))
            metadata = {
                'population_count': segment_data.get('population_count'),
                'targeted_count': segment_data.get('targeted_count'),
                'feature_stats': segment_data.get('feature_stats')
            }
            segments.append(User(
                user_id=segment_id,
                features=features,
                weight=weight,
                metadata=metadata
            ))

        return segments
    
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
            if self.use_segment_data:
                iter_obs_file = os.path.join(
                    self.results_dir,
                    f"iteration_{i}",
                    "segment_data",
                    "segment_action_observations.csv"
                )
            else:
                iter_obs_file = os.path.join(
                    self.results_dir,
                    f"iteration_{i}",
                    "observations",
                    "observations.csv"
                )

            if os.path.exists(iter_obs_file):
                iter_df = pd.read_csv(iter_obs_file)
                iter_df['iteration'] = i
                all_observations.append(iter_df)

        if all_observations:
            return pd.concat(all_observations, ignore_index=True)
        return pd.DataFrame()
    
    def _prepare_training_dataframe(self, observations_df: pd.DataFrame,
                                    action_lookup: Dict[str, Action]) -> pd.DataFrame:
        """Convert raw observation records into a model-friendly DataFrame."""
        records: List[Dict[str, Any]] = []

        if observations_df is None or len(observations_df) == 0:
            return pd.DataFrame(records)

        if self.use_segment_data:
            for _, row in observations_df.iterrows():
                action_id = row.get('action_id')
                action = action_lookup.get(action_id)
                if action is None:
                    continue

                features = self._parse_array(row.get('segment_features') or row.get('feature_vector', []))
                action_embedding = self._parse_array(row.get('action_embedding_array') or row.get('action_embedding', []))
                if action_embedding.size == 0:
                    action_embedding = action.embedding

                weight = float(row.get('sample_weight', row.get('count', row.get('targeted_count', 1.0))))
                reward = float(row.get('reward', row.get('conversion_rate', 0.0)))
                record = {
                    'user_features': features,
                    'action_embedding': action_embedding,
                    'sample_weight': weight,
                    'reward': reward,
                    'segment_id': row.get('segment_id'),
                    'iteration': row.get('iteration')
                }

                if self.task_type == 'binary':
                    conversions = row.get('conversions')
                    if conversions is None:
                        conversions = reward * weight
                    failures = max(0.0, weight - conversions)
                    record['conversions'] = float(conversions)
                    record['failures'] = float(failures)

                records.append(record)
        else:
            for _, row in observations_df.iterrows():
                features = self._parse_array(row.get('user_features'))
                action_embedding = self._parse_array(row.get('action_embedding'))
                reward = float(row.get('reward', 0.0))
                record = {
                    'user_features': features,
                    'action_embedding': action_embedding,
                    'reward': reward,
                    'sample_weight': 1.0,
                    'iteration': row.get('iteration')
                }
                if self.task_type == 'binary':
                    record['conversions'] = float(reward)
                    record['failures'] = float(1.0 - reward)
                records.append(record)

        return pd.DataFrame(records)

    def _train_models(self, training_df: pd.DataFrame) -> Dict[str, Any]:
        """Train reward model on prepared observation data."""
        training_results = {}

        if training_df is None or len(training_df) == 0:
            print("   No observation data available for training")
            return training_results

        # Train reward model
        try:
            print(f"   Training reward model on {len(training_df)} aggregated samples...")

            if self.task_type == 'binary' and 'conversions' in training_df.columns and 'failures' in training_df.columns:
                expanded_records = []
                for _, row in training_df.iterrows():
                    successes = float(row['conversions'])
                    failures = float(row['failures'])
                    features = row['user_features']
                    action_embedding = row['action_embedding']

                    if successes > 0:
                        expanded_records.append({
                            'user_features': features,
                            'action_embedding': action_embedding,
                            'reward': 1.0,
                            'sample_weight': successes
                        })
                    if failures > 0:
                        expanded_records.append({
                            'user_features': features,
                            'action_embedding': action_embedding,
                            'reward': 0.0,
                            'sample_weight': failures
                        })

                training_input = pd.DataFrame(expanded_records)
            else:
                training_input = training_df.copy()

            reward_metrics = self.reward_model.fit(training_input)
            training_results['reward_model'] = reward_metrics
            if self.task_type == 'binary':
                auc = reward_metrics.get('val_auc') or reward_metrics.get('train_auc') or reward_metrics.get('auc')
                if auc is not None:
                    print(f"   Reward model trained - AUC: {float(auc):.4f}")
            else:
                rmse = reward_metrics.get('val_rmse') or reward_metrics.get('train_rmse')
                if rmse is not None:
                    print(f"   Reward model trained - RMSE: {float(rmse):.4f}")
        except Exception as e:
            print(f"   Error training reward model: {e}")
            training_results['reward_model'] = {'error': str(e)}

        # Store training history
        self.training_history.append({
            'iteration': len(self.training_history) + 1,
            'training_data_size': len(training_df),
            'results': training_results,
            'use_segment_data': self.use_segment_data
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
        from src.simulation.ground_truth import create_ground_truth_utility
        
        # Initialize ground truth utility (same as used in simulation)
        # Determine action dimension from actual actions
        action_dim = len(actions[0].embedding) if actions else 3072  # Default to 3072 for text-embedding-3-large
        ground_truth = create_ground_truth_utility(
            ground_truth_type="mixture_of_experts",
            random_seed=self.config['random_seed'],
            user_dim=self.config.get('user_dim', 8),
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
                                current_action_bank: List[Action]) -> Dict[str, Any]:
        """Generate new optimized action bank and include selection diagnostics (e.g., uncertainty).
        Always returns a dictionary with selection and evaluation info.
        """
        
        # If action_bank_size is 0, keep current bank and return evaluation/diagnostics
        if self.config['action_bank_size'] == 0:
            print("   action_bank_size=0: Returning current action bank unchanged")
            evaluation = self._evaluate_action_bank(current_action_bank, users)
            # Include uncertainty summaries if supported
            uncertainty = None
            try:
                uncertainty = self.action_selector._summarize_uncertainty(current_action_bank, users, self.reward_model)
            except Exception:
                pass
            return {
                'selected_actions': current_action_bank,
                'evaluation': evaluation,
                'uncertainty': uncertainty,
                'selection_summary': {
                    'pool_size': 0,
                    'selected_count': len(current_action_bank),
                    'users_count': len(users),
                    'value_mode': getattr(self.action_selector, 'value_mode', 'direct_reward'),
                    'diversity_weight': getattr(self.reward_model, 'diversity_weight', None)
                }
            }
        
        if not self.reward_model.is_trained:
            raise RuntimeError("Reward model not trained")
        
        # Generate large candidate pool
        action_pool = self.action_generator.generate_action_pool(
            pool_size=self.config['action_pool_size'],
            previous_best=previous_best,
            strategy_mix=self.config.get('action_strategy_mix')
        )
        
        # Select optimal subset using our algorithm (start with current action bank)
        selection_result = self.action_selector.select_actions_with_evaluation(
            action_pool=action_pool,
            action_bank_size_per_iter=self.config['action_bank_size'],
            users=users,
            reward_model=self.reward_model,
            current_action_bank=current_action_bank
        )
        return selection_result
    
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
            'selection_uncertainty': results.get('selection_uncertainty'),
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
