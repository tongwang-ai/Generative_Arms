import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from src.util.user_generator import MeaningfulUserGenerator, MeaningfulUser
from src.util.action_embedder import OpenAIActionEmbedder, EmbeddedAction, create_marketing_action_bank
from ..ground_truth import create_ground_truth_utility, CompanyObservation
from ..strategies import BaseCompanyStrategy, LinUCBStrategy


class CompanySimulator:
    """
    Main simulator for company's marketing campaign behavior.
    Simulates the complete cycle from user generation to strategy updates.
    """
    
    def __init__(self, 
                 results_dir: str = "results",
                 openai_api_key: str = None,
                 strategy_type: str = "linucb",
                 strategy_config: Dict[str, Any] = None,
                 ground_truth_type: str = "mixture_of_experts",
                 ground_truth_config: Dict[str, Any] = None,
                 random_seed: int = 42,
                 batch_update_size: int = 1,
                 use_chatgpt_actions: bool = True,
                 performance_tracking_interval: int = 100,
                 use_segment_data: bool = False,
                 number_of_segments: Optional[int] = None,
                 users_per_segment_per_iteration: Optional[float] = None,
                 use_static_user_base: bool = False):
        """
        Initialize company simulator.
        
        Args:
            results_dir: Directory to store iteration results
            openai_api_key: OpenAI API key for embeddings
            strategy_type: 'linucb', 'bootstrapped_dqn', or 'legacy'
            strategy_config: Configuration for the strategy
            ground_truth_type: 'mixture_of_experts' or 'gmm'
            ground_truth_config: Configuration for ground truth model
            random_seed: Random seed for reproducibility
            batch_update_size: Update policy every N observations (1=online, >1=batch)
            use_chatgpt_actions: Whether to use ChatGPT for generating diverse actions
            performance_tracking_interval: Print policy performance every N users
        """
        
        # Store original parameters
        self.results_dir = results_dir
        self.random_seed = random_seed
        self.strategy_type = strategy_type
        self.ground_truth_type = ground_truth_type
        self.batch_update_size = batch_update_size
        self.use_chatgpt_actions = use_chatgpt_actions
        self.performance_tracking_interval = performance_tracking_interval
        self.use_segment_data = use_segment_data
        if number_of_segments is not None and number_of_segments != 1024:
            raise ValueError("Only 1024 segments are supported in the current configuration.")
        self.number_of_segments = 1024
        self.users_per_segment_per_iteration = users_per_segment_per_iteration
        self.use_static_user_base = use_static_user_base
        self._base_users: List[MeaningfulUser] = []
        self.segment_history: List[Dict[str, Any]] = []
        self.rng = np.random.default_rng(random_seed)
        np.random.seed(random_seed)
        
        # Create results directory
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.user_generator = MeaningfulUserGenerator(random_seed=random_seed)
        self.action_embedder = OpenAIActionEmbedder(
            api_key=openai_api_key,
            cache_file=os.path.join(results_dir, "embedding_cache.json")
        )
        
        # Initialize strategy based on type
        if strategy_config is None:
            strategy_config = {}
        strategy_config['random_seed'] = random_seed
        
        if strategy_type == "linucb":
            self.company_strategy = LinUCBStrategy(**strategy_config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}. Currently supported: 'linucb'")
        
        # Initialize ground truth utility model based on type
        if ground_truth_config is None:
            ground_truth_config = {}
        
        # Use dimensions from config (set by run_simulation_from_data.py)
        # Fall back to detection if not provided
        if 'action_dim' not in ground_truth_config:
            embedding_model = self.action_embedder.model
            if embedding_model == "text-embedding-3-large":
                action_dim = 3072
            elif embedding_model == "text-embedding-ada-002":
                action_dim = 1536
            else:
                action_dim = 1536
            ground_truth_config['action_dim'] = action_dim
        
        if 'user_dim' not in ground_truth_config:
            ground_truth_config['user_dim'] = 8
        
        self.ground_truth = create_ground_truth_utility(
            ground_truth_type=ground_truth_type,
            random_seed=random_seed,
            **ground_truth_config
        )
        self.segment_feature_names = self.user_generator.segment_feature_names
        
        # Simulation state
        self.current_action_bank = []
        self.all_observations = []
        self.iteration_results = []
        
        print(f"Company simulator initialized with results directory: {results_dir}")
        print(f"  Strategy type: {strategy_type}")
        print(f"  Ground truth type: {ground_truth_type}")
        if hasattr(self.ground_truth, 'get_model_info'):
            model_info = self.ground_truth.get_model_info()
            if ground_truth_type == "gmm":
                print(f"  GMM components: {model_info.get('n_components', 'N/A')}")
                print(f"  Utility range: {model_info.get('utility_range', 'N/A')}")
            elif ground_truth_type == "mixture_of_experts":
                print(f"  Expert count: {model_info.get('n_experts', 'N/A')}")
    
    def initialize_simulation(self, 
                            n_users: int = 1000,
                            n_initial_actions: int = 30) -> Dict[str, Any]:
        """
        Initialize the simulation with users and initial action bank.
        
        Args:
            n_users: Number of users to generate
            n_initial_actions: Number of initial actions in the bank
            
        Returns:
            Initialization summary
        """
        init_start = time.time()
        print("=== Initializing Company Simulation ===")
        
        # Store configuration for later use in each iteration
        self.n_users = n_users
        self.n_initial_actions = n_initial_actions
        print(f"Configured to generate {n_users} new users per iteration")
        print(f"Configured to generate {n_initial_actions} initial actions in first iteration")
        
        # Create initialization directory
        init_dir = os.path.join(self.results_dir, "initialization")
        os.makedirs(init_dir, exist_ok=True)
        
        # Generate initial action bank during initialization
        print(f"Creating initial action bank with {self.n_initial_actions} actions...")
        action_gen_start = time.time()
        self.current_action_bank = create_marketing_action_bank(
            self.action_embedder, 
            n_actions=self.n_initial_actions,
            use_chatgpt=self.use_chatgpt_actions,  # Use ChatGPT for diverse action generation
            product_focus="professional platform membership"  # Focus all actions on same product
        )
        action_gen_time = time.time() - action_gen_start
        print(f"   ⏱️  Initial action bank creation completed in {action_gen_time:.2f}s")
        
        # Save initial action bank to initialization directory
        action_bank_dir = os.path.join(init_dir, "action_bank")
        os.makedirs(action_bank_dir, exist_ok=True)
        
        action_bank_file = os.path.join(action_bank_dir, "action_bank.json")
        self.action_embedder.save_embedded_actions(self.current_action_bank, action_bank_file)
        
        # Create initialization summary
        init_summary = {
            'timestamp': datetime.now().isoformat(),
            'n_users_per_iteration': self.n_users,
            'n_initial_actions': self.n_initial_actions,
            'initial_action_bank_size': len(self.current_action_bank),
            'embedding_model': self.action_embedder.model,
            'use_segment_data': self.use_segment_data,
            'number_of_segments': self.number_of_segments,
            'users_per_segment_per_iteration': self.users_per_segment_per_iteration,
            'use_static_user_base': self.use_static_user_base
        }

        summary_file = os.path.join(init_dir, "initialization_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(init_summary, f, indent=2)

        # Optionally pre-generate a reusable user base when static cohorts are requested
        if self.use_static_user_base:
            base_users_dir = os.path.join(init_dir, "users")
            os.makedirs(base_users_dir, exist_ok=True)

            print("Generating static user base for all iterations...")
            self._base_users = self.user_generator.generate_users(self.n_users)
            base_users_file = os.path.join(base_users_dir, "users.json")
            self.user_generator.save_users(self._base_users, base_users_file)

            base_segments = self.user_generator.group_users_by_segment(self._base_users)
            base_segment_summary = self.user_generator.build_segment_summary_payload(
                base_segments, len(self._base_users)
            )
            with open(os.path.join(base_users_dir, "segment_summary.json"), 'w') as f:
                json.dump(base_segment_summary, f, indent=2)

            print(f"Static user base created with {len(self._base_users)} users across "
                  f"{len(base_segments)} segments")

        init_time = time.time() - init_start
        print(f"Initialization complete!")
        print(f"  Configuration: {self.n_initial_actions} initial actions, {self.n_users} users per iteration")
        print(f"  Initial action bank created with {len(self.current_action_bank)} actions")
        print(f"  Results saved to: {os.path.join(self.results_dir, 'initialization')}")
        print(f"  ⏱️  Total initialization time: {init_time:.2f}s")
        print(f"  Users will be generated fresh for each iteration")
        
        return init_summary

    def _prepare_iteration_users(self, iteration: int) -> Tuple[List[MeaningfulUser], Dict[str, List[MeaningfulUser]], Dict[str, List[MeaningfulUser]]]:
        """Generate or reuse users and return targeted subsets by segment."""
        if self.use_static_user_base and self._base_users:
            base_population = self._base_users
        else:
            base_population = self.user_generator.generate_users(self.n_users)
            if self.use_static_user_base:
                self._base_users = base_population

        # Reset weights for consistency across iterations
        for user in base_population:
            user.weight = 1.0

        segments = self.user_generator.group_users_by_segment(base_population)

        targeted_segments: Dict[str, List[MeaningfulUser]] = {}
        targeted_users: List[MeaningfulUser] = []

        for segment_id, members in segments.items():
            selected_members = self._sample_segment_users(segment_id, members)
            targeted_segments[segment_id] = selected_members
            for user in selected_members:
                user.weight = 1.0
            targeted_users.extend(selected_members)

        if not targeted_users:
            # Fallback to full population if sampling removed everything
            targeted_users = base_population
            targeted_segments = segments

        return targeted_users, segments, targeted_segments

    def _sample_segment_users(self, segment_id: str, members: List[MeaningfulUser]) -> List[MeaningfulUser]:
        """Sample a subset of users from a segment based on configuration."""
        if not members:
            return []

        sample_param = self.users_per_segment_per_iteration
        if sample_param is None:
            return list(members)

        if isinstance(sample_param, float) and sample_param < 1.0:
            sample_size = max(1, int(round(len(members) * sample_param)))
        else:
            sample_size = int(sample_param)
            sample_size = max(1, sample_size)

        sample_size = min(sample_size, len(members))
        if sample_size >= len(members):
            return list(members)

        indices = self.rng.choice(len(members), size=sample_size, replace=False)
        return [members[int(idx)] for idx in indices]

    def _save_segment_data(self, iteration: int, iteration_dir: str,
                           all_segments: Dict[str, List[MeaningfulUser]],
                           targeted_segments: Dict[str, List[MeaningfulUser]],
                           observations_df: pd.DataFrame,
                           current_actions: List[EmbeddedAction]):
        """Persist segment-level aggregates for downstream algorithms."""
        segment_dir = os.path.join(iteration_dir, "segment_data")
        os.makedirs(segment_dir, exist_ok=True)

        action_embedding_lookup = {
            action.action_id: action.embedding for action in current_actions
        }

        total_population = sum(len(members) for members in all_segments.values())
        population_summary = self.user_generator.build_segment_summary_payload(all_segments, total_population)
        targeted_summary = self.user_generator.build_segment_summary_payload(targeted_segments, total_population)

        combined_summary: Dict[str, Any] = {}
        segment_rows: List[Dict[str, Any]] = []
        feature_vector_length = 0

        for segment_id, members in population_summary.items():
            population_count = len(all_segments.get(segment_id, []))
            targeted_count = len(targeted_segments.get(segment_id, []))
            targeted_stats = targeted_summary.get(segment_id)
            feature_means = None
            if targeted_stats and targeted_stats.get('feature_mean'):
                feature_means = targeted_stats['feature_mean']
            elif members.get('feature_mean'):
                feature_means = members['feature_mean']

            if feature_means is None:
                # Fallback to zeros if features unavailable
                dim = len(all_segments.get(segment_id, [])[0].feature_vector) if all_segments.get(segment_id) else 0
                feature_means = [0.0] * dim

            feature_vector_length = max(feature_vector_length, len(feature_means))

            combined_summary[segment_id] = {
                'population': members,
                'targeted': targeted_stats,
                'population_count': population_count,
                'targeted_count': targeted_count,
                'targeting_rate': targeted_count / population_count if population_count else 0.0,
                'feature_mean_for_model': feature_means
            }

            segment_rows.append({
                'segment_id': segment_id,
                'feature_vector': feature_means,
                'population_count': population_count,
                'targeted_count': targeted_count,
                'weight': targeted_count if targeted_count > 0 else population_count,
                'feature_stats': targeted_stats or members
            })

        # Save summary JSON for transparency
        with open(os.path.join(segment_dir, "segment_summary.json"), 'w') as f:
            json.dump({
                'iteration': iteration,
                'segments': combined_summary,
                'total_population': total_population
            }, f, indent=2)

        # Build per-feature statistics DataFrame
        feature_stats_df = self.user_generator.build_segment_feature_dataframe(targeted_segments)
        if feature_stats_df.empty:
            feature_stats_df = self.user_generator.build_segment_feature_dataframe(all_segments)

        if not observations_df.empty and not feature_stats_df.empty:
            segment_conversion_rates = observations_df.groupby('segment_id')['reward'].mean().to_dict()
            feature_stats_df['conversion_rate'] = feature_stats_df['segment_id'].map(lambda sid: float(segment_conversion_rates.get(sid, 0.0)))
            feature_stats_df['expected_reward'] = feature_stats_df['conversion_rate']
            feature_stats_df['target_metric_value'] = feature_stats_df['conversion_rate']
            feature_stats_df['iteration'] = iteration
        else:
            feature_stats_df['conversion_rate'] = 0.0
            feature_stats_df['expected_reward'] = 0.0
            feature_stats_df['target_metric_value'] = 0.0
            feature_stats_df['iteration'] = iteration

        feature_stats_path = os.path.join(segment_dir, "feature_segment_stats.csv")
        feature_stats_df.to_csv(feature_stats_path, index=False)

        # Prepare observations DataFrame with parsed features
        obs_df = observations_df.copy()
        if not obs_df.empty:
            def _ensure_array(val):
                if isinstance(val, str):
                    return np.array([float(x) for x in val.split(',')])
                if isinstance(val, list):
                    return np.array(val)
                return val

            obs_df['user_features_vector'] = obs_df['user_features'].apply(_ensure_array)
            obs_df['segment_id'] = obs_df['segment_id'].fillna('unknown')

        # Aggregate per segment-action
        segment_action_records: List[Dict[str, Any]] = []
        feature_segment_action_records: List[Dict[str, Any]] = []

        if not obs_df.empty:
            grouped = obs_df.groupby(['segment_id', 'action_id'])
            for (segment_id, action_id), group in grouped:
                conversions = float(group['reward'].sum())
                count = int(len(group))
                conversion_rate = float(group['reward'].mean()) if count else 0.0
                targeted_count = combined_summary.get(segment_id, {}).get('targeted_count', count)
                population_count = combined_summary.get(segment_id, {}).get('population_count', targeted_count)
                action_probability = conversion_rate * 0.0
                if targeted_count:
                    action_probability = count / targeted_count

                feature_means = combined_summary.get(segment_id, {}).get('feature_mean_for_model', [])
                segment_action_records.append({
                    'segment_id': segment_id,
                    'action_id': action_id,
                    'expected_reward': conversion_rate,
                    'target_metric_value': conversion_rate,
                    'conversion_rate': conversion_rate,
                    'count': count,
                    'conversions': conversions,
                    'targeted_count': targeted_count,
                    'segment_population': population_count,
                    'action_probability': action_probability,
                    'feature_vector': ','.join(map(str, feature_means)),
                    'action_embedding': ','.join(map(str, action_embedding_lookup.get(action_id, []))),
                    'iteration': iteration
                })

                feature_matrix = np.vstack(group['user_features_vector'].values)
                feature_stats = {
                    'mean': np.mean(feature_matrix, axis=0),
                    'median': np.median(feature_matrix, axis=0),
                    'p25': np.percentile(feature_matrix, 25, axis=0),
                    'p75': np.percentile(feature_matrix, 75, axis=0),
                    'min': np.min(feature_matrix, axis=0),
                    'max': np.max(feature_matrix, axis=0)
                }

                for idx in range(feature_matrix.shape[1]):
                    feature_segment_action_records.append({
                        'segment_id': segment_id,
                        'action_id': action_id,
                        'feature_index': idx,
                        'feature_value_mean': float(feature_stats['mean'][idx]),
                        'feature_value_median': float(feature_stats['median'][idx]),
                        'feature_value_p25': float(feature_stats['p25'][idx]),
                        'feature_value_p75': float(feature_stats['p75'][idx]),
                        'feature_value_min': float(feature_stats['min'][idx]),
                        'feature_value_max': float(feature_stats['max'][idx]),
                        'expected_reward': conversion_rate,
                        'target_metric_value': conversion_rate,
                        'conversion_rate': conversion_rate,
                        'count': count,
                        'iteration': iteration
                    })

        segment_action_df = pd.DataFrame(segment_action_records)
        segment_action_path = os.path.join(segment_dir, "segment_action_stats.csv")
        if segment_action_records:
            segment_action_df.to_csv(segment_action_path, index=False)
        else:
            # Create empty file with headers for consistency
            pd.DataFrame(columns=[
                'segment_id', 'action_id', 'expected_reward', 'target_metric_value', 'conversion_rate',
                'count', 'conversions', 'targeted_count', 'segment_population', 'action_probability',
                'feature_vector', 'action_embedding', 'iteration'
            ]).to_csv(segment_action_path, index=False)

        feature_segment_action_df = pd.DataFrame(feature_segment_action_records)
        feature_segment_action_path = os.path.join(segment_dir, "feature_segment_action_stats.csv")
        if feature_segment_action_records:
            feature_segment_action_df.to_csv(feature_segment_action_path, index=False)
        else:
            pd.DataFrame(columns=[
                'segment_id', 'action_id', 'feature_index', 'feature_value_mean',
                'feature_value_median', 'feature_value_p25', 'feature_value_p75',
                'feature_value_min', 'feature_value_max', 'expected_reward',
                'target_metric_value', 'conversion_rate', 'count', 'iteration'
            ]).to_csv(feature_segment_action_path, index=False)

        # Save segment-centric payload for algorithm consumption
        segment_users_payload = {
            'iteration': iteration,
            'feature_vector_length': feature_vector_length,
            'segments': [
                {
                    'segment_id': row['segment_id'],
                    'feature_vector': row['feature_vector'],
                    'population_count': row['population_count'],
                    'targeted_count': row['targeted_count'],
                    'weight': row['weight'],
                    'feature_stats': row['feature_stats']
                }
                for row in segment_rows
            ]
        }

        segment_users_file = os.path.join(segment_dir, "segment_users.json")
        with open(segment_users_file, 'w') as f:
            json.dump(segment_users_payload, f, indent=2)

        # Persist list for algorithm training convenience
        training_file = os.path.join(segment_dir, "segment_action_observations.csv")
        if segment_action_records:
            segment_action_df.to_csv(training_file, index=False)
        else:
            pd.DataFrame(columns=[
                'segment_id', 'action_id', 'expected_reward', 'target_metric_value', 'conversion_rate',
                'count', 'conversions', 'targeted_count', 'segment_population', 'action_probability',
                'feature_vector', 'action_embedding', 'iteration'
            ]).to_csv(training_file, index=False)

        self.segment_history.append({
            'iteration': iteration,
            'summary': combined_summary,
            'files': {
                'segment_summary': os.path.relpath(os.path.join(segment_dir, "segment_summary.json"), self.results_dir),
                'segment_users': os.path.relpath(segment_users_file, self.results_dir),
                'segment_action_stats': os.path.relpath(segment_action_path, self.results_dir),
                'feature_segment_stats': os.path.relpath(feature_stats_path, self.results_dir),
                'feature_segment_action_stats': os.path.relpath(feature_segment_action_path, self.results_dir)
            }
        })
    def filter_and_balance_actions(self, iteration_dir: str, 
                                 users: List[MeaningfulUser],
                                 min_conversion: float = 0.30, 
                                 max_conversion: float = 0.75,
                                 max_attempts: int = 5):
        """
        Filter out actions with extreme conversion rates and regenerate replacements.
        
        Args:
            iteration_dir: Directory to save filtered actions info
            min_conversion: Minimum acceptable conversion rate (30%)
            max_conversion: Maximum acceptable conversion rate (75%)
            max_attempts: Maximum attempts to find balanced actions
        """
        print(f"\n🎯 Filtering actions for optimal conversion rates ({min_conversion:.0%}-{max_conversion:.0%})...")
        
        original_count = len(self.current_action_bank)
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            conversion_rates = self.compute_ground_truth_conversion_rates(users)
            
            # Identify actions to remove
            actions_to_remove = []
            for action_id, rate in conversion_rates.items():
                if rate < min_conversion or rate > max_conversion:
                    actions_to_remove.append((action_id, rate))
            
            if not actions_to_remove:
                print(f"✅ All actions have balanced conversion rates!")
                break
                
            print(f"  Attempt {attempt}: Found {len(actions_to_remove)} actions with extreme rates")
            for action_id, rate in actions_to_remove:
                action_text = next((a.text for a in self.current_action_bank if a.action_id == action_id), "Unknown")
                reason = "too low" if rate < min_conversion else "too high"
                print(f"    - {action_id} ({rate:.1%} - {reason}): {action_text[:50]}...")
            
            # Remove extreme actions
            self.current_action_bank = [a for a in self.current_action_bank 
                                      if a.action_id not in [item[0] for item in actions_to_remove]]
            
            # Generate replacement actions
            n_replacements = len(actions_to_remove)
            print(f"  Generating {n_replacements} replacement actions...")
            
            replacement_actions = create_marketing_action_bank(
                self.action_embedder,
                n_actions=n_replacements,
                use_chatgpt=self.use_chatgpt_actions,
                product_focus="professional platform membership"
            )
            
            # Add replacements to current bank
            self.current_action_bank.extend(replacement_actions)
            
            # Update action IDs to avoid conflicts
            for i, action in enumerate(self.current_action_bank):
                action.action_id = f"action_{i:04d}"
        
        if attempt >= max_attempts:
            print(f"⚠️  Reached maximum attempts ({max_attempts}). Using current action bank.")
        
        # Save filtering results
        final_conversion_rates = self.compute_ground_truth_conversion_rates(users)
        rates = list(final_conversion_rates.values())
        
        filtering_results = {
            'original_action_count': original_count,
            'final_action_count': len(self.current_action_bank),
            'filtering_attempts': attempt,
            'min_conversion_threshold': min_conversion,
            'max_conversion_threshold': max_conversion,
            'final_conversion_rates': final_conversion_rates,
            'final_stats': {
                'min_rate': min(rates),
                'max_rate': max(rates),
                'avg_rate': sum(rates) / len(rates),
                'actions_in_range': sum(1 for r in rates if min_conversion <= r <= max_conversion)
            }
        }
        
        filtering_file = os.path.join(iteration_dir, "action_filtering_results.json")
        with open(filtering_file, 'w') as f:
            json.dump(filtering_results, f, indent=2)
        
        balanced_count = filtering_results['final_stats']['actions_in_range']
        total_count = len(rates)
        print(f"🎯 Final result: {balanced_count}/{total_count} actions in optimal range")
        print(f"   Range: {min(rates):.1%} - {max(rates):.1%} (target: {min_conversion:.0%}-{max_conversion:.0%})")
    
    def _print_policy_performance(self, users_processed: int, recent_rewards: List[int], 
                                cumulative_reward: int, all_observations: List):
        """Print policy performance metrics every N users."""
        
        # Calculate recent performance (last N users based on tracking interval)
        recent_window = min(self.performance_tracking_interval, len(recent_rewards))
        recent_n_rewards = recent_rewards[-recent_window:] if recent_window > 0 else recent_rewards
        recent_avg = sum(recent_n_rewards) / len(recent_n_rewards) if recent_n_rewards else 0
        
        # Calculate overall performance
        overall_avg = cumulative_reward / users_processed if users_processed > 0 else 0
        
        # Get strategy-specific stats
        try:
            strategy_stats = self.company_strategy.get_strategy_stats()
            strategy_info = f"total_reward={strategy_stats.get('total_reward', 'N/A')}"
        except Exception:
            strategy_info = "strategy_stats=unavailable"
        
        # Action selection distribution (last N selections based on tracking interval)
        recent_window = min(self.performance_tracking_interval, len(all_observations))
        recent_actions = [obs.action_id for obs in all_observations[-recent_window:]] if recent_window > 0 else [obs.action_id for obs in all_observations]
        from collections import Counter
        action_counts = Counter(recent_actions)
        most_selected = action_counts.most_common(3)
        
        # Format action distribution
        action_dist = ", ".join([f"{aid.split('_')[-1]}:{count}" for aid, count in most_selected])
        
        print(f"   📊 Users {users_processed:4d}: recent_avg={recent_avg:.3f}, overall_avg={overall_avg:.3f}, "
              f"top_actions=[{action_dist}], {strategy_info}")
    
    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Run a single iteration of the company's marketing campaign.
        
        Args:
            iteration: Iteration number (1-based)
            
        Returns:
            Iteration results and observations for the algorithm
        """
        print(f"\n=== Company Iteration {iteration} ===")
        
        # Create iteration directory
        iteration_dir = os.path.join(self.results_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Prepare users for this iteration (optionally reusing and segmenting the base population)
        iteration_users, all_segments, targeted_segments = self._prepare_iteration_users(iteration)
        total_population = sum(len(users) for users in all_segments.values()) or len(iteration_users)
        print(f"Prepared {len(iteration_users)} targeted users across {len(targeted_segments)} segments"
              f" (population pool: {total_population})")
        user_segment_map = {user.user_id: user.segment for user in iteration_users}

        # Save iteration users
        users_dir = os.path.join(iteration_dir, "users")
        os.makedirs(users_dir, exist_ok=True)
        users_file = os.path.join(users_dir, "users.json")
        self.user_generator.save_users(iteration_users, users_file)
        
        # Generate user segment summary for this iteration
        segment_summary = self.user_generator.get_segment_summary(iteration_users)
        segment_file = os.path.join(users_dir, "user_segments.json")
        with open(segment_file, 'w') as f:
            json.dump(segment_summary, f, indent=2)
        
        # Use the action bank that was initialized during setup or updated from previous iterations
        print(f"Using action bank with {len(self.current_action_bank)} actions for iteration {iteration}")
        
        # Print ground truth preference analysis (beginning of iteration)
        print(f"\n🎯 Pre-Simulation Analysis (Start of Iteration {iteration}):")
        self._print_ground_truth_preferences(iteration_users)
        
        # Online contextual bandit: select actions and learn one user at a time
        update_mode = "online" if self.batch_update_size == 1 else f"batch ({self.batch_update_size})"
        print(f"1. Running contextual bandit simulation ({update_mode} updates)...")
        
        observations = []
        batch_observations = []
        
        # Performance tracking variables
        recent_rewards = []
        cumulative_reward = 0
        
        for i, user in tqdm(enumerate(iteration_users), total=len(iteration_users)):
            # 🔥 ONLINE ACTION SELECTION: Select action for this specific user using current policy
            action_assignments = self.company_strategy.select_actions([user], self.current_action_bank)
            
            if user.user_id in action_assignments:
                action_id = action_assignments[user.user_id]
                
                # Find the action object
                action = None
                for a in self.current_action_bank:
                    if a.action_id == action_id:
                        action = a
                        break
                
                if action:
                    # Simulate response
                    response = self.ground_truth.simulate_response(user, action)
                    
                    # Create observation
                    obs = CompanyObservation(
                        user_id=user.user_id,
                        action_id=action_id,
                        user_features=user.feature_vector,
                        action_embedding=action.embedding,
                        action_text=action.text,
                        reward=response,
                        timestamp=datetime.now().isoformat(),
                        iteration=iteration
                    )
                    
                    observations.append(obs)
                    batch_observations.append(obs)
                    
                    # Track performance metrics
                    recent_rewards.append(response)
                    cumulative_reward += response
                    
                    # Update strategy based on batch_update_size
                    if len(batch_observations) >= self.batch_update_size:
                        self.company_strategy.update_strategy(batch_observations)
                        batch_observations = []  # Clear batch
                    
                    # Print performance every N users (configurable)
                    if (i + 1) % self.performance_tracking_interval == 0:
                        self._print_policy_performance(i + 1, recent_rewards, cumulative_reward, observations)
        
        # Update with any remaining observations in batch
        if batch_observations:
            self.company_strategy.update_strategy(batch_observations)
        
        # 3. Finalize iteration
        print("3. Finalizing iteration...")
        self.company_strategy.iteration = iteration
        
        # 4. Calculate iteration metrics
        total_reward = sum(obs.reward for obs in observations)
        avg_reward = total_reward / len(observations) if observations else 0
        
        strategy_stats = self.company_strategy.get_strategy_stats()
        
        # 5. Save observation data (this is what gets sent to the algorithm)
        observations_dir = os.path.join(iteration_dir, "observations")
        os.makedirs(observations_dir, exist_ok=True)
        
        # Save as CSV for algorithm consumption
        obs_data = []
        for obs in observations:
            obs_dict = {
                'user_id': obs.user_id,
                'action_id': obs.action_id,
                'user_features': ','.join(map(str, obs.user_features)),  # Serialize as comma-separated string
                'action_embedding': ','.join(map(str, obs.action_embedding)),  # Serialize as comma-separated string
                'action_text': obs.action_text,
                'reward': obs.reward,
                'timestamp': obs.timestamp,
                'iteration': obs.iteration,
                'segment_id': user_segment_map.get(obs.user_id, 'unknown')
            }
            obs_data.append(obs_dict)
        
        obs_df = pd.DataFrame(obs_data)
        obs_file = os.path.join(observations_dir, "observations.csv")
        obs_df.to_csv(obs_file, index=False)

        # Persist aggregated segment-level datasets when requested
        if self.use_segment_data or self.users_per_segment_per_iteration is not None:
            self._save_segment_data(
                iteration,
                iteration_dir,
                all_segments,
                targeted_segments,
                obs_df,
                self.current_action_bank
            )

        # Save as JSON for detailed analysis
        obs_json_file = os.path.join(observations_dir, "observations.json")
        with open(obs_json_file, 'w') as f:
            json.dump(obs_data, f, indent=2)
        
        # 6. Save current action bank state
        action_bank_dir = os.path.join(iteration_dir, "action_bank")
        os.makedirs(action_bank_dir, exist_ok=True)
        
        action_bank_file = os.path.join(action_bank_dir, "action_bank.json")
        self.action_embedder.save_embedded_actions(self.current_action_bank, action_bank_file)
        
        # 7. Save company strategy state
        model_checkpoint_dir = os.path.join(iteration_dir, "model_checkpoint")
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        
        strategy_state = {
            'strategy_stats': strategy_stats,
            'exploration_rate': getattr(self.company_strategy, 'exploration_rate', 'N/A'),
            'is_trained': getattr(self.company_strategy, 'is_trained', True),
            'total_observations': len(self.company_strategy.observation_history),
            'model_type': getattr(self.company_strategy, 'model_type', self.strategy_type)
        }
        
        strategy_file = os.path.join(model_checkpoint_dir, "company_strategy.json")
        with open(strategy_file, 'w') as f:
            json.dump(strategy_state, f, indent=2)
        
        # 8. Compile iteration results
        iteration_results = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'company_metrics': {
                'total_users_targeted': len(observations),
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'cumulative_reward': strategy_stats['total_reward'],
                'cumulative_avg_reward': strategy_stats['avg_reward'],
                'targeted_user_count': len(iteration_users),
                'segment_population': total_population,
                'num_segments': len(targeted_segments)
            },
            'strategy_state': strategy_state,
            'action_bank_size': len(self.current_action_bank),
            'observations_generated': len(observations),
            'files_created': {
                'observations_csv': obs_file,
                'observations_json': obs_json_file,
                'action_bank': action_bank_file,
                'strategy_checkpoint': strategy_file,
                'segment_data_dir': os.path.join(iteration_dir, "segment_data") if (self.use_segment_data or self.users_per_segment_per_iteration is not None) else None
            }
        }
        
        # Save iteration summary
        summary_file = os.path.join(iteration_dir, "iteration_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(iteration_results, f, indent=2)
        
        # Add to overall results
        self.iteration_results.append(iteration_results)
        self.all_observations.extend(observations)
        
        print(f"Iteration {iteration} complete!")
        print(f"  Users targeted: {len(observations)}")
        print(f"  Total reward: {total_reward}")
        print(f"  Average reward: {avg_reward:.4f}")
        print(f"  Cumulative reward: {strategy_stats['total_reward']}")
        exploration_rate = strategy_stats.get('exploration_rate', 'N/A')
        if exploration_rate != 'N/A':
            print(f"  Exploration rate: {exploration_rate:.3f}")
        else:
            print(f"  Exploration rate: {exploration_rate}")
        print(f"  Results saved to: {iteration_dir}")
        
        # DEBUG: Analyze company strategy behavior
        self._debug_company_strategy_behavior(observations, iteration, iteration_users)
        
        return iteration_results
    
    def _debug_company_strategy_behavior(self, observations: List, iteration: int, iteration_users: List[MeaningfulUser]):
        """DEBUG: Analyze company strategy action selection behavior."""
        from collections import Counter
        
        print(f"\n🔍 DEBUG: Company Strategy Behavior Analysis (Iteration {iteration})")
        
        # 1. Analyze actual behavior during simulation (action usage counts)
        action_usage = Counter(obs.action_id for obs in observations)
        print(f"1. Actual action usage during simulation:")
        print(f"   Total assignments: {len(observations)}")
        for action_id, count in action_usage.most_common():
            percentage = count / len(observations) * 100
            action_text = next((a.text for a in self.current_action_bank if a.action_id == action_id), "Unknown")[:40]
            print(f"   {action_id}: {count:4d} times ({percentage:5.1f}%) - {action_text}...")
        
        # 2. Analyze final policy behavior (how it would assign all users now)
        print(f"2. Final policy assignments (if all users were processed now):")
        try:
            final_assignments = self.company_strategy.select_actions(iteration_users, self.current_action_bank)
            final_usage = Counter(final_assignments.values())
            
            for action_id, count in final_usage.most_common():
                percentage = count / len(iteration_users) * 100
                action_text = next((a.text for a in self.current_action_bank if a.action_id == action_id), "Unknown")[:40]
                print(f"   {action_id}: {count:4d} users ({percentage:5.1f}%) - {action_text}...")
                
        except Exception as e:
            print(f"   Error getting final assignments: {e}")
        
        # 3. Compare with ground truth performance
        print(f"3. Ground truth comparison:")
        conversion_rates = self.compute_ground_truth_conversion_rates(iteration_users)
        
        # Show top actions by usage vs ground truth performance
        for action_id, count in action_usage.most_common(5):
            gt_rate = conversion_rates.get(action_id, 0.0)
            actual_reward = sum(obs.reward for obs in observations if obs.action_id == action_id) / count if count > 0 else 0
            print(f"   {action_id}: Used {count:4d} times, GT conversion: {gt_rate:.3f}, Actual reward: {actual_reward:.3f}")
    
    def compute_ground_truth_conversion_rates(self, users: List[MeaningfulUser]) -> Dict[str, float]:
        """
        Compute ground truth conversion rate for each action across all users.
        
        Args:
            users: List of users to compute conversion rates for.
        
        Returns:
            Dictionary mapping action_id to conversion rate (0.0 to 1.0)
        """
        if not users or not hasattr(self, 'current_action_bank'):
            raise ValueError("Must provide users and initialize action bank before computing conversion rates")
        
        print("⚡ Computing ground truth conversion rates (vectorized)...")
        
        # Use vectorized batch calculation for massive speedup
        utility_matrix = self.ground_truth.calculate_utility_batch(users, self.current_action_bank)
        
        # Calculate conversion rates (mean utility for each action across all users)
        conversion_rates = {}
        for j, action in enumerate(self.current_action_bank):
            conversion_rate = np.mean(utility_matrix[:, j])
            conversion_rates[action.action_id] = conversion_rate
        
        return conversion_rates
    
    def print_ground_truth_analysis(self, users: List[MeaningfulUser]):
        """Print ground truth conversion rates and top performing users for all actions."""
        print("\n=== Ground Truth Action Analysis ===")
        
        conversion_rates = self.compute_ground_truth_conversion_rates(users)
        
        # Sort actions by conversion rate
        sorted_actions = sorted(conversion_rates.items(), key=lambda x: x[1], reverse=True)
            
        print(f"Conversion rates across {len(users)} users:")
        print("-" * 80)
        
        for action_id, conversion_rate in sorted_actions:
            # Find action object
            action = next((a for a in self.current_action_bank if a.action_id == action_id), None)
            if not action:
                continue
                
            action_text_short = action.text[:50] + "..." if len(action.text) > 50 else action.text
            print(f"{action_id}: {conversion_rate:.1%} | {action_text_short}")
            
            # Find top performing users for this action
            user_utilities = []
            for user in users:
                utility = self.ground_truth.calculate_utility(user, action)
                user_utilities.append((user, utility))
            
            # Sort by utility and get top 3
            user_utilities.sort(key=lambda x: x[1], reverse=True)
            top_users = user_utilities[:3]
            
            print(f"  Top performing users:")
            for i, (user, utility) in enumerate(top_users, 1):
                # Get user segment and key features
                segment = getattr(user, 'segment', 'unknown')
                features = user.feature_vector
                
                # Identify dominant user characteristics (top 2 feature values)
                feature_names = ['age', 'income', 'tech_savviness', 'price_sensitivity', 
                               'brand_loyalty', 'social_influence', 'urgency_response', 'quality_focus']
                feature_pairs = list(zip(feature_names, features))
                feature_pairs.sort(key=lambda x: x[1], reverse=True)
                top_traits = f"{feature_pairs[0][0]}={feature_pairs[0][1]:.2f}, {feature_pairs[1][0]}={feature_pairs[1][1]:.2f}"
                
                print(f"    {i}. {user.user_id} ({segment}): {utility:.3f} | {top_traits}")
            print()
        
        # Summary statistics
        rates = list(conversion_rates.values())
        avg_rate = sum(rates) / len(rates)
        min_rate = min(rates)
        max_rate = max(rates)
        
        print("-" * 80)
        print(f"Summary: Avg={avg_rate:.1%}, Min={min_rate:.1%}, Max={max_rate:.1%}")
        print(f"Rate spread: {max_rate - min_rate:.1%} (good diversity if > 20%)")
        print()
    
    def update_action_bank(self, new_action_bank: List[EmbeddedAction], iteration: int):
        """
        Update the action bank with new actions from the algorithm.
        
        Args:
            new_action_bank: New action bank from the algorithm
            iteration: Current iteration number
        """
        print(f"Updating action bank for iteration {iteration + 1}")
        print(f"  Previous bank size: {len(self.current_action_bank)}")
        print(f"  New bank size: {len(new_action_bank)}")
        
        self.current_action_bank = new_action_bank
        
        # Print ground truth analysis for the new action bank (end-of-iteration)
        print(f"\n🔄 Post-Algorithm Analysis (End of Iteration {iteration}):")
        # Load users from saved file for this iteration
        users_file = os.path.join(self.results_dir, f"iteration_{iteration}", "users", "users.json")
        if os.path.exists(users_file):
            loaded_users = self.user_generator.load_users(users_file)
            self._print_ground_truth_preferences(loaded_users)
        else:
            print("  Users file not found - skipping ground truth analysis")
        
        # Save the new action bank
        iteration_dir = os.path.join(self.results_dir, f"iteration_{iteration}")
        new_action_bank_dir = os.path.join(iteration_dir, "new_action_bank")
        os.makedirs(new_action_bank_dir, exist_ok=True)
        
        new_bank_file = os.path.join(new_action_bank_dir, "new_action_bank.json")
        self.action_embedder.save_embedded_actions(new_action_bank, new_bank_file)
        
        print(f"  New action bank saved to: {new_bank_file}")
    
    def update_action_bank_preserve_ids(self, new_action_bank: List[EmbeddedAction], iteration: int):
        """
        Update the action bank while preserving IDs for identical actions to maintain LinUCB knowledge.
        
        Args:
            new_action_bank: New action bank from the algorithm
            iteration: Current iteration number
        """
        print(f"Updating action bank for iteration {iteration + 1} (preserving IDs)")
        print(f"  Previous bank size: {len(self.current_action_bank)}")
        print(f"  New bank size: {len(new_action_bank)}")
        
        # Create mapping from action text/content to existing IDs
        existing_action_map = {}
        for action in self.current_action_bank:
            # Use action text as unique identifier (could also use embedding hash)
            existing_action_map[action.text] = action.action_id
        
        
        # Assign IDs to new actions, preserving existing IDs where possible
        id_preserved_count = 0
        id_reassigned_count = 0
        new_actions_count = 0
        next_new_id = 0
        
        # Track used IDs to avoid conflicts
        used_ids = set()
        
        for action in new_action_bank:
            if action.text in existing_action_map:
                # This action existed before - preserve its ID
                preserved_id = existing_action_map[action.text]
                action.action_id = preserved_id
                used_ids.add(preserved_id)
                id_preserved_count += 1
            else:
                # This is a new action - assign new ID
                while f"action_{next_new_id:04d}" in used_ids or f"action_{next_new_id:04d}" in existing_action_map.values():
                    next_new_id += 1
                action.action_id = f"action_{next_new_id:04d}"
                used_ids.add(action.action_id)
                next_new_id += 1
                new_actions_count += 1
        
        # Handle any remaining actions that need reassignment (should only be new actions without IDs)
        for action in new_action_bank:
            if action.action_id is None:  # FIXED: Only reassign actions with no ID
                old_id = action.action_id
                while f"action_{next_new_id:04d}" in used_ids:
                    next_new_id += 1
                new_id = f"action_{next_new_id:04d}"
                action.action_id = new_id
                used_ids.add(action.action_id)
                next_new_id += 1
                id_reassigned_count += 1
        
        print(f"  ID preservation stats:")
        print(f"    - Preserved existing IDs: {id_preserved_count}")
        print(f"    - New actions: {new_actions_count}")
        print(f"    - Reassigned IDs: {id_reassigned_count}")
        
        
        # Update the current action bank
        self.current_action_bank = new_action_bank
        
        # Save the new action bank
        iteration_dir = os.path.join(self.results_dir, f"iteration_{iteration}")
        new_action_bank_dir = os.path.join(iteration_dir, "new_action_bank")
        os.makedirs(new_action_bank_dir, exist_ok=True)
        
        new_bank_file = os.path.join(new_action_bank_dir, "new_action_bank.json")
        self.action_embedder.save_embedded_actions(new_action_bank, new_bank_file)
        
        print(f"  New action bank saved to: {new_bank_file}")
        
        # Print sample of preserved vs new action IDs for debugging
        if id_preserved_count > 0:
            preserved_actions = [a for a in new_action_bank if a.text in existing_action_map][:3]
            print(f"  Sample preserved actions:")
            for action in preserved_actions:
                print(f"    - {action.action_id}: {action.text[:50]}...")
        
        if new_actions_count > 0:
            new_actions = [a for a in new_action_bank if a.text not in existing_action_map][:3]
            print(f"  Sample new actions:")
            for action in new_actions:
                print(f"    - {action.action_id}: {action.text[:50]}...")
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get overall simulation summary."""
        if not self.iteration_results:
            return {'status': 'no_iterations_completed'}
        
        total_reward = sum(result['company_metrics']['total_reward'] 
                          for result in self.iteration_results)
        
        total_observations = sum(result['observations_generated'] 
                               for result in self.iteration_results)
        
        avg_rewards = [result['company_metrics']['avg_reward'] 
                      for result in self.iteration_results]
        
        # Calculate total users across all iterations (assuming n_users per iteration)
        total_users = len(self.iteration_results) * getattr(self, 'n_users', 0)
        
        summary = {
            'total_iterations': len(self.iteration_results),
            'total_users': total_users,
            'total_observations': total_observations,
            'total_cumulative_reward': total_reward,
            'overall_avg_reward': total_reward / total_observations if total_observations > 0 else 0,
            'avg_reward_by_iteration': avg_rewards,
            'final_exploration_rate': getattr(self.company_strategy, 'exploration_rate', 'N/A'),
            'company_learned': getattr(self.company_strategy, 'is_trained', True),
            'user_segments': 'N/A (users not stored persistently)',  # Can't calculate without persistent users
            'current_action_bank_size': len(self.current_action_bank)
        }
        
        return summary
    
    def _print_ground_truth_preferences(self, users: List[MeaningfulUser]):
        """Print comprehensive ground truth analysis - preferences, average utilities, and conversion rates."""
        if not users or not self.current_action_bank:
            return
        
        print("📊 Ground Truth Analysis:")
        print("=" * 80)
        
        # Use vectorized batch calculation for massive speedup
        utility_matrix = self.ground_truth.calculate_utility_batch(users, self.current_action_bank)
        
        # Calculate average utility for each action across all users
        avg_utilities = np.mean(utility_matrix, axis=0)
        
        # Calculate max conversion rate using utility > 0.5 threshold (removed from output)
        # conversion_matrix = (utility_matrix > 0.5).astype(int)
        # conversion_rates = np.mean(conversion_matrix, axis=0)
        
        # Find best action for each user (argmax along action dimension)
        best_action_indices = np.argmax(utility_matrix, axis=1)
        
        # Count preferences for each action
        action_preferences = {action.action_id: 0 for action in self.current_action_bank}
        
        for user_idx, best_action_idx in enumerate(best_action_indices):
            best_action_id = self.current_action_bank[best_action_idx].action_id
            action_preferences[best_action_id] += 1
        
        total_users = len(users)
        
        # Create comprehensive results for each action
        action_results = []
        for i, action in enumerate(self.current_action_bank):
            action_results.append({
                'action_id': action.action_id,
                'text': action.text,
                'avg_utility': avg_utilities[i],
                'preference_count': action_preferences[action.action_id],
                'preference_pct': (action_preferences[action.action_id] / total_users) * 100
            })
        
        # Sort by average utility (descending)
        action_results.sort(key=lambda x: x['avg_utility'], reverse=True)
        
        print("Action Performance (sorted by average utility):")
        print(f"{'Action ID':<12} {'Avg Util':<9} {'Preference':<11} {'Text':<40}")
        print("-" * 70)
        
        for result in action_results:
            action_text = result['text'][:35] + "..." if len(result['text']) > 35 else result['text']
            preference_str = f"{result['preference_count']:2d}/{total_users} ({result['preference_pct']:4.1f}%)"
            
            print(f"{result['action_id']:<12} "
                  f"{result['avg_utility']:<9.3f} "
                  f"{preference_str:<11} "
                  f"{action_text}")
        
        # Summary statistics
        print("-" * 70)
        max_avg_utility = max(result['avg_utility'] for result in action_results)
        avg_utility_overall = np.mean([result['avg_utility'] for result in action_results])
        
        print(f"Summary: Max Avg Utility: {max_avg_utility:.3f}, "
              f"Overall Avg Utility: {avg_utility_overall:.3f}")
        print("=" * 70)
        print()
    
    def save_final_results(self):
        """Save final simulation results."""
        summary = self.get_simulation_summary()
        
        final_results = {
            'simulation_summary': summary,
            'all_iterations': self.iteration_results,
            'final_timestamp': datetime.now().isoformat()
        }
        
        results_file = os.path.join(self.results_dir, "final_simulation_results.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nFinal simulation results saved to: {results_file}")
        print("Simulation Summary:")
        print(f"  Total iterations: {summary['total_iterations']}")
        print(f"  Total observations: {summary['total_observations']}")
        print(f"  Overall avg reward: {summary['overall_avg_reward']:.4f}")
        print(f"  Company strategy learned: {summary['company_learned']}")
        
        return results_file
