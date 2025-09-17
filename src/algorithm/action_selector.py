import numpy as np
from typing import List, Dict, Any, Optional
from src.data.entities import User, Action
from src.algorithm.models.base_user_preference_model import BaseUserPreferenceModel
import time
from tqdm import tqdm

class ActionSelector:
    """
    Core component that selects K actions from a candidate pool using greedy marginal gain algorithm.
    Includes diversity penalty in reward calculation.
    """
    
    def __init__(self, value_mode: str = 'direct_reward'):
        """
        Initialize ActionSelector.
        
        Args:
            value_mode: 'direct_reward' or 'causal_value' (uplift)
        """
        self.value_mode = value_mode
        if value_mode not in ['direct_reward', 'causal_value']:
            raise ValueError(f"Invalid value_mode: {value_mode}")
            
    def select_actions(self, action_pool: List[Action], action_bank_size_per_iter: int, users: List[User], 
                      reward_model: BaseUserPreferenceModel, current_action_bank: List[Action] = None,
                      control_action: Optional[Action] = None) -> List[Action]:
        """
        Select new actions from the pool using greedy marginal gain algorithm with diversity penalty.
        Start with current action bank and add the best additional actions for this iteration.
        
        Args:
            action_pool: Candidate actions to select from
            action_bank_size_per_iter: Number of new actions to add in this iteration
            users: Users for evaluation
            reward_model: Trained reward model (with diversity penalty) 
            current_action_bank: Current action bank to start with (from previous iterations)
            control_action: Control action for uplift calculation (required for causal_value mode)
            
        Returns:
            List of selected actions forming the new Action Bank (current + new actions)
        """
        if action_bank_size_per_iter <= 0:
            # Return current action bank unchanged when no new actions needed
            if current_action_bank is None:
                return []
            else:
                print("No new actions to add - returning current action bank unchanged")
                return current_action_bank.copy()
            
        if action_bank_size_per_iter >= len(action_pool):
            return action_pool.copy()
            
        if self.value_mode == 'causal_value' and control_action is None:
            raise ValueError("control_action required for causal_value mode")
        
        # Start with current action bank (from previous iterations)
        if current_action_bank is None:
            current_action_bank = []
        
        selected_actions = current_action_bank.copy()

        user_weights = np.array([getattr(u, 'weight', 1.0) for u in users], dtype=float)
        total_weight = float(user_weights.sum()) if len(user_weights) else 0.0

        # Remove actions already in current bank from the pool
        remaining_actions = [a for a in action_pool if a.action_id not in {act.action_id for act in selected_actions}]
        
        actions_to_add = action_bank_size_per_iter
        
        print(f"Starting greedy selection: current bank has {len(selected_actions)} actions")
        print(f"Adding {actions_to_add} more actions from pool of {len(remaining_actions)} candidates")
        
        if actions_to_add == 0:
            print("No new actions to add this iteration")
            return selected_actions
        
        # Pre-compute max rewards for current bank (optimization) - DEBUG VERSION
        # current_user_max_rewards = self._compute_user_max_rewards(selected_actions, users, reward_model)
        current_user_max_rewards, current_best_actions, current_reward_matrix, current_action_stats = self._compute_user_max_rewards_debug(selected_actions, users, reward_model, user_weights)
        
        # Print debug info about current action bank
        print(f"Current action bank analysis:")
        if selected_actions:
            print(f"  Actions in current bank: {len(selected_actions)} (weighted users: {total_weight:.2f})")
            for action_id, stats in current_action_stats.items():
                print(f"    {action_id}: {stats['action_text']}")
                best_weight = stats['users_best_count']
                share = (best_weight / total_weight * 100) if total_weight else 0.0
                print(f"      Mean reward: {stats['mean_reward']:.4f}, Weighted users selecting this: {best_weight:.2f}/{total_weight:.2f} ({share:.1f}%)")
            print(f"  Overall user rewards: min={np.min(current_user_max_rewards):.4f}, "
                  f"max={np.max(current_user_max_rewards):.4f}, "
                  f"mean={np.mean(current_user_max_rewards):.4f}")
        else:
            print(f"  No actions in current bank - all users have 0.0 reward")
        
        # Precompute reward matrix for remaining candidates: shape (n_users, n_remaining)
        n_users = len(users)
        n_remaining = len(remaining_actions)
        if n_remaining == 0:
            print("No candidates available to add; returning current action bank")
            return selected_actions

        reward_matrix = np.zeros((n_users, n_remaining), dtype=float)
        for j, action in enumerate(remaining_actions):
            reward_matrix[:, j] = self._compute_action_rewards_for_users(action, users, reward_model)

        # Active mask to track which candidates are still available
        active = np.ones(n_remaining, dtype=bool)

        for step in range(actions_to_add):
            print(f"Selection step {step + 1}/{actions_to_add}")
            step_start = time.time()

            # Vectorized marginal gains for all remaining candidates
            improvement = reward_matrix - current_user_max_rewards[:, None]
            np.maximum(improvement, 0.0, out=improvement)
            weighted_improvement = improvement
            if len(user_weights):
                weighted_improvement = improvement * user_weights[:, None]
            gains = weighted_improvement.sum(axis=0)
            gains[~active] = -np.inf

            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]

            if not np.isfinite(best_gain) or best_gain <= 0:
                print(f"  Stopping early: best remaining action has non-positive gain ({best_gain:.4f})")
                print(f"  Selected {len(selected_actions)} actions (requested {actions_to_add})")
                break

            best_action = remaining_actions[best_idx]
            selected_actions.append(best_action)
            active[best_idx] = False

            # Update per-user max rewards incrementally
            current_user_max_rewards = np.maximum(current_user_max_rewards, reward_matrix[:, best_idx])

            elapsed = time.time() - step_start
            print(f"  Selected: {best_action.text[:50]}... (weighted gain: {best_gain:.4f}, time: {elapsed:.2f}s)")

        print(f"Selection complete. Selected {len(selected_actions)} actions.")
        return selected_actions
    
    def calculate_marginal_gain(self, action_to_add: Action, current_bank: List[Action], 
                              users: List[User], reward_model: BaseUserPreferenceModel,
                              control_action: Optional[Action] = None,
                              current_user_max_rewards: np.ndarray = None) -> float:
        """
        Calculate marginal gain of adding an action to the current bank.
        Optimized version that avoids redundant calculations.
        
        Args:
            action_to_add: Candidate action to evaluate
            current_bank: Current action bank
            users: Users for evaluation
            reward_model: Trained reward model (using direct prediction)
            control_action: Control action for uplift calculation
            current_user_max_rewards: Pre-computed max rewards for each user from current bank
            
        Returns:
            Marginal gain value
        """
        try:
            # If no current bank, just compute total value of new action
            user_weights = np.array([getattr(u, 'weight', 1.0) for u in users], dtype=float)
            if not current_bank:
                new_action_rewards = self._compute_action_rewards_for_users(action_to_add, users, reward_model)
                if self.value_mode == 'causal_value' and control_action is not None:
                    control_rewards = self._compute_action_rewards_for_users(control_action, users, reward_model)
                    return float(np.sum((new_action_rewards - control_rewards) * user_weights))
                else:
                    return float(np.sum(new_action_rewards * user_weights))
            
            # Compute rewards for new action across all users
            new_action_rewards = self._compute_action_rewards_for_users(action_to_add, users, reward_model)
            
            # If current_user_max_rewards not provided, compute it
            if current_user_max_rewards is None:
                current_user_max_rewards = self._compute_user_max_rewards(current_bank, users, reward_model)
            
            # Calculate marginal gain: sum of improvements for each user
            marginal_gain = 0.0
            for i, user in enumerate(users):
                current_best = current_user_max_rewards[i]
                new_reward = new_action_rewards[i]
                weight = user_weights[i] if i < len(user_weights) else 1.0
                
                if self.value_mode == 'causal_value' and control_action is not None:
                    control_reward = reward_model.predict(user, control_action)
                    # Marginal gain is improvement in uplift
                    new_uplift = new_reward - control_reward
                    current_uplift = current_best - control_reward
                    marginal_gain += weight * max(0, new_uplift - current_uplift)
                else:
                    # Marginal gain is improvement in direct reward
                    marginal_gain += weight * max(0, new_reward - current_best)

            return marginal_gain
            
        except Exception as e:
            print(f"Error calculating marginal gain: {e}")
            return 0.0
    
    def _calculate_bank_value(self, action_bank: List[Action], users: List[User], 
                            reward_model: BaseUserPreferenceModel, control_action: Optional[Action] = None) -> float:
        """
        Calculate total value of an action bank using direct reward-based assignment.
        Optimized version that computes rewards in batches.
        
        Args:
            action_bank: List of actions in the bank
            users: Users for evaluation
            reward_model: Trained reward model (using direct prediction)
            control_action: Control action for uplift calculation
            
        Returns:
            Total expected value of the action bank
        """
        if not action_bank or not users:
            return 0.0
            
        try:
            # Compute reward matrix: (users, actions)
            reward_matrix = np.zeros((len(users), len(action_bank)))

            # For each action, compute rewards for all users at once
            for j, action in enumerate(action_bank):
                action_rewards = self._compute_action_rewards_for_users(action, users, reward_model)
                reward_matrix[:, j] = action_rewards

            # Find best action for each user
            user_max_rewards = np.max(reward_matrix, axis=1)
            user_weights = np.array([getattr(u, 'weight', 1.0) for u in users], dtype=float)

            # Apply value mode
            if self.value_mode == 'causal_value' and control_action is not None:
                control_rewards = self._compute_action_rewards_for_users(control_action, users, reward_model)
                total_value = np.sum((user_max_rewards - control_rewards) * user_weights)
            else:
                total_value = np.sum(user_max_rewards * user_weights)

            return float(total_value)
            
        except Exception as e:
            print(f"Error calculating bank value: {e}")
            return 0.0

    def _compute_action_rewards_for_users(self, action: Action, users: List[User], 
                                        reward_model: BaseUserPreferenceModel) -> np.ndarray:
        """
        Compute reward for a single action across all users.
        Uses direct prediction without diversity penalty.
        
        Args:
            action: Single action to evaluate
            users: List of users
            reward_model: Trained reward model
            
        Returns:
            Array of rewards for each user
        """
        rewards = np.zeros(len(users))
        for i, user in enumerate(users):
            # Use direct prediction (no diversity penalty)
            rewards[i] = reward_model.predict(user, action)
        return rewards
    
    def _compute_user_max_rewards(self, action_bank: List[Action], users: List[User],
                                reward_model: BaseUserPreferenceModel) -> np.ndarray:
        """
        Compute maximum reward for each user across all actions in the bank.
        
        Args:
            action_bank: List of actions in the bank
            users: List of users
            reward_model: Trained reward model
            
        Returns:
            Array of maximum rewards for each user
        """
        if not action_bank:
            return np.zeros(len(users))
        
        # Compute reward matrix: (users, actions)
        reward_matrix = np.zeros((len(users), len(action_bank)))
        
        # For each action, compute rewards for all users
        for j, action in enumerate(action_bank):
            action_rewards = self._compute_action_rewards_for_users(action, users, reward_model)
            reward_matrix[:, j] = action_rewards
        
        # Return maximum reward for each user
        return np.max(reward_matrix, axis=1)
    
    def _compute_user_max_rewards_debug(self, action_bank: List[Action], users: List[User],
                                      reward_model: BaseUserPreferenceModel,
                                      user_weights: Optional[np.ndarray] = None) -> tuple:
        """
        DEBUG VERSION: Compute maximum reward for each user and track best actions.
        
        Args:
            action_bank: List of actions in the bank
            users: List of users
            reward_model: Trained reward model
            
        Returns:
            Tuple of (max_rewards_array, best_action_ids_array, reward_matrix, action_stats)
        """
        if not action_bank:
            return (np.zeros(len(users)), 
                   np.array([None] * len(users)), 
                   np.zeros((len(users), 0)),
                   {})
        
        # Compute reward matrix: (users, actions)
        reward_matrix = np.zeros((len(users), len(action_bank)))
        action_stats = {}
        
        # For each action, compute rewards for all users
        for j, action in enumerate(action_bank):
            action_rewards = self._compute_action_rewards_for_users(action, users, reward_model)
            reward_matrix[:, j] = action_rewards
            
            # Track action statistics
            action_stats[action.action_id] = {
                'action_text': action.text[:50] + "..." if len(action.text) > 50 else action.text,
                'mean_reward': float(np.mean(action_rewards)),
                'max_reward': float(np.max(action_rewards)),
                'min_reward': float(np.min(action_rewards)),
                'std_reward': float(np.std(action_rewards)),
                'users_best_count': 0  # Will be filled later
            }
        
        # Find best action for each user
        best_action_indices = np.argmax(reward_matrix, axis=1)
        max_rewards = np.max(reward_matrix, axis=1)
        best_action_ids = [action_bank[idx].action_id for idx in best_action_indices]
        
        # Count how many users each action is best for
        if user_weights is None:
            user_weights = np.ones(len(users))
        for action_id, weight in zip(best_action_ids, user_weights):
            action_stats[action_id]['users_best_count'] += float(weight)

        return max_rewards, np.array(best_action_ids), reward_matrix, action_stats

    
    
    def evaluate_action_bank(self, action_bank: List[Action], users: List[User], 
                            reward_model: BaseUserPreferenceModel, control_action: Optional[Action] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a selected action bank using direct reward-based assignment.
        
        Args:
            action_bank: Selected action bank
            users: Users for evaluation
            reward_model: Trained reward model (with diversity penalty)
            control_action: Control action for uplift calculation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not action_bank or not users:
            return {'total_value': 0.0, 'avg_value_per_user': 0.0, 'coverage': 1.0}
            
        try:
            # Calculate total value using direct reward-based assignment
            total_value = self._calculate_bank_value(action_bank, users, reward_model, control_action)

            user_weights = np.array([getattr(u, 'weight', 1.0) for u in users], dtype=float)
            total_weight = float(user_weights.sum()) if len(user_weights) else float(len(users))

            # Calculate average value per (weighted) user
            avg_value_per_user = total_value / total_weight if total_weight else 0.0

            # Calculate direct assignments (each user gets their best action)
            action_usage: Dict[str, float] = {}
            for user, weight in zip(users, user_weights):
                best_reward = 0.0
                best_action_id = None

                for action in action_bank:
                    # Use direct prediction (no diversity penalty)
                    reward = reward_model.predict(user, action)

                    if reward > best_reward:
                        best_reward = reward
                        best_action_id = action.action_id

                if best_action_id:
                    action_usage[best_action_id] = action_usage.get(best_action_id, 0.0) + float(weight)
            
            # Coverage is always 100% with direct assignment (everyone gets assigned)
            coverage = 1.0
            
            # Calculate diversity (embedding-based)
            embedding_diversity = self._calculate_embedding_diversity(action_bank)
            
            # Calculate text diversity (simple word-based)
            text_diversity = self._calculate_text_diversity(action_bank)
            
            return {
                'total_value': total_value,
                'avg_value_per_user': avg_value_per_user,
                'coverage': coverage,
                'embedding_diversity': embedding_diversity,
                'text_diversity': text_diversity,
                'action_count': len(action_bank),
                'action_usage': action_usage,
                'total_weight': total_weight,
                'value_mode': self.value_mode
            }
            
        except Exception as e:
            print(f"Error evaluating action bank: {e}")
            return {'total_value': 0.0, 'avg_value_per_user': 0.0, 'coverage': 0.0, 'error': str(e)}
    
    def _calculate_embedding_diversity(self, actions: List[Action]) -> float:
        """Calculate diversity based on action embeddings."""
        if len(actions) <= 1:
            return 1.0
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                sim = np.dot(actions[i].embedding, actions[j].embedding) / (
                    np.linalg.norm(actions[i].embedding) * np.linalg.norm(actions[j].embedding)
                )
                similarities.append(max(0, sim))  # Only positive similarities
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1.0 - avg_similarity
    
    def _calculate_text_diversity(self, actions: List[Action]) -> float:
        """Calculate diversity based on action text."""
        if len(actions) <= 1:
            return 1.0
        
        # Calculate pairwise text similarities (Jaccard similarity)
        similarities = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                words_i = set(actions[i].text.lower().split())
                words_j = set(actions[j].text.lower().split())
                
                if len(words_i) == 0 and len(words_j) == 0:
                    sim = 1.0
                elif len(words_i.union(words_j)) == 0:
                    sim = 0.0
                else:
                    sim = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                
                similarities.append(sim)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1.0 - avg_similarity
    
    def select_actions_with_evaluation(self, action_pool: List[Action], action_bank_size_per_iter: int, users: List[User], 
                                     reward_model: BaseUserPreferenceModel, current_action_bank: List[Action] = None,
                                     control_action: Optional[Action] = None) -> Dict[str, Any]:
        """
        Select actions and return both the selection and evaluation metrics.
        
        Returns:
            Dictionary containing 'selected_actions' and 'evaluation'
        """
        selected_actions = self.select_actions(action_pool, action_bank_size_per_iter, users, reward_model, current_action_bank, control_action)
        evaluation = self.evaluate_action_bank(selected_actions, users, reward_model, control_action)
        # Compute uncertainty summaries if the model supports it
        uncertainty = self._summarize_uncertainty(selected_actions, users, reward_model)

        total_weight = float(sum(getattr(u, 'weight', 1.0) for u in users)) if users else 0.0
        
        return {
            'selected_actions': selected_actions,
            'evaluation': evaluation,
            'uncertainty': uncertainty,
            'selection_summary': {
                'pool_size': len(action_pool),
                'selected_count': len(selected_actions),
                'users_count': len(users),
                'total_weight': total_weight,
                'value_mode': self.value_mode,
                'diversity_weight': reward_model.diversity_weight
            }
        }

    def _summarize_uncertainty(self, selected_actions: List[Action], users: List[User],
                               reward_model: BaseUserPreferenceModel, max_users: int = 200) -> Dict[str, Any]:
        """
        If available, compute simple uncertainty summaries for selected actions.
        Returns a lightweight dict suitable for logging without affecting selection.
        """
        # Detect if uncertainty is supported
        supports_uncertainty = hasattr(reward_model, 'predict_with_uncertainty')
        if not supports_uncertainty or not selected_actions or not users:
            return {
                'supported': False,
                'per_action': [],
                'sampled_users': 0,
                'note': 'Model does not provide uncertainty or empty inputs.'
            }

        # Limit users for efficiency
        sampled_users = users[:max(1, min(max_users, len(users)))]

        per_action = []
        for action in selected_actions:
            stds = []
            for u in sampled_users:
                # Fallback is handled by base class which returns (mean, 0.0)
                _, s = reward_model.predict_with_uncertainty(u, action)
                stds.append(float(s))

            if stds:
                arr = np.array(stds)
                per_action.append({
                    'action_id': action.action_id,
                    'mean_std': float(arr.mean()),
                    'median_std': float(np.median(arr)),
                    'max_std': float(arr.max()),
                    'n_users': len(stds)
                })
            else:
                per_action.append({
                    'action_id': action.action_id,
                    'mean_std': 0.0,
                    'median_std': 0.0,
                    'max_std': 0.0,
                    'n_users': 0
                })

        return {
            'supported': True,
            'per_action': per_action,
            'sampled_users': len(sampled_users),
            'note': 'Uncertainty not used for selection yet.'
        }
