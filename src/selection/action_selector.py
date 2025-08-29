import numpy as np
from typing import List, Dict, Any, Optional
from ..data.entities import User, Action
from ..models.base_user_preference_model import BaseUserPreferenceModel
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
        current_user_max_rewards, current_best_actions, current_reward_matrix, current_action_stats = self._compute_user_max_rewards_debug(selected_actions, users, reward_model)
        
        # Print debug info about current action bank
        print(f"🔍 DEBUG: Current action bank analysis:")
        if selected_actions:
            print(f"  Actions in current bank: {len(selected_actions)}")
            for action_id, stats in current_action_stats.items():
                print(f"    {action_id}: {stats['action_text']}")
                print(f"      Mean reward: {stats['mean_reward']:.4f}, Users selecting this: {stats['users_best_count']}/{len(users)} ({stats['users_best_count']/len(users)*100:.1f}%)")
            print(f"  Overall user rewards: min={np.min(current_user_max_rewards):.4f}, "
                  f"max={np.max(current_user_max_rewards):.4f}, "
                  f"mean={np.mean(current_user_max_rewards):.4f}")
        else:
            print(f"  No actions in current bank - all users have 0.0 reward")
        
        for step in range(actions_to_add):
            print(f"Selection step {step + 1}/{actions_to_add}")
            start_time = time.time()
            
            best_action = None
            best_marginal_gain = float('-inf')
            
            # Evaluate each remaining action
            for i, candidate_action in tqdm(enumerate(remaining_actions), total=len(remaining_actions), desc="Adding actions"):
                marginal_gain = self.calculate_marginal_gain(
                    candidate_action, selected_actions, users, reward_model, control_action, 
                    current_user_max_rewards
                )
                
                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_action = candidate_action
                    
                if (i + 1) % 20 == 0:
                    print(f"  Evaluated {i + 1}/{len(remaining_actions)} candidates")
            
            # Only add actions with positive marginal gain
            if best_action is not None and best_marginal_gain > 0:
                selected_actions.append(best_action)
                remaining_actions.remove(best_action)
                
                # Update max rewards for next iteration (incremental update)
                new_action_rewards = self._compute_action_rewards_for_users(best_action, users, reward_model)
                current_user_max_rewards = np.maximum(current_user_max_rewards, new_action_rewards)
                
                elapsed = time.time() - start_time
                print(f"  Selected: {best_action.text[:50]}... (gain: {best_marginal_gain:.4f}, time: {elapsed:.2f}s)")
            elif best_marginal_gain <= 0:
                print(f"  Stopping early: best remaining action has negative/zero gain ({best_marginal_gain:.4f})")
                print(f"  Selected {len(selected_actions)} actions (requested {actions_to_add})")
                break
            else:
                print(f"  Warning: No valid action found in step {step + 1}")
                break
                
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
            if not current_bank:
                new_action_rewards = self._compute_action_rewards_for_users(action_to_add, users, reward_model)
                if self.value_mode == 'causal_value' and control_action is not None:
                    control_rewards = self._compute_action_rewards_for_users(control_action, users, reward_model)
                    return np.sum(new_action_rewards - control_rewards)
                else:
                    return np.sum(new_action_rewards)
            
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
                
                if self.value_mode == 'causal_value' and control_action is not None:
                    control_reward = reward_model.predict(user, control_action)
                    # Marginal gain is improvement in uplift
                    new_uplift = new_reward - control_reward
                    current_uplift = current_best - control_reward
                    marginal_gain += max(0, new_uplift - current_uplift)
                else:
                    # Marginal gain is improvement in direct reward
                    marginal_gain += max(0, new_reward - current_best)
            
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
            
            # Apply value mode
            if self.value_mode == 'causal_value' and control_action is not None:
                control_rewards = self._compute_action_rewards_for_users(control_action, users, reward_model)
                total_value = np.sum(user_max_rewards - control_rewards)
            else:
                total_value = np.sum(user_max_rewards)
                        
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
                                      reward_model: BaseUserPreferenceModel) -> tuple:
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
        for action_id in best_action_ids:
            action_stats[action_id]['users_best_count'] += 1
        
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
            
            # Calculate average value per user
            avg_value_per_user = total_value / len(users) if users else 0.0
            
            # Calculate direct assignments (each user gets their best action)
            action_usage = {}
            for user in users:
                best_reward = 0.0
                best_action_id = None
                
                for action in action_bank:
                    # Use direct prediction (no diversity penalty)
                    reward = reward_model.predict(user, action)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_action_id = action.action_id
                
                if best_action_id:
                    action_usage[best_action_id] = action_usage.get(best_action_id, 0) + 1
            
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
        
        return {
            'selected_actions': selected_actions,
            'evaluation': evaluation,
            'selection_summary': {
                'pool_size': len(action_pool),
                'selected_count': len(selected_actions),
                'users_count': len(users),
                'value_mode': self.value_mode,
                'diversity_weight': reward_model.diversity_weight
            }
        }