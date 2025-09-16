"""
Base Company Strategy Interface

This module provides a uniform interface for all company targeting strategies in the personalized targeting system.
It follows the same pattern as BaseUserPreferenceModel to ensure consistency across different strategy types.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import json

from src.data.entities import User, Action


# CompanyObservation is imported from src.simulation.ground_truth to avoid circular imports


class BaseCompanyStrategy(ABC):
    """
    Base class for all company targeting strategies in the personalized targeting system.
    
    This interface provides consistency across different strategy types including:
    - LinUCB (Linear Upper Confidence Bound)
    - Bootstrapped DQN (Deep Q-Network with uncertainty)
    - Thompson Sampling variants
    - Custom strategies
    
    The interface supports both online learning (update after each observation)
    and batch learning (update after collecting multiple observations).
    """
    
    def __init__(self, random_seed: int = 42, **kwargs):
        """
        Initialize the base company strategy.
        
        Args:
            random_seed: Random seed for reproducibility
            **kwargs: Additional strategy-specific parameters
        """
        self.random_seed = random_seed
        self.strategy_params = kwargs
        self.is_initialized = False
        self.total_observations = 0
        self.total_reward = 0.0
        self.recent_rewards = []
        
        # Set random seed
        np.random.seed(random_seed)
    
    @abstractmethod
    def select_action(self, user: 'MeaningfulUser', action_bank: List['EmbeddedAction']) -> str:
        """
        Select the best action for a given user from the available action bank.
        
        Args:
            user: User object with features and demographics
            action_bank: List of available actions with embeddings
            
        Returns:
            action_id: ID of the selected action
        """
        pass
    
    def update_strategy(self, observations):
        """
        Update the strategy based on new observation(s).
        Supports both single observation and list of observations for compatibility.
        
        Args:
            observations: Single CompanyObservation or List[CompanyObservation]
        """
        if isinstance(observations, list):
            self.update_strategy_batch(observations)
        else:
            self._update_single_observation(observations)
    
    @abstractmethod  
    def _update_single_observation(self, observation: 'CompanyObservation'):
        """
        Update the strategy based on a single observation.
        This is the method that subclasses should implement.
        
        Args:
            observation: Single observation with user, action, and reward
        """
        pass
    
    def update_strategy_batch(self, observations: List['CompanyObservation']):
        """
        Update the strategy based on multiple observations (batch update).
        Default implementation: apply individual updates sequentially.
        
        Args:
            observations: List of observations to process
        """
        for observation in observations:
            self._update_single_observation(observation)
    
    def update_strategy_list(self, observations: List['CompanyObservation']):
        """
        Update strategy with list of observations (compatibility method).
        Alias for update_strategy_batch for backward compatibility.
        
        Args:
            observations: List of observations to process  
        """
        self.update_strategy_batch(observations)
    
    def select_actions_batch(self, users: List['MeaningfulUser'], 
                           action_bank: List['EmbeddedAction']) -> List[str]:
        """
        Select actions for multiple users efficiently.
        Default implementation: apply individual selections sequentially.
        
        Args:
            users: List of User objects
            action_bank: List of available actions
            
        Returns:
            List of selected action IDs (same order as users)
        """
        selected_actions = []
        for user in users:
            action_id = self.select_action(user, action_bank)
            selected_actions.append(action_id)
        return selected_actions
    
    def select_actions(self, users: List['MeaningfulUser'], 
                      action_bank: List['EmbeddedAction']) -> Dict[str, str]:
        """
        Select actions for multiple users (compatibility method).
        
        Args:
            users: List of User objects
            action_bank: List of available actions
            
        Returns:
            Dictionary mapping user_id to selected action_id
        """
        assignments = {}
        for user in users:
            action_id = self.select_action(user, action_bank)
            assignments[user.user_id] = action_id
        return assignments
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """
        Get the current state of the strategy for saving/loading.
        
        Returns:
            Dictionary containing strategy state and parameters
        """
        state = {
            'strategy_type': self.__class__.__name__,
            'random_seed': self.random_seed,
            'is_initialized': self.is_initialized,
            'total_observations': self.total_observations,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.total_observations),
            'recent_avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'strategy_params': self.strategy_params
        }
        return state
    
    def load_strategy_state(self, state: Dict[str, Any]):
        """
        Load strategy state from a saved dictionary.
        
        Args:
            state: Dictionary containing strategy state
        """
        self.is_initialized = state.get('is_initialized', False)
        self.total_observations = state.get('total_observations', 0)
        self.total_reward = state.get('total_reward', 0.0)
        self.recent_rewards = state.get('recent_rewards', [])
        
        # Load strategy-specific parameters
        if 'strategy_params' in state:
            self.strategy_params.update(state['strategy_params'])
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the strategy.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.total_observations == 0:
            return {
                'total_observations': 0,
                'total_reward': 0.0,
                'avg_reward': 0.0,
                'recent_avg_reward': 0.0
            }
        
        return {
            'total_observations': self.total_observations,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / self.total_observations,
            'recent_avg_reward': np.mean(self.recent_rewards[-100:]) if self.recent_rewards else 0.0
        }
    
    def reset_strategy(self):
        """
        Reset the strategy to initial state (useful for new experiments).
        """
        self.is_initialized = False
        self.total_observations = 0
        self.total_reward = 0.0
        self.recent_rewards = []
        np.random.seed(self.random_seed)
    
    def save_strategy(self, filepath: str):
        """
        Save the strategy state to disk.
        
        Args:
            filepath: Path to save the strategy state
        """
        state = self.get_strategy_state()
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    @classmethod
    def load_strategy(cls, filepath: str, **kwargs):
        """
        Load a strategy from disk.
        
        Args:
            filepath: Path to the saved strategy state
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            Loaded strategy instance
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Extract strategy parameters
        strategy_params = state.get('strategy_params', {})
        strategy_params.update(kwargs)
        
        # Create new instance
        instance = cls(**strategy_params)
        instance.load_strategy_state(state)
        
        return instance
    
    def _track_observation(self, observation: 'CompanyObservation'):
        """
        Track observation for performance statistics.
        
        Args:
            observation: Observation to track
        """
        self.total_observations += 1
        self.total_reward += observation.reward
        self.recent_rewards.append(observation.reward)
        
        # Keep recent rewards list bounded
        if len(self.recent_rewards) > 1000:
            self.recent_rewards = self.recent_rewards[-500:]
    
    def _create_context_features(self, user: 'MeaningfulUser', action: 'EmbeddedAction') -> np.ndarray:
        """
        Create context features for user-action pair (common utility method).
        
        Args:
            user: User object with features
            action: Action object with embedding
            
        Returns:
            Combined feature vector
        """
        user_features = user.feature_vector
        action_embedding = action.embedding
        
        # Calculate norms
        user_norm = np.linalg.norm(user_features)
        action_norm = np.linalg.norm(action_embedding)
        
        # Combine features: user + action + norms
        combined_features = np.concatenate([
            user_features,
            action_embedding,
            [user_norm, action_norm]
        ])
        
        return combined_features
    
    def __str__(self):
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(seed={self.random_seed}, params={self.strategy_params})"
    
    def __repr__(self):
        """String representation of the strategy."""
        return self.__str__()
