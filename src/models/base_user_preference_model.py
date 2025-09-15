"""
Base User Preference Model Interface

This module provides a uniform interface for all user preference models in the personalized targeting system.
It is based on the BaseUserPreferenceModel from test_preference_model.py and provides
consistent methods for training, prediction, and evaluation across all model types.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..data.entities import User, Action


class BaseUserPreferenceModel(ABC):
    """
    Base class for all user preference models in the personalized targeting system.
    
    This interface provides consistency across different model types including:
    - Neural networks
    - Doubly-robust models
    - LightGBM models
    - Gaussian processes
    - Custom models
    
    The interface supports both parametric models (neural nets) and non-parametric models (GPs)
    that may not require explicit training.
    """
    
    def __init__(self, 
                 diversity_weight: float = 0.15,
                 random_seed: int = 42,
                 **kwargs):
        """
        Initialize the base user preference model.
        
        Args:
            diversity_weight: Weight for diversity penalty in action selection
            random_seed: Random seed for reproducibility
            **kwargs: Additional model-specific parameters
        """
        self.diversity_weight = diversity_weight
        self.random_seed = random_seed
        self.is_fitted = False
        self.requires_training = True  # Set to False for non-parametric models like GP
        
        # Store additional parameters
        self.model_params = kwargs
        
        # Set random seed
        np.random.seed(random_seed)
    
    @abstractmethod
    def fit(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit the user preference model on observation data.
        For parametric models: train parameters
        For non-parametric models: store training data for inference
        
        Args:
            observations_df: DataFrame with columns:
                - user_features: numpy array of user characteristics (8-dim)
                - action_embedding: numpy array of action embeddings (1536-dim)
                - action_text: original marketing content text
                - reward/outcome: binary outcome (0/1)
                - (optional) iteration: for temporal splitting
        
        Returns:
            Dict with fitting metrics (loss, accuracy, etc. or fitting info)
        """
        pass
    
    def train(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        """Backward compatibility alias for fit()."""
        return self.fit(observations_df)
    
    @abstractmethod
    def predict(self, user: User, action: Action) -> float:
        """
        Predict user preference probability for a user-action pair.
        
        Args:
            user: User object with features attribute
            action: Action object with embedding and text attributes
            
        Returns:
            Predicted probability of positive outcome (0.0 to 1.0)
        """
        pass
    
    def predict_with_text(self, user: User, action_text: str) -> float:
        """
        Predict user preference using action text directly via OpenAI embeddings.
        Requires OPENAI_API_KEY to be set. Creates an Action with the true embedding.
        """
        # Lazy import to avoid hard dependency at module import time
        try:
            from simulation.action_embedder import OpenAIActionEmbedder
        except Exception as e:
            raise RuntimeError(f"OpenAI embedder not available: {e}")

        embedder = OpenAIActionEmbedder()
        embedding = embedder._get_openai_embedding(action_text)
        action = Action(action_id="temp", text=action_text, embedding=embedding)
        return self.predict(user, action)
    
    def predict_batch(self, users: List[User], actions: List[Action]) -> np.ndarray:
        """
        Predict user preferences for multiple user-action pairs efficiently.
        
        Args:
            users: List of User objects
            actions: List of Action objects (same length as users)
            
        Returns:
            Array of predicted probabilities
        """
        predictions = []
        for user, action in zip(users, actions):
            predictions.append(self.predict(user, action))
        return np.array(predictions)

    # --- Uncertainty-aware prediction (optional) ---
    def predict_with_uncertainty(self, user: User, action: Action) -> tuple:
        """
        Optional: Predict (mean, uncertainty) for a user-action pair.
        Default implementation returns deterministic prediction with zero uncertainty.
        Subclasses that support uncertainty should override this.
        """
        mean = self.predict(user, action)
        return float(mean), 0.0

    def predict_batch_with_uncertainty(self, users: List[User], actions: List[Action]) -> tuple:
        """
        Optional batched uncertainty predictions.
        Returns (means: np.ndarray, stds: np.ndarray) both shaped (N,).
        Default falls back to per-item calls.
        """
        means = []
        stds = []
        for user, action in zip(users, actions):
            m, s = self.predict_with_uncertainty(user, action)
            means.append(m)
            stds.append(s)
        return np.array(means), np.array(stds)
    
    def predict_with_diversity_penalty(self, user: User, action: Action, 
                                     current_bank: List[Action]) -> float:
        """
        Predict user preference with diversity penalty.
        
        Args:
            user: User object
            action: Action to evaluate
            current_bank: Current actions in the bank
            
        Returns:
            Adjusted preference score with diversity penalty
        """
        base_preference = self.predict(user, action)
        
        if not current_bank:
            return base_preference
        
        # Calculate diversity penalty
        diversity_penalty = self._calculate_diversity_penalty(action, current_bank)
        
        # Apply penalty
        adjusted_preference = base_preference - self.diversity_weight * diversity_penalty
        
        return max(0.0, adjusted_preference)  # Ensure non-negative
    
    def predict_batch_with_diversity(self, users: List[User], actions: List[Action], 
                                   current_bank: List[Action] = None) -> np.ndarray:
        """
        Predict user preferences for multiple user-action pairs with optional diversity penalty.
        
        Args:
            users: List of User objects
            actions: List of Action objects
            current_bank: Optional current action bank for diversity penalty
            
        Returns:
            2D array of shape (len(users), len(actions)) with predicted scores
        """
        if not self.is_fitted and self.requires_training:
            raise ValueError("Model must be trained before making predictions")
            
        scores = np.zeros((len(users), len(actions)))
        
        for i, user in enumerate(users):
            for j, action in enumerate(actions):
                if current_bank is not None:
                    scores[i, j] = self.predict_with_diversity_penalty(user, action, current_bank)
                else:
                    scores[i, j] = self.predict(user, action)
                
        return scores
    
    def _calculate_diversity_penalty(self, action: Action, current_bank: List[Action]) -> float:
        """
        Calculate diversity penalty based on similarity to existing actions.
        
        Args:
            action: Action to evaluate
            current_bank: Current actions in the bank
            
        Returns:
            Diversity penalty score (higher = more similar)
        """
        if not current_bank:
            return 0.0
        
        similarities = []
        
        for bank_action in current_bank:
            # Embedding similarity (cosine similarity)
            embedding_sim = np.dot(action.embedding, bank_action.embedding) / (
                np.linalg.norm(action.embedding) * np.linalg.norm(bank_action.embedding)
            )
            
            # Text similarity (simple Jaccard similarity on words)
            words_action = set(action.text.lower().split())
            words_bank = set(bank_action.text.lower().split())
            
            if len(words_action) == 0 and len(words_bank) == 0:
                text_sim = 1.0
            elif len(words_action.union(words_bank)) == 0:
                text_sim = 0.0
            else:
                text_sim = len(words_action.intersection(words_bank)) / len(words_action.union(words_bank))
            
            # Combined similarity
            combined_sim = 0.7 * embedding_sim + 0.3 * text_sim
            similarities.append(max(0, combined_sim))
        
        # Return maximum similarity (highest penalty for most similar action)
        return max(similarities) if similarities else 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the trained model."""
        if not self.is_fitted:
            return {"status": "not_trained"}
            
        info = {
            "status": "trained",
            "model_type": self.__class__.__name__,
            "diversity_weight": self.diversity_weight,
            "requires_training": self.requires_training,
            "random_seed": self.random_seed
        }
        
        # Add model-specific parameters
        info.update(self.model_params)
        
        return info
    
    @property
    def is_trained(self):
        """Backward compatibility property."""
        return self.is_fitted
    
    def save_model(self, filepath: str):
        """Save the trained model to disk (optional, can be overridden)."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model from disk (optional, can be overridden)."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class UserPreferenceModelWrapper(BaseUserPreferenceModel):
    """
    Wrapper for existing reward/preference models to provide uniform interface.
    This can be used to wrap legacy models that don't inherit from BaseUserPreferenceModel.
    """
    
    def __init__(self, wrapped_model, **kwargs):
        """
        Initialize wrapper with an existing model.
        
        Args:
            wrapped_model: Existing model instance
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.wrapped_model = wrapped_model
        
        # Copy attributes from wrapped model if they exist
        if hasattr(wrapped_model, 'diversity_weight'):
            self.diversity_weight = wrapped_model.diversity_weight
        if hasattr(wrapped_model, 'is_trained'):
            self.is_fitted = wrapped_model.is_trained
        if hasattr(wrapped_model, 'is_fitted'):
            self.is_fitted = wrapped_model.is_fitted
    
    def fit(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit the wrapped model."""
        # Handle column name compatibility
        df_for_training = observations_df.copy()
        if 'reward' in df_for_training.columns and 'outcome' not in df_for_training.columns:
            df_for_training['outcome'] = df_for_training['reward']
        
        # Use the wrapped model's training method
        if hasattr(self.wrapped_model, 'fit'):
            result = self.wrapped_model.fit(df_for_training)
        elif hasattr(self.wrapped_model, 'train'):
            result = self.wrapped_model.train(df_for_training)
        else:
            raise ValueError("Wrapped model has no 'fit' or 'train' method")
        
        # Update fitted status
        if hasattr(self.wrapped_model, 'is_trained'):
            self.is_fitted = self.wrapped_model.is_trained
        elif hasattr(self.wrapped_model, 'is_fitted'):
            self.is_fitted = self.wrapped_model.is_fitted
        else:
            self.is_fitted = True  # Assume training was successful
        
        return result
    
    def predict(self, user: User, action: Action) -> float:
        """Predict using the wrapped model."""
        return self.wrapped_model.predict(user, action)
    
    def predict_batch(self, users: List[User], actions: List[Action]) -> np.ndarray:
        """Predict batch using wrapped model if available, otherwise use base implementation."""
        if hasattr(self.wrapped_model, 'predict_batch'):
            return self.wrapped_model.predict_batch(users, actions)
        else:
            return super().predict_batch(users, actions)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info from wrapped model if available."""
        base_info = super().get_model_info()
        
        if hasattr(self.wrapped_model, 'get_model_info'):
            wrapped_info = self.wrapped_model.get_model_info()
            base_info.update(wrapped_info)
        
        base_info['wrapped_model_type'] = type(self.wrapped_model).__name__
        
        return base_info
