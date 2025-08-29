"""
LinUCB Strategy Implementation

Linear Upper Confidence Bound (LinUCB) contextual bandit algorithm for personalized targeting.
Maintains a linear model for each action and uses confidence intervals for exploration-exploitation balance.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .base_strategy import BaseCompanyStrategy


class LinUCBStrategy(BaseCompanyStrategy):
    """
    Linear Upper Confidence Bound (LinUCB) contextual bandit algorithm.
    
    Maintains a linear model for each action and uses confidence intervals
    for exploration-exploitation balance with the formula:
    
    UCB(x,a) = θ_a^T x + α * sqrt(x^T A_a^(-1) x)
    
    Where:
    - θ_a: parameter vector for action a
    - x: context features (user + action)
    - A_a: covariance matrix for action a
    - α: confidence parameter (higher = more exploration)
    """
    
    def __init__(self, alpha: float = 1.0, use_pca: bool = True, 
                 pca_components: int = 64, random_seed: int = 42):
        """
        Initialize LinUCB strategy.
        
        Args:
            alpha: Confidence parameter (higher = more exploration)
            use_pca: Whether to use PCA for action embedding dimensionality reduction
            pca_components: Number of PCA components (if use_pca=True)
            random_seed: Random seed for reproducibility
        """
        super().__init__(random_seed=random_seed, alpha=alpha, 
                         use_pca=use_pca, pca_components=pca_components)
        
        self.alpha = alpha
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # LinUCB parameters per action
        self.A = {}  # action_id -> A matrix (d x d)
        self.b = {}  # action_id -> b vector (d x 1)
        self.theta = {}  # action_id -> theta vector (d x 1)
        self.A_inv_cache = {}  # Cached inverse matrices for speed
        self.cache_valid = {}  # Track if cached inverse is valid
        
        # PCA components for dimensionality reduction
        self.pca_model = None
        self.pca_fitted = False
        
        self.feature_dim = None
        
        # Observation history for tracking
        self.observation_history = []
        
        print(f"LinUCB strategy initialized with alpha={alpha}, use_pca={use_pca}, pca_components={pca_components}")
    
    def _initialize_action(self, action_id: str, feature_dim: int):
        """Initialize LinUCB parameters for a new action."""
        self.A[action_id] = np.eye(feature_dim)  # Identity matrix
        self.b[action_id] = np.zeros(feature_dim)
        self.theta[action_id] = np.zeros(feature_dim)
        self.cache_valid[action_id] = False  # Cache not valid initially
    
    def _get_cached_inverse(self, action_id: str):
        """Get cached inverse matrix or compute if not valid."""
        if not self.cache_valid.get(action_id, False):
            self.A_inv_cache[action_id] = np.linalg.pinv(self.A[action_id])
            self.cache_valid[action_id] = True
        return self.A_inv_cache[action_id]
    
    def _fit_pca_if_needed(self, action_bank: List['EmbeddedAction']):
        """Fit PCA model on action embeddings if not already fitted."""
        if not self.use_pca or self.pca_fitted:
            return
        
        try:
            from sklearn.decomposition import PCA
            
            # Collect all action embeddings
            embeddings = np.array([action.embedding for action in action_bank])
            n_samples, n_features = embeddings.shape
            
            # Adjust PCA components to not exceed available data
            max_components = min(n_samples, n_features)
            actual_components = min(self.pca_components, max_components)
            
            if actual_components < self.pca_components:
                print(f"Warning: Reducing PCA components from {self.pca_components} to {actual_components} "
                      f"(limited by {n_samples} samples and {n_features} features)")
            
            # Fit PCA model with adjusted components
            self.pca_model = PCA(n_components=actual_components, random_state=self.random_seed)
            self.pca_model.fit(embeddings)
            self.pca_fitted = True
            
            explained_var = np.sum(self.pca_model.explained_variance_ratio_)
            print(f"PCA fitted: {n_features} -> {actual_components} dims, "
                  f"explained variance: {explained_var:.3f}")
        
        except ImportError:
            print("Warning: sklearn not available, disabling PCA")
            self.use_pca = False
    
    def _reduce_action_embedding(self, action_embedding: np.ndarray) -> np.ndarray:
        """Reduce action embedding dimensionality using PCA if enabled."""
        if not self.use_pca or not self.pca_fitted:
            return action_embedding
        
        # Transform using fitted PCA
        embedding_reshaped = action_embedding.reshape(1, -1)
        reduced_embedding = self.pca_model.transform(embedding_reshaped).flatten()
        return reduced_embedding
    
    def _create_context_features(self, user: 'MeaningfulUser', action: 'EmbeddedAction') -> np.ndarray:
        """
        Create context features for user-action pair with optional PCA reduction.
        
        Args:
            user: User object with features
            action: Action object with embedding
            
        Returns:
            Combined feature vector
        """
        user_features = user.feature_vector
        action_embedding = self._reduce_action_embedding(action.embedding)
        
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
    
    def select_action(self, user: 'MeaningfulUser', action_bank: List['EmbeddedAction']) -> str:
        """
        Select action using LinUCB algorithm.
        
        Args:
            user: User object with features
            action_bank: List of available actions
            
        Returns:
            Selected action ID
        """
        if not action_bank:
            raise ValueError("Action bank cannot be empty")
        
        # Fit PCA if needed and not already fitted
        self._fit_pca_if_needed(action_bank)
        
        best_action_id = None
        best_ucb_value = -np.inf
        
        for action in action_bank:
            action_id = action.action_id
            
            # Create context features
            context = self._create_context_features(user, action)
            
            # Initialize action parameters if new
            if action_id not in self.A:
                if self.feature_dim is None:
                    self.feature_dim = len(context)
                    self.is_initialized = True
                self._initialize_action(action_id, self.feature_dim)
            
            # Get cached inverse matrix
            A_inv = self._get_cached_inverse(action_id)
            
            # Calculate theta (parameter estimate)
            self.theta[action_id] = A_inv @ self.b[action_id]
            
            # Calculate UCB value: θ^T x + α * sqrt(x^T A^(-1) x)
            mean_reward = self.theta[action_id] @ context
            confidence_width = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_value = mean_reward + confidence_width
            
            # Track best action
            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_action_id = action_id
        
        return best_action_id
    
    def _update_single_observation(self, observation: 'CompanyObservation'):
        """
        Update LinUCB parameters based on new observation.
        
        Args:
            observation: Observation with user, action, and reward
        """
        action_id = observation.action_id
        reward = observation.reward
        
        # Create context features (need to reconstruct from observation)
        # Note: We use the stored embeddings, applying PCA if configured
        context_features = observation.user_features
        action_embedding = self._reduce_action_embedding(observation.action_embedding)
        
        # Calculate norms
        user_norm = np.linalg.norm(context_features)
        action_norm = np.linalg.norm(action_embedding)
        
        # Combine features
        context = np.concatenate([
            context_features,
            action_embedding,
            [user_norm, action_norm]
        ])
        
        # Initialize action if not seen before
        if action_id not in self.A:
            if self.feature_dim is None:
                self.feature_dim = len(context)
                self.is_initialized = True
            self._initialize_action(action_id, self.feature_dim)
        
        # Update LinUCB parameters
        self.A[action_id] += np.outer(context, context)
        self.b[action_id] += reward * context
        
        # Invalidate cached inverse
        self.cache_valid[action_id] = False
        
        # Track observation for performance stats
        self._track_observation(observation)
        
        # Add to observation history
        self.observation_history.append(observation)
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """
        Get the current state of the LinUCB strategy.
        
        Returns:
            Dictionary containing strategy state and parameters
        """
        base_state = super().get_strategy_state()
        
        # Add LinUCB-specific state
        linucb_state = {
            'alpha': self.alpha,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'pca_fitted': self.pca_fitted,
            'feature_dim': self.feature_dim,
            'n_actions_learned': len(self.A),
            'action_ids': list(self.A.keys())
        }
        
        base_state.update(linucb_state)
        return base_state
    
    def load_strategy_state(self, state: Dict[str, Any]):
        """
        Load LinUCB strategy state from a saved dictionary.
        
        Args:
            state: Dictionary containing strategy state
        """
        super().load_strategy_state(state)
        
        # Load LinUCB-specific state
        self.alpha = state.get('alpha', self.alpha)
        self.use_pca = state.get('use_pca', self.use_pca)
        self.pca_components = state.get('pca_components', self.pca_components)
        self.pca_fitted = state.get('pca_fitted', False)
        self.feature_dim = state.get('feature_dim', None)
        
        # Note: A, b, theta matrices would need special serialization
        # For now, we start fresh but keep the configuration
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Get LinUCB-specific strategy statistics.
        
        Returns:
            Dictionary containing LinUCB strategy statistics
        """
        base_stats = self.get_performance_stats()
        
        # Add LinUCB-specific statistics
        linucb_stats = {
            'alpha': self.alpha,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components if self.use_pca else 'N/A',
            'pca_fitted': self.pca_fitted,
            'feature_dim': self.feature_dim or 'Not initialized',
            'n_actions_learned': len(self.A),
            'action_ids_learned': list(self.A.keys()) if len(self.A) <= 10 else f"{len(self.A)} actions",
            'strategy_initialized': self.is_initialized
        }
        
        # Combine base stats with LinUCB stats
        base_stats.update(linucb_stats)
        return base_stats
    
    def reset_strategy(self):
        """
        Reset the LinUCB strategy to initial state.
        """
        super().reset_strategy()
        
        # Reset LinUCB-specific state
        self.A = {}
        self.b = {}
        self.theta = {}
        self.A_inv_cache = {}
        self.cache_valid = {}
        self.pca_model = None
        self.pca_fitted = False
        self.feature_dim = None
        self.observation_history = []
    
    def get_action_confidence(self, action_id: str, context: np.ndarray) -> float:
        """
        Get confidence width for a specific action and context.
        
        Args:
            action_id: ID of the action
            context: Context feature vector
            
        Returns:
            Confidence width (uncertainty estimate)
        """
        if action_id not in self.A:
            return float('inf')  # Maximum uncertainty for unseen actions
        
        A_inv = self._get_cached_inverse(action_id)
        confidence_width = self.alpha * np.sqrt(context @ A_inv @ context)
        return confidence_width
    
    def get_action_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all learned actions.
        
        Returns:
            Dictionary with action statistics
        """
        stats = {}
        
        for action_id in self.A.keys():
            A_inv = self._get_cached_inverse(action_id)
            theta = self.theta.get(action_id, np.zeros(self.feature_dim))
            
            stats[action_id] = {
                'parameter_norm': np.linalg.norm(theta),
                'matrix_condition': np.linalg.cond(self.A[action_id]),
                'confidence_scale': np.sqrt(np.trace(A_inv))
            }
        
        return stats