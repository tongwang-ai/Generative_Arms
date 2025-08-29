"""
Ground Truth Utility Models

This module contains the ground truth models that simulate realistic user preferences
for different marketing actions in the personalized targeting system.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

from .user_generator import MeaningfulUser
from .action_embedder import EmbeddedAction


@dataclass
class CompanyObservation:
    """Single observation from company's field experiment."""
    user_id: str
    action_id: str
    user_features: np.ndarray
    action_embedding: np.ndarray
    action_text: str
    reward: int  # 0 or 1
    timestamp: str
    iteration: int


class GroundTruthUtility:
    """
    Base class for ground truth utility models that simulate user preferences.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def calculate_utility(self, user: MeaningfulUser, action: EmbeddedAction) -> float:
        """
        Calculate true utility for user-action pair.
        
        Args:
            user: User with features and preferences
            action: Action with text and embedding
            
        Returns:
            Utility value (higher = more preferred)
        """
        raise NotImplementedError
    
    def calculate_utility_batch(self, users: List[MeaningfulUser], 
                               actions: List[EmbeddedAction]) -> np.ndarray:
        """
        Calculate utilities for all user-action pairs efficiently.
        
        Args:
            users: List of users
            actions: List of actions
            
        Returns:
            2D array of utilities (users x actions)
        """
        utilities = np.zeros((len(users), len(actions)))
        
        for i, user in enumerate(users):
            for j, action in enumerate(actions):
                utilities[i, j] = self.calculate_utility(user, action)
        
        return utilities
    
    def simulate_response(self, user: MeaningfulUser, action: EmbeddedAction) -> int:
        """
        Simulate binary response (0/1) based on utility using Bernoulli distribution.
        
        Args:
            user: User with features and preferences
            action: Action with text and embedding
            
        Returns:
            Binary response (0 or 1)
        """
        utility = self.calculate_utility(user, action)
        
        # Use steeper sigmoid to make responses less random but still probabilistic
        # probability = np.clip(1.0 / (1.0 + np.exp(-2.0 * utility)), 0.0, 1.0)
        
        # Original version: use utility directly as probability (more random)
        probability = np.clip(utility, 0.0, 1.0)
        
        # Sample from Bernoulli distribution
        response = np.random.binomial(1, probability)
        
        return response


class MixtureOfExpertsUtility(GroundTruthUtility):
    """
    Mixture of experts model where different users prefer different types of actions.
    Enhanced with diversity constraints to prevent single action dominance.
    """
    
    def __init__(self, n_experts: int = 4, random_seed: int = 42, user_dim: int = 8, action_dim: int = 3072,
                 diversity_regularization: float = 0.2, max_preference_share: float = 0.4):
        super().__init__(random_seed)
        self.n_experts = n_experts
        self.diversity_regularization = diversity_regularization
        self.max_preference_share = max_preference_share
        
        # Generate expert weight matrices with better diversity
        self.expert_weights = []
        for i in range(n_experts):
            # Use orthogonal initialization to promote diversity
            W_base = np.random.randn(user_dim, action_dim) * 0.15
            # Add expert-specific bias to create distinct preferences
            expert_bias = np.random.randn(action_dim) * 0.1
            W = W_base + expert_bias / user_dim
            self.expert_weights.append(W)
        
        # Expert assignment vectors - make them more distinct
        self.expert_assignment_vectors = []
        for i in range(n_experts):
            v = np.random.randn(user_dim) * 0.8
            # Add expert-specific focus
            v[i % user_dim] += 1.0  # Each expert focuses more on different user features
            self.expert_assignment_vectors.append(v)
            
        # Track utilities for diversity constraint
        self._utility_history = {}
    
    def _get_expert_probabilities(self, user_features: np.ndarray) -> np.ndarray:
        """Get expert assignment probabilities for a user."""
        logits = []
        for v in self.expert_assignment_vectors:
            logit = np.dot(user_features, v)
            logits.append(logit)
        
        # Softmax
        logits = np.array(logits)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def calculate_utility(self, user: MeaningfulUser, action: EmbeddedAction) -> float:
        """Calculate utility using mixture of experts with diversity constraints."""
        user_features = user.feature_vector
        action_embedding = action.embedding
        
        # Get expert probabilities
        expert_probs = self._get_expert_probabilities(user_features)
        
        # Calculate weighted utility across experts
        total_utility = 0.0
        for k in range(self.n_experts):
            # Expert k's utility
            expert_utility = np.dot(user_features, self.expert_weights[k] @ action_embedding)
            total_utility += expert_probs[k] * expert_utility
        
        # Add diversity bonus
        diversity_bonus = self._calculate_diversity_bonus(user_features, action_embedding)
        
        # Apply diversity regularization to prevent dominance
        regularized_utility = self._apply_diversity_regularization(
            total_utility + diversity_bonus, action.action_id
        )
        
        return regularized_utility
    
    def _calculate_diversity_bonus(self, user_features: np.ndarray, 
                                 action_embedding: np.ndarray) -> float:
        """Add diversity bonus based on user characteristics."""
        # Users with high social influence prefer more diverse content
        social_influence = user_features[5]  # social_influence feature
        tech_savviness = user_features[2]   # tech_savviness feature
        
        # Diversity score based on embedding variance
        embedding_var = np.var(action_embedding)
        diversity_bonus = 0.1 * social_influence * embedding_var
        diversity_bonus += 0.05 * tech_savviness * embedding_var
        
        return diversity_bonus
    
    def _apply_diversity_regularization(self, raw_utility: float, action_id: str) -> float:
        """Apply diversity regularization to prevent single action dominance."""
        # Track utility for this action
        if action_id not in self._utility_history:
            self._utility_history[action_id] = []
        
        # Add adaptive random noise to break ties and add diversity
        base_noise = 0.08  # Increased base noise
        adaptive_noise = base_noise * (1 + len(self._utility_history[action_id]) / 1000)
        noise = np.random.normal(0, adaptive_noise)
        
        # Apply stronger regularization if this action is getting too popular
        if len(self._utility_history[action_id]) > 20:  # Earlier intervention
            avg_utility = np.mean(self._utility_history[action_id])
            recent_utilities = self._utility_history[action_id][-50:]  # Focus on recent trend
            recent_avg = np.mean(recent_utilities)
            
            # Progressive penalty based on popularity
            if recent_avg > 0.1:  # Lower threshold for intervention
                popularity_factor = min(recent_avg / 0.1, 3.0)  # Cap the factor
                regularization_penalty = self.diversity_regularization * popularity_factor * (recent_avg - 0.1)
                regularized_utility = raw_utility - regularization_penalty
            else:
                regularized_utility = raw_utility
        else:
            regularized_utility = raw_utility
        
        # Add noise to final utility
        final_utility = regularized_utility + noise
        
        # Store this utility for future regularization
        self._utility_history[action_id].append(final_utility)
        
        # Keep only recent history to adapt to changes
        if len(self._utility_history[action_id]) > 300:
            self._utility_history[action_id] = self._utility_history[action_id][-300:]
        
        return final_utility
    
    def calculate_utility_batch(self, users: List[MeaningfulUser], 
                               actions: List[EmbeddedAction]) -> np.ndarray:
        """Vectorized batch calculation for efficiency."""
        n_users = len(users)
        n_actions = len(actions)
        
        # Extract features
        user_features = np.array([user.feature_vector for user in users])
        action_embeddings = np.array([action.embedding for action in actions])
        
        # Calculate utilities efficiently
        utilities = np.zeros((n_users, n_actions))
        
        for i, user in enumerate(users):
            user_feat = user_features[i]
            expert_probs = self._get_expert_probabilities(user_feat)
            
            for j, action in enumerate(actions):
                action_emb = action_embeddings[j]
                
                # Calculate expert utilities
                total_utility = 0.0
                for k in range(self.n_experts):
                    expert_utility = np.dot(user_feat, self.expert_weights[k] @ action_emb)
                    total_utility += expert_probs[k] * expert_utility
                
                # Add diversity bonus
                diversity_bonus = self._calculate_diversity_bonus(user_feat, action_emb)
                
                # Apply diversity regularization
                regularized_utility = self._apply_diversity_regularization(
                    total_utility + diversity_bonus, action.action_id
                )
                utilities[i, j] = regularized_utility
        
        return utilities


class GMMUtility(GroundTruthUtility):
    """
    Gaussian Mixture Model in joint user-action feature space.
    """
    
    def __init__(self, n_components: int = 8, random_seed: int = 42):
        super().__init__(random_seed)
        self.n_components = n_components
        self.gmm = None
        self.is_fitted = False
    
    def _fit_gmm_if_needed(self, users: List[MeaningfulUser], 
                          actions: List[EmbeddedAction]):
        """Fit GMM on joint user-action space if not already fitted."""
        if self.is_fitted:
            return
        
        # Create joint feature space
        joint_features = []
        for user in users:
            for action in actions:
                joint_feat = np.concatenate([user.feature_vector, action.embedding])
                joint_features.append(joint_feat)
        
        joint_features = np.array(joint_features)
        
        # Fit GMM
        self.gmm = GaussianMixture(n_components=self.n_components, 
                                  random_state=self.random_seed)
        self.gmm.fit(joint_features)
        self.is_fitted = True
    
    def calculate_utility(self, user: MeaningfulUser, action: EmbeddedAction) -> float:
        """Calculate utility based on GMM density."""
        if not self.is_fitted:
            # Need to fit GMM first - return random utility for now
            return np.random.randn() * 0.1
        
        # Joint features
        joint_feat = np.concatenate([user.feature_vector, action.embedding])
        
        # Get log probability from GMM
        log_prob = self.gmm.score_samples([joint_feat])[0]
        
        # Convert to utility (scale and shift)
        utility = (log_prob + 10) / 20  # Normalize roughly to [-0.5, 0.5]
        
        return utility


def create_ground_truth_utility(ground_truth_type: str = "mixture_of_experts", 
                               random_seed: int = 42,
                               **kwargs) -> GroundTruthUtility:
    """
    Factory function to create ground truth utility models.
    
    Args:
        ground_truth_type: Type of ground truth model
        random_seed: Random seed for reproducibility
        **kwargs: Additional parameters for specific models
                  - user_dim: User feature dimensions (default: 8)
                  - action_dim: Action embedding dimensions (default: 1536, but 3072 for text-embedding-3-large)
        
    Returns:
        GroundTruthUtility instance
    """
    if ground_truth_type == "mixture_of_experts":
        n_experts = kwargs.get('n_experts', 4)
        user_dim = kwargs.get('user_dim', 8)
        action_dim = kwargs.get('action_dim', 1536)
        diversity_regularization = kwargs.get('diversity_regularization', 0.2)
        max_preference_share = kwargs.get('max_preference_share', 0.4)
        return MixtureOfExpertsUtility(
            n_experts=n_experts, 
            random_seed=random_seed, 
            user_dim=user_dim, 
            action_dim=action_dim,
            diversity_regularization=diversity_regularization,
            max_preference_share=max_preference_share
        )
    
    elif ground_truth_type == "gmm":
        n_components = kwargs.get('n_components', 8)
        return GMMUtility(n_components=n_components, random_seed=random_seed)
    
    else:
        raise ValueError(f"Unknown ground truth type: {ground_truth_type}")