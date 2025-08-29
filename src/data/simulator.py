import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import random
from .entities import User, Action


class DataSimulator:
    def __init__(self, n_user_features: int = 10, n_action_embedding_dim: int = 8, 
                 n_experts: int = 3, random_seed: int = 42):
        self.n_user_features = n_user_features
        self.n_action_embedding_dim = n_action_embedding_dim
        self.n_experts = n_experts
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Expert mixture model parameters as described in proposal
        # W_k matrices for each expert (n_user_features x n_action_embedding_dim)
        self._expert_weights = []
        for k in range(n_experts):
            W_k = np.random.normal(0, 0.5, (n_user_features, n_action_embedding_dim))
            self._expert_weights.append(W_k)
        
        # Expert prototype vectors v_k (archetype users)
        self._expert_prototypes = []
        for k in range(n_experts):
            # Create diverse archetype users
            if k == 0:  # Price-sensitive users
                prototype = np.array([1.0, -0.5, 0.0, 0.5, -1.0] + [0.0] * (n_user_features - 5))
            elif k == 1:  # Quality-focused users  
                prototype = np.array([-0.5, 1.0, 0.8, -0.2, 0.5] + [0.0] * (n_user_features - 5))
            else:  # Engagement-focused users
                prototype = np.array([0.0, 0.2, -0.8, 1.0, 0.3] + [0.0] * (n_user_features - 5))
            
            # Pad or truncate to correct size and normalize
            if len(prototype) < n_user_features:
                prototype = np.concatenate([prototype, np.zeros(n_user_features - len(prototype))])
            else:
                prototype = prototype[:n_user_features]
            
            prototype = prototype / np.linalg.norm(prototype)
            self._expert_prototypes.append(prototype)
        
        # Legacy fallback (for backward compatibility)
        self._interaction_weights = self._expert_weights[0]
        self._bias_vector = np.random.normal(0, 0.2, n_action_embedding_dim)
        
    def generate_users(self, n_users: int) -> List[User]:
        """Generate a batch of synthetic users with feature vectors."""
        users = []
        
        for i in range(n_users):
            # Generate user features (normalized)
            features = np.random.normal(0, 1, self.n_user_features)
            features = features / np.linalg.norm(features)
            
            user = User(
                user_id=f"user_{i}",
                features=features
            )
            users.append(user)
            
        return users
    
    def generate_actions(self, n_actions: int, action_texts: List[str] = None) -> List[Action]:
        """Generate a set of actions with embeddings."""
        actions = []
        
        if action_texts is None:
            action_texts = [f"Action {i}" for i in range(n_actions)]
        
        for i in range(min(n_actions, len(action_texts))):
            # Generate action embedding based on text characteristics
            embedding = self._text_to_embedding(action_texts[i])
            
            action = Action(
                action_id=f"action_{i}",
                text=action_texts[i],
                embedding=embedding
            )
            actions.append(action)
            
        return actions
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector (simulated)."""
        # Simple text features to embedding conversion
        text_lower = text.lower()
        
        # Base embedding from text hash
        text_hash = hash(text) % 10000
        np.random.seed(text_hash)
        embedding = np.random.normal(0, 1, self.n_action_embedding_dim)
        
        # Modify based on text characteristics
        if any(word in text_lower for word in ['discount', 'sale', 'off', '%']):
            embedding[0] += 0.5  # Promotional feature
        
        if any(word in text_lower for word in ['learn', 'discover', 'understand']):
            embedding[1] += 0.5  # Educational feature
            
        if any(word in text_lower for word in ['join', 'community', 'share']):
            embedding[2] += 0.5  # Social feature
            
        if any(word in text_lower for word in ['urgent', 'limited', 'now']):
            embedding[3] += 0.5  # Urgency feature
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        # Reset random seed
        np.random.seed(self.random_seed)
        
        return embedding
    
    def ground_truth_utility(self, user: User, action: Action) -> float:
        """
        Ground truth utility function implementing the expert mixture model from proposal.
        U(x,a) = sum_k π_k(x) * σ(x^T W_k a)
        Returns the true probability of conversion for user-action pair.
        """
        # Calculate expert assignment probabilities π_k(x)
        expert_probs = self._calculate_expert_probabilities(user.features)
        
        # Calculate utility as weighted sum over experts
        total_utility = 0.0
        
        for k in range(self.n_experts):
            # Calculate x^T W_k a for expert k
            interaction_score = np.dot(user.features, np.dot(self._expert_weights[k], action.embedding))
            
            # Apply sigmoid activation
            expert_utility = 1 / (1 + np.exp(-interaction_score))
            
            # Weight by expert assignment probability
            total_utility += expert_probs[k] * expert_utility
        
        return total_utility
    
    def _calculate_expert_probabilities(self, user_features: np.ndarray) -> np.ndarray:
        """
        Calculate π_k(x) = exp(x^T v_k) / sum_j exp(x^T v_j)
        Expert assignment probabilities using softmax over user affinities.
        """
        # Calculate affinities x^T v_k for each expert
        affinities = []
        for k in range(self.n_experts):
            affinity = np.dot(user_features, self._expert_prototypes[k])
            affinities.append(affinity)
        
        # Apply softmax to get probabilities
        affinities = np.array(affinities)
        # Numerical stability: subtract max before exp
        affinities = affinities - np.max(affinities)
        exp_affinities = np.exp(affinities)
        probabilities = exp_affinities / np.sum(exp_affinities)
        
        return probabilities
    
    def simulate_outcome(self, user: User, action: Action) -> int:
        """Simulate binary outcome based on ground truth utility."""
        probability = self.ground_truth_utility(user, action)
        return int(np.random.random() < probability)
    
    def generate_historical_data(self, users: List[User], actions: List[Action], 
                               n_interactions: int) -> pd.DataFrame:
        """
        Generate historical interaction data.
        Returns DataFrame with columns: user_id, action_id, outcome, user_features, action_embedding
        """
        data = []
        
        for _ in range(n_interactions):
            user = random.choice(users)
            action = random.choice(actions)
            outcome = self.simulate_outcome(user, action)
            
            data.append({
                'user_id': user.user_id,
                'action_id': action.action_id,
                'outcome': outcome,
                'user_features': user.features,
                'action_embedding': action.embedding,
                'action_text': action.text
            })
            
        return pd.DataFrame(data)
    
    def simulate_deployment(self, action_bank: List[Action], users: List[User], 
                          assignments: Dict[str, str]) -> pd.DataFrame:
        """
        Simulate deployment of action bank and collect feedback data.
        assignments: dict mapping user_id to action_id
        """
        action_dict = {action.action_id: action for action in action_bank}
        data = []
        
        for user in users:
            if user.user_id in assignments:
                action_id = assignments[user.user_id]
                if action_id in action_dict:
                    action = action_dict[action_id]
                    outcome = self.simulate_outcome(user, action)
                    
                    data.append({
                        'user_id': user.user_id,
                        'action_id': action_id,
                        'outcome': outcome,
                        'user_features': user.features,
                        'action_embedding': action.embedding,
                        'action_text': action.text
                    })
        
        return pd.DataFrame(data)
    
    def generate_policy_training_data(self, users: List[User], actions: List[Action], 
                                    n_samples: int) -> pd.DataFrame:
        """
        Generate training data for proxy policy model.
        This simulates the data that would come from the black-box targeting policy.
        """
        data = []
        
        for _ in range(n_samples):
            user = random.choice(users)
            
            # Generate probability distribution over actions for this user
            # This simulates what the black-box policy would provide
            action_probs = []
            
            for action in actions:
                # Base probability from user-action similarity
                similarity = np.dot(user.features[:min(len(user.features), len(action.embedding))], 
                                  action.embedding[:min(len(user.features), len(action.embedding))])
                
                # Add some randomness and bias
                prob = max(0, similarity + np.random.normal(0, 0.2))
                action_probs.append(prob)
            
            # Normalize to get probability distribution
            total_prob = sum(action_probs)
            if total_prob > 0:
                action_probs = [p / total_prob for p in action_probs]
            else:
                action_probs = [1.0 / len(actions)] * len(actions)
            
            # Store the data point
            data.append({
                'user_id': user.user_id,
                'user_features': user.features,
                'action_probabilities': action_probs,
                'action_ids': [action.action_id for action in actions]
            })
        
        return pd.DataFrame(data)