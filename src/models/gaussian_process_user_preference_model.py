"""
Gaussian Process User Preference Model

A Gaussian Process model for user preference prediction (non-parametric).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score

from .base_user_preference_model import BaseUserPreferenceModel
from ..data.entities import User, Action


class GaussianProcessUserPreferenceModel(BaseUserPreferenceModel):
    """
    Gaussian Process user preference model (non-parametric).
    
    This model doesn't require explicit training - it uses the training data
    directly for inference via GP regression.
    """
    
    def __init__(self, kernel_type='rbf', length_scale=1.0, use_pca=False, 
                 pca_components=50, max_samples=1000, **kwargs):
        super().__init__(**kwargs)
        self.requires_training = False  # Non-parametric model
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.max_samples = max_samples
        self.X_train = None
        self.y_train = None
        self.gp_model = None
        self.pca = None
        
    def fit(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        """Store training data for GP inference."""
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, Matern
        from sklearn.decomposition import PCA
        
        # Extract features and labels
        user_features = np.vstack(observations_df['user_features'].values)  # (N, 8)
        action_embeddings = np.vstack(observations_df['action_embedding'].values)  # (N, 1536) 
        rewards = observations_df['reward'].values  # (N,)
        
        print(f"GP fitting on {len(observations_df)} samples")
        print(f"User features shape: {user_features.shape}")
        print(f"Action embeddings shape: {action_embeddings.shape}")
        print(f"Using PCA: {self.use_pca}")
        if self.use_pca:
            print(f"PCA components: {self.pca_components}")
        print(f"Positive rate: {rewards.mean():.3f}")
        
        # Apply PCA to action embeddings if requested
        if self.use_pca:
            print(f"Applying PCA to reduce action embeddings from {action_embeddings.shape[1]} to {self.pca_components} dimensions")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            action_embeddings = self.pca.fit_transform(action_embeddings)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Combine features
        self.X_train = np.concatenate([user_features, action_embeddings], axis=1)
        self.y_train = rewards
        
        # Create GP with appropriate kernel
        if self.kernel_type == 'rbf':
            kernel = RBF(length_scale=self.length_scale)
        elif self.kernel_type == 'matern':
            kernel = Matern(length_scale=self.length_scale, nu=1.5)
        else:
            kernel = RBF(length_scale=self.length_scale)  # Default
            
        # For large datasets, use subset for GP (computationally intensive)
        if len(self.X_train) > self.max_samples:
            print(f"Using subset of {self.max_samples} samples for GP (computational efficiency)")
            indices = np.random.choice(len(self.X_train), self.max_samples, replace=False)
            X_subset = self.X_train[indices]
            y_subset = self.y_train[indices]
        else:
            X_subset = self.X_train
            y_subset = self.y_train
            
        self.gp_model = GaussianProcessClassifier(kernel=kernel, random_state=42)
        self.gp_model.fit(X_subset, y_subset)
        
        self.is_fitted = True
        
        # Evaluate on training data
        train_pred = self.gp_model.predict_proba(X_subset)[:, 1]
        auc = roc_auc_score(y_subset, train_pred)
        accuracy = accuracy_score(y_subset, train_pred > 0.5)
        
        return {
            'train_auc': auc,
            'train_accuracy': accuracy,
            'n_samples': len(observations_df),
            'n_gp_samples': len(X_subset),
            'positive_rate': rewards.mean(),
            'use_pca': self.use_pca,
            'pca_components': self.pca_components if self.use_pca else None,
            'pca_explained_variance': self.pca.explained_variance_ratio_.sum() if self.use_pca else None,
            'feature_dim': self.X_train.shape[1],
            'model_type': f'gaussian_process_{self.kernel_type}_pca{self.pca_components}' if self.use_pca else f'gaussian_process_{self.kernel_type}',
            'kernel_length_scale': self.length_scale,
            'kernel_type': self.kernel_type
        }
    
    def predict(self, user: User, action: Action) -> float:
        """Predict using GP posterior."""
        if not self.is_fitted or self.gp_model is None:
            return 0.5  # Random guess
            
        # Prepare features
        action_embedding = action.embedding
        if self.use_pca:
            action_embedding = self.pca.transform(action_embedding.reshape(1, -1)).flatten()
        
        # Combine features same way as training
        features = np.concatenate([user.features, action_embedding]).reshape(1, -1)
        
        # Get GP prediction probability
        prob = self.gp_model.predict_proba(features)[0, 1]
        return float(prob)

    def predict_with_uncertainty(self, user: User, action: Action) -> tuple:
        """
        Approximate uncertainty for GP classifier by Bernoulli variance p*(1-p).
        """
        p = self.predict(user, action)
        # Standard deviation of Bernoulli as a proxy
        std = float(np.sqrt(max(0.0, p * (1.0 - p))))
        return float(p), std
