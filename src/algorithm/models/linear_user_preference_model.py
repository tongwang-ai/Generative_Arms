"""
Linear Regression User Preference Model

A linear regression model for user preference prediction with optional PCA dimension reduction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from .base_user_preference_model import BaseUserPreferenceModel
from src.data.entities import User, Action


class LinearUserPreferenceModel(BaseUserPreferenceModel):
    """
    Linear Regression User Preference Model with optional PCA dimension reduction.
    """
    
    def __init__(self, use_pca=False, pca_components=50, solver='lbfgs', max_iter=1000,
                 C=1.0, **kwargs):
        super().__init__(**kwargs)
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.model = None
        self.pca = None
        self.scaler = None
        
    def fit(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit linear regression with optional PCA."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        # Extract features and labels
        user_features = np.vstack(observations_df['user_features'].values)  # (N, 8)
        action_embeddings = np.vstack(observations_df['action_embedding'].values)  # (N, 1536)
        rewards = observations_df['reward'].values  # (N,)
        
        print(f"Linear Regression fitting on {len(observations_df)} samples")
        print(f"User features shape: {user_features.shape}")
        print(f"Action embeddings shape: {action_embeddings.shape}")
        print(f"Using PCA: {self.use_pca}")
        if self.use_pca:
            print(f"PCA components: {self.pca_components}")
        
        # Apply PCA to action embeddings if requested
        if self.use_pca:
            print(f"Applying PCA to reduce action embeddings from {action_embeddings.shape[1]} to {self.pca_components} dimensions")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            action_embeddings = self.pca.fit_transform(action_embeddings)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Combine features
        features = np.concatenate([user_features, action_embeddings], axis=1)
        
        # Scale features
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        
        # Fit logistic regression
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=self.max_iter, 
            solver=self.solver,
            C=self.C
        )
        self.model.fit(features, rewards)
        
        # Evaluate on training data
        train_pred = self.model.predict_proba(features)[:, 1]
        auc = roc_auc_score(rewards, train_pred)
        accuracy = accuracy_score(rewards, train_pred > 0.5)
        
        self.is_fitted = True
        
        return {
            'train_auc': auc,
            'train_accuracy': accuracy,
            'n_samples': len(observations_df),
            'positive_rate': rewards.mean(),
            'use_pca': self.use_pca,
            'pca_components': self.pca_components if self.use_pca else None,
            'pca_explained_variance': self.pca.explained_variance_ratio_.sum() if self.use_pca else None,
            'feature_dim': features.shape[1],
            'model_type': f'linear_regression_pca{self.pca_components}' if self.use_pca else 'linear_regression',
            'solver': self.solver,
            'C': self.C
        }
    
    def predict(self, user: User, action: Action) -> float:
        """Predict using linear regression."""
        if not self.is_fitted:
            return 0.5
            
        # Prepare features
        action_embedding = action.embedding
        if self.use_pca:
            action_embedding = self.pca.transform(action_embedding.reshape(1, -1)).flatten()
        
        features = np.concatenate([user.features, action_embedding]).reshape(1, -1)
        features = self.scaler.transform(features)
        
        prob = self.model.predict_proba(features)[0, 1]
        return float(prob)
