"""
LightGBM User Preference Model

A fast and accurate gradient boosting model for user preference prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import lightgbm as lgb

from .base_user_preference_model import BaseUserPreferenceModel
from src.data.entities import User, Action


class LightGBMUserPreferenceModel(BaseUserPreferenceModel):
    """
    LightGBM User Preference Model with optional PCA dimension reduction.
    """
    
    def __init__(self, use_pca=False, pca_components=50, **kwargs):
        super().__init__(**kwargs)
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.model = None
        self.pca = None
        
    def fit(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit LightGBM with optional PCA."""
        from sklearn.decomposition import PCA
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        # Extract features and labels
        user_features = np.vstack(observations_df['user_features'].values)  # (N, 8)
        action_embeddings = np.vstack(observations_df['action_embedding'].values)  # (N, 3072)
        rewards = observations_df['reward'].values  # (N,)
        
        print(f"LightGBM fitting on {len(observations_df)} samples")
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
        
        # LightGBM parameters
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_seed': 42
        }
        
        # Split for validation if enough data
        if len(features) > 100:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    features, rewards, test_size=0.2, random_state=42, stratify=rewards
                )
            except ValueError:
                # Fall back to random split if stratification fails
                X_train, X_val, y_train, y_val = train_test_split(
                    features, rewards, test_size=0.2, random_state=42
                )
                
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Evaluate on validation data
            val_pred = self.model.predict(X_val)
            auc = roc_auc_score(y_val, val_pred)
            accuracy = accuracy_score(y_val, val_pred > 0.5)
        else:
            # Not enough data for validation split
            train_data = lgb.Dataset(features, label=rewards)
            self.model = lgb.train(lgb_params, train_data, num_boost_round=50)
            
            # Evaluate on training data
            train_pred = self.model.predict(features)
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
            'model_type': f'lightgbm_pca{self.pca_components}' if self.use_pca else 'lightgbm'
        }
    
    def predict(self, user: User, action: Action) -> float:
        """Predict using LightGBM."""
        if not self.is_fitted:
            return 0.5
            
        # Prepare features
        action_embedding = action.embedding

        if self.use_pca:
            action_embedding = self.pca.transform(action_embedding.reshape(1, -1)).flatten()
        
        features = np.concatenate([user.features, action_embedding]).reshape(1, -1)
        
        prob = self.model.predict(features)[0]
        return float(prob)
