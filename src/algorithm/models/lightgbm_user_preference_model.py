"""
LightGBM User Preference Model

A fast and accurate gradient boosting model for user preference prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

import lightgbm as lgb

from .base_user_preference_model import BaseUserPreferenceModel
from src.data.entities import User, Action


class LightGBMUserPreferenceModel(BaseUserPreferenceModel):
    """
    LightGBM User Preference Model with optional PCA dimension reduction.
    """
    
    def __init__(self, use_pca=False, pca_components=50, task_type: str = 'binary', lightgbm_config: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.model = None
        self.pca = None
        self.task_type = task_type
        self.lightgbm_config = lightgbm_config or {}
        
    def fit(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit LightGBM with optional PCA."""
        from sklearn.decomposition import PCA
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        # Extract features and labels
        user_features = np.vstack(observations_df['user_features'].values)  # (N, 8)
        action_embeddings = np.vstack(observations_df['action_embedding'].values)  # (N, 3072)
        rewards = observations_df['reward'].values  # (N,)
        sample_weights = observations_df['sample_weight'].values if 'sample_weight' in observations_df.columns else None
        
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
        if self.task_type == 'binary':
            objective = 'binary'
            metric = 'binary_logloss'
        else:
            objective = 'regression'
            metric = 'l2'

        extra_params = dict(self.lightgbm_config)
        custom_rounds = extra_params.pop('num_boost_round', None)

        lgb_params = {
            'objective': objective,
            'metric': metric,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_seed': self.random_seed
        }
        lgb_params.update(extra_params)
        
        # Split for validation if enough data
        had_validation = len(features) > 100
        if had_validation:
            try:
                if sample_weights is not None:
                    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                        features, rewards, sample_weights,
                        test_size=0.2,
                        random_state=42,
                        stratify=rewards if self.task_type == 'binary' else None
                    )
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        features, rewards,
                        test_size=0.2,
                        random_state=42,
                        stratify=rewards if self.task_type == 'binary' else None
                    )
                    w_train = w_val = None
            except ValueError:
                if sample_weights is not None:
                    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                        features, rewards, sample_weights,
                        test_size=0.2,
                        random_state=42
                    )
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        features, rewards,
                        test_size=0.2,
                        random_state=42
                    )
                    w_train = w_val = None

            train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
            val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)

            self.model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=custom_rounds or 100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

            val_pred = self.model.predict(X_val)
            if self.task_type == 'binary':
                auc = roc_auc_score(y_val, val_pred)
                accuracy = accuracy_score(y_val, val_pred > 0.5)
                rmse = None
            else:
                rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                auc = accuracy = None
        else:
            train_data = lgb.Dataset(features, label=rewards, weight=sample_weights)
            self.model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=custom_rounds or 50
            )

            pred = self.model.predict(features)
            if self.task_type == 'binary':
                auc = roc_auc_score(rewards, pred)
                accuracy = accuracy_score(rewards, pred > 0.5)
                rmse = None
            else:
                rmse = np.sqrt(mean_squared_error(rewards, pred))
                auc = accuracy = None

        self.is_fitted = True

        metrics: Dict[str, Any] = {
            'n_samples': len(observations_df),
            'use_pca': self.use_pca,
            'pca_components': self.pca_components if self.use_pca else None,
            'pca_explained_variance': self.pca.explained_variance_ratio_.sum() if self.use_pca and self.pca is not None else None,
            'feature_dim': features.shape[1],
            'model_type': f'lightgbm_pca{self.pca_components}' if self.use_pca else 'lightgbm',
            'task_type': self.task_type
        }

        if self.task_type == 'binary':
            metrics['val_auc'] = auc
            metrics['val_accuracy'] = accuracy
        else:
            if had_validation:
                metrics['val_rmse'] = rmse
            else:
                metrics['train_rmse'] = rmse

        return metrics
    
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
        if self.task_type == 'binary':
            return float(prob)
        return float(np.clip(prob, 0.0, 1.0))
