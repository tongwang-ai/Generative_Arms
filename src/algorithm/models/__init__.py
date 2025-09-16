"""
User Preference Models for Personalized Targeting

This package provides user preference modeling capabilities with a uniform interface.

Available Models:
- LightGBMUserPreferenceModel: Fast gradient boosting (recommended)
- NeuralUserPreferenceModel: Deep neural network with PyTorch
- LinearUserPreferenceModel: Logistic regression baseline
- GaussianProcessUserPreferenceModel: Non-parametric GP model
- BaseUserPreferenceModel: Base class for custom model development
- UserPreferenceModelWrapper: Wrapper for legacy models

Usage:
    from src.algorithm.models import LightGBMUserPreferenceModel
    
    # Create model with optional PCA
    model = LightGBMUserPreferenceModel(use_pca=True, pca_components=50)
    
    # Train on observation data
    model.fit(observations_df)
    
    # Make predictions
    probability = model.predict(user, action)
"""

from .base_user_preference_model import BaseUserPreferenceModel, UserPreferenceModelWrapper
from .lightgbm_user_preference_model import LightGBMUserPreferenceModel
from .neural_user_preference_model import NeuralUserPreferenceModel
from .bayesian_neural_user_preference_model import BayesianNeuralUserPreferenceModel
from .linear_user_preference_model import LinearUserPreferenceModel
from .gaussian_process_user_preference_model import GaussianProcessUserPreferenceModel
from .user_preference_model_tester import UserPreferenceModelTester
from .ft_transformer_user_preference_model import FTTransformerUserPreferenceModel

__all__ = [
    'BaseUserPreferenceModel', 
    'UserPreferenceModelWrapper',
    'LightGBMUserPreferenceModel',
    'NeuralUserPreferenceModel', 
    'BayesianNeuralUserPreferenceModel',
    'LinearUserPreferenceModel',
    'GaussianProcessUserPreferenceModel',
    'UserPreferenceModelTester',
    'FTTransformerUserPreferenceModel'
]
