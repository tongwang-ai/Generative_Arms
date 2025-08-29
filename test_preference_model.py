#!/usr/bin/env python3
"""
User Preference Model Testing Script

This script loads observation data from experiments and provides a clean interface
for testing and evaluating custom user preference models. It allows you to:

1. Load historical observation data from any experiment (with action text + embeddings)
2. Implement and test custom user preference models (parametric and non-parametric)
3. Compare different modeling approaches (neural nets, GPs, ensemble methods)
4. Test different embedding methods using original action text
5. Evaluate model performance against ground truth (if available)

Usage:
    python test_preference_model.py --experiment_dir results/simulation_20250819_185550 --model_type custom
    python test_preference_model.py --experiment_dir results/simulation_20250819_185550 --model_type compare_all
"""

import numpy as np
import pandas as pd
import os
import argparse
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.data.entities import User, Action
from src.models.base_user_preference_model import BaseUserPreferenceModel
from src.models.lightgbm_user_preference_model import LightGBMUserPreferenceModel
from src.models.neural_user_preference_model import NeuralUserPreferenceModel
from src.models.linear_user_preference_model import LinearUserPreferenceModel
from src.models.gaussian_process_user_preference_model import GaussianProcessUserPreferenceModel
from src.models.user_preference_model_tester import UserPreferenceModelTester


class CustomUserPreferenceModel(BaseUserPreferenceModel):
    """
    Template for custom user preference model implementation.
    
    TODO: Implement your custom preference model here!
    This is where you can experiment with:
    - Different embedding methods (using action_text instead of pre-computed embeddings)
    - Novel architectures (transformers, attention, etc.)  
    - Advanced regularization techniques
    - Domain-specific inductive biases for marketing/advertising
    - Multi-task learning objectives
    - Non-parametric approaches (kernel methods, k-NN, etc.)
    - Ensemble methods combining multiple approaches
    """
    
    def __init__(self, use_text_features=False, **kwargs):
        super().__init__(**kwargs)
        # TODO: Add your hyperparameters here
        self.hyperparams = kwargs
        self.use_text_features = use_text_features  # Whether to re-embed action text
        self.model = None
        
    def fit(self, observations_df: pd.DataFrame) -> dict:
        """
        TODO: Implement your model training here!
        
        Args:
            observations_df: DataFrame with columns:
                - user_features: numpy array of user characteristics (8-dim)
                - action_embedding: numpy array of action embeddings (1536-dim)
                - action_text: original marketing content text
                - reward: binary outcome (0/1)
        
        Returns:
            Dictionary with fitting metrics (loss, accuracy, etc.)
        """
        print("TODO: Implement custom model training!")
        print(f"Training data shape: {observations_df.shape}")
        print(f"Positive rate: {observations_df['reward'].mean():.3f}")
        
        # TODO: Replace this with your actual model training
        self.is_fitted = True
        
        return {
            'status': 'custom_model_not_implemented',
            'n_samples': len(observations_df),
            'positive_rate': observations_df['reward'].mean()
        }
    
    def predict(self, user: User, action: Action) -> float:
        """
        TODO: Implement your model prediction here!
        
        Args:
            user: User object with features attribute
            action: Action object with embedding and text attributes
            
        Returns:
            Predicted probability of positive outcome (0.0 to 1.0)
        """
        if not self.is_fitted:
            return 0.5
            
        # TODO: Replace this with your actual prediction logic
        # For now, return random prediction based on user features
        user_preference = np.mean(user.features)  # Simple baseline
        return float(np.clip(user_preference, 0, 1))
    
    @property
    def is_trained(self):
        """Backward compatibility property."""
        return self.is_fitted


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(description='Test reward models on observation data')
    parser.add_argument('--experiment_dir', required=True, 
                       help='Path to experiment directory containing observation data')
    parser.add_argument('--model_type', default='custom', 
                       choices=['custom', 'neural', 'lightgbm', 'gaussian_process', 'linear_regression', 'compare_all'],
                       help='Type of model to test')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = UserPreferenceModelTester(args.experiment_dir)
    
    if args.model_type == 'custom':
        print(f"\\n{'='*60}")
        print("TESTING CUSTOM USER PREFERENCE MODEL")
        print(f"{'='*60}")
        
        model = CustomUserPreferenceModel()
        results = tester.train_and_evaluate_model(model, test_size=args.test_size)
        
        print(f"\\nFinal Results:")
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Train Status: {results['train_metrics']}")
            if 'eval_metrics' in results:
                eval_metrics = results['eval_metrics']
                print(f"Test AUC: {eval_metrics.get('test_auc', 'N/A')}")
                print(f"Test Accuracy: {eval_metrics.get('test_accuracy', 'N/A')}")
    
    elif args.model_type == 'neural':
        print(f"\\n{'='*60}")
        print("TESTING NEURAL USER PREFERENCE MODEL")
        print(f"{'='*60}")
        
        model = NeuralUserPreferenceModel(diversity_weight=0.1, random_seed=42)
        results = tester.train_and_evaluate_model(model, test_size=args.test_size)
        
        print(f"\\nFinal Results:")
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Train AUC: {results['train_metrics'].get('val_auc', 'N/A')}")
            print(f"Test AUC: {results['eval_metrics'].get('test_auc', 'N/A')}")
    
    elif args.model_type == 'lightgbm':
        print(f"\\n{'='*60}")
        print("TESTING LIGHTGBM USER PREFERENCE MODEL")
        print(f"{'='*60}")
        
        model = LightGBMUserPreferenceModel(diversity_weight=0.1, random_seed=42)
        results = tester.train_and_evaluate_model(model, test_size=args.test_size)
        
        print(f"\\nFinal Results:")
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Train AUC: {results['train_metrics']['train_auc']:.4f}")
            print(f"Test AUC: {results['eval_metrics']['test_auc']:.4f}")
    
    elif args.model_type == 'gaussian_process':
        print(f"\\n{'='*60}")
        print("TESTING GAUSSIAN PROCESS USER PREFERENCE MODEL")
        print(f"{'='*60}")
        
        model = GaussianProcessUserPreferenceModel(kernel_type='rbf', diversity_weight=0.1, random_seed=42)
        results = tester.train_and_evaluate_model(model, test_size=args.test_size)
        
        print(f"\\nFinal Results:")
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Train AUC: {results['train_metrics']['train_auc']:.4f}")
            print(f"Test AUC: {results['eval_metrics']['test_auc']:.4f}")
    
    elif args.model_type == 'linear_regression':
        print(f"\\n{'='*60}")
        print("TESTING LINEAR REGRESSION USER PREFERENCE MODEL")
        print(f"{'='*60}")
        
        model = LinearUserPreferenceModel(diversity_weight=0.1, random_seed=42)
        results = tester.train_and_evaluate_model(model, test_size=args.test_size)
        
        print(f"\\nFinal Results:")
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Train AUC: {results['train_metrics']['train_auc']:.4f}")
            print(f"Test AUC: {results['eval_metrics']['test_auc']:.4f}")
    
    elif args.model_type == 'compare_all':
        print(f"\\n{'='*60}")
        print("COMPARING ALL USER PREFERENCE MODELS")
        print(f"{'='*60}")
        
        models = {
            # 'Custom Model': CustomUserPreferenceModel(),
            'Linear Regression': LinearUserPreferenceModel(diversity_weight=0.1, random_seed=42),
            'Linear Reg + PCA': LinearUserPreferenceModel(use_pca=True, pca_components=50, diversity_weight=0.1, random_seed=42),
            'LightGBM': LightGBMUserPreferenceModel(diversity_weight=0.1, random_seed=42),
            'LightGBM + PCA': LightGBMUserPreferenceModel(use_pca=True, pca_components=50, diversity_weight=0.1, random_seed=42),
            'Neural Network': NeuralUserPreferenceModel(diversity_weight=0.1, random_seed=42),
            'Neural + PCA': NeuralUserPreferenceModel(use_pca=True, pca_components=50, diversity_weight=0.1, random_seed=42),
            'Gaussian Process RBF': GaussianProcessUserPreferenceModel(kernel_type='rbf', diversity_weight=0.1, random_seed=42),
            'Gaussian Process Matern': GaussianProcessUserPreferenceModel(kernel_type='matern', diversity_weight=0.1, random_seed=42)
        }
        
        comparison_df = tester.compare_models(models)
        print(f"\\n{'='*60}")
        print("USER PREFERENCE MODEL COMPARISON RESULTS")
        print(f"{'='*60}")
        print("\\nTop performing models:")
        print(comparison_df.head(3).to_string(index=False, float_format='%.4f'))
        
    print("\\nDone!")


if __name__ == "__main__":
    main()