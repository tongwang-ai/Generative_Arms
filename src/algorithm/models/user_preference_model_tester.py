"""
User Preference Model Testing Framework

This module provides evaluation capabilities for user preference models
with methods for loading data, training models, and comparing performance.
"""

import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings

from .base_user_preference_model import BaseUserPreferenceModel
from src.data.entities import User, Action


class UserPreferenceModelTester:
    """
    Main testing framework for user preference models.
    
    This class provides comprehensive evaluation capabilities including:
    - Loading observation data from experiment directories
    - Training and evaluating different model types
    - Comparing model performance across multiple metrics
    - Action-based train/test splitting for realistic evaluation
    """
    
    def __init__(self, experiment_dir: str):
        """Initialize tester with experiment directory."""
        self.experiment_dir = Path(experiment_dir)
        self.observations_df = None
        self.users = None
        self.actions = None
        
    def load_data(self) -> pd.DataFrame:
        """Load all observation data from the experiment."""
        print(f"Loading data from {self.experiment_dir}")
        
        # Find all observation files
        obs_files = list(self.experiment_dir.glob("iteration_*/observations/observations.csv"))
        
        if not obs_files:
            raise FileNotFoundError(f"No observation files found in {self.experiment_dir}")
            
        print(f"Found {len(obs_files)} observation files")
        
        # Load and combine all observations
        all_observations = []
        for obs_file in sorted(obs_files):
            df = pd.read_csv(obs_file)
            
            # Parse string representations back to arrays
            if 'user_features' in df.columns:
                df['user_features'] = df['user_features'].apply(
                    lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
                )
            if 'action_embedding' in df.columns:
                df['action_embedding'] = df['action_embedding'].apply(
                    lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
                )
                
            all_observations.append(df)
            
        self.observations_df = pd.concat(all_observations, ignore_index=True)
        
        print(f"Loaded {len(self.observations_df)} total observations")
        print(f"Unique users: {self.observations_df['user_id'].nunique()}")
        print(f"Unique actions: {self.observations_df['action_id'].nunique()}")
        print(f"Overall positive rate: {self.observations_df['reward'].mean():.3f}")
        
        # Create User and Action objects for compatibility
        self._create_entities()
        
        return self.observations_df
    
    def _create_entities(self):
        """Create User and Action objects from DataFrame."""
        # Create unique users
        unique_users = self.observations_df.drop_duplicates('user_id')
        self.users = []
        
        for _, row in unique_users.iterrows():
            user = User(
                user_id=row['user_id'],
                features=row['user_features']
            )
            self.users.append(user)
        
        # Create unique actions
        unique_actions = self.observations_df.drop_duplicates('action_id')
        self.actions = []
        
        for _, row in unique_actions.iterrows():
            action = Action(
                action_id=row['action_id'],
                text=row.get('action_text', f"Action {row['action_id']}"),
                embedding=row['action_embedding']
            )
            self.actions.append(action)
    
    def action_based_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by actions for more realistic evaluation.
        
        Args:
            test_size: Fraction of actions to use for test
            random_state: Random seed
        
        Returns:
            train_df, test_df
        """
        if self.observations_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get unique actions
        unique_actions = self.observations_df['action_id'].unique()
        
        # Split actions
        train_actions, test_actions = train_test_split(
            unique_actions, test_size=test_size, random_state=random_state
        )
        
        # Filter observations based on action split
        train_df = self.observations_df[self.observations_df['action_id'].isin(train_actions)].copy()
        test_df = self.observations_df[self.observations_df['action_id'].isin(test_actions)].copy()
        
        print(f"Action-based split:")
        print(f"  Train: {len(train_df)} observations, {len(train_actions)} actions")
        print(f"  Test: {len(test_df)} observations, {len(test_actions)} actions")
        print(f"  Train positive rate: {train_df['reward'].mean():.3f}")
        print(f"  Test positive rate: {test_df['reward'].mean():.3f}")
        
        return train_df, test_df
    
    def train_and_evaluate_model(self, model: BaseUserPreferenceModel, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train and evaluate a user preference model.
        
        Args:
            model: Model to train and evaluate
            test_size: Fraction for test split
        
        Returns:
            Dictionary with training results and metrics
        """
        if self.observations_df is None:
            self.load_data()
        
        print(f"\n{'='*50}")
        print(f"Training and Evaluating Model: {type(model).__name__}")
        print(f"{'='*50}")
        
        # Split data by actions
        train_df, test_df = self.action_based_train_test_split(test_size=test_size)
        
        # Train model
        print(f"\nTraining model...")
        try:
            train_metrics = model.fit(train_df)
            print(f"Training completed successfully")
            if isinstance(train_metrics, dict):
                for key, value in train_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        except Exception as e:
            print(f"Training failed: {e}")
            return {'error': str(e), 'status': 'failed'}
        
        # Evaluate on test data
        print(f"\nEvaluating on test data...")
        try:
            test_predictions = []
            test_actuals = []
            
            for _, row in test_df.iterrows():
                # Create User and Action objects
                user = User(user_id=row['user_id'], features=row['user_features'])
                action = Action(action_id=row['action_id'], 
                               text=row.get('action_text', f"Action {row['action_id']}"),
                               embedding=row['action_embedding'])
                
                # Get prediction
                pred = model.predict(user, action)
                test_predictions.append(pred)
                test_actuals.append(row['reward'])
            
            # Calculate metrics
            test_predictions = np.array(test_predictions)
            test_actuals = np.array(test_actuals)
            
            test_auc = roc_auc_score(test_actuals, test_predictions)
            test_accuracy = accuracy_score(test_actuals, test_predictions > 0.5)
            
            print(f"Test Results:")
            print(f"  AUC: {test_auc:.4f}")
            print(f"  Accuracy: {test_accuracy:.4f}")
            
            eval_metrics = {
                'test_auc': test_auc,
                'test_accuracy': test_accuracy,
                'n_test_samples': len(test_df),
                'predictions': test_predictions,
                'actuals': test_actuals
            }
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            eval_metrics = {'error': str(e)}
        
        return {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'model': model,
            'train_data': train_df,
            'test_data': test_df
        }
    
    def compare_models(self, models: Dict[str, BaseUserPreferenceModel], test_size: float = 0.2) -> pd.DataFrame:
        """
        Compare multiple user preference models.
        
        Args:
            models: Dictionary of model_name -> model_instance
            test_size: Test split fraction
        
        Returns:
            DataFrame with comparison results
        """
        if self.observations_df is None:
            self.load_data()
        
        print(f"\n{'='*60}")
        print(f"COMPARING {len(models)} USER PREFERENCE MODELS")
        print(f"{'='*60}")
        
        results = []
        
        for model_name, model in models.items():
            print(f"\n--- Testing {model_name} ---")
            try:
                result = self.train_and_evaluate_model(model, test_size=test_size)
                
                if 'error' in result:
                    results.append({
                        'Model': model_name,
                        'Status': 'Failed',
                        'Error': result['error'],
                        'Train_AUC': np.nan,
                        'Test_AUC': np.nan,
                        'Test_Accuracy': np.nan
                    })
                else:
                    train_metrics = result['train_metrics']
                    eval_metrics = result['eval_metrics']
                    
                    results.append({
                        'Model': model_name,
                        'Status': 'Success',
                        'Error': '',
                        'Train_AUC': train_metrics.get('train_auc', np.nan),
                        'Test_AUC': eval_metrics.get('test_auc', np.nan),
                        'Test_Accuracy': eval_metrics.get('test_accuracy', np.nan),
                        'N_Test_Samples': eval_metrics.get('n_test_samples', np.nan)
                    })
                    
            except Exception as e:
                print(f"Failed to test {model_name}: {e}")
                results.append({
                    'Model': model_name,
                    'Status': 'Failed',
                    'Error': str(e),
                    'Train_AUC': np.nan,
                    'Test_AUC': np.nan,
                    'Test_Accuracy': np.nan
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Sort by Test_AUC descending
        comparison_df = comparison_df.sort_values('Test_AUC', ascending=False, na_last=True)
        
        print(f"\n{'='*60}")
        print("FINAL COMPARISON RESULTS")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def evaluate_action_selection_quality(self, model: BaseUserPreferenceModel, 
                                         ground_truth_model: BaseUserPreferenceModel,
                                         n_actions_to_select: int = 5,
                                         n_users_sample: int = 100) -> Dict[str, Any]:
        """
        Evaluate how well the model's action selection aligns with ground truth optimal actions.
        
        This compares:
        1. Actions selected by the model being evaluated
        2. Actions that would be selected by the ground truth model
        3. The ground truth scores for both sets of actions
        
        Args:
            model: The model being evaluated
            ground_truth_model: The ground truth model (oracle)
            n_actions_to_select: Number of top actions to select for each user
            n_users_sample: Number of users to sample for evaluation
            
        Returns:
            Dictionary with action selection quality metrics
        """
        if self.observations_df is None:
            self.load_data()
        
        if not model.is_fitted:
            raise ValueError("Model must be fitted before action selection evaluation")
            
        print(f"\n{'='*50}")
        print("EVALUATING ACTION SELECTION QUALITY")
        print(f"{'='*50}")
        print(f"Comparing model vs ground truth action selection")
        print(f"Actions to select per user: {n_actions_to_select}")
        print(f"Users to evaluate: {n_users_sample}")
        
        # Sample users for evaluation
        sampled_users = np.random.choice(self.users, size=min(n_users_sample, len(self.users)), replace=False)
        
        results = {
            'model_selected_actions': [],
            'gt_selected_actions': [],
            'model_gt_scores': [],  # Ground truth scores for model-selected actions
            'optimal_gt_scores': [],  # Ground truth scores for GT-selected actions  
            'action_overlap': [],  # Overlap between model and GT selections
            'users_evaluated': len(sampled_users),
            'actions_per_user': n_actions_to_select
        }
        
        for user in sampled_users:
            # Get model predictions for all actions for this user
            model_scores = []
            gt_scores = []
            
            for action in self.actions:
                model_score = model.predict(user, action)
                gt_score = ground_truth_model.predict(user, action)
                model_scores.append(model_score)
                gt_scores.append(gt_score)
            
            model_scores = np.array(model_scores)
            gt_scores = np.array(gt_scores)
            
            # Select top actions according to each model
            model_top_indices = np.argsort(model_scores)[-n_actions_to_select:]
            gt_top_indices = np.argsort(gt_scores)[-n_actions_to_select:]
            
            # Store selected actions
            model_selected = [self.actions[i].action_id for i in model_top_indices]
            gt_selected = [self.actions[i].action_id for i in gt_top_indices]
            
            results['model_selected_actions'].append(model_selected)
            results['gt_selected_actions'].append(gt_selected)
            
            # Calculate ground truth scores for selected actions
            model_selected_gt_scores = gt_scores[model_top_indices]
            optimal_gt_scores = gt_scores[gt_top_indices]
            
            results['model_gt_scores'].append(model_selected_gt_scores.mean())
            results['optimal_gt_scores'].append(optimal_gt_scores.mean())
            
            # Calculate action overlap (Jaccard similarity)
            model_set = set(model_selected)
            gt_set = set(gt_selected)
            overlap = len(model_set.intersection(gt_set)) / len(model_set.union(gt_set))
            results['action_overlap'].append(overlap)
        
        # Calculate aggregate metrics
        results['mean_model_gt_score'] = np.mean(results['model_gt_scores'])
        results['mean_optimal_gt_score'] = np.mean(results['optimal_gt_scores'])
        results['mean_action_overlap'] = np.mean(results['action_overlap'])
        
        # Calculate relative performance
        optimal_scores = np.array(results['optimal_gt_scores'])
        model_scores = np.array(results['model_gt_scores'])
        
        # Avoid division by zero
        relative_performance = np.where(optimal_scores > 0, 
                                       model_scores / optimal_scores, 
                                       np.nan)
        results['mean_relative_performance'] = np.nanmean(relative_performance)
        
        # Score gap (how much worse the model's selections are)
        results['mean_score_gap'] = results['mean_optimal_gt_score'] - results['mean_model_gt_score']
        
        print(f"\nAction Selection Quality Results:")
        print(f"  Mean GT score for model selections: {results['mean_model_gt_score']:.4f}")
        print(f"  Mean GT score for optimal selections: {results['mean_optimal_gt_score']:.4f}")
        print(f"  Mean score gap (optimal - model): {results['mean_score_gap']:.4f}")
        print(f"  Mean relative performance: {results['mean_relative_performance']:.4f}")
        print(f"  Mean action overlap (Jaccard): {results['mean_action_overlap']:.4f}")
        
        # Interpretation
        if results['mean_relative_performance'] > 0.9:
            interpretation = "Excellent - model selections are very close to optimal"
        elif results['mean_relative_performance'] > 0.8:
            interpretation = "Good - model selections are reasonably close to optimal"
        elif results['mean_relative_performance'] > 0.7:
            interpretation = "Fair - model selections have some gap from optimal"
        else:
            interpretation = "Poor - model selections are far from optimal"
            
        print(f"  Interpretation: {interpretation}")
        
        return results
    
    def plot_predictions(self, predictions: np.ndarray, actuals: np.ndarray, title: str = "Model Predictions"):
        """Plot prediction distribution and calibration."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Prediction distribution by class
        ax1.hist(predictions[actuals == 0], bins=30, alpha=0.6, label='Negative', density=True)
        ax1.hist(predictions[actuals == 1], bins=30, alpha=0.6, label='Positive', density=True)
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{title} - Prediction Distribution')
        ax1.legend()
        
        # Calibration plot
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(actuals, predictions, n_bins=10)
        ax2.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Model')
        ax2.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        ax2.set_xlabel('Mean Predicted Probability')
        ax2.set_ylabel('Fraction of Positives')
        ax2.set_title(f'{title} - Calibration Plot')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_action_selection_comparison(self, action_selection_results: Dict[str, Any], title: str = "Action Selection Quality"):
        """
        Plot action selection quality comparison results.
        
        Args:
            action_selection_results: Results from evaluate_action_selection_quality()
            title: Plot title
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Score comparison - model vs optimal
        model_scores = action_selection_results['model_gt_scores']
        optimal_scores = action_selection_results['optimal_gt_scores']
        
        ax1.scatter(optimal_scores, model_scores, alpha=0.6)
        ax1.plot([min(optimal_scores), max(optimal_scores)], [min(optimal_scores), max(optimal_scores)], 
                'r--', label='Perfect Agreement')
        ax1.set_xlabel('Optimal GT Score')
        ax1.set_ylabel('Model Selection GT Score')
        ax1.set_title('Ground Truth Scores: Model vs Optimal Selections')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Action overlap distribution
        overlaps = action_selection_results['action_overlap']
        ax2.hist(overlaps, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(overlaps), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(overlaps):.3f}')
        ax2.set_xlabel('Action Overlap (Jaccard Similarity)')
        ax2.set_ylabel('Number of Users')
        ax2.set_title('Distribution of Action Selection Overlap')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Score gap distribution
        score_gaps = np.array(optimal_scores) - np.array(model_scores)
        ax3.hist(score_gaps, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax3.axvline(np.mean(score_gaps), color='red', linestyle='--', 
                   label=f'Mean Gap: {np.mean(score_gaps):.4f}')
        ax3.set_xlabel('Score Gap (Optimal - Model)')
        ax3.set_ylabel('Number of Users')
        ax3.set_title('Distribution of Score Gaps')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Relative performance distribution  
        relative_perf = np.array(model_scores) / np.array(optimal_scores)
        relative_perf = relative_perf[np.isfinite(relative_perf)]  # Remove inf/nan
        
        ax4.hist(relative_perf, bins=20, alpha=0.7, edgecolor='black', color='green')
        ax4.axvline(np.mean(relative_perf), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(relative_perf):.3f}')
        ax4.axvline(1.0, color='blue', linestyle='-', alpha=0.5,
                   label='Perfect (1.0)')
        ax4.set_xlabel('Relative Performance (Model/Optimal)')
        ax4.set_ylabel('Number of Users')  
        ax4.set_title('Distribution of Relative Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title} - Action Selection Analysis', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\\nAction Selection Summary Statistics:")
        print(f"  Score gap - Mean: {np.mean(score_gaps):.4f}, Std: {np.std(score_gaps):.4f}")
        print(f"  Relative performance - Mean: {np.mean(relative_perf):.4f}, Std: {np.std(relative_perf):.4f}")
        print(f"  Action overlap - Mean: {np.mean(overlaps):.4f}, Std: {np.std(overlaps):.4f}")
        print(f"  Users with >80% relative performance: {np.sum(relative_perf > 0.8) / len(relative_perf) * 100:.1f}%")
        print(f"  Users with >50% action overlap: {np.sum(np.array(overlaps) > 0.5) / len(overlaps) * 100:.1f}%")
