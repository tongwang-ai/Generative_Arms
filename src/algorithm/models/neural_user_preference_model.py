"""
Neural Network User Preference Model

A deep neural network model for user preference prediction using PyTorch.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base_user_preference_model import BaseUserPreferenceModel
from src.data.entities import User, Action


class NeuralNetwork(nn.Module):
    """Neural network architecture for user preference prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class NeuralUserPreferenceModel(BaseUserPreferenceModel):
    """
    Neural Network User Preference Model with configurable architecture.
    """
    
    def __init__(self, hidden_dims=[128, 64], dropout_rate=0.3, learning_rate=0.001,
                 weight_decay=0.01, use_batch_norm=True, use_pca=False, pca_components=50,
                 batch_size=32, epochs=100, patience=10, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_batch_norm = use_batch_norm
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Model components
        self.model = None
        self.pca = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
    
    def fit(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit neural network with optional PCA."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        # Extract features and labels
        user_features = np.vstack(observations_df['user_features'].values)  # (N, 8)
        action_embeddings = np.vstack(observations_df['action_embedding'].values)  # (N, 1536)
        rewards = observations_df['reward'].values  # (N,)
        
        print(f"Neural Network fitting on {len(observations_df)} samples")
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
        features = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, rewards, test_size=0.2, random_state=42, stratify=rewards
        )
        
        # Create model
        input_dim = features.shape[1]
        self.model = NeuralNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # Calculate metrics
                val_pred = val_outputs.cpu().numpy().flatten()
                val_auc = roc_auc_score(y_val, val_pred)
                val_accuracy = accuracy_score(y_val, val_pred > 0.5)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            train_outputs = self.model(X_train_tensor)
            train_pred = train_outputs.cpu().numpy().flatten()
            train_auc = roc_auc_score(y_train, train_pred)
            train_accuracy = accuracy_score(y_train, train_pred > 0.5)
        
        self.is_fitted = True
        
        return {
            'train_auc': train_auc,
            'train_accuracy': train_accuracy,
            'val_auc': val_auc,
            'val_accuracy': val_accuracy,
            'n_samples': len(observations_df),
            'positive_rate': rewards.mean(),
            'use_pca': self.use_pca,
            'pca_components': self.pca_components if self.use_pca else None,
            'pca_explained_variance': self.pca.explained_variance_ratio_.sum() if self.use_pca else None,
            'feature_dim': features.shape[1],
            'model_type': f'neural_pca{self.pca_components}' if self.use_pca else 'neural',
            'hidden_dims': self.hidden_dims,
            'epochs_trained': epoch + 1
        }
    
    def predict(self, user: User, action: Action) -> float:
        """Predict using neural network."""
        if not self.is_fitted:
            return 0.5
            
        # Prepare features
        action_embedding = action.embedding
        if self.use_pca:
            action_embedding = self.pca.transform(action_embedding.reshape(1, -1)).flatten()
        
        features = np.concatenate([user.features, action_embedding]).reshape(1, -1)
        features = self.scaler.transform(features)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            output = self.model(features_tensor)
            prob = output.cpu().numpy()[0, 0]
        
        return float(prob)
