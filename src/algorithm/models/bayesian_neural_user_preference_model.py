"""
Bayesian Neural Network User Preference Model (MC Dropout)

Extends the neural model by estimating predictive uncertainty via Monte Carlo Dropout.
Returns both mean prediction and an uncertainty estimate (std over MC samples).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

import torch
import torch.nn as nn

from .neural_user_preference_model import NeuralUserPreferenceModel, NeuralNetwork
from src.data.entities import User, Action


class BayesianNeuralUserPreferenceModel(NeuralUserPreferenceModel):
    """
    Bayesian Neural Network using MC Dropout for uncertainty.
    """

    def __init__(self, mc_samples: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.mc_samples = mc_samples

    def _forward_with_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dropout layers active (keeps other layers in eval mode).
        """
        # Ensure base model is in eval to keep e.g. BatchNorm stable, then
        # flip Dropout submodules to train mode to activate stochasticity.
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        with torch.no_grad():
            return self.model(x)

    def predict(self, user: User, action: Action) -> float:
        """Return MC mean prediction as the point estimate."""
        if not self.is_fitted:
            return 0.5

        # Prepare features as in base class
        action_embedding = action.embedding
        if self.use_pca:
            action_embedding = self.pca.transform(action_embedding.reshape(1, -1)).flatten()
        features = np.concatenate([user.features, action_embedding]).reshape(1, -1)
        features = self.scaler.transform(features)

        x = torch.FloatTensor(features).to(self.device)

        preds = []
        # Use MC dropout for mean estimate
        for _ in range(max(1, self.mc_samples)):
            out = self._forward_with_dropout(x)
            preds.append(out.cpu().numpy()[0, 0])
        mean = float(np.mean(preds))
        return mean

    def predict_with_uncertainty(self, user: User, action: Action) -> tuple:
        """
        Returns (mean, std) using MC Dropout.
        """
        if not self.is_fitted:
            return 0.5, 0.5

        action_embedding = action.embedding
        if self.use_pca:
            action_embedding = self.pca.transform(action_embedding.reshape(1, -1)).flatten()
        features = np.concatenate([user.features, action_embedding]).reshape(1, -1)
        features = self.scaler.transform(features)

        x = torch.FloatTensor(features).to(self.device)

        preds = []
        for _ in range(max(2, self.mc_samples)):
            out = self._forward_with_dropout(x)
            preds.append(out.cpu().numpy()[0, 0])

        preds = np.array(preds)
        mean = float(preds.mean())
        std = float(preds.std())
        return mean, std

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'bayesian': True,
            'mc_samples': self.mc_samples
        })
        return info
