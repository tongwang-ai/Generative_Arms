"""
FT-Transformer User Preference Model

This module implements a user preference model that uses an FT-Transformer
encoder for tabular user features, a lightweight encoder for (precomputed)
OpenAI text embeddings of actions, an information fusion module (concat or
attention-based), and a prediction head to estimate P(y=1 | user, action).

References:
- FT-Transformer (Gorishniy et al., 2021): feature tokenization + Transformer
"""

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from .base_user_preference_model import BaseUserPreferenceModel
from ..data.entities import User, Action


class FeatureTokenizer(nn.Module):
    """
    Tokenizes continuous features into per-feature tokens for Transformer.

    For each continuous feature x_i (scalar), produce a token vector:
        t_i = x_i * W_i + b_i,  where W_i and b_i are learnable (d_model dims)

    Adds a learned CLS token at position 0.
    Output shape: (batch, n_features + 1, d_model)
    """

    def __init__(self, n_features: int, d_model: int, cls: bool = True):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.use_cls = cls

        # Per-feature weights and bias
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

        # Learned CLS token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter('cls_token', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features) continuous features
        Returns:
            tokens: (batch, n_features+1, d_model) if cls, else (batch, n_features, d_model)
        """
        # x -> (batch, n_features, 1)
        x_expanded = x.unsqueeze(-1)
        # Broadcast multiply and add bias -> (batch, n_features, d_model)
        tokens = x_expanded * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        if self.use_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens


class FTTransformerEncoder(nn.Module):
    """
    A small Transformer encoder for tabular tokens.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ff_multiplier: int = 4, dropout: float = 0.1, use_cls_pool: bool = True):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.use_cls_pool = use_cls_pool

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, seq_len, d_model)
        h = self.encoder(tokens)
        h = self.norm(h)
        if self.use_cls_pool:
            # Return CLS token representation (position 0)
            return h[:, 0, :]
        else:
            # Mean pool all tokens
            return h.mean(dim=1)


class ActionEncoder(nn.Module):
    """Lightweight encoder for precomputed action embeddings."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        h = self.proj(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class AttentionFusion(nn.Module):
    """
    Attention-based fusion between user and action representations.

    Uses MultiheadAttention with user as Query and action as Key/Value.
    Returns a fused vector via concatenation + small MLP projection.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1, fused_dim: Optional[int] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        fused_dim = fused_dim or hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, user_repr: torch.Tensor, action_repr: torch.Tensor) -> torch.Tensor:
        # user_repr, action_repr: (batch, hidden_dim)
        q = user_repr.unsqueeze(1)   # (batch, 1, hidden)
        k = action_repr.unsqueeze(1) # (batch, 1, hidden)
        v = action_repr.unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)  # (batch, 1, hidden)
        fused = torch.cat([user_repr, attn_out.squeeze(1)], dim=-1)
        return self.ffn(fused)


class ConcatFusion(nn.Module):
    """Simple concatenation fusion with a small projection network."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, fused_dim: Optional[int] = None):
        super().__init__()
        fused_dim = fused_dim or hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, user_repr: torch.Tensor, action_repr: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([user_repr, action_repr], dim=-1))


class PreferenceHead(nn.Module):
    """Prediction head producing a single logit for BCEWithLogits."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, head_dims: Optional[List[int]] = None):
        super().__init__()
        layers: List[nn.Module] = []
        last = hidden_dim
        head_dims = head_dims or [hidden_dim]
        for dim in head_dims:
            layers += [nn.Linear(last, dim), nn.GELU(), nn.Dropout(dropout)]
            last = dim
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Returns logits (no sigmoid)
        return self.net(h).squeeze(-1)


class FTTransformerPreferenceTorchModel(nn.Module):
    """
    End-to-end torch model: FT-Transformer user encoder + action encoder + fusion + head.
    """

    def __init__(self, n_user_features: int,
                 action_in_dim: int,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 fusion_type: str = 'concat',  # 'concat' or 'attention'
                 head_dims: Optional[List[int]] = None):
        super().__init__()

        self.tokenizer = FeatureTokenizer(n_user_features, hidden_dim, cls=True)
        self.user_encoder = FTTransformerEncoder(
            d_model=hidden_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout, use_cls_pool=True
        )
        self.action_encoder = ActionEncoder(action_in_dim, hidden_dim, dropout=dropout)

        if fusion_type not in ('concat', 'attention'):
            raise ValueError("fusion_type must be 'concat' or 'attention'")
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(hidden_dim, dropout=dropout)
        else:
            self.fusion = AttentionFusion(hidden_dim, n_heads=n_heads, dropout=dropout)

        self.head = PreferenceHead(hidden_dim, dropout=dropout, head_dims=head_dims)

    def forward(self, user_x: torch.Tensor, action_x: torch.Tensor) -> torch.Tensor:
        # user_x: (batch, n_feat), action_x: (batch, action_in_dim)
        user_tokens = self.tokenizer(user_x)
        user_repr = self.user_encoder(user_tokens)           # (batch, hidden)
        action_repr = self.action_encoder(action_x)          # (batch, hidden)
        fused = self.fusion(user_repr, action_repr)          # (batch, hidden)
        logits = self.head(fused)                            # (batch,)
        return logits


class FTTransformerUserPreferenceModel(BaseUserPreferenceModel):
    """
    User preference model with FT-Transformer encoder for user tabular features,
    action embedding projection, fusion (concat/attention), and a prediction head.
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout_rate: float = 0.1,
                 fusion_type: str = 'concat',  # 'concat' or 'attention'
                 head_dims: Optional[List[int]] = None,
                 use_pca: bool = False,
                 pca_components: int = 256,
                 batch_size: int = 128,
                 epochs: int = 50,
                 patience: int = 8,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.fusion_type = fusion_type
        self.head_dims = head_dims
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.user_scaler = StandardScaler()
        self.pca = None  # Fitted on action embeddings if use_pca

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        self.model: Optional[FTTransformerPreferenceTorchModel] = None

    # --- Fitting ---
    def fit(self, observations_df: pd.DataFrame) -> Dict[str, Any]:
        # Extract arrays
        user_features = np.vstack(observations_df['user_features'].values).astype(np.float32)
        action_embeddings = np.vstack(observations_df['action_embedding'].values).astype(np.float32)
        rewards = observations_df['reward'].values.astype(np.float32)

        n_user_features = user_features.shape[1]
        action_in_dim = action_embeddings.shape[1]

        # Optional PCA on action embeddings (can reduce 1536/3072 -> pca_components)
        if self.use_pca:
            from sklearn.decomposition import PCA
            target_components = min(self.pca_components, action_in_dim)
            self.pca = PCA(n_components=target_components, random_state=42)
            action_embeddings = self.pca.fit_transform(action_embeddings).astype(np.float32)
            action_in_dim = action_embeddings.shape[1]

        # Scale user features
        user_features = self.user_scaler.fit_transform(user_features).astype(np.float32)

        # Train/val split
        try:
            X_u_train, X_u_val, X_a_train, X_a_val, y_train, y_val = train_test_split(
                user_features, action_embeddings, rewards, test_size=0.2, random_state=42, stratify=rewards
            )
        except ValueError:
            X_u_train, X_u_val, X_a_train, X_a_val, y_train, y_val = train_test_split(
                user_features, action_embeddings, rewards, test_size=0.2, random_state=42
            )

        # Datasets and loaders
        train_ds = TensorDataset(torch.from_numpy(X_u_train), torch.from_numpy(X_a_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_u_val), torch.from_numpy(X_a_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Build model
        self.model = FTTransformerPreferenceTorchModel(
            n_user_features=n_user_features,
            action_in_dim=action_in_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout_rate,
            fusion_type=self.fusion_type,
            head_dims=self.head_dims
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_val_loss = float('inf')
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_losses: List[float] = []
            for u_batch, a_batch, y_batch in train_loader:
                u_batch = u_batch.to(self.device)
                a_batch = a_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(u_batch, a_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_losses: List[float] = []
                all_logits: List[np.ndarray] = []
                all_targets: List[np.ndarray] = []
                for u_batch, a_batch, y_batch in val_loader:
                    u_batch = u_batch.to(self.device)
                    a_batch = a_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = self.model(u_batch, a_batch)
                    loss = criterion(logits, y_batch)
                    val_losses.append(float(loss.item()))
                    all_logits.append(logits.detach().cpu().numpy())
                    all_targets.append(y_batch.detach().cpu().numpy())

                val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                preds = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits))) if all_logits else np.array([])
                targs = np.concatenate(all_targets) if all_targets else np.array([])
                if targs.size > 0 and len(np.unique(targs)) > 1:
                    val_auc = float(roc_auc_score(targs, preds))
                    val_acc = float(accuracy_score(targs, preds > 0.5))
                else:
                    val_auc = float('nan')
                    val_acc = float('nan')

            # Early stopping on val loss
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Load best state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final metrics on train (quick pass)
        self.model.eval()
        with torch.no_grad():
            train_logits = []
            for u_batch, a_batch, _ in train_loader:
                u_batch = u_batch.to(self.device)
                a_batch = a_batch.to(self.device)
                train_logits.append(self.model(u_batch, a_batch).detach().cpu().numpy())
            train_preds = 1.0 / (1.0 + np.exp(-np.concatenate(train_logits))) if train_logits else np.array([])
            train_auc = float(roc_auc_score(y_train, train_preds)) if y_train.size and len(np.unique(y_train)) > 1 else float('nan')
            train_acc = float(accuracy_score(y_train, train_preds > 0.5)) if y_train.size else float('nan')

        self.is_fitted = True

        return {
            'status': 'trained',
            'n_samples': int(len(observations_df)),
            'train_auc': train_auc,
            'train_accuracy': train_acc,
            'val_auc': val_auc,
            'val_accuracy': val_acc,
            'use_pca': self.use_pca,
            'pca_components': (self.pca_components if self.use_pca else None),
            'model_type': 'ft_transformer',
            'fusion_type': self.fusion_type,
            'hidden_dim': self.hidden_dim,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers
        }

    # --- Prediction ---
    def _prepare_single(self, user: User, action: Action) -> Tuple[np.ndarray, np.ndarray]:
        u = user.features.astype(np.float32)
        a = action.embedding.astype(np.float32)
        u = self.user_scaler.transform(u.reshape(1, -1)).astype(np.float32)
        if self.use_pca and self.pca is not None:
            a = self.pca.transform(a.reshape(1, -1)).astype(np.float32)
        else:
            a = a.reshape(1, -1)
        return u, a

    def predict(self, user: User, action: Action) -> float:
        if not self.is_fitted or self.model is None:
            return 0.5
        u, a = self._prepare_single(user, action)
        u_t = torch.from_numpy(u).to(self.device)
        a_t = torch.from_numpy(a).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logit = self.model(u_t, a_t).cpu().numpy()[0]
            prob = 1.0 / (1.0 + np.exp(-logit))
            return float(prob)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'ft_transformer': True,
            'fusion_type': self.fusion_type,
            'hidden_dim': self.hidden_dim,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components
        })
        return info

