import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class User:
    user_id: str
    features: np.ndarray
    weight: float = 1.0
    metadata: Optional[Dict[str, Any]] = field(default=None)


@dataclass 
class Action:
    action_id: str
    text: str
    embedding: np.ndarray
