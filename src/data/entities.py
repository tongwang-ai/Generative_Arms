import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    user_id: str
    features: np.ndarray


@dataclass 
class Action:
    action_id: str
    text: str
    embedding: np.ndarray