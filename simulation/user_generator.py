import numpy as np
import pandas as pd
from typing import List, Dict, Any
import json
from dataclasses import dataclass


@dataclass
class MeaningfulUser:
    """User with meaningful, interpretable features."""
    user_id: str
    age: float  # Normalized age (0-1)
    income_level: float  # Normalized income level (0-1) 
    tech_savviness: float  # How comfortable with technology (0-1)
    price_sensitivity: float  # How price-conscious (0-1)
    brand_loyalty: float  # Tendency to stick with known brands (0-1)
    social_influence: float  # Influenced by social factors (0-1)
    urgency_response: float  # Response to urgent/limited time offers (0-1)
    quality_focus: float  # Focus on quality over price (0-1)
    feature_vector: np.ndarray  # Combined feature vector
    segment: str  # User segment for interpretability


class MeaningfulUserGenerator:
    """
    Generates users with meaningful, interpretable features.
    Users are created with realistic demographic and behavioral characteristics.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define user archetypes
        self.archetypes = {
            'young_tech_savvy': {
                'age': (0.2, 0.4),  # 20-40% of age range (young adults)
                'income_level': (0.3, 0.7),
                'tech_savviness': (0.7, 0.95),
                'price_sensitivity': (0.6, 0.9),
                'brand_loyalty': (0.2, 0.5),
                'social_influence': (0.7, 0.9),
                'urgency_response': (0.6, 0.8),
                'quality_focus': (0.4, 0.7),
                'weight': 0.25
            },
            'middle_aged_professional': {
                'age': (0.4, 0.7),  # 40-70% of age range
                'income_level': (0.6, 0.9),
                'tech_savviness': (0.5, 0.8),
                'price_sensitivity': (0.3, 0.6),
                'brand_loyalty': (0.6, 0.9),
                'social_influence': (0.3, 0.6),
                'urgency_response': (0.4, 0.7),
                'quality_focus': (0.7, 0.95),
                'weight': 0.30
            },
            'budget_conscious_family': {
                'age': (0.3, 0.6),  # 30-60% of age range
                'income_level': (0.2, 0.5),
                'tech_savviness': (0.3, 0.7),
                'price_sensitivity': (0.8, 0.95),
                'brand_loyalty': (0.4, 0.7),
                'social_influence': (0.5, 0.8),
                'urgency_response': (0.7, 0.9),
                'quality_focus': (0.3, 0.6),
                'weight': 0.25
            },
            'premium_customer': {
                'age': (0.5, 0.8),  # 50-80% of age range (older, established)
                'income_level': (0.8, 0.95),
                'tech_savviness': (0.4, 0.7),
                'price_sensitivity': (0.1, 0.3),
                'brand_loyalty': (0.8, 0.95),
                'social_influence': (0.2, 0.5),
                'urgency_response': (0.2, 0.5),
                'quality_focus': (0.8, 0.95),
                'weight': 0.20
            }
        }
    
    def generate_users(self, n_users: int) -> List[MeaningfulUser]:
        """
        Generate users with meaningful features based on realistic archetypes.
        
        Args:
            n_users: Number of users to generate
            
        Returns:
            List of MeaningfulUser objects
        """
        users = []
        
        # Determine how many users of each archetype to create
        archetype_names = list(self.archetypes.keys())
        weights = [self.archetypes[name]['weight'] for name in archetype_names]
        
        # Sample archetype assignments
        archetype_assignments = np.random.choice(
            archetype_names, 
            size=n_users, 
            p=weights
        )
        
        for i in range(n_users):
            archetype_name = archetype_assignments[i]
            archetype = self.archetypes[archetype_name]
            
            # Generate features based on archetype ranges
            age = np.random.uniform(*archetype['age'])
            income_level = np.random.uniform(*archetype['income_level'])
            tech_savviness = np.random.uniform(*archetype['tech_savviness'])
            price_sensitivity = np.random.uniform(*archetype['price_sensitivity'])
            brand_loyalty = np.random.uniform(*archetype['brand_loyalty'])
            social_influence = np.random.uniform(*archetype['social_influence'])
            urgency_response = np.random.uniform(*archetype['urgency_response'])
            quality_focus = np.random.uniform(*archetype['quality_focus'])
            
            # Add some noise to make users more realistic
            noise_factor = 0.1
            features = [age, income_level, tech_savviness, price_sensitivity, 
                       brand_loyalty, social_influence, urgency_response, quality_focus]
            
            # Add correlated noise
            noise = np.random.normal(0, noise_factor, len(features))
            features = np.array(features) + noise
            features = np.clip(features, 0, 1)  # Keep in [0,1] range
            
            # Create feature vector
            feature_vector = features
            
            # Create user
            user = MeaningfulUser(
                user_id=f"user_{i:06d}",
                age=features[0],
                income_level=features[1],
                tech_savviness=features[2],
                price_sensitivity=features[3],
                brand_loyalty=features[4],
                social_influence=features[5],
                urgency_response=features[6],
                quality_focus=features[7],
                feature_vector=feature_vector,
                segment=archetype_name
            )
            
            users.append(user)
        
        return users
    
    def save_users(self, users: List[MeaningfulUser], filepath: str):
        """Save users to JSON file with interpretable format."""
        users_data = []
        
        for user in users:
            user_dict = {
                'user_id': user.user_id,
                'segment': user.segment,
                'demographics': {
                    'age': float(user.age),
                    'income_level': float(user.income_level)
                },
                'behavioral_traits': {
                    'tech_savviness': float(user.tech_savviness),
                    'price_sensitivity': float(user.price_sensitivity),
                    'brand_loyalty': float(user.brand_loyalty),
                    'social_influence': float(user.social_influence),
                    'urgency_response': float(user.urgency_response),
                    'quality_focus': float(user.quality_focus)
                },
                'feature_vector': user.feature_vector.tolist()
            }
            users_data.append(user_dict)
        
        with open(filepath, 'w') as f:
            json.dump({
                'users': users_data,
                'feature_dimensions': [
                    'age', 'income_level', 'tech_savviness', 'price_sensitivity',
                    'brand_loyalty', 'social_influence', 'urgency_response', 'quality_focus'
                ],
                'segments': list(self.archetypes.keys()),
                'total_users': len(users)
            }, f, indent=2)
    
    def load_users(self, filepath: str) -> List[MeaningfulUser]:
        """Load users from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        users = []
        for user_data in data['users']:
            user = MeaningfulUser(
                user_id=user_data['user_id'],
                age=user_data['demographics']['age'],
                income_level=user_data['demographics']['income_level'],
                tech_savviness=user_data['behavioral_traits']['tech_savviness'],
                price_sensitivity=user_data['behavioral_traits']['price_sensitivity'],
                brand_loyalty=user_data['behavioral_traits']['brand_loyalty'],
                social_influence=user_data['behavioral_traits']['social_influence'],
                urgency_response=user_data['behavioral_traits']['urgency_response'],
                quality_focus=user_data['behavioral_traits']['quality_focus'],
                feature_vector=np.array(user_data['feature_vector']),
                segment=user_data['segment']
            )
            users.append(user)
        
        return users
    
    def get_segment_summary(self, users: List[MeaningfulUser]) -> Dict[str, Any]:
        """Generate summary statistics by user segment."""
        segment_stats = {}
        
        for segment in self.archetypes.keys():
            segment_users = [u for u in users if u.segment == segment]
            
            if segment_users:
                # Calculate average characteristics
                features = np.array([u.feature_vector for u in segment_users])
                feature_means = np.mean(features, axis=0)
                
                segment_stats[segment] = {
                    'count': len(segment_users),
                    'percentage': len(segment_users) / len(users) * 100,
                    'avg_characteristics': {
                        'age': float(feature_means[0]),
                        'income_level': float(feature_means[1]),
                        'tech_savviness': float(feature_means[2]),
                        'price_sensitivity': float(feature_means[3]),
                        'brand_loyalty': float(feature_means[4]),
                        'social_influence': float(feature_means[5]),
                        'urgency_response': float(feature_means[6]),
                        'quality_focus': float(feature_means[7])
                    }
                }
        
        return segment_stats