import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Optional, Tuple
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
    cluster_id: str = ""
    weight: float = 1.0


class MeaningfulUserGenerator:
    """
    Generates users with meaningful, interpretable features.
    Users are created with realistic demographic and behavioral characteristics.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)

        # Define consistent 30-dimensional feature schema
        self.feature_names = [
            'age_norm',
            'income_level',
            'tech_savviness',
            'price_sensitivity',
            'brand_loyalty',
            'social_influence',
            'urgency_response',
            'quality_focus',
            'budget_flexibility',
            'loyalty_program_participation',
            'remote_work_adoption',
            'professional_network_size',
            'content_consumption_depth',
            'mobile_usage_intensity',
            'email_engagement_rate',
            'webinar_interest_level',
            'ai_tool_interest',
            'experimentation_willingness',
            'risk_tolerance',
            'support_preference',
            'customization_preference',
            'collaboration_orientation',
            'decision_speed',
            'upgrade_frequency',
            'satisfaction_score',
            'referral_likelihood',
            'analytics_usage',
            'automation_adoption',
            'compliance_sensitivity',
            'sustainability_interest'
        ]

        self.segment_feature_names = [
            'tech_savviness',
            'price_sensitivity',
            'brand_loyalty',
            'social_influence',
            'urgency_response',
            'quality_focus',
            'ai_tool_interest',
            'experimentation_willingness',
            'risk_tolerance',
            'automation_adoption'
        ]
        self.segment_thresholds = {
            'tech_savviness': 0.6,
            'price_sensitivity': 0.55,
            'brand_loyalty': 0.6,
            'social_influence': 0.55,
            'urgency_response': 0.55,
            'quality_focus': 0.6,
            'ai_tool_interest': 0.6,
            'experimentation_willingness': 0.5,
            'risk_tolerance': 0.5,
            'automation_adoption': 0.6
        }

        # Define user archetypes to ground base distributions
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
    
    def _clip(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _compute_segment_id(self, feature_values: Dict[str, float]) -> str:
        bits = []
        for name in self.segment_feature_names:
            threshold = self.segment_thresholds[name]
            value = feature_values.get(name, 0.0)
            bits.append('1' if value >= threshold else '0')
        return f"segment_{''.join(bits)}"

    def _build_feature_profile(self, base_traits: Dict[str, float]) -> Dict[str, float]:
        clip = self._clip

        budget_flexibility = clip(1.0 - base_traits['price_sensitivity'] + self.rng.normal(0, 0.05))
        loyalty_program_participation = clip((base_traits['brand_loyalty'] + base_traits['social_influence']) / 2 + self.rng.normal(0, 0.05))
        remote_work_adoption = clip((base_traits['tech_savviness'] + base_traits['income_level']) / 2 + self.rng.normal(0, 0.05))
        professional_network_size = clip((base_traits['social_influence'] + base_traits['brand_loyalty']) / 2 + self.rng.normal(0, 0.05))
        content_consumption_depth = clip((base_traits['tech_savviness'] + base_traits['quality_focus']) / 2 + self.rng.normal(0, 0.05))
        mobile_usage_intensity = clip((base_traits['tech_savviness'] + base_traits['urgency_response']) / 2 + self.rng.normal(0, 0.05))
        email_engagement_rate = clip((base_traits['brand_loyalty'] + (1 - base_traits['price_sensitivity'])) / 2 + self.rng.normal(0, 0.05))
        webinar_interest_level = clip((base_traits['quality_focus'] + base_traits['social_influence']) / 2 + self.rng.normal(0, 0.05))
        ai_tool_interest = clip((base_traits['tech_savviness'] + base_traits['quality_focus']) / 2 + self.rng.normal(0, 0.05))
        experimentation_willingness = clip((1 - base_traits['brand_loyalty']) * 0.4 + base_traits['tech_savviness'] * 0.6 + self.rng.normal(0, 0.05))
        risk_tolerance = clip((1 - base_traits['price_sensitivity']) * 0.6 + base_traits['tech_savviness'] * 0.4 + self.rng.normal(0, 0.05))
        support_preference = clip((1 - risk_tolerance) * 0.6 + base_traits['quality_focus'] * 0.4 + self.rng.normal(0, 0.05))
        customization_preference = clip((base_traits['quality_focus'] + base_traits['income_level']) / 2 + self.rng.normal(0, 0.05))
        collaboration_orientation = clip((base_traits['social_influence'] + loyalty_program_participation) / 2 + self.rng.normal(0, 0.05))
        decision_speed = clip((base_traits['urgency_response'] + risk_tolerance) / 2 + self.rng.normal(0, 0.05))
        upgrade_frequency = clip((decision_speed + base_traits['tech_savviness']) / 2 + self.rng.normal(0, 0.05))
        satisfaction_score = clip((base_traits['quality_focus'] + loyalty_program_participation) / 2 + self.rng.normal(0, 0.05))
        referral_likelihood = clip((satisfaction_score + collaboration_orientation) / 2 + self.rng.normal(0, 0.05))
        analytics_usage = clip((base_traits['tech_savviness'] + base_traits['income_level']) / 2 + self.rng.normal(0, 0.05))
        automation_adoption = clip((analytics_usage + ai_tool_interest) / 2 + self.rng.normal(0, 0.05))
        compliance_sensitivity = clip((1 - risk_tolerance) * 0.6 + base_traits['quality_focus'] * 0.4 + self.rng.normal(0, 0.05))
        sustainability_interest = clip((base_traits['quality_focus'] + base_traits['social_influence']) / 2 + self.rng.normal(0, 0.05))

        feature_values = {
            'age_norm': clip(base_traits['age_norm']),
            'income_level': clip(base_traits['income_level']),
            'tech_savviness': clip(base_traits['tech_savviness']),
            'price_sensitivity': clip(base_traits['price_sensitivity']),
            'brand_loyalty': clip(base_traits['brand_loyalty']),
            'social_influence': clip(base_traits['social_influence']),
            'urgency_response': clip(base_traits['urgency_response']),
            'quality_focus': clip(base_traits['quality_focus']),
            'budget_flexibility': budget_flexibility,
            'loyalty_program_participation': loyalty_program_participation,
            'remote_work_adoption': remote_work_adoption,
            'professional_network_size': professional_network_size,
            'content_consumption_depth': content_consumption_depth,
            'mobile_usage_intensity': mobile_usage_intensity,
            'email_engagement_rate': email_engagement_rate,
            'webinar_interest_level': webinar_interest_level,
            'ai_tool_interest': ai_tool_interest,
            'experimentation_willingness': experimentation_willingness,
            'risk_tolerance': risk_tolerance,
            'support_preference': support_preference,
            'customization_preference': customization_preference,
            'collaboration_orientation': collaboration_orientation,
            'decision_speed': decision_speed,
            'upgrade_frequency': upgrade_frequency,
            'satisfaction_score': satisfaction_score,
            'referral_likelihood': referral_likelihood,
            'analytics_usage': analytics_usage,
            'automation_adoption': automation_adoption,
            'compliance_sensitivity': compliance_sensitivity,
            'sustainability_interest': sustainability_interest
        }

        return feature_values

    def generate_users(self, n_users: int) -> List[MeaningfulUser]:
        """
        Generate users with meaningful features based on realistic archetypes.
        Handles its own logging so callers don't need to print generation messages.
        When monkeypatched to return pre-generated users, these logs are naturally suppressed.
        
        Args:
            n_users: Number of users to generate
            
        Returns:
            List of MeaningfulUser objects
        """
        print(f"Generating {n_users} NEW users...")
        _start_time = time.time()
        users = []
        
        # Determine how many users of each archetype to create
        archetype_names = list(self.archetypes.keys())
        weights = [self.archetypes[name]['weight'] for name in archetype_names]
        
        # Sample archetype assignments
        archetype_assignments = self.rng.choice(
            archetype_names,
            size=n_users,
            p=weights
        )
        
        for i in range(n_users):
            archetype_name = archetype_assignments[i]
            archetype = self.archetypes[archetype_name]

            # Generate base traits with archetype-driven ranges
            age = self.rng.uniform(*archetype['age'])
            income_level = self.rng.uniform(*archetype['income_level'])
            tech_savviness = self.rng.uniform(*archetype['tech_savviness'])
            price_sensitivity = self.rng.uniform(*archetype['price_sensitivity'])
            brand_loyalty = self.rng.uniform(*archetype['brand_loyalty'])
            social_influence = self.rng.uniform(*archetype['social_influence'])
            urgency_response = self.rng.uniform(*archetype['urgency_response'])
            quality_focus = self.rng.uniform(*archetype['quality_focus'])

            noise = self.rng.normal(0, 0.05, 8)
            base_traits = np.clip(
                np.array([age, income_level, tech_savviness, price_sensitivity,
                          brand_loyalty, social_influence, urgency_response, quality_focus]) + noise,
                0, 1
            )

            base_trait_map = {
                'age_norm': base_traits[0],
                'income_level': base_traits[1],
                'tech_savviness': base_traits[2],
                'price_sensitivity': base_traits[3],
                'brand_loyalty': base_traits[4],
                'social_influence': base_traits[5],
                'urgency_response': base_traits[6],
                'quality_focus': base_traits[7]
            }

            feature_values = self._build_feature_profile(base_trait_map)
            feature_vector = np.array([feature_values[name] for name in self.feature_names], dtype=float)
            segment_id = self._compute_segment_id(feature_values)

            user = MeaningfulUser(
                user_id=f"user_{i:06d}",
                age=feature_values['age_norm'],
                income_level=feature_values['income_level'],
                tech_savviness=feature_values['tech_savviness'],
                price_sensitivity=feature_values['price_sensitivity'],
                brand_loyalty=feature_values['brand_loyalty'],
                social_influence=feature_values['social_influence'],
                urgency_response=feature_values['urgency_response'],
                quality_focus=feature_values['quality_focus'],
                feature_vector=feature_vector,
                segment=segment_id,
                cluster_id=segment_id,
                weight=1.0
            )
            users.append(user)
        
        _elapsed = time.time() - _start_time
        print(f"   ⏱️  User generation completed in {_elapsed:.2f}s")
        return users
    
    def save_users(self, users: List[MeaningfulUser], filepath: str):
        """Save users to JSON file with interpretable format."""
        users_data = []
        
        latent_dim = len(users[0].feature_vector) - 8 if users else 22
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
                'feature_vector': user.feature_vector.tolist(),
                'cluster_id': getattr(user, 'cluster_id', user.segment),
                'weight': float(getattr(user, 'weight', 1.0))
            }
            users_data.append(user_dict)

        with open(filepath, 'w') as f:
            json.dump({
                'users': users_data,
                'feature_dimensions': self.feature_names,
                'feature_vector_length': len(users[0].feature_vector) if users else 0,
                'segment_feature_names': self.segment_feature_names,
                'segments': sorted({user.segment for user in users}),
                'cluster_ids': sorted({getattr(user, 'cluster_id', user.segment) for user in users}),
                'total_users': len(users)
            }, f, indent=2)
    
    def load_users(self, filepath: str) -> List[MeaningfulUser]:
        """Load users from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        users = []
        for user_data in data['users']:
            feature_vector = np.array(user_data['feature_vector'])
            feature_map = dict(zip(data.get('feature_dimensions', self.feature_names), feature_vector))
            segment_id = user_data.get('segment') or self._compute_segment_id(feature_map)

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
                feature_vector=feature_vector,
                segment=segment_id,
                cluster_id=user_data.get('cluster_id', segment_id),
                weight=float(user_data.get('weight', 1.0))
            )
            users.append(user)

        return users

    def group_users_by_segment(self, users: List[MeaningfulUser]) -> Dict[str, List[MeaningfulUser]]:
        segments: Dict[str, List[MeaningfulUser]] = {}
        for user in users:
            feature_map = dict(zip(self.feature_names, user.feature_vector))
            segment_id = user.segment or self._compute_segment_id(feature_map)
            user.segment = segment_id
            user.cluster_id = segment_id
            segments.setdefault(segment_id, []).append(user)
        return segments

    def compute_segment_feature_statistics(self, segment_users: List[MeaningfulUser]) -> Dict[str, np.ndarray]:
        """Compute descriptive statistics for a set of users belonging to a segment."""
        if not segment_users:
            return {}

        feature_matrix = np.array([u.feature_vector for u in segment_users])
        stats = {
            'mean': np.mean(feature_matrix, axis=0),
            'median': np.median(feature_matrix, axis=0),
            'p25': np.percentile(feature_matrix, 25, axis=0),
            'p75': np.percentile(feature_matrix, 75, axis=0),
            'min': np.min(feature_matrix, axis=0),
            'max': np.max(feature_matrix, axis=0),
            'std': np.std(feature_matrix, axis=0)
        }
        return stats

    def build_segment_feature_dataframe(self, segments: Dict[str, List[MeaningfulUser]]) -> pd.DataFrame:
        """Create per-feature statistics for each segment as a DataFrame."""
        records: List[Dict[str, Any]] = []
        for segment_id, members in segments.items():
            stats = self.compute_segment_feature_statistics(members)
            if not stats:
                continue
            count = len(members)
            for idx, feature_name in enumerate(self.feature_names):
                records.append({
                    'segment_id': segment_id,
                    'feature_index': idx,
                    'feature_name': feature_name,
                    'mean': float(stats['mean'][idx]),
                    'median': float(stats['median'][idx]),
                    'p25': float(stats['p25'][idx]),
                    'p75': float(stats['p75'][idx]),
                    'min': float(stats['min'][idx]),
                    'max': float(stats['max'][idx]),
                    'std': float(stats['std'][idx]),
                    'count': int(count)
                })

        return pd.DataFrame(records)

    def build_segment_summary_payload(self, segments: Dict[str, List[MeaningfulUser]],
                                      total_population: int) -> Dict[str, Any]:
        """Create JSON-friendly summary for a segment mapping."""
        summary: Dict[str, Any] = {}
        for segment_id, members in segments.items():
            stats = self.compute_segment_feature_statistics(members)
            if not stats:
                continue
            count = len(members)
            summary[segment_id] = {
                'count': count,
                'proportion': count / total_population if total_population else 0,
                'feature_mean': stats['mean'].tolist(),
                'feature_median': stats['median'].tolist(),
                'feature_p25': stats['p25'].tolist(),
                'feature_p75': stats['p75'].tolist(),
                'feature_min': stats['min'].tolist(),
                'feature_max': stats['max'].tolist(),
                'feature_std': stats['std'].tolist(),
                'feature_names': self.feature_names
            }
        return summary
    
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
