import numpy as np
import random
from typing import List, Dict, Any, Optional
from ..data.entities import Action, User
import re
import os
import sys

# Import OpenAI embedder for LLM-based action generation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from simulation.action_embedder import OpenAIActionEmbedder


class ActionGenerator:
    """
    Generates candidate action pools using multiple strategies.
    """
    
    def __init__(self, random_seed: int = 42, use_llm: bool = True, embedder_model: Optional[str] = None):
        self.random_seed = random_seed
        self.use_llm = use_llm
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize OpenAI embedder (required for embeddings)
        # This will raise if OpenAI is unavailable or no API key is set
        self.llm_embedder = OpenAIActionEmbedder(model=embedder_model or "text-embedding-3-large")
        if self.use_llm:
            print("LLM-based action generation enabled")
        
        # Base action templates for different categories
        self.base_templates = {
            'promotional': [
                "Get {discount}% off {product}",
                "Limited time: {discount}% discount on {product}",
                "Save big on {product} - {discount}% off",
                "Flash sale: {product} at {discount}% off",
                "Exclusive {discount}% discount for you",
                "Special offer: {product} reduced by {discount}%"
            ],
            'informational': [
                "Learn about {topic}",
                "Discover the benefits of {product}",
                "New insights on {topic}",
                "Everything you need to know about {product}",
                "Expert tips on {topic}",
                "Understanding {product} better"
            ],
            'engagement': [
                "Join our {community} community",
                "Share your thoughts on {topic}",
                "Tell us about your {experience}",
                "Rate your experience with {product}",
                "Connect with other {community} members",
                "Participate in our {event}"
            ],
            'retention': [
                "We miss you! Come back for {incentive}",
                "Your {product} is waiting for you",
                "Special comeback offer: {incentive}",
                "We've improved {product} - check it out",
                "Return and get {incentive}",
                "Your account needs attention"
            ]
        }
        
        # Variables for template filling
        self.template_vars = {
            'discount': ['10', '15', '20', '25', '30', '50'],
            'product': ['Premium Plan', 'Pro Features', 'Advanced Tools', 'New Release', 'Popular Items'],
            'topic': ['productivity', 'innovation', 'growth', 'success', 'efficiency'],
            'community': ['user', 'customer', 'expert', 'professional'],
            'experience': ['journey', 'feedback', 'story', 'success'],
            'event': ['webinar', 'workshop', 'challenge', 'contest'],
            'incentive': ['20% off', 'free trial', 'bonus content', 'priority support']
        }
        
        # RLHF-style high-quality actions (simulated)
        self.rlhf_actions = [
            "Unlock your potential with personalized recommendations",
            "Transform your workflow with AI-powered insights", 
            "Experience the future of productivity tools",
            "Join thousands who've already upgraded their experience",
            "Your success story starts with the right tools",
            "Discover features designed specifically for your needs",
            "Take the next step in your professional journey",
            "Get ahead with cutting-edge solutions"
        ]
        
    def _fill_template(self, template: str) -> str:
        """Fill template with random variables."""
        filled = template
        for var, options in self.template_vars.items():
            if f'{{{var}}}' in filled:
                filled = filled.replace(f'{{{var}}}', random.choice(options))
        return filled
    
    # All embeddings are produced via OpenAIActionEmbedder; no local/random embeddings.
    
    def _generate_exploit_actions(self, previous_best: List[Action], target_count: int) -> List[str]:
        """Generate actions by exploiting (varying) previous best performers."""
        if not previous_best:
            return self._generate_from_templates(target_count)
        
        if self.use_llm and self.llm_embedder:
            return self._generate_llm_exploit_actions(previous_best, target_count)
        else:
            return self._generate_template_exploit_actions(previous_best, target_count)
    
    def _generate_llm_exploit_actions(self, previous_best: List[Action], target_count: int) -> List[str]:
        """Generate exploit actions using LLM based on previous best performers."""
        if not self.llm_embedder.use_openai:
            return self._generate_template_exploit_actions(previous_best, target_count)
        
        # Extract top performing action texts
        best_texts = [action.text for action in previous_best[:5]]  # Use top 5
        best_examples = "\n".join([f"- \"{text}\"" for text in best_texts])
        
        prompt = f"""Generate {target_count} new marketing messages by exploiting and improving upon these top-performing messages:

{best_examples}

Requirements:
- Create variations that maintain the core appeal of the successful messages
- Use similar messaging strategies, tones, and value propositions
- Make small but meaningful improvements (better wording, clearer benefits, stronger calls-to-action)
- Keep the same general length and style as the originals
- Focus on the same product: professional platform membership
- Each message should be 5-15 words long
- Return only the messages, one per line, no numbering

Generate {target_count} exploit strategy messages now:"""

        try:
            response = self.llm_embedder.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7
            )
            
            actions_text = response.choices[0].message.content.strip()
            actions = [line.strip() for line in actions_text.split('\n') if line.strip()]
            
            # Clean up formatting
            cleaned_actions = []
            for action in actions:
                action = action.strip()
                if action.startswith(('-', '*', '•')):
                    action = action[1:].strip()
                if '. ' in action and action.split('. ')[0].isdigit():
                    action = '. '.join(action.split('. ')[1:])
                if action.startswith('"') and action.endswith('"'):
                    action = action[1:-1]
                cleaned_actions.append(action)
            
            return cleaned_actions[:target_count]
            
        except Exception as e:
            print(f"Error generating LLM exploit actions: {e}")
            return self._generate_template_exploit_actions(previous_best, target_count)
    
    def _generate_template_exploit_actions(self, previous_best: List[Action], target_count: int) -> List[str]:
        """Generate exploit actions using template variations (fallback method)."""
        exploit_actions = []
        
        for _ in range(target_count):
            base_action = random.choice(previous_best)
            base_text = base_action.text
            
            # Apply variations
            variation_type = random.choice(['synonym', 'structure', 'emphasis', 'detail'])
            
            if variation_type == 'synonym':
                variations = {
                    'get': 'receive', 'save': 'earn', 'special': 'exclusive',
                    'new': 'latest', 'best': 'top', 'great': 'amazing',
                    'discover': 'explore', 'learn': 'understand'
                }
                varied_text = base_text.lower()
                for original, replacement in variations.items():
                    if original in varied_text:
                        varied_text = varied_text.replace(original, replacement, 1)
                        break
                exploit_actions.append(varied_text.capitalize())
                
            elif variation_type == 'structure':
                if '!' in base_text:
                    exploit_actions.append(base_text.replace('!', '.'))
                elif '?' in base_text:
                    exploit_actions.append(base_text.replace('?', '!'))
                else:
                    exploit_actions.append(base_text + '!')
                    
            elif variation_type == 'emphasis':
                emphasis_words = ['Now:', 'Today:', 'Don\'t miss:', 'Urgent:', 'Last chance:']
                exploit_actions.append(f"{random.choice(emphasis_words)} {base_text}")
                
            else:  # detail
                details = ['for limited time', 'while supplies last', 'just for you', 'no strings attached']
                exploit_actions.append(f"{base_text} - {random.choice(details)}")
        
        return exploit_actions
    
    def _generate_explore_actions(self, target_count: int, existing_actions: List[str] = None) -> List[str]:
        """Generate entirely new, diverse actions for exploration."""
        if self.use_llm and self.llm_embedder:
            return self._generate_llm_explore_actions(target_count, existing_actions)
        else:
            return self._generate_template_explore_actions(target_count, existing_actions)
    
    def _generate_llm_explore_actions(self, target_count: int, existing_actions: List[str] = None) -> List[str]:
        """Generate explore actions using LLM for maximum creativity and diversity."""
        if not self.llm_embedder.use_openai:
            return self._generate_template_explore_actions(target_count, existing_actions)
        
        existing_text = ""
        if existing_actions:
            existing_sample = existing_actions[:5]  # Show a few examples to avoid
            existing_text = f"\nAvoid creating messages similar to these existing ones:\n" + "\n".join([f"- \"{action}\"" for action in existing_sample])
        
        prompt = f"""Generate {target_count} highly creative and diverse marketing messages for professional platform membership using EXPLORATION strategy.

Requirements:
- Be innovative and experimental with messaging approaches
- Explore completely new angles, benefits, and value propositions
- Try different emotional appeals (curiosity, ambition, FOMO, social proof, etc.)
- Experiment with different message structures and formats
- Use unexpected but relevant hooks and angles
- Focus on professional platform membership
- Each message should be 5-15 words long
- Be creative but still professional and compelling
- Return only the messages, one per line, no numbering{existing_text}

Generate {target_count} exploratory strategy messages now:"""

        try:
            response = self.llm_embedder.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.9  # Higher temperature for more creativity
            )
            
            actions_text = response.choices[0].message.content.strip()
            actions = [line.strip() for line in actions_text.split('\n') if line.strip()]
            
            # Clean up formatting
            cleaned_actions = []
            for action in actions:
                action = action.strip()
                if action.startswith(('-', '*', '•')):
                    action = action[1:].strip()
                if '. ' in action and action.split('. ')[0].isdigit():
                    action = '. '.join(action.split('. ')[1:])
                if action.startswith('"') and action.endswith('"'):
                    action = action[1:-1]
                cleaned_actions.append(action)
            
            return cleaned_actions[:target_count]
            
        except Exception as e:
            print(f"Error generating LLM explore actions: {e}")
            return self._generate_template_explore_actions(target_count, existing_actions)
    
    def _generate_template_explore_actions(self, target_count: int, existing_actions: List[str] = None) -> List[str]:
        """Generate explore actions using creative templates (fallback method)."""
        explore_actions = []
        
        creative_patterns = [
            "What if {subject} could {action}?",
            "Imagine {outcome} in just {timeframe}",
            "The secret to {goal} is {method}",
            "Why {audience} choose {solution}",
            "Before you {action}, consider {alternative}",
            "The {adjective} way to {achieve} {goal}"
        ]
        
        creative_vars = {
            'subject': ['you', 'your team', 'your business', 'your workflow'],
            'action': ['automate tasks', 'increase efficiency', 'save time', 'boost productivity'],
            'outcome': ['success', 'growth', 'improvement', 'transformation'],
            'timeframe': ['30 days', 'one week', 'minutes', 'hours'],
            'goal': ['success', 'efficiency', 'growth', 'innovation'],
            'method': ['the right tools', 'smart planning', 'expert guidance', 'automation'],
            'audience': ['professionals', 'experts', 'leaders', 'innovators'],
            'solution': ['our platform', 'advanced features', 'premium tools', 'AI assistance'],
            'alternative': ['these options', 'our approach', 'this solution', 'expert advice'],
            'adjective': ['smartest', 'fastest', 'most effective', 'proven'],
            'achieve': ['reaching', 'attaining', 'accomplishing', 'realizing']
        }
        
        for _ in range(target_count):
            pattern = random.choice(creative_patterns)
            filled_action = pattern
            
            for var, options in creative_vars.items():
                if f'{{{var}}}' in filled_action:
                    filled_action = filled_action.replace(f'{{{var}}}', random.choice(options))
                    
            explore_actions.append(filled_action)
            
        return explore_actions
    
    def _generate_targeted_actions(self, target_segments: List[str], target_count: int) -> List[str]:
        """Generate actions targeted at specific user segments."""
        if self.use_llm and self.llm_embedder:
            return self._generate_llm_targeted_actions(target_segments, target_count)
        else:
            return self._generate_template_targeted_actions(target_segments, target_count)
    
    def _generate_llm_targeted_actions(self, target_segments: List[str], target_count: int) -> List[str]:
        """Generate targeted actions using LLM for precise segment targeting."""
        if not self.llm_embedder.use_openai:
            return self._generate_template_targeted_actions(target_segments, target_count)
        
        # Define segment characteristics for better targeting
        segment_descriptions = {
            'high_value': 'high-value customers who spend significantly and seek premium experiences',
            'churn_risk': 'users at risk of leaving who need re-engagement and retention offers',
            'new_user': 'new users who need onboarding, guidance, and welcome incentives',
            'professional': 'career-focused professionals seeking growth and advancement',
            'enterprise': 'business decision-makers looking for team and organization solutions',
            'budget_conscious': 'cost-sensitive users who prioritize value and affordability'
        }
        
        if not target_segments:
            target_segments = ['high_value', 'churn_risk', 'new_user']
        
        # Create segment-specific messages
        segments_text = ", ".join([segment_descriptions.get(seg, seg) for seg in target_segments])
        
        prompt = f"""Generate {target_count} highly targeted marketing messages for professional platform membership using TARGETED strategy.

Target these specific user segments: {segments_text}

Requirements:
- Create messages that speak directly to each segment's specific needs, pain points, and motivations
- Use language and value propositions that resonate with each target audience
- Address the unique benefits that matter most to each segment
- Focus on professional platform membership
- Each message should be 5-15 words long
- Make the targeting clear but not overly obvious
- Return only the messages, one per line, no numbering

Generate {target_count} targeted strategy messages now:"""

        try:
            response = self.llm_embedder.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.8
            )
            
            actions_text = response.choices[0].message.content.strip()
            actions = [line.strip() for line in actions_text.split('\n') if line.strip()]
            
            # Clean up formatting
            cleaned_actions = []
            for action in actions:
                action = action.strip()
                if action.startswith(('-', '*', '•')):
                    action = action[1:].strip()
                if '. ' in action and action.split('. ')[0].isdigit():
                    action = '. '.join(action.split('. ')[1:])
                if action.startswith('"') and action.endswith('"'):
                    action = action[1:-1]
                cleaned_actions.append(action)
            
            return cleaned_actions[:target_count]
            
        except Exception as e:
            print(f"Error generating LLM targeted actions: {e}")
            return self._generate_template_targeted_actions(target_segments, target_count)
    
    def _generate_template_targeted_actions(self, target_segments: List[str], target_count: int) -> List[str]:
        """Generate targeted actions using segment templates (fallback method)."""
        segment_templates = {
            'high_value': [
                "Exclusive VIP offer just for you",
                "Premium benefits for our top members",
                "Your elite status unlocks these perks",
                "High-value customers get priority access",
                "Luxury experience tailored for you"
            ],
            'churn_risk': [
                "We want to win you back",
                "Special offer to keep you with us",
                "Don't leave without seeing this",
                "Your feedback matters - let's talk",
                "One more chance to impress you"
            ],
            'new_user': [
                "Welcome! Here's your starter guide",
                "New user bonus waiting for you",
                "Get started with these easy steps",
                "Your onboarding journey begins here",
                "First-time user special offer"
            ]
        }
        
        targeted_actions = []
        
        for _ in range(target_count):
            if target_segments:
                segment = random.choice(target_segments)
                if segment in segment_templates:
                    template = random.choice(segment_templates[segment])
                    targeted_actions.append(template)
                else:
                    targeted_actions.append(self._fill_template(random.choice(self.base_templates['promotional'])))
            else:
                targeted_actions.append(self._fill_template(random.choice(self.base_templates['promotional'])))
                
        return targeted_actions
    
    def _generate_rlhf_actions(self, target_count: int) -> List[str]:
        selected = random.choices(self.rlhf_actions, k=min(target_count, len(self.rlhf_actions)))
        
        if target_count > len(selected):
            remaining = target_count - len(selected)
            for _ in range(remaining):
                base = random.choice(self.rlhf_actions)
                variation = base.replace('your', 'the').replace('you', 'users')
                selected.append(variation)
                
        return selected[:target_count]
    
    def _generate_from_templates(self, target_count: int) -> List[str]:
        """Generate actions from base templates."""
        actions = []
        categories = list(self.base_templates.keys())
        
        for _ in range(target_count):
            category = random.choice(categories)
            template = random.choice(self.base_templates[category])
            filled_template = self._fill_template(template)
            actions.append(filled_template)
            
        return actions
    
    def _remove_duplicates(self, actions: List[str]) -> List[str]:
        """Remove duplicate actions using sophisticated similarity detection."""
        unique_actions = []
        seen_normalized = set()
        
        for action in actions:
            # Normalize for comparison
            normalized = self._normalize_action_text(action)
            
            if normalized not in seen_normalized:
                unique_actions.append(action)
                seen_normalized.add(normalized)
        
        return unique_actions
    
    def _normalize_action_text(self, text: str) -> str:
        """Normalize action text for duplicate detection."""
        import re
        
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove punctuation except for essential ones
        normalized = re.sub(r'[^\w\s%!?.-]', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common variations that don't change meaning
        replacements = {
            'get ': 'receive ',
            'grab ': 'get ',
            'snag ': 'get ',
            'secure ': 'get ',
            'claim ': 'get ',
            'unlock ': 'access ',
            'discover ': 'find ',
            'explore ': 'find ',
            'learn about ': 'learn ',
            'find out ': 'learn ',
            'join our ': 'join ',
            'become part of ': 'join ',
            'sign up for ': 'join ',
            'limited time': 'limited',
            'for a limited time': 'limited',
            'dont miss': 'dont miss',
            "don't miss": 'dont miss'
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def generate_action_pool(self, pool_size: int = 200, 
                           previous_best: Optional[List[Action]] = None,
                           user_segments: Optional[List[str]] = None,
                           embed_batch_size: int = 500) -> List[Action]:
        """
        Generate a large candidate action pool using multiple strategies.
        
        Args:
            pool_size: Total number of actions to generate
            previous_best: List of best performing actions from previous iteration
            user_segments: List of user segments to target
            embedding_dim: Dimension of action embeddings
            
        Returns:
            List of Action objects
        """
        if user_segments is None:
            user_segments = ['high_value', 'churn_risk', 'new_user']
            
        # Calculate action counts for each strategy
        exploit_count = int(pool_size * 0.4)
        explore_count = int(pool_size * 0.3) 
        targeted_count = int(pool_size * 0.3)
        
        all_action_texts = []
        
        # Generate actions using each strategy
        exploit_actions = self._generate_exploit_actions(previous_best or [], exploit_count)
        explore_actions = self._generate_explore_actions(explore_count, exploit_actions)
        targeted_actions = self._generate_targeted_actions(user_segments, targeted_count)
        # rlhf_actions = self._generate_rlhf_actions(rlhf_count)
        
        all_action_texts.extend(exploit_actions)
        all_action_texts.extend(explore_actions) 
        all_action_texts.extend(targeted_actions)
        # all_action_texts.extend(rlhf_actions)
        
        # Remove duplicates while preserving order (enhanced deduplication)
        unique_actions = self._remove_duplicates(all_action_texts)
        
        # Convert to Action objects with OpenAI embeddings (batching)
        if not getattr(self.llm_embedder, 'use_openai', False):
            raise RuntimeError("OpenAI embeddings are required but not available. Set OPENAI_API_KEY.")

        embeddings = self.llm_embedder.embed_texts_in_batch(unique_actions, batch_size=embed_batch_size)

        action_pool = []
        for i, (text, embedding) in enumerate(zip(unique_actions, embeddings)):
            action = Action(action_id=f"gen_action_{i}", text=text, embedding=embedding)
            action_pool.append(action)
            
        return action_pool[:pool_size]  # Ensure we don't exceed requested size
