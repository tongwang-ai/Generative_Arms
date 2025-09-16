import numpy as np
import random
from typing import List, Dict, Any, Optional
from src.data.entities import Action, User
import re

from src.util.action_embedder import OpenAIActionEmbedder


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
        
        # Internal controls for LLM generation loops
        self._llm_max_attempts = 10
        self._llm_model = "gpt-5"  # Keep as-is; override in embedder if needed
        
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
        """Generate exploit actions using LLM based on previous best performers.
        Accumulates until target_count is reached, with retries and fallback.
        """
        if not self.llm_embedder.use_openai:
            return self._generate_template_exploit_actions(previous_best, target_count)
        
        # Helper: accumulate unique cleaned lines up to target_count
        best_texts = [action.text for action in previous_best[:5]]  # Use top 5
        best_examples = "\n".join([f"- \"{text}\"" for text in best_texts])

        collected: List[str] = []
        seen_norm = set()

        attempts = 0
        while len(collected) < target_count and attempts < self._llm_max_attempts:
            prompt = f"""Generate several improved marketing messages by exploiting and improving upon these top-performing messages:

{best_examples}

Requirements:
- Create variations that maintain the core appeal of the successful messages
- Use similar messaging strategies, tones, and value propositions
- Make small but meaningful improvements (better wording, clearer benefits, stronger calls-to-action)
- Keep the same general length and style as the originals
- Focus on the same product: professional platform membership
- Each message should be 5-15 words long
- Return the messages, one per line, no numbering or bullets
"""
            try:
                response = self.llm_embedder.client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                actions_text = response.choices[0].message.content.strip()
                new_lines = self._clean_llm_lines(actions_text)
                for line in new_lines:
                    norm = self._normalize_action_text(line)
                    if norm and norm not in seen_norm:
                        collected.append(line)
                        seen_norm.add(norm)
                        if len(collected) >= target_count:
                            break
            except Exception as e:
                print(f"Error generating LLM exploit actions (attempt {attempts+1}): {e}")
                # Continue to next attempt; fallback after loop if needed
            attempts += 1

        if len(collected) < target_count:
            # Fallback to template variations to fill the gap
            shortfall = target_count - len(collected)
            filler = self._generate_template_exploit_actions(previous_best, shortfall)
            collected.extend(filler)

        return collected[:target_count]
    
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
        """Generate explore actions using LLM for maximum creativity and diversity.
        Accumulates until target_count is reached, with retries and fallback.
        """
        if not self.llm_embedder.use_openai:
            return self._generate_template_explore_actions(target_count, existing_actions)
        
        collected: List[str] = []
        seen_norm = set()
        attempts = 0

        while len(collected) < target_count and attempts < self._llm_max_attempts:
            existing_text = ""
            if existing_actions:
                existing_sample = (existing_actions + collected)[:5]
                existing_text = "\nAvoid creating messages similar to these existing ones:\n" + "\n".join([f"- \"{a}\"" for a in existing_sample])

            prompt = f"""Generate several creative and diverse marketing messages for professional platform membership using EXPLORATION strategy.

Requirements:
- Be innovative and experimental with messaging approaches
- Explore completely new angles, benefits, and value propositions
- Try different emotional appeals (curiosity, ambition, FOMO, social proof, etc.)
- Experiment with different message structures and formats
- Use unexpected but relevant hooks and angles
- Focus on professional platform membership
- Each message should be 5-15 words long
- Be creative but still professional and compelling
- Return the messages, one per line, no numbering{existing_text}
"""
            try:
                response = self.llm_embedder.client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                actions_text = response.choices[0].message.content.strip()
                new_lines = self._clean_llm_lines(actions_text)
                for line in new_lines:
                    norm = self._normalize_action_text(line)
                    if norm and norm not in seen_norm:
                        collected.append(line)
                        seen_norm.add(norm)
                        if len(collected) >= target_count:
                            break
            except Exception as e:
                print(f"Error generating LLM explore actions (attempt {attempts+1}): {e}")
            attempts += 1

        if len(collected) < target_count:
            shortfall = target_count - len(collected)
            filler = self._generate_template_explore_actions(shortfall, existing_actions)
            collected.extend(filler)

        return collected[:target_count]
    
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
        """Generate targeted actions using LLM for precise segment targeting.
        Accumulates until target_count is reached, with retries and fallback.
        """
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
        
        collected: List[str] = []
        seen_norm = set()
        attempts = 0

        while len(collected) < target_count and attempts < self._llm_max_attempts:
            prompt = f"""Generate several highly targeted marketing messages for professional platform membership using TARGETED strategy.

Target these specific user segments: {segments_text}

Requirements:
- Create messages that speak directly to each segment's specific needs, pain points, and motivations
- Use language and value propositions that resonate with each target audience
- Address the unique benefits that matter most to each segment
- Focus on professional platform membership
- Each message should be 5-15 words long
- Make the targeting clear but not overly obvious
- Return the messages, one per line, no numbering
"""
            try:
                response = self.llm_embedder.client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                actions_text = response.choices[0].message.content.strip()
                new_lines = self._clean_llm_lines(actions_text)
                for line in new_lines:
                    norm = self._normalize_action_text(line)
                    if norm and norm not in seen_norm:
                        collected.append(line)
                        seen_norm.add(norm)
                        if len(collected) >= target_count:
                            break
            except Exception as e:
                print(f"Error generating LLM targeted actions (attempt {attempts+1}): {e}")
            attempts += 1

        if len(collected) < target_count:
            shortfall = target_count - len(collected)
            filler = self._generate_template_targeted_actions(target_segments, shortfall)
            collected.extend(filler)

        return collected[:target_count]
    
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
                           embed_batch_size: int = 500,
                           strategy_mix: Optional[dict] = None) -> List[Action]:
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
        # Allow caller to pass ratios via strategy_mix = {"exploit": 0.4, "explore": 0.3, "targeted": 0.3}
        if strategy_mix is None:
            strategy_mix = {"exploit": 0.4, "explore": 0.3, "targeted": 0.3}
        # Normalize any provided ratios to sum to 1.0
        mix_total = sum(max(0.0, float(strategy_mix.get(k, 0.0))) for k in ("exploit", "explore", "targeted"))
        if mix_total <= 0:
            # Fallback to defaults if invalid
            strategy_mix = {"exploit": 0.4, "explore": 0.3, "targeted": 0.3}
            mix_total = 1.0
        norm_mix = {k: max(0.0, float(strategy_mix.get(k, 0.0))) / mix_total for k in ("exploit", "explore", "targeted")}

        # Allocate integer counts and fix rounding to match pool_size
        exploit_count = int(round(pool_size * norm_mix["exploit"]))
        explore_count = int(round(pool_size * norm_mix["explore"]))
        targeted_count = int(round(pool_size * norm_mix["targeted"]))
        allocated = exploit_count + explore_count + targeted_count
        # Adjust by adding/removing from the largest bucket to hit exact pool_size
        if allocated != pool_size:
            diff = pool_size - allocated
            # Find largest category by ratio to adjust
            largest_key = max(norm_mix.keys(), key=lambda k: norm_mix[k])
            if largest_key == "exploit":
                exploit_count += diff
            elif largest_key == "explore":
                explore_count += diff
            else:
                targeted_count += diff
        
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

        # Ensure we have at least pool_size items; fill shortfall deterministically
        attempts = 0
        while len(unique_actions) < pool_size and attempts < 3:
            shortfall = pool_size - len(unique_actions)
            # Use template-based generation to quickly fill any gap
            unique_actions.extend(self._generate_from_templates(shortfall * 2))  # over-generate for more uniqueness
            unique_actions = self._remove_duplicates(unique_actions)
            attempts += 1
        
        # Convert to Action objects with OpenAI embeddings (batching)
        if not getattr(self.llm_embedder, 'use_openai', False):
            raise RuntimeError("OpenAI embeddings are required but not available. Set OPENAI_API_KEY.")

        embeddings = self.llm_embedder.embed_texts_in_batch(unique_actions[:pool_size], batch_size=embed_batch_size)

        action_pool = []
        for i, (text, embedding) in enumerate(zip(unique_actions[:pool_size], embeddings)):
            action = Action(action_id=f"gen_action_{i}", text=text, embedding=embedding)
            action_pool.append(action)
            
        return action_pool[:pool_size]  # Ensure we don't exceed requested size

    # --- helpers ---
    def _clean_llm_lines(self, raw_text: str) -> List[str]:
        """Split LLM output into lines and normalize common formatting artifacts."""
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        cleaned: List[str] = []
        for action in lines:
            a = action.strip()
            if a.startswith(('-', '*', '•')):
                a = a[1:].strip()
            # Remove '1. ', '2) ' etc.
            if re.match(r"^\d+[\).]\s+", a):
                a = re.sub(r"^\d+[\).]\s+", "", a)
            if a.startswith('"') and a.endswith('"') and len(a) >= 2:
                a = a[1:-1].strip()
            cleaned.append(a)
        return cleaned
