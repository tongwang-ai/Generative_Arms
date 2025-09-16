try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    OpenAI = None

import numpy as np
import pandas as pd
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class EmbeddedAction:
    """Action with OpenAI text embedding."""
    action_id: str
    text: str
    embedding: np.ndarray
    category: str
    metadata: Dict[str, Any]


class OpenAIActionEmbedder:
    """
    Generates action embeddings using OpenAI's text embedding API.
    Handles rate limiting, caching, and batch processing.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-3-large",
                 cache_file: str = "action_embeddings_cache.json",
                 require_openai: bool = True):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
            model: OpenAI embedding model to use
            cache_file: File to cache embeddings to avoid re-computation
        """
        # Set up OpenAI client (required when require_openai is True)
        if not OPENAI_AVAILABLE:
            if require_openai:
                raise RuntimeError("OpenAI package not installed. Install `openai` and set OPENAI_API_KEY.")
            self.use_openai = False
            self.client = None
        else:
            # Get API key
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                if require_openai:
                    raise RuntimeError("OPENAI_API_KEY is required for embeddings. Set it in env or pass api_key.")
                self.use_openai = False
                self.client = None
            else:
                try:
                    self.client = OpenAI(api_key=api_key)
                    self.use_openai = True
                except Exception as e:
                    if require_openai:
                        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
                    self.use_openai = False
                    self.client = None
        
        self.model = model
        self.cache_file = cache_file
        self.embedding_cache = self._load_cache()
        
        # Rate limiting
        self.requests_per_minute = 3000  # OpenAI limit for text-embedding-3-small
        self.last_request_time = 0
        
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache(self):
        """Save embedding cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.embedding_cache, f, indent=2)
    
    def _rate_limit(self):
        """Simple rate limiting to avoid API limits (only when using OpenAI)."""
        if not self.use_openai:
            return  # Should not happen when require_openai=True
            
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60 / self.requests_per_minute  # seconds between requests
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API with caching and rate limiting."""
        # Check cache first
        if text in self.embedding_cache:
            return np.array(self.embedding_cache[text])
        
        if not self.use_openai:
            raise RuntimeError("OpenAI embeddings required but client is not initialized.")
        
        try:
            self._rate_limit()
            
            # Make API call
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            embedding_array = np.array(embedding)
            
            # Cache the result
            self.embedding_cache[text] = embedding_array.tolist()
            
            return embedding_array
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error while embedding text: {e}")
    
    # Note: Fallback embedding method removed; embeddings must use OpenAI.
    
    def embed_actions(self, action_texts: List[str], 
                     action_categories: Optional[List[str]] = None,
                     batch_size: int = 50) -> List[EmbeddedAction]:
        """
        Embed a list of action texts using OpenAI API.
        
        Args:
            action_texts: List of action text strings
            action_categories: Optional list of categories for each action
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of EmbeddedAction objects
        """
        embedded_actions = []
        
        if action_categories is None:
            action_categories = ['general'] * len(action_texts)
        
        print(f"Embedding {len(action_texts)} actions using OpenAI API...")
        
        for i in range(0, len(action_texts), batch_size):
            batch_texts = action_texts[i:i+batch_size]
            batch_categories = action_categories[i:i+batch_size]

            print(f"Processing batch {i//batch_size + 1}/{(len(action_texts)-1)//batch_size + 1}")

            # Embed the entire batch with a single API call (with cache support)
            embeddings_batch = self.embed_texts_in_batch(batch_texts, batch_size=len(batch_texts))

            for j, (text, embedding) in enumerate(zip(batch_texts, embeddings_batch)):
                action = EmbeddedAction(
                    action_id=f"action_{i+j:04d}",
                    text=text,
                    embedding=embedding,
                    category=batch_categories[j],
                    metadata={
                        'embedding_model': self.model,
                        'embedding_dim': len(embedding),
                        'text_length': len(text)
                    }
                )
                embedded_actions.append(action)
            # Save cache after each batch
            self._save_cache()
        
        # Final cache save
        self._save_cache()
        
        print(f"Embedding complete! Generated {len(embedded_actions)} embedded actions.")
        return embedded_actions

    def embed_texts_in_batch(self, texts: List[str], batch_size: int = 500) -> List[np.ndarray]:
        """
        Embed raw texts efficiently using the OpenAI embeddings API in batches.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per API call

        Returns:
            List of numpy arrays (embeddings) aligned with input order
        """
        if not self.use_openai or self.client is None:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        embeddings: List[Optional[np.ndarray]] = [None] * len(texts)

        start = 0
        total = len(texts)
        batch_num = 0
        while start < total:
            batch_num += 1
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            # Determine which texts need API calls (not cached)
            missing_mask = [t not in self.embedding_cache for t in batch_texts]
            missing_texts = [t for t, miss in zip(batch_texts, missing_mask) if miss]

            if missing_texts:
                # Single API call for missing texts in this batch
                self._rate_limit()
                resp = self.client.embeddings.create(model=self.model, input=missing_texts)
                # resp.data is aligned with input order
                for i, item in enumerate(resp.data):
                    vec = np.array(item.embedding)
                    txt = missing_texts[i]
                    self.embedding_cache[txt] = vec.tolist()

                # Save cache periodically
                if (batch_num % 5) == 0:
                    self._save_cache()

            # Fill outputs from cache (all batch_texts should now be cached)
            for i, t in enumerate(batch_texts):
                cached = self.embedding_cache.get(t)
                if cached is None:
                    # Fallback to one-off embed if something went wrong
                    vec = self._get_openai_embedding(t)
                else:
                    vec = np.array(cached)
                embeddings[start + i] = vec

            start = end

        # Final cache save
        try:
            self._save_cache()
        except Exception:
            pass

        # type: ignore - ensured populated
        return embeddings  # type: ignore

    def embed_single_action(self,
                            action_id: str,
                            text: str,
                            category: str = 'general',
                            metadata: Optional[Dict[str, Any]] = None) -> EmbeddedAction:
        """
        Embed a single action text and return an EmbeddedAction object.

        Args:
            action_id: Identifier to assign to the embedded action
            text: The action text to embed
            category: Optional category label
            metadata: Optional metadata to attach to the action

        Returns:
            EmbeddedAction with embedding populated.
        """
        # Generate embedding using OpenAI
        embedding = self._get_openai_embedding(text)

        # Merge provided metadata with embedder metadata
        base_metadata: Dict[str, Any] = {
            'embedding_model': self.model,
            'embedding_dim': int(len(embedding)),
            'text_length': int(len(text))
        }
        if metadata:
            # Do not mutate caller-provided dict
            combined_metadata = {**metadata, **base_metadata}
        else:
            combined_metadata = base_metadata

        # Create and return the embedded action
        action = EmbeddedAction(
            action_id=action_id,
            text=text,
            embedding=embedding,
            category=category,
            metadata=combined_metadata
        )

        # Persist cache so repeated calls benefit across runs
        try:
            self._save_cache()
        except Exception:
            # Cache failures should not break the embedding path
            pass

        return action
    
    def save_embedded_actions(self, actions: List[EmbeddedAction], filepath: str):
        """Save embedded actions to JSON file."""
        actions_data = []
        
        for action in actions:
            action_dict = {
                'action_id': action.action_id,
                'text': action.text,
                'embedding': action.embedding.tolist(),
                'category': action.category,
                'metadata': action.metadata
            }
            actions_data.append(action_dict)
        
        with open(filepath, 'w') as f:
            json.dump({
                'actions': actions_data,
                'embedding_model': self.model,
                'total_actions': len(actions),
                'embedding_dimension': len(actions[0].embedding) if actions else 0
            }, f, indent=2)
    
    def load_embedded_actions(self, filepath: str) -> List[EmbeddedAction]:
        """Load embedded actions from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        actions = []
        for action_data in data['actions']:
            action = EmbeddedAction(
                action_id=action_data['action_id'],
                text=action_data['text'],
                embedding=np.array(action_data['embedding']),
                category=action_data['category'],
                metadata=action_data['metadata']
            )
            actions.append(action)
        
        return actions
    
    def generate_actions_with_chatgpt(self, 
                                    n_actions: int = 50,
                                    categories: List[str] = None,
                                    product_focus: str = "professional platform membership") -> List[str]:
        """
        Generate diverse marketing actions using ChatGPT focused on a specific product.
        
        Args:
            n_actions: Number of actions to generate
            categories: List of categories to generate actions for
            product_focus: The specific product/service to focus all actions on
            
        Returns:
            List of marketing action texts
        """
        if not self.use_openai or not self.client:
            print("Warning: ChatGPT not available. Using fallback action generation.")
            return self._generate_fallback_actions(n_actions, categories, product_focus)
        
        if categories is None:
            categories = ['promotional', 'educational', 'social', 'quality', 'urgency']
        
        actions_per_category = max(1, n_actions // len(categories))
        remaining_actions = n_actions - (actions_per_category * len(categories))
        
        all_actions = []
        
        for i, category in enumerate(categories):
            # Add extra actions to first few categories if needed
            category_count = actions_per_category + (1 if i < remaining_actions else 0)
            
            try:
                category_actions = self._generate_category_actions(category, category_count, product_focus)
                all_actions.extend(category_actions)
                print(f"Generated {len(category_actions)} {category} actions for {product_focus}")
                
                # Rate limiting between categories
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating {category} actions: {e}")
                # Fallback to predefined actions for this category
                fallback_actions = self._get_fallback_category_actions(category, category_count, product_focus)
                all_actions.extend(fallback_actions)
        
        # Shuffle to avoid category clustering
        np.random.shuffle(all_actions)
        
        return all_actions[:n_actions]
    
    def _generate_category_actions(self, category: str, count: int, product_focus: str = "professional platform membership") -> List[str]:
        """Generate actions for a specific category using ChatGPT."""
        
        # Define product-focused category descriptions
        category_prompts = {
            'promotional': f"promotional offers and discounts for {product_focus}",
            'educational': f"educational benefits and learning opportunities through {product_focus}",
            'social': f"community and networking benefits of {product_focus}",
            'quality': f"premium features and quality benefits of {product_focus}",
            'urgency': f"time-sensitive membership offers for {product_focus}"
        }
        
        category_desc = category_prompts.get(category, f"{category} aspects of {product_focus}")
        
        # Create focused examples based on the product
        category_examples = {
            'promotional': [
                "Save 30% on annual professional platform membership today",
                "Limited-time offer: 50% off premium platform access",
                "Early bird special: Join the platform at founder pricing"
            ],
            'educational': [
                "Access 500+ expert courses with platform membership",
                "Master new skills through our premium learning platform",
                "Unlock professional development with platform access"
            ],
            'social': [
                "Connect with 100K+ professionals on our platform",
                "Join industry leaders in our exclusive member community",
                "Network with peers through our professional platform"
            ],
            'quality': [
                "Experience enterprise-grade platform security and reliability",
                "Premium platform features trusted by Fortune 500 companies",
                "Professional-grade tools and analytics in our platform"
            ],
            'urgency': [
                "Membership spots filling fast - secure yours today",
                "Last chance: Platform early access ends tonight",
                "Only 48 hours left for exclusive platform pricing"
            ]
        }
        
        examples = category_examples.get(category, category_examples['promotional'])
        example_text = "\n".join([f"- \"{ex}\"" for ex in examples[:3]])
        
        prompt = f"""Generate {count} diverse and compelling marketing messages for {category_desc}.

IMPORTANT: All messages must focus on the same product: "{product_focus}"

Requirements:
- Each message should be 5-15 words long
- Focus specifically on {category_desc} 
- Make them appealing to different customer segments (young professionals, families, premium customers, budget-conscious)
- All messages should be about the SAME platform/product, just different marketing approaches
- Avoid generic phrases
- Make each message unique and specific
- Return only the messages, one per line, no numbering

Examples for this category:
{example_text}

Generate {count} marketing messages for {product_focus} now:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}]
            )
            
            actions_text = response.choices[0].message.content.strip()
            actions = [line.strip() for line in actions_text.split('\n') if line.strip()]
            
            # Clean up any numbering or formatting
            cleaned_actions = []
            for action in actions:
                # Remove common prefixes
                action = action.strip()
                if action.startswith(('-', '*', '•')):
                    action = action[1:].strip()
                if '. ' in action and action.split('. ')[0].isdigit():
                    action = '. '.join(action.split('. ')[1:])
                cleaned_actions.append(action)
            
            return cleaned_actions[:count]
            
        except Exception as e:
            print(f"ChatGPT API error for {category} category: {e}")
            return self._get_fallback_category_actions(category, count, product_focus)
    
    def _get_fallback_category_actions(self, category: str, count: int, product_focus: str = "professional platform membership") -> List[str]:
        """Fallback predefined actions when ChatGPT is unavailable."""
        fallback_actions = {
            'promotional': [
                f"Save 25% on your {product_focus} - limited time offer",
                f"Flash sale: 40% off premium {product_focus} today only",
                f"Exclusive member discount: 30% off {product_focus}",
                f"Special weekend deal: Half-price {product_focus} access",
                f"Early bird pricing: Join {product_focus} at 50% off",
                f"Black Friday exclusive: {product_focus} at lowest price ever"
            ],
            'educational': [
                f"Master new skills with {product_focus} learning resources",
                f"Access 1000+ courses through {product_focus}",
                f"Professional development made easy with {product_focus}",
                f"Learn from industry experts via {product_focus}",
                f"Unlock career growth through {product_focus} training",
                f"Expert-led workshops included in {product_focus}"
            ],
            'social': [
                f"Join 100K+ professionals in our {product_focus} community",
                f"Network with industry leaders through {product_focus}",
                f"Connect with peers in our exclusive {product_focus}",
                f"Build your network with {product_focus} connections",
                f"Professional community awaits in {product_focus}",
                f"Collaborate with experts through {product_focus}"
            ],
            'quality': [
                f"Enterprise-grade {product_focus} trusted by Fortune 500",
                f"Premium {product_focus} with 99.9% uptime guarantee",
                f"Professional-grade security in our {product_focus}",
                f"Award-winning {product_focus} used by industry leaders",
                f"Reliable {product_focus} with 24/7 expert support",
                f"Top-tier {product_focus} with advanced analytics"
            ],
            'urgency': [
                f"Limited spots left for {product_focus} - secure yours today",
                f"Last chance: {product_focus} early access ends tonight",
                f"Only 24 hours left for exclusive {product_focus} pricing",
                f"Membership filling fast - join {product_focus} now",
                f"Final hours: {product_focus} founder pricing expires soon",
                f"Act now: {product_focus} beta access closing tomorrow"
            ]
        }
        
        category_actions = fallback_actions.get(category, fallback_actions['promotional'])
        return (category_actions * ((count // len(category_actions)) + 1))[:count]
    
    def _generate_fallback_actions(self, n_actions: int, categories: List[str], product_focus: str = "professional platform membership") -> List[str]:
        """Generate fallback actions when ChatGPT is unavailable."""
        if categories is None:
            categories = ['promotional', 'educational', 'social', 'quality', 'urgency']
        
        actions_per_category = max(1, n_actions // len(categories))
        all_actions = []
        
        for category in categories:
            category_actions = self._get_fallback_category_actions(category, actions_per_category, product_focus)
            all_actions.extend(category_actions)
        
        return all_actions[:n_actions]


def create_marketing_action_bank(embedder: OpenAIActionEmbedder, 
                               n_actions: int = 50,
                               use_chatgpt: bool = True,
                               product_focus: str = "professional platform membership") -> List[EmbeddedAction]:
    """
    Create a diverse bank of marketing actions using ChatGPT or fallback content.
    
    Args:
        embedder: OpenAI embedder instance
        n_actions: Number of actions to create
        use_chatgpt: Whether to use ChatGPT for action generation
        product_focus: The specific product/service to focus all actions on
        
    Returns:
        List of embedded marketing actions
    """
    print(f"Creating action bank with {n_actions} actions...")
    
    if use_chatgpt and embedder.use_openai:
        print(f"🤖 Using ChatGPT to generate diverse marketing actions for {product_focus}...")
        
        # Use multiple requests for better diversity
        categories = ['promotional', 'educational', 'social', 'quality', 'urgency']
        action_texts = embedder.generate_actions_with_chatgpt(n_actions, categories, product_focus)
        
        # Assign categories based on content
        action_categories = []
        for action in action_texts:
            # Simple keyword-based category assignment
            action_lower = action.lower()
            if any(word in action_lower for word in ['%', 'off', 'sale', 'discount', 'deal', 'free']):
                category = 'promotional'
            elif any(word in action_lower for word in ['learn', 'guide', 'tutorial', 'course', 'webinar']):
                category = 'educational'
            elif any(word in action_lower for word in ['community', 'join', 'connect', 'share', 'network']):
                category = 'social'
            elif any(word in action_lower for word in ['premium', 'quality', 'trusted', 'guarantee', 'security']):
                category = 'quality'
            elif any(word in action_lower for word in ['last', 'limited', 'expires', 'hurry', 'fast', 'now']):
                category = 'urgency'
            else:
                category = 'general'
            action_categories.append(category)
        
    else:
        print(f"📋 Using predefined diverse marketing actions for {product_focus}...")
        
        # Fallback to predefined diverse actions
        action_texts = embedder._generate_fallback_actions(n_actions, None, product_focus)
        
        # Assign categories for fallback actions
        actions_per_category = n_actions // 5
        action_categories = []
        categories = ['promotional', 'educational', 'social', 'quality', 'urgency']
        
        for i, category in enumerate(categories):
            count = actions_per_category + (1 if i < (n_actions % 5) else 0)
            action_categories.extend([category] * count)
        
        # Ensure we have the right number of categories
        action_categories = action_categories[:len(action_texts)]
    
    # Print category distribution
    category_counts = {}
    for cat in action_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"Action bank composition:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} actions")
    
    # Embed all actions
    return embedder.embed_actions(action_texts, action_categories)
