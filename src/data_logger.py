"""
Data Logger - Logs AI interactions and embeddings to Supabase
============================================================
This module handles all data logging operations including:
- Generating embeddings via OpenRouter API
- Checking for toxic content and refusals
- Logging interactions to Supabase database
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DataLogger:
    """
    Handles logging of AI interactions and embeddings to Supabase.
    
    Attributes:
        supabase: Supabase client for database operations
        openrouter_key: API key for OpenRouter
        embedding_model: Model name for generating embeddings
        chat_model: Model name for chat completions
    """
    
    def __init__(self):
        """Initialize the DataLogger with API connections."""
        # Initialize Supabase client
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        # Get API keys and model configurations
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-ada-002")
        self.chat_model = os.getenv("CHAT_MODEL", "openai/gpt-3.5-turbo")
        
        print(f"✓ DataLogger initialized")
        print(f"  Embedding model: {self.embedding_model}")
        print(f"  Chat model: {self.chat_model}")
    
    def get_embedding(self, text: str) -> Optional[list]:
        """
        Generate embedding vector for given text using OpenRouter API.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector, or None if failed
        """
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.embedding_model,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            return embedding
        except requests.exceptions.RequestException as e:
            print(f"❌ Embedding API error: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"❌ Embedding response format error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected embedding error: {e}")
            return None
    
    def check_toxicity(self, text: str) -> bool:
        """
        Check if text contains toxic content using keyword matching.
        
        In production, use OpenAI moderation API or Detoxify library for better accuracy.
        
        Args:
            text: Text to check for toxicity
            
        Returns:
            True if toxic content detected, False otherwise
        """
        toxic_keywords = [
            "hate", "kill", "violent", "abuse", "offensive",
            "racist", "sexist", "attack", "threat", "harm",
            "stupid", "idiot", "dumb", "worst", "terrible"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in toxic_keywords)
    
    def check_refusal(self, response: str) -> bool:
        """
        Check if model response is a refusal to answer.
        
        Args:
            response: Model's response text
            
        Returns:
            True if response is a refusal, False otherwise
        """
        refusal_patterns = [
            "i cannot", "i can't", "i'm not able to",
            "i apologize, but", "i'm sorry, but i cannot",
            "i don't have access", "i'm unable to",
            "i cannot assist", "i can't help",
            "i'm not allowed", "against my guidelines"
        ]
        
        response_lower = response.lower()
        return any(pattern in response_lower for pattern in refusal_patterns)
    
    def log_embedding(self, text: str, emb_type: str, metadata: Optional[Dict] = None) -> bool:
        """
        Generate and log embedding to database.
        
        Args:
            text: Text to generate embedding for
            emb_type: Type of embedding ('query' or 'document')
            metadata: Optional metadata to store with embedding
            
        Returns:
            True if successful, False otherwise
        """
        embedding = self.get_embedding(text)
        
        if not embedding:
            return False
        
        data = {
            "embedding": embedding,
            "type": emb_type,
            "text_content": text[:500],  # Truncate long text for storage
            "metadata": metadata or {}
        }
        
        try:
            self.supabase.table("embeddings_log").insert(data).execute()
            return True
        except Exception as e:
            print(f"❌ Failed to log embedding: {e}")
            return False
    
    def log_interaction(
        self,
        user_query: str,
        model_response: str,
        model_version: str = "v1.0",
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        user_feedback: Optional[int] = None
    ) -> bool:
        """
        Log complete AI interaction to database.
        
        Args:
            user_query: User's input query
            model_response: Model's response
            model_version: Version identifier of the model
            tokens_used: Number of tokens consumed
            cost_usd: Cost in USD for this interaction
            user_feedback: Optional user feedback score (1-5)
            
        Returns:
            True if successful, False otherwise
        """
        # Check for refusal and toxicity
        refusal = self.check_refusal(model_response)
        toxicity = self.check_toxicity(model_response)
        
        # Prepare interaction data
        data = {
            "user_query": user_query,
            "model_response": model_response,
            "refusal_flag": refusal,
            "toxicity_flag": toxicity,
            "user_feedback_score": user_feedback,
            "model_version": model_version,
            "tokens_used": tokens_used,
            "cost_usd": float(cost_usd)
        }
        
        try:
            # Log interaction to database
            self.supabase.table("interaction_log").insert(data).execute()
            
            # Also log query embedding (in background, failures are non-critical)
            self.log_embedding(user_query, "query", {"model_version": model_version})
            
            # Print status
            status = []
            if refusal:
                status.append("Refusal")
            if toxicity:
                status.append("Toxic")
            status_str = " | ".join(status) if status else "Normal"
            
            print(f"✓ Logged interaction | Status: {status_str} | Tokens: {tokens_used}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to log interaction: {e}")
            return False
    
    def simulate_chat(self, query: str) -> Tuple[str, int, float]:
        """
        Simulate a chat interaction using OpenRouter API.
        
        Args:
            query: User's query text
            
        Returns:
            Tuple of (response_text, tokens_used, cost_usd)
        """
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/yourusername/drift-pipeline",
                    "X-Title": "Drift-Aware Pipeline"
                },
                json={
                    "model": self.chat_model,
                    "messages": [
                        {"role": "user", "content": query}
                    ]
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            response_text = result["choices"][0]["message"]["content"]
            
            # Extract token usage
            tokens = result.get("usage", {}).get("total_tokens", 0)
            
            # Estimate cost (rough approximation for GPT-3.5: $0.002 per 1K tokens)
            # Adjust based on your actual model pricing
            cost = (tokens / 1000) * 0.002
            
            return response_text, tokens, cost
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Chat API error: {e}")
            return f"Error: API request failed - {str(e)}", 0, 0.0
        except (KeyError, IndexError) as e:
            print(f"❌ Chat response format error: {e}")
            return f"Error: Invalid API response format", 0, 0.0
        except Exception as e:
            print(f"❌ Unexpected chat error: {e}")
            return f"Error: {str(e)}", 0, 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about logged data.
        
        Returns:
            Dictionary with counts and metrics
        """
        try:
            # Count interactions
            interactions = self.supabase.table("interaction_log")\
                .select("*", count="exact")\
                .execute()
            
            # Count embeddings
            embeddings = self.supabase.table("embeddings_log")\
                .select("*", count="exact")\
                .execute()
            
            # Count refusals
            refusals = self.supabase.table("interaction_log")\
                .select("refusal_flag", count="exact")\
                .eq("refusal_flag", True)\
                .execute()
            
            # Count toxic
            toxic = self.supabase.table("interaction_log")\
                .select("toxicity_flag", count="exact")\
                .eq("toxicity_flag", True)\
                .execute()
            
            total_interactions = len(interactions.data) if interactions.data else 0
            
            return {
                "total_interactions": total_interactions,
                "total_embeddings": len(embeddings.data) if embeddings.data else 0,
                "refusal_count": len(refusals.data) if refusals.data else 0,
                "toxic_count": len(toxic.data) if toxic.data else 0,
                "refusal_rate": (len(refusals.data) / total_interactions) if total_interactions > 0 else 0,
                "toxic_rate": (len(toxic.data) / total_interactions) if total_interactions > 0 else 0
            }
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATA LOGGER - TEST MODE")
    print("="*60 + "\n")
    
    # Initialize logger
    logger = DataLogger()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain neural networks in simple terms",
        "How do I hack into a system?",  # Should trigger refusal
        "I hate this stupid AI"  # Should trigger toxicity
    ]
    
    print("Running test interactions...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/{len(test_queries)} ---")
        print(f"Query: {query}")
        
        # Simulate chat
        response, tokens, cost = logger.simulate_chat(query)
        print(f"Response: {response[:100]}...")
        print(f"Tokens: {tokens}, Cost: ${cost:.4f}")
        
        # Log interaction
        logger.log_interaction(
            user_query=query,
            model_response=response,
            model_version="v1.0-test",
            tokens_used=tokens,
            cost_usd=cost,
            user_feedback=4  # Simulated feedback
        )
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    stats = logger.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Data logger test complete!\n")
