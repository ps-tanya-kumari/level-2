import os
import logging
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiEmbeddings:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info("GeminiEmbeddings initialized successfully.")
        else:
            logger.warning("GEMINI_API_KEY not configured. Will use Ollama fallback for embeddings.")
        
        self.model = "models/gemini-embedding-001"

    def _call_ollama_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama's OpenAI-compatible /v1/embeddings endpoint."""
        api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        
        import httpx
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "input": texts
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(f"{api_base}/v1/embeddings", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                data_list = data["data"]
                # Sort by index to maintain ordering
                data_list.sort(key=lambda x: x.get("index", 0))
                return [x["embedding"] for x in data_list]
        except Exception as e:
            logger.error(f"Ollama embeddings API call failed: {e}")
            raise e

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text chunk or query."""
        if self.api_key:
            task_type = "retrieval_query" if is_query else "retrieval_document"
            try:
                response = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type=task_type
                )
                return response.get('embedding', [])
            except Exception as e:
                logger.warning(f"Gemini embedding failed: {e}. Falling back to Ollama...")

        # Fallback to Ollama
        try:
            results = self._call_ollama_embeddings_sync([text])
            return results[0] if results else []
        except Exception as oe:
            logger.error(f"Ollama embedding fallback failed: {oe}")
            raise oe

    def get_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings in batches to improve processing speed."""
        import time
        if not texts:
            return []
            
        if self.api_key:
            task_type = "retrieval_query" if is_query else "retrieval_document"
            batch_size = 100
            all_embeddings = []
            
            try:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    
                    # Retry with exponential backoff on 429
                    for attempt in range(5):
                        try:
                            response = genai.embed_content(
                                model=self.model,
                                content=batch,
                                task_type=task_type
                            )
                            embeddings = response.get('embedding', [])
                            all_embeddings.extend(embeddings)
                            break
                        except Exception as e:
                            if "429" in str(e) and attempt < 4:
                                sleep_time = (attempt + 1) * 3
                                logger.warning(f"Rate limit 429 hit. Sleeping for {sleep_time}s and retrying (Attempt {attempt+1}/5)...")
                                time.sleep(sleep_time)
                            else:
                                raise e
                    
                    # Small delay between batches to respect rate limits
                    time.sleep(0.5)
                    
                return all_embeddings
            except Exception as e:
                logger.warning(f"Gemini batch embedding failed: {e}. Falling back to Ollama...")

        # Fallback to Ollama
        try:
            return self._call_ollama_embeddings_sync(texts)
        except Exception as oe:
            logger.error(f"Ollama batch embedding fallback failed: {oe}")
            raise oe
