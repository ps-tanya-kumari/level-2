import os
import asyncio
import logging
from dotenv import load_dotenv

# Setup logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestBackend")

# Load environment
load_dotenv()

async def test_all():
    logger.info("Starting Backend Component Verification...")
    
    # 1. Environment Check
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("[FAIL] GEMINI_API_KEY is not configured in .env file.")
    else:
        logger.info("[SUCCESS] GEMINI_API_KEY is configured.")
        
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.warning("[WARN] GITHUB_TOKEN is not configured. Rate limits will be restricted to 60 requests/hour.")
    else:
        logger.info("[SUCCESS] GITHUB_TOKEN is configured.")

    # Import modules dynamically to check dependencies
    try:
        from app.github_client import GitHubClient
        from app.embeddings import GeminiEmbeddings
        from app.vector_store import VectorStore
        from app.gemini_client import GeminiClient
        logger.info("[SUCCESS] All backend Python modules imported successfully.")
    except ImportError as e:
        logger.error(f"[FAIL] Dependency import error: {e}")
        return

    # 2. Test GitHub API client
    logger.info("Verifying GitHub API Client...")
    try:
        client = GitHubClient()
        profile = await client.get_user_profile("octocat")
        logger.info(f"[SUCCESS] Fetched GitHub user profile: {profile['name']} (@{profile['username']})")
        
        repos = await client.get_user_repos("octocat")
        logger.info(f"[SUCCESS] Fetched public repositories: {len(repos)} repositories found.")
        await client.close()
    except Exception as e:
        logger.error(f"[FAIL] GitHub Client verification failed: {e}")

    # 3. Test Gemini Embeddings Client
    if api_key:
        logger.info("Verifying Gemini Embedding Model (text-embedding-004)...")
        try:
            embedder = GeminiEmbeddings()
            vector = embedder.get_embedding("Verify backend connection.")
            logger.info(f"[SUCCESS] Vector generated successfully! Vector length: {len(vector)}")
        except Exception as e:
            logger.error(f"[FAIL] Embedding generation failed: {e}")
    else:
        logger.warning("[SKIP] Embedding verification skipped (No API key).")

    # 4. Test ChromaDB Vector Store
    logger.info("Verifying ChromaDB Vector Store...")
    try:
        store = VectorStore()
        collection = store.get_or_create_collection("test", "sandbox")
        logger.info(f"[SUCCESS] Created sandbox collection: {collection.name}")
        
        # Test cleaning sandbox
        store.delete_collection("test", "sandbox")
        logger.info("[SUCCESS] Cleaned sandbox collection.")
    except Exception as e:
        logger.error(f"[FAIL] ChromaDB verification failed: {e}")

    # 5. Test Gemini Text Agent
    if api_key:
        logger.info("Verifying Gemini Text Generation (gemini-3.5-flash)...")
        try:
            gemini = GeminiClient()
            response = await gemini.generate(
                system_instruction="You are a helpful test assistant. Respond in exactly 3 words.",
                prompt="Ping"
            )
            logger.info(f"[SUCCESS] Gemini response: {response.strip()}")
        except Exception as e:
            logger.error(f"[FAIL] Gemini text generation failed: {e}")
    else:
        logger.warning("[SKIP] Gemini text generation skipped (No API key).")

    logger.info("Verification complete.")

if __name__ == "__main__":
    asyncio.run(test_all())
