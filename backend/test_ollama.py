import os
import asyncio
import logging
from dotenv import load_dotenv

# Setup logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestOllama")

# Load environment
load_dotenv()

async def test_ollama():
    logger.info("Starting Ollama Fallback Verification...")
    
    # Override credentials/keys to force fallback
    os.environ["GEMINI_API_KEY"] = "" # Force fallback
    
    # 1. Import modules dynamically to check correctness
    try:
        from app.embeddings import GeminiEmbeddings
        from app.gemini_client import GeminiClient
        from app.tools import MCPTools
        from app.github_client import GitHubClient
        from app.vector_store import VectorStore
        logger.info("[SUCCESS] All backend modules imported successfully.")
    except ImportError as e:
        logger.error(f"[FAIL] Dependency import error: {e}")
        return

    # Check if local Ollama is reachable
    import httpx
    api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{api_base}/api/tags")
            if res.status_code == 200:
                logger.info(f"[SUCCESS] Ollama is reachable on {api_base}. Models available: {res.json().get('models', [])}")
            else:
                logger.warning(f"[WARN] Ollama is reachable but returned status {res.status_code}.")
    except Exception as e:
        logger.error(f"[FAIL] Local Ollama is NOT reachable on {api_base}. Make sure Ollama is running! Error: {e}")
        logger.warning("[INFO] We will run the tests anyway, but they will likely report fallback failures unless Ollama is started.")

    # 2. Test direct text generation fallback
    logger.info("Verifying direct text generation fallback...")
    try:
        client = GeminiClient()
        # Verify api_key is cleared
        client.api_key = ""
        response = await client.generate(
            system_instruction="You are a helpful test assistant. Respond in exactly 3 words.",
            prompt="Ping"
        )
        logger.info(f"[SUCCESS] Ollama text generation response: {response.strip()}")
    except Exception as e:
        logger.error(f"[FAIL] Direct text generation fallback failed: {e}")

    # 3. Test embeddings fallback
    logger.info("Verifying embeddings fallback...")
    try:
        embedder = GeminiEmbeddings()
        embedder.api_key = ""
        vector = embedder.get_embedding("Verify Ollama embeddings fallback.")
        logger.info(f"[SUCCESS] Ollama vector generated successfully! Vector length: {len(vector)}")
    except Exception as e:
        logger.error(f"[FAIL] Embeddings fallback failed: {e}")

    # 4. Test tool calling resolution loop fallback
    logger.info("Verifying tool calling loop fallback...")
    try:
        # Create a mock tools/clients setup
        gh = GitHubClient()
        vs = VectorStore()
        mcp = MCPTools(gh, embedder, vs)
        
        history = [
            {"role": "user", "content": "What is the metadata description of octocat/Spoon-Knife?"}
        ]
        
        client = GeminiClient()
        client.api_key = ""
        
        response = await client.generate_with_mcp(
            system_instruction="You are an expert repository explorer agent.",
            chat_history=history,
            mcp_tools=mcp,
            owner="octocat",
            repo="Spoon-Knife"
        )
        logger.info(f"[SUCCESS] Ollama tool-calling chat response: {response}")
        await gh.close()
    except Exception as e:
        logger.error(f"[FAIL] Tool-calling loop fallback failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
