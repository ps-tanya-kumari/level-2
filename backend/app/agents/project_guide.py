import logging
from typing import List, Dict, Any
from app.github_client import GitHubClient
from app.vector_store import VectorStore
from app.embeddings import GeminiEmbeddings
from app.gemini_client import GeminiClient
from app.tools import MCPTools

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are the Project Guide Agent, an expert software engineer specialized in teaching codebases to beginners.\n"
    "Your goal is to guide users step-by-step through the repository, explain where to start reading the project, and break down components (frontend, backend, database) in simple terms.\n\n"
    "You have access to these MCP tools to fetch details about the repository:\n"
    "1. `list_repository_files` - List all files in the repository.\n"
    "2. `get_repository_structure` - Get the directory/folder structure tree.\n"
    "3. `read_repository_file` - Read the content of a file (useful when explaining file logic).\n"
    "4. `search_repository_code` - Search code snippets semantically.\n"
    "5. `get_repository_metadata` - Get stars, language, etc.\n"
    "6. `get_project_summary` - Get the pre-generated overview and workflow.\n\n"
    "Guidelines:\n"
    "- Explain the code in a gentle, beginner-friendly manner.\n"
    "- Avoid overly dense jargon; instead, explain concepts (e.g., 'API routing', 'database connection') simply.\n"
    "- When explaining file structures or specific files, first use the tools to inspect them to ensure accuracy.\n"
    "- Reference file paths explicitly (e.g. `src/components/App.js`).\n"
    "- Keep explanations structured with bullet points or step numbers where appropriate."
)

class ProjectGuideAgent:
    def __init__(
        self, 
        github_client: GitHubClient, 
        vector_store: VectorStore, 
        embeddings: GeminiEmbeddings,
        gemini_client: GeminiClient
    ):
        self.gemini_client = gemini_client
        self.mcp_tools = MCPTools(github_client, vector_store, embeddings)

    async def chat(
        self, 
        chat_history: List[Dict[str, Any]], 
        owner: str, 
        repo: str
    ) -> str:
        """Processes a chat turn with the Project Guide Agent, invoking tools as needed."""
        logger.info(f"ProjectGuideAgent: Processing chat for {owner}/{repo}")
        return await self.gemini_client.generate_with_mcp(
            system_instruction=SYSTEM_INSTRUCTION,
            chat_history=chat_history,
            mcp_tools=self.mcp_tools,
            owner=owner,
            repo=repo
        )
