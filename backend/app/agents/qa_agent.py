import logging
from typing import List, Dict, Any
from app.github_client import GitHubClient
from app.vector_store import VectorStore
from app.embeddings import GeminiEmbeddings
from app.gemini_client import GeminiClient
from app.tools import MCPTools

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are the Q&A Agent, an expert software architect who answers questions about code repositories.\n"
    "Your goal is to answer specific user queries about the codebase accurately, referencing relevant file paths, classes, and function names.\n\n"
    "You have access to these MCP tools to inspect the repository:\n"
    "1. `list_repository_files` - List all files in the repository.\n"
    "2. `get_repository_structure` - Get the directory structure tree.\n"
    "3. `read_repository_file` - Read the content of a file.\n"
    "4. `search_repository_code` - Search code snippets semantically.\n"
    "5. `get_repository_metadata` - Get stars, language, etc.\n"
    "6. `get_project_summary` - Get the pre-generated overview and workflow.\n\n"
    "Guidelines:\n"
    "- Base your answers ONLY on the actual codebase. Never invent or hallucinate functions or files.\n"
    "- Mention specific file paths (e.g., `app/main.py`) in your explanations whenever possible.\n"
    "- If you need more details to answer a question (e.g., you see an import statement and need to inspect the imported module), call the `read_repository_file` tool to fetch it.\n"
    "- If the answer is not present in the repository, state clearly: 'I could not find information about this in the repository.'"
)

class QAAgent:
    def __init__(
        self, 
        github_client: GitHubClient, 
        vector_store: VectorStore, 
        embeddings: GeminiEmbeddings,
        gemini_client: GeminiClient
    ):
        self.github_client = github_client
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.gemini_client = gemini_client
        self.mcp_tools = MCPTools(github_client, vector_store, embeddings)

    async def ask(
        self, 
        chat_history: List[Dict[str, Any]], 
        owner: str, 
        repo: str
    ) -> str:
        """
        Executes a Q&A question turn.
        First executes a similarity search (RAG) on ChromaDB using the user's latest query,
        prepends the retrieved chunks to the query, and triggers the Gemini tool loop.
        """
        logger.info(f"QAAgent: Processing question for {owner}/{repo}")
        
        if not chat_history:
            return "Error: No chat history provided."
            
        # 1. Grab the latest user query
        latest_user_msg = next((msg for msg in reversed(chat_history) if msg["role"] == "user"), None)
        if not latest_user_msg:
            return "Error: No user message found in history."
            
        query = latest_user_msg["content"]
        
        # 2. Perform ChromaDB similarity search (RAG)
        rag_context = ""
        try:
            query_emb = self.embeddings.get_embedding(query, is_query=True)
            results = self.vector_store.query_similar_chunks(owner, repo, query_emb, n_results=5)
            
            if results:
                context_parts = []
                for r in results:
                    file_path = r["metadata"].get("file_path", "Unknown File")
                    context_parts.append(
                        f"--- Code Context from {file_path} ---\n"
                        f"{r['content']}\n"
                        f"--------------------------------------"
                    )
                rag_context = "\n\n".join(context_parts)
                logger.info(f"QAAgent: Successfully retrieved {len(results)} chunks from ChromaDB.")
            else:
                logger.info("QAAgent: No matching chunks found in ChromaDB.")
        except Exception as e:
            logger.error(f"Error performing RAG similarity search: {e}")
            
        # 3. Create a modified copy of the chat history where we prepend the RAG context to the latest query.
        # This keeps the model grounded without breaking the conversation flow.
        history_copy = [dict(msg) for msg in chat_history]
        
        augmented_prompt = query
        if rag_context:
            augmented_prompt = (
                f"Use the following retrieved code snippets to help answer the user's question.\n"
                f"You can also use tools if you need to read more files or search further.\n\n"
                f"{rag_context}\n\n"
                f"User Question: {query}"
            )
            
        # Replace the content of the last user message with our augmented prompt
        for msg in reversed(history_copy):
            if msg["role"] == "user":
                msg["content"] = augmented_prompt
                break
                
        # 4. Run through Gemini tool calling loop
        return await self.gemini_client.generate_with_mcp(
            system_instruction=SYSTEM_INSTRUCTION,
            chat_history=history_copy,
            mcp_tools=self.mcp_tools,
            owner=owner,
            repo=repo
        )
