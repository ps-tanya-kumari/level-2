import os
import json
import logging
from typing import List, Dict, Any, Optional
from app.github_client import GitHubClient
from app.repository_parser import RepositoryParser
from app.vector_store import VectorStore
from app.embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)

class MCPTools:
    def __init__(self, github_client: GitHubClient, vector_store: VectorStore, embeddings: GeminiEmbeddings):
        self.github_client = github_client
        self.vector_store = vector_store
        self.embeddings = embeddings
        # Local JSON cache for summaries to avoid re-generating
        self.cache_dir = "./analysis_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, owner: str, repo: str) -> str:
        return os.path.join(self.cache_dir, f"{owner}_{repo}_summary.json")

    async def list_repository_files(self, owner: str, repo: str) -> List[str]:
        """
        Retrieves a flat list of all file paths in the repository that are analyzed.
        Use this tool to see what source code files exist in the project.
        """
        try:
            tree = await self.github_client.get_repo_tree(owner, repo)
            filtered = RepositoryParser.filter_files(tree)
            return [f.get("path", "") for f in filtered if f.get("path")]
        except Exception as e:
            logger.error(f"Error in list_repository_files: {e}")
            return [f"Error listing repository files: {str(e)}"]

    async def get_repository_structure(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Gets a tree-like directory structure of the repository.
        Useful for understanding how files are organized into folders.
        """
        try:
            tree = await self.github_client.get_repo_tree(owner, repo)
            structure = RepositoryParser.build_tree_structure(tree)
            return structure
        except Exception as e:
            logger.error(f"Error in get_repository_structure: {e}")
            return {"error": f"Failed to build repository structure: {str(e)}"}

    async def read_repository_file(self, owner: str, repo: str, path: str) -> str:
        """
        Reads the full content of a specific file in the repository.
        Use this tool to inspect the implementation details of a code file.
        """
        try:
            tree = await self.github_client.get_repo_tree(owner, repo)
            # Find the file with the matching path to get its sha
            sha = None
            for item in tree:
                if item.get("path") == path and item.get("type") == "blob":
                    sha = item.get("sha")
                    break
                    
            if not sha:
                return f"Error: File '{path}' not found in the repository."
                
            content = await self.github_client.get_file_content(owner, repo, sha)
            return content
        except Exception as e:
            logger.error(f"Error in read_repository_file for {path}: {e}")
            return f"Error reading file '{path}': {str(e)}"

    async def search_repository_code(self, owner: str, repo: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search over the codebase using the vector database.
        Returns code snippets, file paths, and chunk info matching the search query.
        """
        try:
            query_emb = self.embeddings.get_embedding(query, is_query=True)
            results = self.vector_store.query_similar_chunks(owner, repo, query_emb, n_results=n_results)
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "file_path": r["metadata"].get("file_path"),
                    "file_name": r["metadata"].get("file_name"),
                    "file_type": r["metadata"].get("file_type"),
                    "content": r["content"],
                    "relevance_distance": r["distance"]
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error in search_repository_code: {e}")
            return [{"error": f"Search failed: {str(e)}"}]

    async def get_repository_metadata(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Fetches repository metadata details (description, stars, primary language, etc.).
        """
        try:
            repos = await self.github_client.get_user_repos(owner)
            for r in repos:
                if r["name"].lower() == repo.lower():
                    return r
            # Fallback direct fetch if not in user list
            default_branch = await self.github_client.get_default_branch(owner, repo)
            return {
                "name": repo,
                "owner": owner,
                "default_branch": default_branch,
                "language": "Unknown",
                "stars": 0,
                "description": "Public Repository"
            }
        except Exception as e:
            logger.error(f"Error in get_repository_metadata: {e}")
            return {"error": f"Failed to fetch metadata: {str(e)}"}

    def save_project_summary(self, owner: str, repo: str, summary_data: Dict[str, Any]):
        """Save a generated overview/summary to local cache."""
        cache_path = self._get_cache_path(owner, repo)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved project summary cache for {owner}/{repo}")
        except Exception as e:
            logger.error(f"Failed to cache project summary: {e}")

    def get_project_summary(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Retrieves the cached project analysis summary (overview, workflow, structure summary) if available.
        """
        cache_path = self._get_cache_path(owner, repo)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading project summary cache: {e}")
        return {"error": "Project overview has not been generated or cached yet."}
