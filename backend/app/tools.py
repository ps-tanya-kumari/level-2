import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.github_client import GitHubClient
from app.repository_parser import RepositoryParser
from app.vector_store import VectorStore
from app.embeddings import GeminiEmbeddings
from app.validator import ToolValidator, ToolValidationError, sanitize_path

logger = logging.getLogger(__name__)

# Maximum file character length to return directly to prevent context-window blowup
MAX_FILE_READ_CHARS = 40_000

class MCPTools:
    """Core tool implementation providing data retrieval and analysis from GitHub and ChromaDB."""

    def __init__(
        self, 
        github_client: GitHubClient, 
        vector_store: VectorStore, 
        embeddings: GeminiEmbeddings,
        cache_dir: Optional[str] = None
    ):
        self.github_client = github_client
        self.vector_store = vector_store
        self.embeddings = embeddings
        
        # Determine reliable absolute cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            env_cache = os.getenv("ANALYSIS_CACHE_DIR")
            if env_cache:
                self.cache_dir = Path(env_cache).resolve()
            else:
                self.cache_dir = (Path(__file__).resolve().parent.parent / "analysis_cache").resolve()
                
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"MCPTools cache directory initialized at: {self.cache_dir}")

    def _get_cache_path(self, owner: str, repo: str) -> Path:
        safe_owner = "".join(c for c in owner if c.isalnum() or c in ("-", "_"))
        safe_repo = "".join(c for c in repo if c.isalnum() or c in ("-", "_"))
        return self.cache_dir / f"{safe_owner}_{safe_repo}_summary.json"

    async def list_repository_files(self, owner: str, repo: str) -> List[str]:
        """
        Retrieves a flat list of all file paths in the repository that are analyzed.
        Use this tool to see what source code files exist in the project.
        """
        try:
            validated = ToolValidator.validate_arguments("list_repository_files", {"owner": owner, "repo": repo})
            tree = await self.github_client.get_repo_tree(validated["owner"], validated["repo"])
            filtered = RepositoryParser.filter_files(tree)
            return [f.get("path", "") for f in filtered if f.get("path")]
        except ToolValidationError as tve:
            logger.warning(f"Validation error in list_repository_files: {tve}")
            return [f"ValidationError: {tve.message}"]
        except Exception as e:
            logger.error(f"Error in list_repository_files: {e}")
            return [f"Error listing repository files: {str(e)}"]

    async def get_repository_structure(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Gets a tree-like directory structure of the repository.
        Useful for understanding how files are organized into folders.
        """
        try:
            validated = ToolValidator.validate_arguments("get_repository_structure", {"owner": owner, "repo": repo})
            tree = await self.github_client.get_repo_tree(validated["owner"], validated["repo"])
            structure = RepositoryParser.build_tree_structure(tree)
            return structure
        except ToolValidationError as tve:
            logger.warning(f"Validation error in get_repository_structure: {tve}")
            return {"error": f"ValidationError: {tve.message}"}
        except Exception as e:
            logger.error(f"Error in get_repository_structure: {e}")
            return {"error": f"Failed to build repository structure: {str(e)}"}

    async def read_repository_file(self, owner: str, repo: str, path: str) -> str:
        """
        Reads the content of a specific file in the repository.
        Includes path normalization, traversal protection, and length truncation.
        """
        try:
            validated = ToolValidator.validate_arguments("read_repository_file", {"owner": owner, "repo": repo, "path": path})
            clean_owner = validated["owner"]
            clean_repo = validated["repo"]
            clean_path = validated["path"]

            tree = await self.github_client.get_repo_tree(clean_owner, clean_repo)
            
            # Find the file with matching path to retrieve its sha
            sha = None
            for item in tree:
                if item.get("path") == clean_path and item.get("type") == "blob":
                    sha = item.get("sha")
                    break
                    
            if not sha:
                return f"Error: File '{clean_path}' not found in repository '{clean_owner}/{clean_repo}'."
                
            raw_content = await self.github_client.get_file_content(clean_owner, clean_repo, sha)
            
            # Enforce content size limits to prevent context blowing
            truncated_note = ""
            if len(raw_content) > MAX_FILE_READ_CHARS:
                raw_content = raw_content[:MAX_FILE_READ_CHARS]
                truncated_note = f"\n\n[Note: File content truncated at {MAX_FILE_READ_CHARS} characters for token safety.]"
                
            return (
                f"<file_content path=\"{clean_path}\">\n"
                f"{raw_content}"
                f"{truncated_note}\n"
                f"</file_content>"
            )
        except ToolValidationError as tve:
            logger.warning(f"Validation error in read_repository_file: {tve}")
            return f"ValidationError: {tve.message}"
        except Exception as e:
            logger.error(f"Error in read_repository_file for {path}: {e}")
            return f"Error reading file '{path}': {str(e)}"

    async def search_repository_code(self, owner: str, repo: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search over the codebase using the vector database.
        Returns code snippets, file paths, and chunk info matching the search query.
        """
        try:
            validated = ToolValidator.validate_arguments(
                "search_repository_code", 
                {"owner": owner, "repo": repo, "query": query, "n_results": n_results}
            )
            query_emb = self.embeddings.get_embedding(validated["query"], is_query=True)
            results = self.vector_store.query_similar_chunks(
                validated["owner"], 
                validated["repo"], 
                query_emb, 
                n_results=validated["n_results"]
            )
            
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
        except ToolValidationError as tve:
            logger.warning(f"Validation error in search_repository_code: {tve}")
            return [{"error": f"ValidationError: {tve.message}"}]
        except Exception as e:
            logger.error(f"Error in search_repository_code: {e}")
            return [{"error": f"Search failed: {str(e)}"}]

    async def get_repository_metadata(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Fetches repository metadata details (description, stars, primary language, etc.).
        """
        try:
            validated = ToolValidator.validate_arguments("get_repository_metadata", {"owner": owner, "repo": repo})
            clean_owner = validated["owner"]
            clean_repo = validated["repo"]

            repos = await self.github_client.get_user_repos(clean_owner)
            for r in repos:
                if r["name"].lower() == clean_repo.lower():
                    return r
                    
            default_branch = await self.github_client.get_default_branch(clean_owner, clean_repo)
            return {
                "name": clean_repo,
                "owner": clean_owner,
                "default_branch": default_branch,
                "language": "Unknown",
                "stars": 0,
                "description": "Public Repository"
            }
        except ToolValidationError as tve:
            logger.warning(f"Validation error in get_repository_metadata: {tve}")
            return {"error": f"ValidationError: {tve.message}"}
        except Exception as e:
            logger.error(f"Error in get_repository_metadata: {e}")
            return {"error": f"Failed to fetch metadata: {str(e)}"}

    def save_project_summary(self, owner: str, repo: str, summary_data: Dict[str, Any]):
        """Save a generated overview/summary to local cache."""
        if not summary_data or not isinstance(summary_data, dict):
            return
        overview_desc = summary_data.get("overview", {}).get("description", "")
        if "Failed to parse analysis JSON" in overview_desc or "Failed to generate analysis" in overview_desc:
            logger.warning(f"Refusing to save failed project summary to cache for {owner}/{repo}")
            return
            
        cache_path = self._get_cache_path(owner, repo)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved project summary cache for {owner}/{repo} at {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache project summary: {e}")

    def get_project_summary(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Retrieves the cached project analysis summary (overview, workflow, structure summary) if available.
        """
        try:
            validated = ToolValidator.validate_arguments("get_project_summary", {"owner": owner, "repo": repo})
            cache_path = self._get_cache_path(validated["owner"], validated["repo"])
            if cache_path.exists():
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        desc = data.get("overview", {}).get("description", "")
                        if "Failed to parse analysis JSON" in desc or "Failed to generate analysis" in desc:
                            cache_path.unlink(missing_ok=True)
                            return {"error": "Previous cached summary was invalid."}
                        return data
                except Exception as e:
                    logger.error(f"Error reading project summary cache: {e}")
            return {"error": "Project overview has not been generated or cached yet."}
        except ToolValidationError as tve:
            return {"error": f"ValidationError: {tve.message}"}
