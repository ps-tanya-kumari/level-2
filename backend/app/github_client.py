import os
import base64
import logging
from typing import Dict, List, Any, Optional
import httpx
from dotenv import load_dotenv

# Load env variables
load_dotenv()

logger = logging.getLogger(__name__)

class GitHubClient:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Multi-Agent-Project-Analyzer"
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
            logger.info("GitHubClient initialized with auth token.")
        else:
            logger.warning("GitHubClient initialized without auth token. Rate limits will apply.")
        
        self.base_url = "https://api.github.com"
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
        full_url = f"{self.base_url}{url}" if url.startswith("/") else url
        response = await self.client.get(full_url, params=params)
        if response.status_code != 200:
            logger.error(f"GitHub API Error on {full_url}: Status {response.status_code} - {response.text}")
            response.raise_for_status()
        return response

    async def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Fetch general details about the user profile."""
        response = await self._get(f"/users/{username}")
        data = response.json()
        return {
            "username": data.get("login"),
            "name": data.get("name") or data.get("login"),
            "avatar_url": data.get("avatar_url"),
            "bio": data.get("bio"),
            "public_repos": data.get("public_repos"),
            "followers": data.get("followers"),
            "following": data.get("following"),
            "html_url": data.get("html_url")
        }

    async def get_user_repos(self, username: str) -> List[Dict[str, Any]]:
        """Fetch all public repositories of a user."""
        # Page through repos up to 100 per page
        response = await self._get(f"/users/{username}/repos", params={"per_page": 100, "sort": "updated"})
        repos = response.json()
        
        result = []
        for r in repos:
            if not r.get("fork"): # Analyze own projects primarily
                result.append({
                    "name": r.get("name"),
                    "owner": username,
                    "description": r.get("description"),
                    "language": r.get("language") or "Unknown",
                    "stars": r.get("stargazers_count", 0),
                    "updated_at": r.get("updated_at"),
                    "default_branch": r.get("default_branch", "main"),
                    "html_url": r.get("html_url")
                })
        return result

    async def get_default_branch(self, owner: str, repo: str) -> str:
        """Get default branch for a repo."""
        try:
            response = await self._get(f"/repos/{owner}/{repo}")
            return response.json().get("default_branch", "main")
        except Exception:
            return "main"

    async def get_repo_tree(self, owner: str, repo: str, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch the repository's file structure recursively using Git Trees API."""
        if not branch:
            branch = await self.get_default_branch(owner, repo)
            
        try:
            response = await self._get(f"/repos/{owner}/{repo}/git/trees/{branch}", params={"recursive": "1"})
            data = response.json()
            return data.get("tree", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # If 'main' doesn't exist, let's try 'master'
                if branch == "main":
                    logger.info("Failed to find 'main' branch, trying 'master'...")
                    return await self.get_repo_tree(owner, repo, "master")
            raise e

    async def get_file_content(self, owner: str, repo: str, sha: str) -> str:
        """Fetch file content using Git Blobs API."""
        response = await self._get(f"/repos/{owner}/{repo}/git/blobs/{sha}")
        data = response.json()
        content_b64 = data.get("content", "")
        encoding = data.get("encoding", "")
        
        if encoding == "base64":
            # Remove newlines before decoding
            cleaned_content = content_b64.replace("\n", "").replace("\r", "")
            decoded = base64.b64decode(cleaned_content)
            try:
                return decoded.decode("utf-8")
            except UnicodeDecodeError:
                # Fallback if it's binary or weird encoding
                return "[Binary content or non-UTF-8 character set detected]"
        return content_b64
