import json
import logging
from typing import Dict, Any, List
from app.github_client import GitHubClient
from app.vector_store import VectorStore
from app.embeddings import GeminiEmbeddings
from app.gemini_client import GeminiClient
from app.tools import MCPTools

logger = logging.getLogger(__name__)

class RepositoryAnalyzerAgent:
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

    async def analyze(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Analyze the repository. Gather files, README, project structure, 
        and semantic code context to generate overview and workflow.
        """
        logger.info(f"RepositoryAnalyzerAgent: Starting analysis for {owner}/{repo}")

        # 1. Fetch README
        readme_content = ""
        tree = await self.github_client.get_repo_tree(owner, repo)
        readme_item = next((item for item in tree if item.get("path", "").lower() in ["readme.md", "readme"]), None)
        if readme_item:
            try:
                readme_content = await self.github_client.get_file_content(owner, repo, readme_item["sha"])
            except Exception as e:
                logger.warning(f"Could not fetch README content: {e}")

        # 2. Fetch project files list and package configs if they exist
        pkg_configs = []
        for item in tree:
            path = item.get("path", "")
            if path in ["package.json", "requirements.txt", "go.mod", "Cargo.toml", "pom.xml", "build.gradle"]:
                try:
                    content = await self.github_client.get_file_content(owner, repo, item["sha"])
                    # Truncate to first 100 lines if too large
                    lines = content.splitlines()[:100]
                    pkg_configs.append(f"File: {path}\nContent Snippet:\n" + "\n".join(lines))
                except Exception:
                    pass

        # 3. Get file structure tree (truncated list for summary context)
        structure = [item.get("path", "") for item in tree if item.get("type") == "blob"]
        structure_str = "\n".join(structure[:100]) # Limit list length to fit token limits
        if len(structure) > 100:
            structure_str += f"\n... and {len(structure) - 100} more files."

        # 4. Perform vector queries for architectural keywords
        queries = ["architecture main process flow", "database connection setup", "api entry point frontend backend"]
        semantic_chunks = []
        for q in queries:
            try:
                emb = self.embeddings.get_embedding(q, is_query=True)
                results = self.vector_store.query_similar_chunks(owner, repo, emb, n_results=2)
                for r in results:
                    semantic_chunks.append(f"File: {r['metadata'].get('file_path')}\nContent:\n{r['content']}")
            except Exception as e:
                logger.warning(f"Failed semantic query during analysis: {e}")

        # 5. Build context
        context_parts = []
        if readme_content:
            context_parts.append(f"### README.md content:\n{readme_content[:3000]}") # Truncate readme if huge
        if pkg_configs:
            context_parts.append("### Project Configuration Files:\n" + "\n\n".join(pkg_configs))
        context_parts.append(f"### Repository Files Structure:\n{structure_str}")
        if semantic_chunks:
            context_parts.append("### Relevant Code Chunks:\n" + "\n\n".join(semantic_chunks))
            
        context = "\n\n".join(context_parts)

        # 6. Prompt Gemini for overview and workflow JSON
        system_instruction = (
            "You are the Repository Analyzer Agent, a specialized AI that understands code repositories.\n"
            "Analyze the provided repository files structure, README, configuration, and code chunks to understand the project.\n"
            "Your output must be a valid, parseable JSON document matching the schema requested below. Do not output anything else but JSON. No markdown backticks, no wrap text."
        )

        prompt = f"""
You are analyzing the GitHub repository: {owner}/{repo}.
Here is the context collected from the repository:
{context}

Please generate the Project Overview and Project Workflow.
You MUST respond with a single, valid JSON object. Do not include markdown code block syntax (like ```json). Just start with {{ and end with }}.

Required JSON Schema:
{{
  "overview": {{
    "project_name": "Project Name (clean, readable title)",
    "description": "High-level summary of what the project does",
    "problem_solved": "What real-world problem or challenge does this project solve?",
    "technologies": ["List of core technologies/frameworks used"],
    "languages": ["Main programming languages used (e.g. Python, Javascript)"],
    "dependencies": ["Key dependencies/libraries installed"],
    "features": ["3 to 5 key features of the application"],
    "important_files": [
      {{
        "path": "file_path_1",
        "description": "Brief description of this file's role in the project"
      }}
    ],
    "explanation": "A short, beginner-friendly explanation of how the project functions end-to-end."
  }},
  "workflow": {{
    "diagram": "A visual text-based flow diagram representing the execution sequence. Format example: User -> Frontend -> API Gateway -> Backend -> Database -> Response",
    "starting_point": "Where does the application execution start? (e.g. main.py, index.html, server.js)",
    "execution_flow": "Step-by-step description of the execution flow when a user interacts with the project",
    "frontend_backend": "How do the frontend and backend communicate? (REST API, WebSockets, none?)",
    "api_flow": "Describe the API routing and endpoints if any are present",
    "database_interaction": "Describe how the database is connected, queried, and updated (or mention if no database is used)",
    "processing_steps": "What are the core processing/business logic steps?",
    "final_output": "What is the final result or output of a successful run?"
  }}
}}
"""
        response_text = await self.gemini_client.generate(system_instruction, prompt)
        
        # Clean up response text in case Gemini wraps in markdown backticks
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            analysis_data = json.loads(response_text)
            # Cache the summary
            self.mcp_tools.save_project_summary(owner, repo, analysis_data)
            return analysis_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Gemini: {response_text}. Error: {e}")
            # Return partial structures if JSON fails
            fallback = {
                "overview": {
                    "project_name": repo,
                    "description": "Failed to parse analysis JSON. Please try again.",
                    "problem_solved": "Unknown",
                    "technologies": [],
                    "languages": [],
                    "dependencies": [],
                    "features": [],
                    "important_files": [],
                    "explanation": response_text
                },
                "workflow": {
                    "diagram": "Error parsing workflow details",
                    "starting_point": "Unknown",
                    "execution_flow": "Unknown",
                    "frontend_backend": "Unknown",
                    "api_flow": "Unknown",
                    "database_interaction": "Unknown",
                    "processing_steps": "Unknown",
                    "final_output": "Unknown"
                }
            }
            self.mcp_tools.save_project_summary(owner, repo, fallback)
            return fallback
