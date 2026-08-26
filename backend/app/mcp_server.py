import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from app.tools import MCPTools
from app.github_client import GitHubClient
from app.vector_store import VectorStore
from app.embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)

# FastAPI Router for MCP
mcp_router = APIRouter(prefix="/api/mcp", tags=["MCP"])

# Pydantic schemas for request validation
class ToolCallRequest(BaseModel):
    name: str = Field(..., description="The name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")

# Dependency injection helpers
def get_github_client():
    from app.main import app_state
    return app_state["github_client"]

def get_vector_store():
    from app.main import app_state
    return app_state["vector_store"]

def get_embeddings():
    from app.main import app_state
    return app_state["embeddings"]

def get_mcp_tools(
    github_client: GitHubClient = Depends(get_github_client),
    vector_store: VectorStore = Depends(get_vector_store),
    embeddings: GeminiEmbeddings = Depends(get_embeddings)
) -> MCPTools:
    return MCPTools(github_client, vector_store, embeddings)

# Tool Schemas matching MCP specification
TOOLS_SCHEMAS = [
    {
        "name": "list_repository_files",
        "description": "Retrieves a flat list of all file paths in the repository that are analyzed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                "repo": {"type": "string", "description": "The name of the GitHub repository"}
            },
            "required": ["owner", "repo"]
        }
    },
    {
        "name": "get_repository_structure",
        "description": "Gets a tree-like directory structure of the repository. Useful for folder explorer views.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                "repo": {"type": "string", "description": "The name of the GitHub repository"}
            },
            "required": ["owner", "repo"]
        }
    },
    {
        "name": "read_repository_file",
        "description": "Reads the full content of a specific file in the repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                "repo": {"type": "string", "description": "The name of the GitHub repository"},
                "path": {"type": "string", "description": "The path to the file inside the repository (e.g. 'src/main.py')"}
            },
            "required": ["owner", "repo", "path"]
        }
    },
    {
        "name": "search_repository_code",
        "description": "Performs a semantic search over the codebase to find relevant code chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                "repo": {"type": "string", "description": "The name of the GitHub repository"},
                "query": {"type": "string", "description": "Semantic query describing the code logic or variables to search"},
                "n_results": {"type": "integer", "description": "Number of results to return", "default": 5}
            },
            "required": ["owner", "repo", "query"]
        }
    },
    {
        "name": "get_repository_metadata",
        "description": "Fetches metadata details (stars, language, description) of the repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                "repo": {"type": "string", "description": "The name of the GitHub repository"}
            },
            "required": ["owner", "repo"]
        }
    },
    {
        "name": "get_project_summary",
        "description": "Retrieves the pre-generated analyzer project overview and workflow.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                "repo": {"type": "string", "description": "The name of the GitHub repository"}
            },
            "required": ["owner", "repo"]
        }
    }
]

@mcp_router.get("/tools")
def list_tools():
    """List all available tools in the MCP Server."""
    return {"tools": TOOLS_SCHEMAS}

@mcp_router.post("/tools/call")
async def call_tool(request: ToolCallRequest, tools: MCPTools = Depends(get_mcp_tools)):
    """Execute a tool request and return the result."""
    name = request.name
    args = request.arguments
    
    logger.info(f"MCP Call: tool={name}, arguments={args}")
    
    if name == "list_repository_files":
        if "owner" not in args or "repo" not in args:
            raise HTTPException(status_code=400, detail="Missing required arguments: owner, repo")
        result = await tools.list_repository_files(args["owner"], args["repo"])
        return {"content": [{"type": "text", "text": json_or_str(result)}]}
        
    elif name == "get_repository_structure":
        if "owner" not in args or "repo" not in args:
            raise HTTPException(status_code=400, detail="Missing required arguments: owner, repo")
        result = await tools.get_repository_structure(args["owner"], args["repo"])
        return {"content": [{"type": "text", "text": json_or_str(result)}]}
        
    elif name == "read_repository_file":
        if "owner" not in args or "repo" not in args or "path" not in args:
            raise HTTPException(status_code=400, detail="Missing required arguments: owner, repo, path")
        result = await tools.read_repository_file(args["owner"], args["repo"], args["path"])
        return {"content": [{"type": "text", "text": result}]}
        
    elif name == "search_repository_code":
        if "owner" not in args or "repo" not in args or "query" not in args:
            raise HTTPException(status_code=400, detail="Missing required arguments: owner, repo, query")
        n = args.get("n_results", 5)
        result = await tools.search_repository_code(args["owner"], args["repo"], args["query"], n)
        return {"content": [{"type": "text", "text": json_or_str(result)}]}
        
    elif name == "get_repository_metadata":
        if "owner" not in args or "repo" not in args:
            raise HTTPException(status_code=400, detail="Missing required arguments: owner, repo")
        result = await tools.get_repository_metadata(args["owner"], args["repo"])
        return {"content": [{"type": "text", "text": json_or_str(result)}]}
        
    elif name == "get_project_summary":
        if "owner" not in args or "repo" not in args:
            raise HTTPException(status_code=400, detail="Missing required arguments: owner, repo")
        result = tools.get_project_summary(args["owner"], args["repo"])
        return {"content": [{"type": "text", "text": json_or_str(result)}]}
        
    else:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")

def json_or_str(data: Any) -> str:
    """Helper to convert objects/lists to string if they aren't raw text."""
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, ensure_ascii=False)
