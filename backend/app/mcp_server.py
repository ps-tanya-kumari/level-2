import json
import logging
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.tools import MCPTools
from app.github_client import GitHubClient
from app.vector_store import VectorStore
from app.embeddings import GeminiEmbeddings
from app.validator import ToolValidator, ToolValidationError

logger = logging.getLogger(__name__)

# FastAPI Router for MCP
mcp_router = APIRouter(prefix="/api/mcp", tags=["MCP"])

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

# Standard MCP Specification Tool Schemas
TOOLS_SCHEMAS = [
    {
        "name": "list_repository_files",
        "description": "Retrieves a flat list of all analyzed file paths in the GitHub repository.",
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
        "description": "Gets a tree-like directory structure of the repository for hierarchical exploration.",
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
        "description": "Reads the text content of a specific file inside the repository with path safety.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                "repo": {"type": "string", "description": "The name of the GitHub repository"},
                "path": {"type": "string", "description": "Relative file path inside the repository (e.g. 'backend/app/main.py')"}
            },
            "required": ["owner", "repo", "path"]
        }
    },
    {
        "name": "search_repository_code",
        "description": "Performs semantic vector search over the codebase to find relevant code snippets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                "repo": {"type": "string", "description": "The name of the GitHub repository"},
                "query": {"type": "string", "description": "Natural language or code query describing desired logic"},
                "n_results": {"type": "integer", "description": "Number of results to return (1-25)", "default": 5}
            },
            "required": ["owner", "repo", "query"]
        }
    },
    {
        "name": "get_repository_metadata",
        "description": "Fetches GitHub repository metadata including stars, primary language, and description.",
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
        "description": "Retrieves the pre-generated architectural overview and execution workflow.",
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

# JSON-RPC 2.0 Request / Response Pydantic Schemas
class JSONRPCRequest(BaseModel):
    jsonrpc: str = Field(default="2.0", description="JSON-RPC protocol version")
    id: Optional[Union[str, int]] = Field(default=None, description="Request ID")
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Method parameters")

class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None

class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None

# REST facade request schema
class ToolCallRequest(BaseModel):
    name: str = Field(..., description="The name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")


def json_or_str(data: Any) -> str:
    """Helper to convert objects/lists to formatted string if they aren't raw text."""
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, ensure_ascii=False)


async def execute_tool_dispatch(name: str, args: Dict[str, Any], tools: MCPTools) -> Any:
    """Internal tool dispatcher with Pydantic validation."""
    validated = ToolValidator.validate_arguments(name, args)
    
    if name == "list_repository_files":
        return await tools.list_repository_files(validated["owner"], validated["repo"])
    elif name == "get_repository_structure":
        return await tools.get_repository_structure(validated["owner"], validated["repo"])
    elif name == "read_repository_file":
        return await tools.read_repository_file(validated["owner"], validated["repo"], validated["path"])
    elif name == "search_repository_code":
        return await tools.search_repository_code(
            validated["owner"], 
            validated["repo"], 
            validated["query"], 
            validated.get("n_results", 5)
        )
    elif name == "get_repository_metadata":
        return await tools.get_repository_metadata(validated["owner"], validated["repo"])
    elif name == "get_project_summary":
        return tools.get_project_summary(validated["owner"], validated["repo"])
    else:
        raise ToolValidationError(name, f"Unknown tool '{name}'.")


# ==========================================
# 1. Standard MCP JSON-RPC 2.0 Endpoint
# ==========================================

@mcp_router.post("", response_model=JSONRPCResponse)
@mcp_router.post("/jsonrpc", response_model=JSONRPCResponse)
async def handle_jsonrpc(request: JSONRPCRequest, tools: MCPTools = Depends(get_mcp_tools)):
    """
    Standard Model Context Protocol JSON-RPC 2.0 Handler.
    Supports: initialize, notifications/initialized, tools/list, tools/call, ping.
    """
    req_id = request.id
    method = request.method
    params = request.params or {}

    logger.info(f"MCP JSON-RPC request: method='{method}', id={req_id}")

    # 1. initialize
    if method == "initialize":
        return JSONRPCResponse(
            id=req_id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": "github-mcp-server",
                    "version": "1.0.0"
                }
            }
        )

    # 2. initialized notification
    elif method == "notifications/initialized":
        return JSONRPCResponse(id=req_id, result={"status": "ready"})

    # 3. ping
    elif method == "ping":
        return JSONRPCResponse(id=req_id, result={})

    # 4. tools/list
    elif method == "tools/list":
        return JSONRPCResponse(
            id=req_id,
            result={"tools": TOOLS_SCHEMAS}
        )

    # 5. tools/call
    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=-32602, message="Missing parameter 'name' for tools/call.")
            )
            
        try:
            result = await execute_tool_dispatch(tool_name, arguments, tools)
            content_text = json_or_str(result)
            return JSONRPCResponse(
                id=req_id,
                result={
                    "content": [
                        {"type": "text", "text": content_text}
                    ],
                    "isError": False
                }
            )
        except ToolValidationError as tve:
            logger.warning(f"MCP JSON-RPC validation error on '{tool_name}': {tve}")
            return JSONRPCResponse(
                id=req_id,
                result={
                    "content": [
                        {"type": "text", "text": f"ToolValidationError: {tve.message}"}
                    ],
                    "isError": True
                }
            )
        except Exception as e:
            logger.error(f"MCP JSON-RPC execution error on '{tool_name}': {e}")
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=-32603, message=f"Internal tool execution error: {str(e)}")
            )

    else:
        return JSONRPCResponse(
            id=req_id,
            error=JSONRPCError(code=-32601, message=f"Method '{method}' not found.")
        )


# ==========================================
# 2. Standard MCP Server-Sent Events (SSE)
# ==========================================

@mcp_router.get("/sse")
async def mcp_sse_endpoint():
    """
    Standard MCP SSE Transport Endpoint.
    Sends initial endpoint discovery and keepalive events.
    """
    async def event_generator():
        # MCP SSE specification handshake
        endpoint_event = {
            "type": "endpoint",
            "uri": "/api/mcp"
        }
        yield f"event: endpoint\ndata: {json.dumps(endpoint_event)}\n\n"
        
        # Initial tools summary
        init_event = {
            "protocolVersion": "2024-11-05",
            "server": "github-mcp-server",
            "tools_count": len(TOOLS_SCHEMAS)
        }
        yield f"event: server_info\ndata: {json.dumps(init_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==========================================
# 3. REST API Compatibility Facade
# ==========================================

@mcp_router.get("/tools")
def list_tools():
    """List all available tools in the MCP Server."""
    return {"tools": TOOLS_SCHEMAS}

@mcp_router.post("/tools/call")
async def call_tool(request: ToolCallRequest, tools: MCPTools = Depends(get_mcp_tools)):
    """Execute a tool request via REST facade with strict validation."""
    try:
        result = await execute_tool_dispatch(request.name, request.arguments, tools)
        return {"content": [{"type": "text", "text": json_or_str(result)}]}
    except ToolValidationError as tve:
        raise HTTPException(status_code=400, detail=tve.message)
    except Exception as e:
        logger.error(f"Error in REST call_tool '{request.name}': {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")
