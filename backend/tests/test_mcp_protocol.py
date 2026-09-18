import pytest
from unittest.mock import AsyncMock, MagicMock
from app.mcp_server import handle_jsonrpc, JSONRPCRequest, TOOLS_SCHEMAS
from app.tools import MCPTools

@pytest.fixture
def mock_mcp_tools():
    tools = MagicMock(spec=MCPTools)
    tools.list_repository_files = AsyncMock(return_value=["backend/app/main.py", "backend/app/tools.py", "README.md"])
    tools.read_repository_file = AsyncMock(return_value="<file_content path='README.md'>\n# Hello MCP\n</file_content>")
    tools.get_repository_structure = AsyncMock(return_value={"type": "tree", "children": [{"name": "README.md", "type": "file"}]})
    tools.search_repository_code = AsyncMock(return_value=[{"file_path": "README.md", "content": "Hello MCP", "distance": 0.1}])
    tools.get_repository_metadata = AsyncMock(return_value={"name": "test-repo", "stars": 42, "language": "Python"})
    tools.get_project_summary = MagicMock(return_value={"overview": {"project_name": "test-repo"}, "workflow": {}})
    return tools

@pytest.mark.asyncio
async def test_mcp_initialize(mock_mcp_tools):
    req = JSONRPCRequest(id=1, method="initialize", params={})
    res = await handle_jsonrpc(req, tools=mock_mcp_tools)
    
    assert res.id == 1
    assert res.error is None
    assert res.result["protocolVersion"] == "2024-11-05"
    assert "tools" in res.result["capabilities"]
    assert res.result["serverInfo"]["name"] == "github-mcp-server"

@pytest.mark.asyncio
async def test_mcp_tools_list(mock_mcp_tools):
    req = JSONRPCRequest(id=2, method="tools/list", params={})
    res = await handle_jsonrpc(req, tools=mock_mcp_tools)
    
    assert res.id == 2
    assert res.error is None
    tools = res.result["tools"]
    assert len(tools) == len(TOOLS_SCHEMAS)
    tool_names = [t["name"] for t in tools]
    assert "list_repository_files" in tool_names
    assert "read_repository_file" in tool_names
    assert "search_repository_code" in tool_names

@pytest.mark.asyncio
async def test_mcp_tools_call_success(mock_mcp_tools):
    req = JSONRPCRequest(
        id=3,
        method="tools/call",
        params={
            "name": "read_repository_file",
            "arguments": {"owner": "octocat", "repo": "Spoon-Knife", "path": "README.md"}
        }
    )
    res = await handle_jsonrpc(req, tools=mock_mcp_tools)
    
    assert res.id == 3
    assert res.error is None
    assert res.result["isError"] is False
    assert len(res.result["content"]) == 1
    assert "Hello MCP" in res.result["content"][0]["text"]

@pytest.mark.asyncio
async def test_mcp_tools_call_validation_error(mock_mcp_tools):
    # Missing required 'path' parameter
    req = JSONRPCRequest(
        id=4,
        method="tools/call",
        params={
            "name": "read_repository_file",
            "arguments": {"owner": "octocat", "repo": "Spoon-Knife"}
        }
    )
    res = await handle_jsonrpc(req, tools=mock_mcp_tools)
    
    assert res.id == 4
    assert res.result["isError"] is True
    assert "ToolValidationError" in res.result["content"][0]["text"]

@pytest.mark.asyncio
async def test_mcp_unknown_method(mock_mcp_tools):
    req = JSONRPCRequest(id=5, method="unknown/method", params={})
    res = await handle_jsonrpc(req, tools=mock_mcp_tools)
    
    assert res.id == 5
    assert res.error is not None
    assert res.error.code == -32601
    assert "not found" in res.error.message
