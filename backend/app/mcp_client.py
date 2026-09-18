import json
import logging
from typing import List, Dict, Any, Optional
from google.generativeai.types import FunctionDeclaration, Tool
from app.mcp_server import handle_jsonrpc, JSONRPCRequest, TOOLS_SCHEMAS
from app.tools import MCPTools

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Standard MCP Client that connects to the MCP tool registry/server.
    Handles dynamic tool discovery, schema translation, and tool execution dispatch.
    """

    def __init__(self, mcp_tools: MCPTools):
        self.mcp_tools = mcp_tools
        self._initialized = False
        self._cached_tools: Optional[List[Dict[str, Any]]] = None

    async def initialize(self) -> Dict[str, Any]:
        """Perform MCP initialize handshake."""
        req = JSONRPCRequest(
            id=1,
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "github-agent-client", "version": "1.0.0"}
            }
        )
        res = await handle_jsonrpc(req, tools=self.mcp_tools)
        if res.error:
            raise RuntimeError(f"MCP Initialize failed: {res.error.message}")
            
        self._initialized = True
        logger.info(f"MCP Client initialized with server: {res.result.get('serverInfo')}")
        return res.result

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Discover all available tools dynamically via MCP 'tools/list'."""
        if not self._initialized:
            await self.initialize()
            
        req = JSONRPCRequest(
            id=2,
            method="tools/list",
            params={}
        )
        res = await handle_jsonrpc(req, tools=self.mcp_tools)
        if res.error:
            logger.error(f"MCP tools/list error: {res.error.message}")
            return TOOLS_SCHEMAS
            
        self._cached_tools = res.result.get("tools", [])
        return self._cached_tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool call via the MCP standard 'tools/call' protocol method.
        Returns the text response string.
        """
        if not self._initialized:
            await self.initialize()

        req = JSONRPCRequest(
            id=3,
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments
            }
        )
        res = await handle_jsonrpc(req, tools=self.mcp_tools)
        if res.error:
            logger.error(f"MCP tool call error on '{name}': {res.error.message}")
            return f"Error ({res.error.code}): {res.error.message}"
            
        result = res.result
        if result and "content" in result and len(result["content"]) > 0:
            first_item = result["content"][0]
            return first_item.get("text", "")
            
        return json.dumps(result, indent=2, ensure_ascii=False)

    def get_gemini_toolset(self, mcp_tools_list: Optional[List[Dict[str, Any]]] = None) -> Tool:
        """
        Dynamically translates standard MCP tool schemas into Google Gemini FunctionDeclarations and Toolset.
        Eliminates duplicate hardcoded schema drifting.
        """
        tools_to_convert = mcp_tools_list or self._cached_tools or TOOLS_SCHEMAS
        func_declarations = []
        
        for tool in tools_to_convert:
            name = tool["name"]
            desc = tool.get("description", "")
            input_schema = tool.get("inputSchema", {})
            
            # Map JSON Schema type keywords to Gemini types
            properties = {}
            for prop_name, prop_def in input_schema.get("properties", {}).items():
                p_type = prop_def.get("type", "string").upper()
                p_desc = prop_def.get("description", "")
                properties[prop_name] = {
                    "type": p_type,
                    "description": p_desc
                }
                
            decl = FunctionDeclaration(
                name=name,
                description=desc,
                parameters={
                    "type": "OBJECT",
                    "properties": properties,
                    "required": input_schema.get("required", [])
                }
            )
            func_declarations.append(decl)
            
        return Tool(function_declarations=func_declarations)

    def get_openai_tools(self, mcp_tools_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Dynamically translates standard MCP tool schemas into OpenAI function tools format.
        """
        tools_to_convert = mcp_tools_list or self._cached_tools or TOOLS_SCHEMAS
        openai_tools = []
        for tool in tools_to_convert:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {})
                }
            })
        return openai_tools
