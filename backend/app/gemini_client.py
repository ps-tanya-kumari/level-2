import os
import logging
import asyncio
from typing import List, Dict, Any, Union
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from google.generativeai.types.content_types import to_content, to_part
from app.tools import MCPTools
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# 1. Define Function Declarations for Gemini
list_repository_files_decl = FunctionDeclaration(
    name="list_repository_files",
    description="Retrieves a flat list of all file paths in the repository that are analyzed.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "owner": {"type": "STRING", "description": "The owner of the GitHub repository"},
            "repo": {"type": "STRING", "description": "The name of the GitHub repository"}
        },
        "required": ["owner", "repo"]
    }
)

get_repository_structure_decl = FunctionDeclaration(
    name="get_repository_structure",
    description="Gets a tree-like directory structure of the repository. Useful for folder explorer views.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "owner": {"type": "STRING", "description": "The owner of the GitHub repository"},
            "repo": {"type": "STRING", "description": "The name of the GitHub repository"}
        },
        "required": ["owner", "repo"]
    }
)

read_repository_file_decl = FunctionDeclaration(
    name="read_repository_file",
    description="Reads the full content of a specific file in the repository.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "owner": {"type": "STRING", "description": "The owner of the GitHub repository"},
            "repo": {"type": "STRING", "description": "The name of the GitHub repository"},
            "path": {"type": "STRING", "description": "The path to the file inside the repository (e.g. 'src/main.py')"}
        },
        "required": ["owner", "repo", "path"]
    }
)

search_repository_code_decl = FunctionDeclaration(
    name="search_repository_code",
    description="Performs a semantic search over the codebase to find relevant code chunks.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "owner": {"type": "STRING", "description": "The owner of the GitHub repository"},
            "repo": {"type": "STRING", "description": "The name of the GitHub repository"},
            "query": {"type": "STRING", "description": "Semantic query describing the code logic or variables to search"},
            "n_results": {"type": "INTEGER", "description": "Number of results to return (default is 5)"}
        },
        "required": ["owner", "repo", "query"]
    }
)

get_repository_metadata_decl = FunctionDeclaration(
    name="get_repository_metadata",
    description="Fetches metadata details (stars, language, description) of the repository.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "owner": {"type": "STRING", "description": "The owner of the GitHub repository"},
            "repo": {"type": "STRING", "description": "The name of the GitHub repository"}
        },
        "required": ["owner", "repo"]
    }
)

get_project_summary_decl = FunctionDeclaration(
    name="get_project_summary",
    description="Retrieves the pre-generated analyzer project overview and workflow.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "owner": {"type": "STRING", "description": "The owner of the GitHub repository"},
            "repo": {"type": "STRING", "description": "The name of the GitHub repository"}
        },
        "required": ["owner", "repo"]
    }
)

# Bundle declarations into a Tool
mcp_toolset = Tool(
    function_declarations=[
        list_repository_files_decl,
        get_repository_structure_decl,
        read_repository_file_decl,
        search_repository_code_decl,
        get_repository_metadata_decl,
        get_project_summary_decl
    ]
)

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info("GeminiClient initialized successfully.")
        else:
            logger.warning("GEMINI_API_KEY is not set.")
        
        self.model_name = "gemini-3.5-flash"

    async def _call_ollama(self, messages: list, tools: list = None, system_instruction: str = None) -> dict:
        """Call Ollama's OpenAI-compatible completions endpoint."""
        api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        
        payload = {
            "model": model,
            "messages": [],
            "stream": False
        }
        
        if system_instruction:
            payload["messages"].append({"role": "system", "content": system_instruction})
            
        payload["messages"].extend(messages)
        
        if tools:
            payload["tools"] = tools
            
        headers = {"Content-Type": "application/json"}
        
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{api_base}/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def generate(self, system_instruction: str, prompt: str) -> str:
        """Standard direct text generation without tool calling."""
        if self.api_key:
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )
                # Run in executor to prevent blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: model.generate_content(prompt)
                )
                return response.text
            except Exception as e:
                logger.warning(f"Error in Gemini direct generation: {e}. Falling back to Ollama...")
        else:
            logger.info("GEMINI_API_KEY not set. Using Ollama for direct generation.")

        # Fallback to Ollama
        try:
            messages = [{"role": "user", "content": prompt}]
            res = await self._call_ollama(messages, system_instruction=system_instruction)
            return res["choices"][0]["message"]["content"]
        except Exception as oe:
            logger.error(f"Ollama direct generation fallback failed: {oe}")
            return f"Error: Gemini failed and Ollama fallback also failed. Gemini: {locals().get('e', 'API key missing')}. Ollama: {str(oe)}"

    async def generate_with_mcp_ollama(
        self, 
        system_instruction: str, 
        chat_history: List[Dict[str, Any]], 
        mcp_tools: MCPTools, 
        owner: str, 
        repo: str
    ) -> str:
        """
        Executes an Ollama conversation, resolving tool/function calls using MCP.
        Supports dual-layer tool calling: native OpenAPI schemas or manual instruction parsing fallback.
        """
        logger.info(f"Ollama generate_with_mcp starting for {owner}/{repo}")
        import httpx
        
        messages = []
        for msg in chat_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_repository_files",
                    "description": "Retrieves a flat list of all file paths in the repository that are analyzed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                            "repo": {"type": "string", "description": "The name of the GitHub repository"}
                        },
                        "required": ["owner", "repo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_repository_structure",
                    "description": "Gets a tree-like directory structure of the repository. Useful for folder explorer views.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                            "repo": {"type": "string", "description": "The name of the GitHub repository"}
                        },
                        "required": ["owner", "repo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_repository_file",
                    "description": "Reads the full content of a specific file in the repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                            "repo": {"type": "string", "description": "The name of the GitHub repository"},
                            "path": {"type": "string", "description": "The path to the file inside the repository (e.g. 'src/main.py')"}
                        },
                        "required": ["owner", "repo", "path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_repository_code",
                    "description": "Performs a semantic search over the codebase to find relevant code chunks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                            "repo": {"type": "string", "description": "The name of the GitHub repository"},
                            "query": {"type": "string", "description": "Semantic query describing the code logic or variables to search"},
                            "n_results": {"type": "integer", "description": "Number of results to return (default is 5)"}
                        },
                        "required": ["owner", "repo", "query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_repository_metadata",
                    "description": "Fetches metadata details (stars, language, description) of the repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                            "repo": {"type": "string", "description": "The name of the GitHub repository"}
                        },
                        "required": ["owner", "repo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_project_summary",
                    "description": "Retrieves the pre-generated analyzer project overview and workflow.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "The owner of the GitHub repository"},
                            "repo": {"type": "string", "description": "The name of the GitHub repository"}
                        },
                        "required": ["owner", "repo"]
                    }
                }
            }
        ]

        use_native_tools = True
        manual_system_instruction = system_instruction + "\n\n" + (
            "You have access to the following MCP tools:\n"
            "1. `list_repository_files(owner, repo)`\n"
            "2. `get_repository_structure(owner, repo)`\n"
            "3. `read_repository_file(owner, repo, path)`\n"
            "4. `search_repository_code(owner, repo, query, n_results=5)`\n"
            "5. `get_repository_metadata(owner, repo)`\n"
            "6. `get_project_summary(owner, repo)`\n\n"
            "If you need to call a tool, output a single line in this exact format:\n"
            "CALL:<tool_name>(<json_arguments>)\n"
            "Example:\n"
            "CALL:read_repository_file({\"owner\": \"octocat\", \"repo\": \"Spoon-Knife\", \"path\": \"README.md\"})\n\n"
            "Do not output anything else in that turn if you output a CALL. Once you get the result (which will be supplied in your next turn as TOOL_RESPONSE: ...), write your final answer or execute another call."
        )

        max_iterations = 8
        for step in range(max_iterations):
            logger.info(f"Ollama Agent reasoning step {step+1}/{max_iterations}")
            
            try:
                if use_native_tools:
                    try:
                        res = await self._call_ollama(messages, tools=openai_tools, system_instruction=system_instruction)
                    except httpx.HTTPStatusError as hse:
                        if hse.response.status_code == 400:
                            logger.warning("Ollama native tool calling rejected (400). Switching to manual tool formatting fallback...")
                            use_native_tools = False
                            res = await self._call_ollama(messages, system_instruction=manual_system_instruction)
                        else:
                            raise hse
                else:
                    res = await self._call_ollama(messages, system_instruction=manual_system_instruction)
            except Exception as e:
                logger.error(f"Ollama API call failed during tool loop: {e}")
                return f"Ollama API call failed: {str(e)}"
                
            choice = res["choices"][0]
            assistant_msg = choice["message"]
            content = assistant_msg.get("content") or ""
            
            if use_native_tools:
                messages.append(assistant_msg)
                tool_calls = assistant_msg.get("tool_calls")
                if not tool_calls:
                    return content
                    
                for call in tool_calls:
                    call_id = call.get("id")
                    func = call["function"]
                    name = func["name"]
                    
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        import json
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                            
                    if "owner" not in args:
                        args["owner"] = owner
                    if "repo" not in args:
                        args["repo"] = repo
                        
                    logger.info(f"Ollama executing native tool call: {name}({args})")
                    
                    try:
                        result = None
                        if name == "list_repository_files":
                            result = await mcp_tools.list_repository_files(args["owner"], args["repo"])
                        elif name == "get_repository_structure":
                            result = await mcp_tools.get_repository_structure(args["owner"], args["repo"])
                        elif name == "read_repository_file":
                            result = await mcp_tools.read_repository_file(args["owner"], args["repo"], args["path"])
                        elif name == "search_repository_code":
                            result = await mcp_tools.search_repository_code(args["owner"], args["repo"], args["query"], args.get("n_results", 5))
                        elif name == "get_repository_metadata":
                            result = await mcp_tools.get_repository_metadata(args["owner"], args["repo"])
                        elif name == "get_project_summary":
                            result = mcp_tools.get_project_summary(args["owner"], args["repo"])
                        else:
                            result = f"Error: Tool '{name}' is not supported."
                            
                        if not isinstance(result, str):
                            import json
                            result_str = json.dumps(result, indent=2, ensure_ascii=False)
                        else:
                            result_str = result
                    except Exception as ex:
                        logger.error(f"Error executing tool {name} under Ollama: {ex}")
                        result_str = f"Error executing tool: {str(ex)}"
                        
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result_str
                    })
            else:
                # Manual tool parsing mode
                messages.append({"role": "assistant", "content": content})
                
                # Search for CALL:tool_name(args)
                import re
                match = re.search(r"CALL:(\w+)\((.*)\)", content)
                if not match:
                    return content
                    
                name = match.group(1)
                args_str = match.group(2).strip()
                
                import json
                try:
                    args = json.loads(args_str)
                except Exception as je:
                    logger.error(f"Ollama manual tool parsing failed to load json arguments: {je}")
                    # Try direct eval of args in case LLM missed double quotes or output single quotes
                    try:
                        import ast
                        args = ast.literal_eval(args_str)
                    except Exception:
                        args = {}
                    
                if "owner" not in args:
                    args["owner"] = owner
                if "repo" not in args:
                    args["repo"] = repo
                    
                logger.info(f"Ollama executing manual tool call: {name}({args})")
                
                try:
                    result = None
                    if name == "list_repository_files":
                        result = await mcp_tools.list_repository_files(args["owner"], args["repo"])
                    elif name == "get_repository_structure":
                        result = await mcp_tools.get_repository_structure(args["owner"], args["repo"])
                    elif name == "read_repository_file":
                        result = await mcp_tools.read_repository_file(args["owner"], args["repo"], args["path"])
                    elif name == "search_repository_code":
                        result = await mcp_tools.search_repository_code(args["owner"], args["repo"], args["query"], args.get("n_results", 5))
                    elif name == "get_repository_metadata":
                        result = await mcp_tools.get_repository_metadata(args["owner"], args["repo"])
                    elif name == "get_project_summary":
                        result = mcp_tools.get_project_summary(args["owner"], args["repo"])
                    else:
                        result = f"Error: Tool '{name}' is not supported."
                        
                    if not isinstance(result, str):
                        result_str = json.dumps(result, indent=2, ensure_ascii=False)
                    else:
                        result_str = result
                except Exception as ex:
                    logger.error(f"Error executing manual tool {name} under Ollama: {ex}")
                    result_str = f"Error executing tool: {str(ex)}"
                    
                messages.append({
                    "role": "user",
                    "content": f"TOOL_RESPONSE: {result_str}"
                })
                
        return "Ollama Agent execution terminated because it reached the maximum tool call limit."


    async def generate_with_mcp(
        self, 
        system_instruction: str, 
        chat_history: List[Dict[str, Any]], 
        mcp_tools: MCPTools, 
        owner: str, 
        repo: str
    ) -> str:
        """
        Executes a Gemini conversation, intercepting and resolving tool/function calls using MCP.
        Supports multi-turn tool loops. If Gemini fails or key is missing, falls back to Ollama.
        """
        if self.api_key:
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction,
                    tools=[mcp_toolset]
                )

                contents = []
                for msg in chat_history:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "assistant":
                        role = "model"
                    contents.append(to_content({"role": role, "parts": [content]}))

                max_iterations = 8
                loop = asyncio.get_event_loop()

                for step in range(max_iterations):
                    logger.info(f"Agent reasoning step {step+1}/{max_iterations}")
                    
                    response = await loop.run_in_executor(
                        None, 
                        lambda: model.generate_content(contents)
                    )

                    if not response.candidates:
                        return "Gemini did not return any response candidate."

                    candidate = response.candidates[0]
                    contents.append(candidate.content)

                    function_calls = [part.function_call for part in candidate.content.parts if part.function_call]
                    
                    if not function_calls:
                        return response.text

                    response_parts = []
                    for call in function_calls:
                        name = call.name
                        args = dict(call.args)
                        
                        if "owner" not in args:
                            args["owner"] = owner
                        if "repo" not in args:
                            args["repo"] = repo
                        
                        logger.info(f"Executing tool call: {name}({args})")
                        
                        try:
                            result = None
                            if name == "list_repository_files":
                                result = await mcp_tools.list_repository_files(args["owner"], args["repo"])
                            elif name == "get_repository_structure":
                                result = await mcp_tools.get_repository_structure(args["owner"], args["repo"])
                            elif name == "read_repository_file":
                                result = await mcp_tools.read_repository_file(args["owner"], args["repo"], args["path"])
                            elif name == "search_repository_code":
                                result = await mcp_tools.search_repository_code(args["owner"], args["repo"], args["query"], args.get("n_results", 5))
                            elif name == "get_repository_metadata":
                                result = await mcp_tools.get_repository_metadata(args["owner"], args["repo"])
                            elif name == "get_project_summary":
                                result = mcp_tools.get_project_summary(args["owner"], args["repo"])
                            else:
                                result = f"Error: Tool '{name}' is not supported."
                                
                            if not isinstance(result, str):
                                import json
                                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                            else:
                                result_str = result
                                
                            response_parts.append(
                                to_part({
                                    "function_response": {
                                        "name": name,
                                        "response": {"result": result_str}
                                    }
                                })
                            )
                        except Exception as ex:
                            logger.error(f"Error executing tool {name}: {ex}")
                            response_parts.append(
                                to_part({
                                    "function_response": {
                                        "name": name,
                                        "response": {"result": f"Error executing tool: {str(ex)}"}
                                    }
                                })
                            )

                    contents.append(to_content({"role": "user", "parts": response_parts}))

                return "Agent execution terminated because it reached the maximum tool call limit."

            except Exception as e:
                logger.warning(f"Gemini API call failed during tool loop: {e}. Falling back to Ollama...")
        else:
            logger.info("GEMINI_API_KEY not configured. Falling back to Ollama...")

        # Run with Ollama E2E
        return await self.generate_with_mcp_ollama(
            system_instruction=system_instruction,
            chat_history=chat_history,
            mcp_tools=mcp_tools,
            owner=owner,
            repo=repo
        )
