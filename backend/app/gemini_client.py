import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types.content_types import to_content, to_part
from app.tools import MCPTools
from app.mcp_client import MCPClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.nvidia_api_base = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
        self.nvidia_model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")

        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            os.environ["GOOGLE_API_KEY"] = self.api_key
            logger.info("GeminiClient initialized with Gemini API.")
            
        if self.nvidia_api_key:
            logger.info(f"GeminiClient configured with NVIDIA NIM model: {self.nvidia_model}")
            
        if not self.api_key and not self.nvidia_api_key:
            logger.warning("Neither GEMINI_API_KEY nor NVIDIA_API_KEY configured. Will use Ollama fallback.")

    async def _call_openai_compatible(
        self, 
        api_key: str, 
        api_base: str, 
        model: str, 
        messages: list, 
        tools: list = None, 
        system_instruction: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.2
    ) -> dict:
        """Call any OpenAI-compatible chat completions endpoint."""
        payload = {
            "model": model,
            "messages": [],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_instruction:
            payload["messages"].append({"role": "system", "content": system_instruction})
            
        payload["messages"].extend(messages)
        
        if tools:
            payload["tools"] = tools
            
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        import httpx
        async with httpx.AsyncClient(timeout=90.0) as client:
            url = api_base.rstrip('/')
            if not url.endswith('/chat/completions'):
                if not url.endswith('/v1'):
                    url = f"{url}/v1/chat/completions"
                else:
                    url = f"{url}/chat/completions"
            
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def _call_ollama(self, messages: list, tools: list = None, system_instruction: str = None) -> dict:
        """Call Ollama's OpenAI-compatible completions endpoint."""
        api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        return await self._call_openai_compatible("", api_base, model, messages, tools, system_instruction)

    async def generate(self, system_instruction: str, prompt: str) -> str:
        """Standard direct text generation without tool calling."""
        nvidia_error = None
        if self.nvidia_api_key:
            try:
                messages = [{"role": "user", "content": prompt}]
                res = await self._call_openai_compatible(
                    api_key=self.nvidia_api_key,
                    api_base=self.nvidia_api_base,
                    model=self.nvidia_model,
                    messages=messages,
                    system_instruction=system_instruction,
                    max_tokens=4096,
                    temperature=0.2
                )
                msg = res["choices"][0]["message"]
                content = msg.get("content")
                if not content and "reasoning_content" in msg:
                    content = msg.get("reasoning_content")
                if content:
                    return content
            except Exception as e:
                nvidia_error = e
                logger.warning(f"Error in NVIDIA NIM direct generation: {e}. Falling back to Gemini...")

        gemini_error = None
        if self.api_key:
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: model.generate_content(prompt)
                )
                return response.text
            except Exception as e:
                gemini_error = e
                logger.warning(f"Error in Gemini direct generation: {e}. Falling back to Ollama...")

        # Fallback to Ollama
        try:
            messages = [{"role": "user", "content": prompt}]
            res = await self._call_ollama(messages, system_instruction=system_instruction)
            return res["choices"][0]["message"]["content"]
        except Exception as oe:
            logger.error(f"Ollama direct generation fallback failed: {oe}")
            errors = []
            if nvidia_error:
                errors.append(f"Nvidia NIM: {nvidia_error}")
            if gemini_error:
                errors.append(f"Gemini: {gemini_error}")
            errors.append(f"Ollama: {oe}")
            return f"Error: All direct generation paths failed.\n" + "\n".join(errors)

    async def generate_with_mcp_openai_compatible(
        self, 
        api_key: str, 
        api_base: str, 
        model: str, 
        system_instruction: str, 
        chat_history: List[Dict[str, Any]], 
        mcp_tools: MCPTools, 
        owner: str, 
        repo: str
    ) -> str:
        """
        Executes an OpenAI-compatible conversation loop using dynamic MCP discovery and protocol dispatch.
        """
        logger.info(f"OpenAI generate_with_mcp starting for {owner}/{repo} using model {model}")
        import httpx
        
        mcp_client = MCPClient(mcp_tools)
        discovered_tools = await mcp_client.list_tools()
        openai_tools = mcp_client.get_openai_tools(discovered_tools)

        messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
        
        use_native_tools = True
        max_iterations = 8
        
        for step in range(max_iterations):
            logger.info(f"OpenAI Agent step {step+1}/{max_iterations}")
            
            try:
                res = await self._call_openai_compatible(
                    api_key=api_key,
                    api_base=api_base,
                    model=model,
                    messages=messages,
                    tools=openai_tools if use_native_tools else None,
                    system_instruction=system_instruction
                )
            except httpx.HTTPStatusError as hse:
                if hse.response.status_code in [400, 404, 501] and use_native_tools:
                    logger.warning(f"Native tool calling failed (status {hse.response.status_code}). Falling back to manual formatting.")
                    use_native_tools = False
                    continue
                else:
                    return f"API call failed: {str(hse)}"
            except Exception as e:
                return f"API call failed: {str(e)}"
                
            choice = res["choices"][0]
            assistant_msg = choice["message"]
            content = assistant_msg.get("content") or ""
            
            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls")
            
            if not tool_calls:
                return content
                
            for call in tool_calls:
                call_id = call.get("id")
                func = call["function"]
                name = func["name"]
                
                raw_args = func.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception as je:
                        logger.warning(f"Invalid JSON arguments from model: {raw_args}. Error: {je}")
                        args = {"owner": owner, "repo": repo, "_raw": raw_args}
                else:
                    args = raw_args
                    
                args["owner"] = owner
                args["repo"] = repo
                
                logger.info(f"MCP Client executing tool: {name}({args})")
                result_str = await mcp_client.call_tool(name, args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result_str
                })
                
        return "Agent execution stopped: Maximum iterations reached."

    async def generate_with_mcp_gemini(
        self,
        system_instruction: str,
        chat_history: List[Dict[str, Any]],
        mcp_tools: MCPTools,
        owner: str,
        repo: str
    ) -> str:
        """
        Executes Gemini multi-turn conversation loop with dynamic MCP tool discovery and execution.
        """
        mcp_client = MCPClient(mcp_tools)
        discovered_tools = await mcp_client.list_tools()
        gemini_toolset = mcp_client.get_gemini_toolset(discovered_tools)

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
            tools=[gemini_toolset]
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
            logger.info(f"Gemini MCP reasoning step {step+1}/{max_iterations}")
            
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
                args["owner"] = owner
                args["repo"] = repo
                
                logger.info(f"MCP Client dispatching Gemini tool: {name}({args})")
                result_str = await mcp_client.call_tool(name, args)
                
                response_parts.append(
                    to_part({
                        "function_response": {
                            "name": name,
                            "response": {"result": result_str}
                        }
                    })
                )

            contents.append(to_content({"role": "user", "parts": response_parts}))

        return "Gemini agent execution reached maximum tool call iterations."

    async def generate_with_mcp(
        self, 
        system_instruction: str, 
        chat_history: List[Dict[str, Any]], 
        mcp_tools: MCPTools, 
        owner: str, 
        repo: str
    ) -> str:
        """
        Executes conversation with MCP tool execution across available LLM backends (NVIDIA, Gemini, Ollama).
        """
        nvidia_error = None
        if self.nvidia_api_key:
            try:
                return await self.generate_with_mcp_openai_compatible(
                    api_key=self.nvidia_api_key,
                    api_base=self.nvidia_api_base,
                    model=self.nvidia_model,
                    system_instruction=system_instruction,
                    chat_history=chat_history,
                    mcp_tools=mcp_tools,
                    owner=owner,
                    repo=repo
                )
            except Exception as e:
                nvidia_error = e
                logger.warning(f"NVIDIA NIM tool calling failed: {e}. Falling back to Gemini...")

        gemini_error = None
        if self.api_key:
            try:
                return await self.generate_with_mcp_gemini(
                    system_instruction=system_instruction,
                    chat_history=chat_history,
                    mcp_tools=mcp_tools,
                    owner=owner,
                    repo=repo
                )
            except Exception as e:
                gemini_error = e
                logger.warning(f"Gemini tool calling failed: {e}. Falling back to Ollama...")

        try:
            api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3")
            return await self.generate_with_mcp_openai_compatible(
                api_key="",
                api_base=api_base,
                model=model,
                system_instruction=system_instruction,
                chat_history=chat_history,
                mcp_tools=mcp_tools,
                owner=owner,
                repo=repo
            )
        except Exception as oe:
            logger.error(f"All tool calling execution paths failed: {oe}")
            errors = []
            if nvidia_error:
                errors.append(f"Nvidia: {nvidia_error}")
            if gemini_error:
                errors.append(f"Gemini: {gemini_error}")
            errors.append(f"Ollama: {oe}")
            return "Error: All tool calling loops failed.\n" + "\n".join(errors)
