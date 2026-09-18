import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.qa_agent import QAAgent
from app.agents.project_guide import ProjectGuideAgent
from app.mcp_client import MCPClient
from app.tools import MCPTools

@pytest.fixture
def mock_dependencies():
    github_client = MagicMock()
    github_client.get_repo_tree = AsyncMock(return_value=[
        {"path": "backend/app/main.py", "type": "blob", "sha": "123"},
        {"path": "backend/app/tools.py", "type": "blob", "sha": "456"},
        {"path": "README.md", "type": "blob", "sha": "789"}
    ])
    github_client.get_file_content = AsyncMock(return_value="# Demo App\nFastAPI Application")
    github_client.get_user_repos = AsyncMock(return_value=[{"name": "demo-repo", "stars": 10}])
    github_client.get_default_branch = AsyncMock(return_value="main")

    vector_store = MagicMock()
    vector_store.query_similar_chunks = MagicMock(return_value=[
        {
            "id": "chunk1",
            "content": "from fastapi import FastAPI\napp = FastAPI()",
            "metadata": {"file_path": "backend/app/main.py"},
            "distance": 0.05
        }
    ])

    embeddings = MagicMock()
    embeddings.get_embedding = MagicMock(return_value=[0.1] * 768)

    gemini_client = MagicMock()
    gemini_client.generate_with_mcp = AsyncMock(
        return_value="The application initializes the FastAPI server in `backend/app/main.py`."
    )

    return {
        "github_client": github_client,
        "vector_store": vector_store,
        "embeddings": embeddings,
        "gemini_client": gemini_client
    }

@pytest.mark.asyncio
async def test_mcp_client_dynamic_discovery_and_call(mock_dependencies):
    tools = MCPTools(
        mock_dependencies["github_client"],
        mock_dependencies["vector_store"],
        mock_dependencies["embeddings"]
    )
    client = MCPClient(tools)
    
    # 1. Discover tools dynamically
    tool_list = await client.list_tools()
    assert len(tool_list) >= 6
    
    # 2. Convert to Gemini toolset dynamically
    gemini_toolset = client.get_gemini_toolset(tool_list)
    assert len(gemini_toolset.function_declarations) >= 6
    
    # 3. Call tool through MCP JSON-RPC boundary
    result_str = await client.call_tool(
        "read_repository_file",
        {"owner": "octocat", "repo": "demo-repo", "path": "README.md"}
    )
    assert "Demo App" in result_str

@pytest.mark.asyncio
async def test_qa_agent_ask_pipeline_with_reflection(mock_dependencies):
    agent = QAAgent(
        github_client=mock_dependencies["github_client"],
        vector_store=mock_dependencies["vector_store"],
        embeddings=mock_dependencies["embeddings"],
        gemini_client=mock_dependencies["gemini_client"]
    )
    
    chat_history = [
        {"role": "user", "content": "Where is the FastAPI app initialized?"}
    ]
    
    answer = await agent.ask(chat_history, owner="octocat", repo="demo-repo")
    assert "backend/app/main.py" in answer
    # Check that reflection did not flag valid file
    assert "Grounding & Reflection Report" not in answer

@pytest.mark.asyncio
async def test_qa_agent_ask_detects_hallucination(mock_dependencies):
    # LLM hallucinates a non-existent file
    mock_dependencies["gemini_client"].generate_with_mcp = AsyncMock(
        return_value="The database models are defined in `backend/app/models/user_account.py`."
    )
    
    agent = QAAgent(
        github_client=mock_dependencies["github_client"],
        vector_store=mock_dependencies["vector_store"],
        embeddings=mock_dependencies["embeddings"],
        gemini_client=mock_dependencies["gemini_client"]
    )
    
    chat_history = [
        {"role": "user", "content": "Where are user models?"}
    ]
    
    answer = await agent.ask(chat_history, owner="octocat", repo="demo-repo")
    assert "user_account.py" in answer
    assert "Grounding & Reflection Report" in answer
    assert "Referenced path(s) not found in indexed repository" in answer

@pytest.mark.asyncio
async def test_project_guide_chat_pipeline(mock_dependencies):
    agent = ProjectGuideAgent(
        github_client=mock_dependencies["github_client"],
        vector_store=mock_dependencies["vector_store"],
        embeddings=mock_dependencies["embeddings"],
        gemini_client=mock_dependencies["gemini_client"]
    )
    
    chat_history = [
        {"role": "user", "content": "Guide me through this project."}
    ]
    
    response = await agent.chat(chat_history, owner="octocat", repo="demo-repo")
    assert "backend/app/main.py" in response
