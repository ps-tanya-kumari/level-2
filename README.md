# GitHub Multi-Agent Project Analyzer

Welcome to the **GitHub Multi-Agent Project Analyzer**! This is an intelligent multi-agent developer system designed to analyze, walk through, and answer queries about public GitHub repositories. 

The application works locally and connects directly to **GitHub**, **Google Gemini**, and local vector storage using a **FastAPI** backend conforming to the **Model Context Protocol (MCP)** specification, coupled with a modern **Vite React** frontend.

---

## 🏗️ Architecture & How It Works

```text
               +-----------------------------------------+
               |            React Frontend               |
               +--------------------+--------------------+
                                    |
                       REST APIs /  | Agent Chat
                       SSE Status   |
                                    v
               +--------------------+--------------------+
               |            FastAPI Backend              |
               +--+-------------------+---------------+--+
                  |                   |               |
                  | Query/Fetch       | Embed/Query   | Reasoning & Tool Calls
                  v                   v               v
           +------+------+     +------+------+ +------+------+
           |  GitHub API |     |  ChromaDB   | | Google      | <--> MCP Server (JSON-RPC 2.0)
           |  Integration|     | (Vector DB) | | Gemini API  |      (/api/mcp & /api/mcp/sse)
           +-------------+     +-------------+ +------+------+
                                                      |
                                                      v
                                              +---------------+
                                              | Reflection &  |
                                              | Verifier Node |
                                              +---------------+
```

### 1. The RAG Pipeline (Retrieval-Augmented Generation)
To answer questions about code accurately without downloading entire folders, the app implements a complete RAG pipeline:
1. **Fetch:** The app requests the list of files in the repository using the GitHub Git Trees API recursively.
2. **Parse & Filter:** A parser filters out non-code files (images, binaries, node_modules, lockfiles) and enforces a 500 KB safety limit.
3. **Chunk:** File content is split into line-preserved chunks of ~1500 characters with 300-character overlap.
4. **Embed:** Chunks are sent to Gemini's `text-embedding-004` model to generate 768-dimensional embeddings.
5. **Store:** Chunks and their vectors are stored in a repository-specific collection in a persistent **ChromaDB** store.
6. **Query:** When a user asks a question, the query is vectorized and matched against ChromaDB code chunks.

### 2. Standard Model Context Protocol (MCP) Server
The application hosts a standard-compliant **MCP Server** inside FastAPI at `/api/mcp`:
- **JSON-RPC 2.0 Standard Methods:** Supports `initialize`, `notifications/initialized`, `tools/list`, `tools/call`, and `ping`.
- **Transports:** Supports standard HTTP POST JSON-RPC 2.0 endpoint (`/api/mcp`), SSE transport (`/api/mcp/sse`), and REST tool routes (`/api/mcp/tools`).
- **Dynamic Tool Discovery:** Agents discover available tools dynamically via the MCP `tools/list` protocol rather than hardcoding static schemas.
- **Available MCP Tools:**
  1. `list_repository_files`: Lists all source files in the project.
  2. `get_repository_structure`: Generates a folder structure JSON tree.
  3. `read_repository_file`: Dynamically fetches and reads a file's content from GitHub with path traversal protection.
  4. `search_repository_code`: Semantically searches ChromaDB.
  5. `get_repository_metadata`: Gets repository stars, language, and branch.
  6. `get_project_summary`: Retrieves the pre-analyzed workflow and overview.

### 3. Reflection & Verification Node (`ReflectionVerifier`)
Every synthesized answer passes through a post-generation evaluator node before returning to the user:
- **Claim Extraction:** Extracts all referenced file paths and symbols.
- **Evidence Cross-Check:** Verifies that referenced files actually exist in the indexed repository tree and matches tool observations.
- **Hallucination Detection:** Flags ungrounded file paths and appends a structured **Grounding & Reflection Report** detailing explicit limitations.

### 4. Tool Argument & Path Security Layer (`ToolValidator`)
- **Pydantic Validation:** Every tool invocation argument is strictly validated against typed Pydantic models before execution.
- **Path Traversal Protection:** Normalizes paths and rejects `../`, absolute drive letters, control characters, and null bytes.
- **Bounded Error Recovery:** Formats structured validation errors back to the agent loop to allow bounded parameter repair.

---

## 🛠️ Project Structure

```text
backend/
├── app/
│   ├── main.py                 # FastAPI Application routes & status polling
│   ├── github_client.py        # GitHub API interface
│   ├── repository_parser.py    # File structure builder & filtering
│   ├── chunker.py              # Line-aligned code text chunking
│   ├── embeddings.py           # Gemini text-embedding-004 client
│   ├── vector_store.py         # ChromaDB client & vector operations
│   ├── validator.py            # Pydantic tool argument & path validator
│   ├── tools.py                # Core tool logic
│   ├── mcp_server.py           # Standard JSON-RPC 2.0 MCP Server & SSE
│   ├── mcp_client.py           # Dynamic MCP discovery & execution client
│   ├── gemini_client.py        # Gemini & OpenAI-compatible LLM agent loops
│   └── agents/
│       ├── reflection.py       # Post-generation grounding verifier node
│       ├── repository_analyzer.py # Architecture overview & workflow generator
│       ├── project_guide.py    # Step-by-step beginner guide agent
│       └── qa_agent.py         # Grounded RAG + tool calling Q&A agent
├── tests/
│   ├── test_validator.py       # Schema & path traversal tests
│   ├── test_mcp_protocol.py    # JSON-RPC 2.0 MCP protocol tests
│   ├── test_reflection.py      # Grounding & hallucination detection tests
│   └── test_agent_mocked.py    # Offline mocked agent end-to-end tests
├── agent_trajectory.json       # Sanitized JSON execution trajectory trace
├── agent_trajectory.md         # Documented execution trajectory
├── requirements.txt            # Python dependencies list
└── .env.example                # Template for environment configuration
```

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.11+
- Node.js 18+ (npm)

---

### Step 1: Backend Configuration & Run

1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```

2. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```

3. Open `.env` and fill in:
   - `GEMINI_API_KEY`: Your key from [Google AI Studio](https://aistudio.google.com/).
   - `GITHUB_TOKEN`: (Recommended) GitHub Personal Access Token to prevent rate limits.

4. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. Run Automated Tests (Offline / Mocked):
   ```bash
   pytest backend/tests/ -v
   ```

6. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```
   The backend runs at `http://127.0.0.1:8000` and the interactive MCP/Swagger docs at `http://127.0.0.1:8000/docs`.

---

### Step 2: Frontend Configuration & Run

1. Open a new terminal and navigate to `frontend/`:
   ```bash
   cd frontend
   ```

2. Install dependencies and start Vite:
   ```bash
   npm install
   npm run dev
   ```
   The React web app will open at `http://localhost:5173`.
