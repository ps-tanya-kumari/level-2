# GitHub Multi-Agent Project Analyzer

Welcome to the **GitHub Multi-Agent Project Analyzer**! This is a simple, beginner-friendly web application designed to help you analyze, walk through, and ask questions about public GitHub repositories. 

The application works entirely locally and connects directly to **GitHub** and **Google Gemini** using a FastAPI backend and a Vite React frontend.

---

## 🏗️ Architecture & How It Works

Here is a look at how the different components of this application work together:

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
           |  GitHub API |     |  ChromaDB   | | Google      | <--> MCP Tools Server
           |  Integration|     | (Vector DB) | | Gemini API  |      (/api/mcp/tools)
           +-------------+     +-------------+ +-------------+
```

### 1. The RAG Pipeline (Retrieval-Augmented Generation)
To answer questions about code accurately without downloading entire folders, the app implements a complete RAG pipeline:
1. **Fetch:** The app requests the list of files in the repository using the GitHub Git Trees API recursively.
2. **Parse:** A parser filters out non-code files (images, binaries, node_modules, lockfiles) and files larger than 500 KB to avoid bloat.
3. **Chunk:** File content is split into line-preserved chunks of ~1500 characters with a 300-character overlap.
4. **Embed:** Chunks are sent to Gemini's `text-embedding-004` model to generate 768-dimensional mathematical vectors representing their meaning.
5. **Store:** Chunks and their vectors are saved to a repository-specific collection in a local persistent instance of **ChromaDB**.
6. **Query:** When a user asks a question, the question is vectorized, and ChromaDB retrieves the top 5 most similar code chunks.

### 2. Model Context Protocol (MCP) Tools Server
The app hosts an HTTP-based **MCP Tools Server** inside FastAPI at `/api/mcp`. It exposes six specialized endpoints:
- `list_repository_files`: Lists all source files in the project.
- `get_repository_structure`: Generates a folder structure JSON tree.
- `read_repository_file`: Dynamically fetches and reads a file's content from GitHub.
- `search_repository_code`: Semantically searches ChromaDB.
- `get_repository_metadata`: Gets repository stars, language, etc.
- `get_project_summary`: Retrieves the pre-analyzed workflow and overview.

### 3. The 3 Specialized AI Agents
We use three cooperative agents to handle separate concerns:
1. **Repository Analyzer Agent:** Executes post-ingestion. It scans the README, directory files, and package configurations, then calls Gemini to generate a structured JSON overview (description, problem solved, features) and a dynamic workflow chart.
2. **Project Guide Agent:** Helps beginners walk through the project. It handles questions like *"Explain this like a beginner"* or *"Where do I start?"*. It dynamically queries the folder structure and reads code using MCP tools.
3. **Q&A Agent:** Answers open-ended developer questions (e.g., *"How does login work?"*). It retrieves relevant context chunks via ChromaDB (RAG) and calls MCP tools to inspect other files referenced in code imports.

---

## 🛠️ Project Structure

The project code is organized as follows:

```text
backend/
├── app/
│   ├── main.py                 # FastAPI Application routes & status polling
│   ├── github_client.py        # GitHub API interface
│   ├── repository_parser.py    # File structure builder & filtering
│   ├── chunker.py              # Line-aligned code text chunking
│   ├── embeddings.py           # Gemini text-embedding-004 client
│   ├── vector_store.py         # ChromaDB client & vector operations
│   ├── tools.py                # MCP tools implementations
│   ├── mcp_server.py           # JSON-RPC MCP Tools Server endpoints
│   ├── gemini_client.py        # Gemini GenerativeModel & manual tool resolution loop
│   └── agents/
│       ├── repository_analyzer.py # Overview & Workflow generator
│       ├── project_guide.py    # Walkthrough Guide agent
│       └── qa_agent.py         # Code search & RAG Q&A agent
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
   - `GEMINI_API_KEY`: Get your key from [Google AI Studio](https://aistudio.google.com/).
   - `GITHUB_TOKEN`: (Optional but highly recommended) Generate a Personal Access Token (classic or fine-grained) on GitHub to prevent API rate limits.

4. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```
   The backend will be running at `http://127.0.0.1:8000`.

---

### Step 2: Frontend Configuration & Run

1. Open a new terminal and navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```

2. Install npm packages:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm run dev
   ```
   The React web app will open at `http://localhost:5173`.

---

## 💡 Troubleshooting & FAQs

* **Rate Limits:** If the analyzer fails to load files, check if you added a `GITHUB_TOKEN` to your `.env` file. Unauthenticated calls are limited to 60 requests per hour.
* **ChromaDB Errors:** If you experience issues starting ChromaDB, check that the folder `./backend/chroma_db` is writable.
* **Gemini Connection issues:** Ensure your `.env` contains a valid API key and that your network can connect to Google AI services.
