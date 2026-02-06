

# Splitwise MCP Server (Local)

This repo runs a **local MCP server** that can read/write to **your Splitwise account**.

✅ Each user runs it locally and puts **their own Splitwise API key** in `.env`  
✅ Includes a **preview → confirm** safety gate for writes  
✅ Works via:
- **HTTP** (default): `http://127.0.0.1:8000/mcp`
- **STDIO** (recommended for desktop MCP clients)

---

## 1) Install prerequisites

- Install Python 3.11+
- Install **uv** (recommended)
  - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## 2) Clone the repo

```bash
git clone https://github.com/yashspatel/splitwise-mcp-server.git
cd splitwise-mcp-server

