# Splitwise MCP Server (Local) — Step-by-step Setup (uv)

This guide tells you exactly how to run this MCP server locally and connect it to your own client.

---

## What you need

- Python **3.11+**
- **uv**
- Your Splitwise keys:
  - `SPLITWISE_CONSUMER_KEY`
  - `SPLITWISE_CONSUMER_SECRET`
  - `SPLITWISE_API_KEY` (your personal Splitwise API key)

---

## 1) Install uv

### Windows (PowerShell)
Open PowerShell and run:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell after installation.

### macOS / Linux (Terminal)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal after installation.

---

## 2) Clone this repo

```bash
git clone <REPLACE_WITH_THIS_REPO_URL>
cd <REPLACE_WITH_THIS_REPO_FOLDER>
```

---

## 3) Create your `.env`

### macOS / Linux
```bash
cp .env.example .env
```

### Windows (PowerShell)
```powershell
copy .env.example .env
```

Now open `.env` and fill in your values:

```env
SPLITWISE_CONSUMER_KEY=PASTE_APP_KEY_HERE
SPLITWISE_CONSUMER_SECRET=PASTE_APP_SECRET_HERE
SPLITWISE_API_KEY=PASTE_YOUR_SPLITWISE_API_KEY_HERE

# Optional server settings
MCP_TRANSPORT=http
MCP_HOST=127.0.0.1
MCP_PORT=8000
```

---

## 4) Install dependencies (uv)

From inside the repo folder:

```bash
uv sync
```

---

## 5) Run the MCP server

You have 2 ways to run it. Pick ONE depending on how you want to connect.

---

### Option A (HTTP) — easiest to test

Run:
```bash
uv run python main.py
```

Your server URL will be:
```text
http://127.0.0.1:8000/mcp
```

If port 8000 is busy, run:
```bash
uv run python main.py --port 8010
```

Then your URL becomes:
```text
http://127.0.0.1:8010/mcp
```

---

### Option B (STDIO) — best for most desktop MCP clients

Run:
```bash
uv run python main.py --transport stdio
```

Keep this running while you use your MCP client.

---

## 6) Connect this MCP server to your local client (JSON configs)

Different clients use different config styles. Use the one that matches your client.

---

### A) STDIO config (command + args)

Use this if your client supports launching the server locally.

#### macOS / Linux example
```json
{
  "mcpServers": {
    "splitwise": {
      "command": "uv",
      "args": ["run", "python", "main.py", "--transport", "stdio"],
      "cwd": "/absolute/path/to/this/repo"
    }
  }
}
```

#### Windows example (IMPORTANT: use double backslashes in the path)
```json
{
  "mcpServers": {
    "splitwise": {
      "command": "uv",
      "args": ["run", "python", "main.py", "--transport", "stdio"],
      "cwd": "C:\\\\Users\\\\YourName\\\\Desktop\\\\splitwise-mcp"
    }
  }
}
```

✅ What to change:
- Replace `cwd` with the folder path where `main.py` exists.

---

### B) HTTP config (URL)

Use this if your client connects using an MCP server URL.

Run the server in HTTP mode first:
```bash
uv run python main.py
```

Then use this JSON:

```json
{
  "mcpServers": {
    "splitwise": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

If you used another port like 8010, update the URL:
```json
{
  "mcpServers": {
    "splitwise": {
      "url": "http://127.0.0.1:8010/mcp"
    }
  }
}
```

---

## 7) Quick test prompts (inside your client)

Try these:

- `who am i on splitwise`
- `list my groups`
- `list my friends`
- `show my last 10 expenses`
- `add an expense of 10 cad between me and honey equally split, i paid, for lunch`

---

## Troubleshooting

### 1) Error: Missing SPLITWISE_API_KEY (or other Splitwise keys)
Fix:
- Make sure `.env` exists in the repo folder
- Make sure it contains:
  - `SPLITWISE_CONSUMER_KEY`
  - `SPLITWISE_CONSUMER_SECRET`
  - `SPLITWISE_API_KEY`

Then restart the server.

---

### 2) Error: Port already in use
Run with a different port:
```bash
uv run python main.py --port 8010
```

---

### 3) Error: Could not resolve participant names
Fix:
- Use the exact first name or full name as shown in Splitwise
- First run in your client:
  - `list my friends`
  - then use the names exactly

---

## Important notes

- Do **NOT** commit your `.env` file to GitHub.
- Keep `.env` private because it contains your personal API key.
- Only commit `.env.example`.

---

## Done ✅

You are ready when:
- Your server is running (`uv run python main.py` OR `--transport stdio`)
- Your client config points to the correct URL or command
