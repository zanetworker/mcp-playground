# MCP Playground Revamp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modernize mcp-playground to use latest MCP Python SDK (1.26.0) with Streamable HTTP as default transport, SSE as fallback, updated dependencies, and latest LLM model names.

**Architecture:** Context-manager-based MCPClient that wraps MCP SDK's ClientSession with transport selection (streamable-http or sse). LLM bridges receive the client and use it within its async context. pyproject.toml replaces setup.py.

**Tech Stack:** mcp>=1.26.0, httpx>=0.28.0, openai>=2.28.0, anthropic>=0.84.0, ollama>=0.6.0, Python >=3.10

---

### Task 1: Replace setup.py with pyproject.toml and update dependencies

**Files:**
- Create: `pyproject.toml`
- Delete: `setup.py`
- Rewrite: `requirements.txt`

**Step 1:** Create `pyproject.toml` with latest dependency versions, python>=3.10
**Step 2:** Update `requirements.txt` to match
**Step 3:** Delete `setup.py`
**Step 4:** Run `pip install -e .` and verify it works
**Step 5:** Commit

### Task 2: Rewrite MCPClient with transport abstraction and context manager

**Files:**
- Rewrite: `mcp_playground/client.py`
- Modify: `mcp_playground/__init__.py`

**Step 1:** Write tests for new client API (context manager, both transports, call_tool with arguments=)
**Step 2:** Rewrite `client.py` with:
  - `transport` param: `"streamable-http"` (default) or `"sse"`
  - Optional `http_client` (httpx.AsyncClient) passthrough
  - `__aenter__`/`__aexit__` for session lifecycle
  - `list_tools()`, `call_tool()`, `list_resources()`, `read_resource()`
  - `call_tool()` uses `arguments=` kwarg, returns structured content
  - Remove all `/sse` URL enforcement/warnings
  - Keep backward-compat one-shot helpers
**Step 3:** Update `__init__.py` exports
**Step 4:** Run tests
**Step 5:** Commit

### Task 3: Update LLM models and bridge implementations

**Files:**
- Modify: `mcp_playground/llm_bridge/models.py`
- Modify: `mcp_playground/llm_bridge/base.py`
- Modify: `mcp_playground/llm_bridge/openai_bridge.py`
- Modify: `mcp_playground/llm_bridge/anthropic_bridge.py`
- Modify: `mcp_playground/llm_bridge/ollama_bridge.py`
- Modify: `mcp_playground/llm_bridge/openrouter_bridge.py`

**Step 1:** Update `models.py` with latest model IDs (claude-sonnet-4-20250514, gpt-4o, o3, etc.)
**Step 2:** Update all bridges to work with new context-manager client
**Step 3:** Fix `parse_tool_call` in all bridges to return tool call `id`
**Step 4:** Run tests
**Step 5:** Commit

### Task 4: Update Streamlit app

**Files:**
- Modify: `mcp-streamlit-app/app.py`
- Modify: `mcp-streamlit-app/requirements.txt`

**Step 1:** Add transport selector dropdown (Streamable HTTP / SSE)
**Step 2:** Remove `/sse` URL enforcement and warnings
**Step 3:** Update model lists from new models.py
**Step 4:** Update client usage to new context-manager API
**Step 5:** Update requirements.txt
**Step 6:** Commit

### Task 5: Update all tests

**Files:**
- Modify: `tests/test_client.py`
- Modify: `tests/unit/*.py`
- Modify: `tests/integration/*.py`

**Step 1:** Update mocks to match new client shape (context manager, streamable_http_client)
**Step 2:** Add tests for SSE fallback transport
**Step 3:** Fix integration tests
**Step 4:** Run full test suite
**Step 5:** Commit

### Task 6: Update README and docs

**Files:**
- Rewrite: `README.md`
- Modify: `DEVELOPMENT.md`

**Step 1:** Rewrite README with new API examples, transport info, updated model names
**Step 2:** Update DEVELOPMENT.md
**Step 3:** Commit
