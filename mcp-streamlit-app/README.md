# MCP Streamlit Tool Tester App

A Streamlit application for interactively testing MCP (Model Context Protocol) servers and tools using LLMs (Large Language Models).

## Features

- **Modern UI**: Clean, responsive interface built with Streamlit, including dark mode styling.
- **LLM Integration**: Support for OpenAI and Anthropic models.
- **Tool Visualization**: Clear display of available tools and their parameters in the sidebar.
- **Educational Display**: Shows LLM reasoning, tool selection, and tool results separately.
- **Smart Output Display**:
    - Tabbed view for JSON and extracted Text content from tool results.
    - Automatic format detection (JSON, Markdown, HTML, Text) in the Text view.
    - Content displayed appropriately based on detected format.
    - Limited display height to prevent excessive scrolling.
    - Easy copy and download options for extracted content.
- **Conversation Management**:
    - History tab in the sidebar to view, load, and delete past conversations.
    - Ability to start new conversations.
    - Server endpoint history for quick switching (though direct input is primary).

## Prerequisites

- Python 3.7+
- Required packages listed in `requirements.txt` (Streamlit, OpenAI, Anthropic, Markdown, etc.)
- The main `mcp-sse-client` package installed (e.g., via `pip install -e .` from the root directory).

## Installation

1. Ensure you are in the root directory of the `mcp-sse-client-python` project.
2. Navigate to the app directory:
   ```bash
   cd mcp-streamlit-app
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

You can run the app using the provided script:

```bash
./run.sh
```

Or directly with Streamlit:

```bash
streamlit run app.py
```

## Usage

1. Start the application using `./run.sh` or `streamlit run app.py`.
2. In the sidebar's "Server Config" tab:
    - Enter the full URL for your MCP server endpoint.
    - Select the LLM provider (OpenAI or Anthropic).
    - Enter the corresponding API key.
    - Click "Connect".
3. Once connected, available tools will appear below the configuration tabs.
4. Use the main chat interface to interact with the LLM and trigger tools.
5. View results, switch between JSON/Text views, and use copy/download options.
6. Use the "History" tab to manage past conversations.
7. Use the "New Conversation" button to save the current chat and start a new one.

## Configuration

- Server endpoints and API keys are configured via the sidebar UI.
- API keys are stored in Streamlit's session state for the duration of the session.
- Conversation history is also stored in session state and will be lost when the app restarts or the browser tab is closed.

## License

This application is part of the `mcp-sse-client-python` project, licensed under the MIT License.
