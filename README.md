# jupyter-ai-hermes-magics

`%%hermes` cell magic — talk to [Hermes Agent](https://github.com/NousResearch/hermes-agent) directly inside Jupyter notebooks.

## Features

- **Two-phase button**: Shift+Enter shows a ▶ button; click to send. No auto-send on run.
- **Stop mid-stream**: Click ⏹ to stop Hermes. No transcript cell created if stopped.
- **Dynamic cell type**: Code responses → code cell. Explanations → markdown cell.
- **No ipywidgets**: Uses IPython comms — no widget-state-saving issues.
- **Notebook context**: Hermes sees cells above the `%%hermes` cell (not below).
- **Session tree**: Dot-notation labels (`main.debug.fix`) for branching conversations.
- **Always new transcript**: Each response creates a fresh cell below — no fragile update logic.
- **Full agent**: Hermes has tools, memory, skills, and can edit notebook cells via MCP.

## Installation

```bash
pip install jupyter-ai-hermes-magics
```

Then load in a notebook:
```python
%load_ext jupyter_ai_hermes_magics
```

Or add to `ipython_config.py` for auto-load:
```python
c.InteractiveShellApp.extensions = ["jupyter_ai_hermes_magics"]
```

## Usage

### Basic chat
```python
%%hermes
Explain what the code in the cell above does
```
→ A ▶ button appears. Click it to send to Hermes. Response goes to a new cell below.

### Branching sessions
```python
%%hermes --label main
Write a function to sort a list

%%hermes --label main.debug
Why does this throw an error?

%%hermes --label main.debug.fix
Fix the error and show the corrected code
```

### Skip context
```python
%%hermes --no-context
What is the capital of France?
```

### Subcommands
```python
%hermes list     # Show session tree
%hermes reset    # Clear all sessions
%hermes version  # Show version
```

## How it works

### Two-phase flow

**Phase 1** (Shift+Enter): The cell output displays a ▶ button and a tip.
No request is sent. The cell's `execution_count` is captured to identify
the magic cell later.

**Phase 2** (Click ▶): Hermes starts via `hermes chat -q "..." -Q`.
The button toggles to ⏹ with a live timer. The user can:
- Wait for completion → response goes to a new transcript cell below.
- Click ⏹ → process killed, no transcript cell created.

### Cell type detection

After Hermes completes, the response is analyzed:
- **Single fenced code block** (no prose) → **code cell** (fences stripped)
- **Raw code** (starts with `def`/`class`/`import`) → **code cell**
- **Prose or mixed** → **markdown cell** (code blocks render inline)

### State consistency

- ▶ only when idle/done/stopped — never during streaming
- ⏹ only during streaming — never when idle
- Transcript always targets the cell below the **magic cell** (identified by `execution_count`)
- On kernel restart, button becomes a static snapshot (no comm target registered)

## Requirements

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed and configured (`hermes` on PATH)
- IPython ≥ 8.0
- Jupyter MCP server (auto-started by jupyter-ai in JupyterLab)

## License

MIT
