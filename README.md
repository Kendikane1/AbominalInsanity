# AI Workspace

This repository is the top-level index for AI study, research, and experiments on
this Mac. Each real software project should be self-contained: its own Git
repository (or clearly defined subproject), dependency manifest, lockfile, local
`.venv`, tests, and README.

## Current map

| Path | Purpose | Environment status |
| --- | --- | --- |
| `ARENA_3.0/` | ARENA deep-learning, interpretability, RL, and evals curriculum | Legacy `requirements.txt`; no local environment yet |
| `Torch-related/` | PyTorch tutorials and reference implementations | Learning/reference material; no declared environment |
| `Literature/` | Research papers and paper-related notes/assets | No runtime needed |
| `bin/ai-project` | One-command bootstrapper for reproducible AI/ML projects | Ready; see below |
| `Projects/` | Master home for new, independently versioned projects | Agent-assisted setup enabled |

The retired `Gen_modelling_project/` remains in repository history but was
intentionally deleted from the working tree.

## One-command project setup

```sh
./bin/ai-project projects/my-experiment dl
```

Choose `minimal`, `ml` (the default), or `dl`. The command uses Python 3.12,
creates a project-local `.venv`, adds development tooling, and generates a lockfile.
See `./bin/ai-project` with no arguments for examples. Once installed on your
shell `PATH`, the leading `./bin/` is unnecessary.

For an agent-driven start, create an empty child under `Projects/`, launch Codex,
Claude Code, or Gemini CLI there, and describe the project normally. Persistent
parent instructions tell the agent to apply the same setup and verification
standard automatically.

## Golden path for a new Python research project

Use one `.venv` per project. `uv` shares downloaded packages in a global cache,
so separate environments are fast and space-efficient while remaining
reproducible.

```sh
mkdir my-project
cd my-project
uv init --python 3.12
uv add numpy pandas matplotlib jupyter ipykernel
uv add --dev pytest ruff basedpyright pre-commit
uv run python -c "import sys; print(sys.executable)"
```

Use `uv add package` instead of `pip install package`. Use `uv run ...` to run
commands without manually activating the environment. If activation is useful:

```sh
source .venv/bin/activate
deactivate
```

Commit `pyproject.toml`, `.python-version`, and `uv.lock`; never commit `.venv`.
Open the project directory itself in Zed, not this entire umbrella directory,
then select `.venv` in Zed's toolchain selector if it is not detected
automatically.

## Common commands

```sh
uv sync                         # Recreate/synchronize the environment
uv add <package>                # Add a runtime dependency
uv add --dev <package>          # Add a development dependency
uv remove <package>             # Remove a dependency
uv run python script.py         # Run in the project environment
uv run pytest                   # Run tests
uv run ruff check .             # Lint
uv run ruff format .            # Format
uv python list --only-installed # Inspect uv-managed Python versions
```

For full conventions, machine audit findings, experiment practices, and the
recommended cleanup sequence, see [docs/AI_WORKSTATION.md](docs/AI_WORKSTATION.md).
