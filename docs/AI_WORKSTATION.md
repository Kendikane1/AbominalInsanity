# AI engineering workstation guide

## The model to use

Treat this directory as an index, not as one giant Python project. A clean
research workstation has three layers:

1. **Machine tools**: Homebrew, `uv`, Git, Zed, and optional container/cloud CLIs.
2. **Project tools**: a `pyproject.toml`, `uv.lock`, `.python-version`, and `.venv`
   inside every project that executes Python.
3. **Research outputs**: data, checkpoints, run logs, plots, and paper notes with
   explicit storage and versioning rules.

Do not install research libraries into global Python. Do not share one virtual
environment between unrelated projects. Virtual environments are disposable;
the manifest and lockfile are the source of truth.

## Audit snapshot (2026-08-14)

- Apple Silicon Mac (`arm64`), macOS 26.3.
- `uv 0.11.7` is installed in `~/.local/bin`.
- The shell currently resolves Python to the python.org 3.11.1 framework install.
- Homebrew is installed under `/opt/homebrew` (native Apple Silicon).
- MacPorts paths under `/opt/local` are also injected by `.zprofile`.
- Node 22.11.0 and npm 10.9.0 resolve from `/usr/local`, which often indicates an
  Intel/Rosetta-era installation on Apple Silicon.
- No project-local Python virtual environments were found.
- Zed has Codex and Claude agent servers configured globally. Python settings in
  this repository now use Zed's built-in basedpyright and Ruff servers.
- Global Git `core.editor` still points to `code --wait`.
- The accidental root npm install containing JavaScript's deprecated `torch`
  package was removed on 2026-08-14.
- `ARENA_3.0/requirements.txt` adds the CUDA 11.8 wheel index. CUDA is unavailable
  on macOS; Apple Silicon acceleration uses PyTorch's MPS backend.
- `Gen_modelling_project/` was intentionally retired and deleted. Git history
  remains available if old work ever needs to be inspected.

## Python and virtual environments

### Automated setup

Use the included command instead of memorizing setup steps:

```sh
ai-project projects/my-project minimal
ai-project projects/my-analysis ml
ai-project projects/my-neural-network dl
```

`ml` is the default. Add `--no-sync` when offline or when you want only the
scaffold and lock metadata; run `uv sync` later. The generated project includes
separate source, test, notebook, configuration, data, and output locations.

### What a virtual environment is

A `.venv` is a project-specific Python executable plus links/install records for
that project's packages. It prevents incompatible package versions from leaking
between projects. It does not replace the source code or lockfile and can always
be deleted and recreated with `uv sync`.

The recommended rule is one environment per project, even when two projects use
the same dependencies. `uv` deduplicates downloads through its cache, so sharing
an environment provides little benefit and introduces dependency drift.

### Standard project layout

```text
project-name/
├── .gitignore
├── .python-version
├── .venv/                 # generated, ignored
├── README.md
├── pyproject.toml         # declared dependencies and tool configuration
├── uv.lock                # exact reproducible resolution
├── src/project_name/      # reusable code
├── notebooks/             # exploration, kept thin
├── tests/
├── configs/               # experiment configuration
└── outputs/                # generated, ignored or externally tracked
```

Prefer Python 3.12 for new general-purpose AI work today. Use 3.11 when older ML
packages require it. Pin the minor version (`3.12`), not normally a patch version,
unless exact interpreter reproduction is essential.

### Jupyter kernels

Install `ipykernel` in each project's environment. Launch Jupyter via
`uv run jupyter lab`; this makes the environment explicit. If an external
notebook UI needs a named kernel:

```sh
uv run python -m ipykernel install --user --name project-name --display-name "Python (project-name)"
jupyter kernelspec list
jupyter kernelspec uninstall project-name
```

Avoid accumulating stale kernels: remove the kernelspec when archiving a project.

## Zed workflow

- Open one executable project as the worktree whenever possible. This improves
  indexing, AI context, environment detection, and search relevance.
- Run `uv sync`, then use Zed's toolchain selector to verify `.venv` is selected.
  New integrated terminals should activate the selected environment.
- Zed supplies basedpyright, Ruff, and debugpy integration; they need not be
  installed globally. Keeping them in project dev dependencies is still useful
  for terminal and CI parity.
- Use `F4` to debug Python files, modules, and pytest tests. Add
  `.zed/debug.json` only when a project needs stable custom launch arguments.
- Keep global Zed settings personal; keep repository-specific formatting and
  language behavior in `.zed/settings.json`.
- Change Git's editor to Zed only after installing Zed's CLI from the command
  palette, then run `git config --global core.editor "zed --wait"`.

The global Zed option `trust_all_worktrees: true` trades away a security boundary.
It is convenient for personal code, but disabling it is safer when opening cloned
or third-party repositories because Zed can then ask before enabling project
features.

## Experiment discipline

Every serious run should make these recoverable:

- Git commit and dirty/clean state.
- Full resolved configuration and random seeds.
- Dataset name, version, preprocessing, and split hashes.
- Python/package lockfile and hardware/backend.
- Metrics plus raw predictions needed to recompute them.
- Checkpoint lineage and a short conclusion, including failed runs.

Use notebooks for exploration and communication, not as the only implementation.
Move stable functions into `src/`, test them, and make the notebook call them.
For configuration, start with plain TOML/YAML plus dataclasses; adopt Hydra only
when composition or large sweeps justify its complexity.

Weights & Biases is already referenced by ARENA and is reasonable for remote run
tracking. For local/offline work, MLflow or simple structured JSON/CSV plus a run
directory is enough. Do not commit API keys: keep a `.env.example` containing
names only and store actual secrets in environment variables, the macOS Keychain,
or the service CLI's credential store.

Large datasets and model weights should not live in ordinary Git. Use an external
data directory or object storage and document acquisition commands/checksums.
Use Git LFS or DVC only when the collaboration/versioning need warrants it.

## Apple Silicon notes

PyTorch on this machine should use MPS when supported:

```python
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

Do not request CUDA-specific wheels or call `.cuda()` unconditionally. Some
operations still fall back to CPU or behave differently, so record the backend
and test numerical behavior. For CUDA-only training, use a Linux NVIDIA cloud
machine or institutional cluster and keep the same locked project manifest.

## Recommended cleanup, requiring confirmation

These changes affect files or global configuration and were intentionally not
performed automatically:

1. Commit the intentional `Gen_modelling_project/` deletion when ready.
2. Remove `.DS_Store` files from the working tree and untrack any that Git knows.
3. Remove the python.org and MacPorts PATH injections from `.zprofile` after
   confirming no other project depends on them. Keep Homebrew plus `~/.local/bin`.
4. Install Zed's CLI and switch Git's editor from VS Code.
5. Rebuild ARENA in its own `.venv` with a macOS-compatible dependency set. Its
   broad, unpinned requirements should ideally be locked after a successful solve.
6. Reinstall Node natively through one manager only (Homebrew or `fnm`) if Node is
   needed; do not mix `/usr/local` and `/opt/homebrew` installations.

## What is worth adding later

- `pre-commit` running Ruff and basic file hygiene per active project.
- GitHub Actions for lint, type checks, and unit tests.
- `pytest`, basedpyright, and a moderate Ruff ruleset in every maintained project.
- Docker only when Linux parity, services, or deployment requires it; it is not a
  replacement for local Python environments.
- `direnv` only when projects need repeatable environment variables; do not add it
  merely to activate `.venv`, because Zed and `uv run` already handle that.
