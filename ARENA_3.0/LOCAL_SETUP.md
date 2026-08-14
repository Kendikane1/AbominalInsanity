# Local setup: Chapters 0 and 1

This Apple Silicon environment intentionally covers only Fundamentals and
Transformer Interpretability. It excludes Chapter 2 RL and Chapter 3 evals.

```sh
cd ARENA_3.0
uv sync
uv run python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
uv run jupyter lab
```

In Zed, open `ARENA_3.0` as its own worktree. After `uv sync`, verify that
`.venv` is selected in the Python toolchain selector.

Apple Silicon uses the `mps` PyTorch device, not CUDA. Some original notebooks
assume CUDA or Google Colab; replace unconditional `cuda` device selection with
MPS/CPU-aware selection where encountered.
