# Projects

This is the home for active AI, ML, deep-learning, and Python engineering
projects. Each child directory is an independent Git repository with its own
`.venv`, dependency lockfile, documentation, tests, and outputs.

## Start a project with an AI coding agent

```sh
cd "/Users/ariz/Abominal Insanity/Projects"
mkdir my-project
cd my-project
codex
```

Claude Code or Gemini CLI can be launched instead if installed. Then use a normal
prompt describing the work, for example:

> Start a computer-vision research project for classifying plant diseases. I want
> a small baseline CNN and a notebook for exploring the dataset.

The parent instruction files tell the agent to configure the environment and
project conventions automatically.

You can also bypass the agent and scaffold directly:

```sh
ai-project my-project dl
```

## Lifecycle

- `Projects/`: active work that runs code.
- `Literature/`: papers and reading material, outside this directory.
- Archive completed projects outside `Projects/` only after documenting how to
  reproduce their final result.

The umbrella repository ignores all child directories here. Each generated child
has its own Git repository, preventing unrelated project histories and
dependencies from being mixed together.
