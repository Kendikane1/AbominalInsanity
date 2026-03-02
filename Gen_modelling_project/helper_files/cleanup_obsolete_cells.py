#!/usr/bin/env python3
"""
Remove obsolete cells from notebook (broken-label ablation, debug artifacts, report text).

Run from project root:
    python helper_files/cleanup_obsolete_cells.py
"""

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
BACKUP_PATH = "COMP2261_ArizMLCW_with_baseline.backup.ipynb"

# Cell indices to remove (0-indexed)
# Group 1: Original ablation with broken labels (cells 81-91)
# Group 2: Debug intermediate steps (cells 92-97)
# Group 3: Report-oriented duplicates (cells 106-108, 117)
CELLS_TO_REMOVE = sorted(
    list(range(81, 92)) +   # 81-91 inclusive
    list(range(92, 98)) +   # 92-97 inclusive
    [106, 107, 108, 117],   # Individual report cells
    reverse=True  # MUST remove from highest index first
)


def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return

    # Backup first
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Backup saved to: {BACKUP_PATH}")

    original_count = len(notebook['cells'])

    # Remove cells from highest index first
    removed = []
    for idx in CELLS_TO_REMOVE:
        if idx < len(notebook['cells']):
            cell = notebook['cells'][idx]
            src = cell['source']
            first_line = src[0].strip()[:60] if src else '(empty)'
            removed.append((idx, cell['cell_type'], first_line))
            del notebook['cells'][idx]

    # Write cleaned notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    print(f"\nRemoved {len(removed)} cells:")
    for idx, typ, line in sorted(removed):
        print(f"  [{idx}] {typ}: {line}")

    print(f"\nOriginal cells: {original_count}")
    print(f"Removed: {len(removed)}")
    print(f"Remaining: {len(notebook['cells'])}")
    print(f"\nRefresh the notebook in Jupyter to see changes.")


if __name__ == "__main__":
    main()
