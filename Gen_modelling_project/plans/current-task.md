# Current Task: Notebook Cleanup — Remove Obsolete Sections

**Status:** ACTIVE
**Prerequisite reading:** `@ANTIGRAVITY_AGENT.md`, `@CLAUDE.md`
**Priority:** Do this before any further experiments

---

## Context

The notebook has accumulated obsolete cells from the coursework development process. Several sections are superseded by later fixes, and some report-oriented content is no longer needed (we're in independent research mode, not writing a report). We need a clean, lean pipeline before proceeding with upstream diagnostics.

The notebook currently has **129 cells** (indices 0–128).

---

## What to Remove

### Group 1: Original Ablation Study (broken labels) — Cells 81–91

These cells train the GZSL classifier and run the ablation study with **uncorrected labels** (unseen labels 1-200 collide with seen labels 1-200). This entire section is superseded by the corrected ablation in cells 109–117 (post-cleanup numbering).

| Cell | Content | Why remove |
|------|---------|------------|
| 81 | Method D: Full GZSL (broken labels) | Superseded by corrected Method D in cell 114 |
| 82 | Ablation summary table | Superseded by corrected table in cell 115 |
| 83 | 2×2 bias table (broken labels) | Superseded by corrected bias table in cells 103-105 |
| 84 | Bias table continued | Part of broken bias table |
| 85 | Bias table continued | Part of broken bias table |
| 86 | (markdown) Comparison heading | Report-oriented |
| 87 | Comparison bar chart | Report-oriented, broken labels |
| 88 | Save comparison figure | Report-oriented |
| 89 | Print comparison analysis | Report text, broken labels |
| 90 | Detailed results discussion | Report text |
| 91 | Heatmap visualization | Broken labels |

**Total: 11 cells removed**

### Group 2: Debug Intermediate Steps — Cells 92–97

These cells are the debugging process that *discovered* the label collision. The actual fix is properly implemented in cells 98–108 (the "FIX: Label Collision Resolution" section). These debug cells include the assertion that fails ("200 overlapping labels") and served their diagnostic purpose — they're no longer needed.

| Cell | Content | Why remove |
|------|---------|------------|
| 92 | (markdown) "Debug + Fixed 2×2 Bias Table" header | Debug section header |
| 93 | DEBUG: Identify label set mismatch | Debug step — the mismatch is now understood |
| 94 | Fixed bias table: use classifier's actual labels | Intermediate fix attempt |
| 95 | Compute boolean masks with actual labels | Intermediate debug |
| 96 | Build 2×2 bias table | Intermediate debug |
| 97 | Heatmap visualization | Intermediate debug |

**Total: 6 cells removed**

### Group 3: Report-Oriented Content in Corrected Sections — Cells 106–108, 117

These cells exist within the corrected sections but are report-specific (ablation bar charts, component contribution analysis text). Since we're in research mode and will be rebuilding the pipeline upstream, these become noise.

| Cell | Content | Why remove |
|------|---------|------------|
| 106 | Corrected ablation table (duplicate of 115) | Redundant — ablation is done properly in cells 115-116 of the corrected complete ablation |
| 107 | Corrected ablation bar chart | Report figure, will be rebuilt |
| 108 | Final summary print | Report text |
| 117 | Component contribution analysis | Report narrative, based on stale metrics |

**Total: 4 cells removed**

---

## What to KEEP

After cleanup, the notebook should have **108 cells** (129 - 21 = 108) with this clean structure:

| New Cells | Old Cells | Section | Notes |
|-----------|-----------|---------|-------|
| 0–8 | 0–8 | Data loading | Sacred, never modify |
| 9–15 | 9–15 | Data exploration | Useful diagnostics |
| 16–22 | 16–22 | Baseline Model A (LogReg on raw EEG) | Core reference |
| 23–28 | 23–28 | GZSL-style baseline evaluation | Motivation for pipeline |
| 29–49 | 29–49 | CLIP encoder (config → training → t-SNE → prototypes) | Core pipeline |
| 50–70 | 50–70 | cWGAN-GP (config → training → synthesis → diagnostics) | Core pipeline |
| 71–80 | 71–80 | Quantitative diagnostics | Useful for upcoming bottleneck analysis |
| 81–90 | 98–105 | Label collision fix (offset + retrain + corrected bias table) | Proper fix |
| 91–100 | 109–116, 118 | Complete corrected ablation (A-D) + Phase 0 harness header | Research infrastructure |
| 101–107 | 119–128 | Phase 0 eval harness + Phase 1 sample balancing | Current experiments |

*(The exact new cell numbers will shift based on removal — this is approximate. What matters is the sections are in the right order.)*

---

## How to Do This

### Method: Create a cleanup helper script

Create `helper_files/cleanup_obsolete_cells.py` that:

1. Reads the notebook JSON
2. Removes cells at indices **81–91, 92–97, 106–108, 117** (21 cells total)
3. Writes the cleaned notebook back

**Critical**: Remove cells by **descending index** to avoid index shifting problems. Delete from highest index first.

```python
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
```

### Verification

After running the script, verify the notebook structure by checking:

1. **Cell count**: Should be 108 (129 - 21)
2. **First cell after quantitative diagnostics (old cell 80)**: Should be the "FIX: Label Collision Resolution" markdown header (old cell 98)
3. **No broken references**: The label fix section (old 98-105) should flow directly after diagnostics (old 71-80) with no intermediate broken cells
4. **Phase 0+1 cells**: Should still be at the end of the notebook, intact
5. **Sanity check the new cell 81**: Print its first line — it should be the label fix markdown

### After Running

Print a summary of the cleaned notebook structure:

```python
# Quick structure check
import json
with open('COMP2261_ArizMLCW_with_baseline.ipynb') as f:
    nb = json.load(f)
print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = cell['source']
    first = src[0].strip()[:70] if src else '(empty)'
    print(f"  [{i}] {cell['cell_type']}: {first}")
```

---

## What NOT to Touch

- Cells 0–80 (data loading through quantitative diagnostics) — these are the core pipeline
- Cells 98–105 (label collision fix) — this is the proper fix, keep all of it
- Cells 109–116 (corrected ablation Methods A-D) — the clean ablation study
- Cells 118–128 (Phase 0 eval harness + Phase 1 sample balancing) — current experiments
- Any `.npy` cached files — leave them as-is
- The `figures/` directory — stale figures will be overwritten naturally

---

## What to Report When Done

1. The final cell count
2. A listing of the first line of every cell in the cleaned notebook (so we can verify structure)
3. Any issues (e.g., if a cell you tried to remove didn't match expectations)
4. Confirm the backup was created at `COMP2261_ArizMLCW_with_baseline.backup.ipynb`
