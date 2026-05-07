import json
import shutil
import re
import os
from datetime import datetime

NOTEBOOK_PATH = 'COMP2261_ArizMLCW_with_baseline.ipynb'
TASK_PATH = 'plans/current-task.md'

def make_source(text):
    """Convert multi-line string into notebook source format."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result

def make_code_cell(source_text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': make_source(source_text)
    }

def make_markdown_cell(source_text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': make_source(source_text)
    }

def main():
    # 1. Read the task spec
    print(f"Reading task spec from {TASK_PATH}...")
    with open(TASK_PATH, 'r') as f:
        md_content = f.read()

    # Find all cells
    # The pattern matches ```markdown ... ``` or ```python ... ```
    pattern = re.compile(r'```(markdown|python)\n(.*?)```', re.DOTALL)
    matches = pattern.findall(md_content)

    new_cells = []
    for block_type, code in matches:
        clean_code = code.strip('\n')
        if block_type == 'markdown':
            new_cells.append(make_markdown_cell(clean_code))
        elif block_type == 'python':
            new_cells.append(make_code_cell(clean_code))

    print(f"Found {len(new_cells)} cell blocks in {TASK_PATH}")
    assert len(new_cells) >= 11, f"Expected at least 11 cells, found {len(new_cells)}"
    
    # We only want the first 11 blocks corresponding to Cell 114 to 124
    new_cells = new_cells[:11]
    
    assert len(new_cells) == 11, "Must have exactly 11 cells to append."

    # 2. Open Notebook
    print(f"Reading notebook from {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Notebook loaded: {len(cells)} cells")
    assert len(cells) == 114, f"Expected exactly 114 cells in the notebook, got {len(cells)}"

    # 3. Append cells
    cells.extend(new_cells)

    # 4. Backup and save
    backup_path = NOTEBOOK_PATH + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(NOTEBOOK_PATH, backup_path)
    print(f"Backup created: {backup_path}")

    nb['cells'] = cells
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    print("="*60)
    print(f"SUCCESS: Appended 11 Phase F cells.")
    print(f"Notebook cell count: 114 -> {len(cells)}")
    print("="*60)
    
    for i, cell in enumerate(new_cells):
        cell_type = cell['cell_type']
        first_line = cell['source'][0].strip() if cell['source'] else ''
        print(f"  Cell {114+i} ({cell_type}): {first_line[:60]}")

if __name__ == '__main__':
    main()
