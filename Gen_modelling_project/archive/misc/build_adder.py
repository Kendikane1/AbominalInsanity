import json
import re
import os

def main():
    with open('plans/current-task.md', 'r') as f:
        text = f.read()

    blocks = re.split(r'### Cell \d+ \(Cell \d+\)[^\n]*\n', text)[1:]
    
    cells = []
    for block in blocks:
        if '```markdown' in block:
            content = block.split('```markdown')[1].split('```')[0].lstrip('\n')
            lines = [line + '\n' for line in content.split('\n')]
            if lines and lines[-1] == '\n':
                lines = lines[:-1]
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines
            })
        elif '```python' in block:
            content = block.split('```python')[1].split('```')[0].lstrip('\n')
            lines = [line + '\n' for line in content.split('\n')]
            if lines and lines[-1] == '\n':
                lines = lines[:-1]
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            })

    if len(cells) != 12:
        print(f"Error: Found {len(cells)} cells instead of 12.")
        return

    json_str = json.dumps(cells, indent=2).replace('\\', '\\\\').replace('`', '\\`')
    
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Script to add CLIP Optimisation cells to the notebook.

Run from project root:
    python helper_files/add_clip_optimisation.py
\"\"\"

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"

NEW_CELLS_JSON = r\"\"\"{json_str}\"\"\"
NEW_CELLS = json.loads(NEW_CELLS_JSON)

def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {{NOTEBOOK_PATH}}")
        return

    print(f"Loading notebook: {{NOTEBOOK_PATH}}")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    original_count = len(notebook['cells'])
    notebook['cells'].extend(NEW_CELLS)

    print(f"Adding {{len(NEW_CELLS)}} new cells...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    print(f"\\nSuccess! Updated: {{OUTPUT_PATH}}")
    print(f"  - Original cells: {{original_count}}")
    print(f"  - New cells added: {{len(NEW_CELLS)}}")
    print(f"  - Total cells: {{len(notebook['cells'])}}")
    print(f"\\nRefresh the notebook in Jupyter and run the new cells.")

if __name__ == "__main__":
    main()
"""
    
    os.makedirs('helper_files', exist_ok=True)
    with open('helper_files/add_clip_optimisation.py', 'w') as f:
        f.write(script_content)
    
    print("Successfully generated helper_files/add_clip_optimisation.py")

if __name__ == '__main__':
    main()
