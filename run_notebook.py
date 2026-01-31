import json
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

def run_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Running notebook: {notebook_path}")
    print("-" * 40)
    
    # Create a global namespace for execution
    globs = {}
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            print(f"Executing Cell {i+1}...")
            try:
                # We exec in the global namespace
                exec(source, globs)
            except Exception as e:
                print(f"Error in Cell {i+1}: {e}")
                import traceback
                traceback.print_exc()
                # Decide if you want to stop or continue. Usually stop on error.
                break
    
    print("-" * 40)
    print("Execution complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_notebook(sys.argv[1])
    else:
        print("Usage: python run_notebook.py <notebook_path>")
