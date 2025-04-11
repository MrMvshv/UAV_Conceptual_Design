import os
import re
from pathlib import Path

# Manually set your SUAVE path (replace with output from step 1)
SUAVE_PATH = "/mnt/g/wrk/conceptual_design/venv/lib/python3.12/site-packages/SUAVE-2.5.2-py3.12.egg/SUAVE"

REPLACEMENTS = [
    (r'from scipy\.integrate\s+import\s+cumtrapz',
     'from scipy.integrate import cumulative_trapezoid as cumtrapz'),
    (r'from scipy\.misc\s+import\s+derivative',
     'from scipy.optimize._numdiff import approx_derivative as derivative'),
    (r'from scipy\.optimize\s+import\s+minimize\s+as\s+minimize_scalar',
     'from scipy.optimize import minimize')
]

def patch_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        modified = False
        for pattern, replacement in REPLACEMENTS:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
        
        if modified:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error patching {filepath}: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Patching SUAVE at: {SUAVE_PATH}")
    patched_files = 0
    
    for root, _, files in os.walk(SUAVE_PATH):
        for file in files:
            if file.endswith('.py'):
                if patch_file(os.path.join(root, file)):
                    patched_files += 1
    
    print(f"Successfully patched {patched_files} files")
    print("Now try importing SUAVE again")