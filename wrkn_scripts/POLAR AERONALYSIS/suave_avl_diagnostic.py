# suave_avl_diagnostic.py
import os
import sys
import shutil
import subprocess
import inspect
import traceback

print("=== ENVIRONMENT CHECK ===")
print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n"," "))
print("PATH contains C:\\Tools\\AVL? ->", any(p.lower().strip() == r"c:\tools\avl" or p.lower().strip() == r"c:\tools\avl\\" or p.lower().strip() == r"c:\tools\\avl" for p in os.environ.get("PATH","").split(os.pathsep)))
print("Full PATH (short):")
for p in os.environ.get("PATH","").split(os.pathsep)[:20]:
    print("  ", p)

print("\n=== which/where checks ===")
which_avl = shutil.which("avl")
print("shutil.which('avl') ->", which_avl)
# try 'where' on Windows
try:
    out = subprocess.run(["where", "avl"], capture_output=True, text=True)
    print("where avl ->", out.stdout.strip() if out.returncode==0 else "(not found)")
except Exception as e:
    print("where avl error:", e)

print("\n=== Try invoking avl.exe with --help (if found) ===")
if which_avl:
    try:
        p = subprocess.run([which_avl, "-h"], capture_output=True, text=True, timeout=5)
        print("avl -h returncode:", p.returncode)
        print("avl -h stdout (first 300 chars):", p.stdout[:300])
        print("avl -h stderr (first 300 chars):", p.stderr[:300])
    except Exception as e:
        print("Error invoking avl:", e)
else:
    print("avl not found via shutil.which; cannot invoke.")

print("\n=== SUAVE INSTALLATION CHECK ===")
try:
    import SUAVE
    base = os.path.dirname(inspect.getfile(SUAVE))
    print("SUAVE package directory:", base)
    candidate_avl_dir = os.path.join(base, "Methods", "Aerodynamics", "AVL")
    print("Expected SUAVE AVL folder:", candidate_avl_dir)
    print("AVL folder exists?:", os.path.isdir(candidate_avl_dir))
    settings_file = os.path.join(candidate_avl_dir, "Data", "Settings.py")
    print("Expected Settings.py:", settings_file)
    print("Settings.py exists?:", os.path.isfile(settings_file))
    if os.path.isfile(settings_file):
        print("\n--- Content excerpt of Settings.py (filenames.avl_bin_name lines) ---")
        with open(settings_file, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        for line in txt.splitlines():
            if "avl_bin_name" in line or "AVL" in line and "bin" in line:
                print(line)
    # inspect any environment override SUAVE might check
    if "AVL_PATH" in os.environ:
        print("Environment AVL_PATH variable:", os.environ["AVL_PATH"])
    else:
        print("No environment variable AVL_PATH set.")
except Exception as e:
    print("Could not import SUAVE or inspect paths.")
    traceback.print_exc()

print("\n=== SUAVE AVL run smoke test (light, catches errors) ===")
try:
    # Attempt to run the SUAVE AVL analysis used earlier (non-invasive)
    from SUAVE.Analyses.Aerodynamics import AVL as SUAVE_AVL_Module
    print("Imported SUAVE AVL module:", SUAVE_AVL_Module)
    # create minimal analysis object and call its check/run method if available
    try:
        analysis = SUAVE_AVL_Module.AVL()
    except Exception:
        # some SUAVE versions expose class differently:
        try:
            analysis = SUAVE_AVL_Module()
        except Exception as e:
            print("Could not instantiate AVL analysis class:", e)
            raise

    # print settings object if present
    if hasattr(analysis, "settings"):
        print("analysis.settings exists. filenames.avl_bin_name:", getattr(getattr(analysis, "settings"), "filenames", None))
    # try calling to run (this may throw; catch it)
    print("Calling analysis.__call__() (will catch errors)...")
    try:
        res = analysis.__call__()  # many SUAVE builds require vehicle/geometry; this may fail but will show where
        print("analysis.__call__ returned:", type(res), getattr(res, "keys", lambda: "(no keys)")())
    except Exception as e:
        print("analysis.__call__ error (expected on minimal test):")
        traceback.print_exc()
except Exception as e:
    print("SUAVE AVL smoke test aborted; details below.")
    traceback.print_exc()

print("\n=== DIAGNOSTIC COMPLETE ===")
