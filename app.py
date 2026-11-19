"""
AquaGuard AI — Root Launcher
============================
Delegates execution to personC_app/app.py so you can run:
    streamlit run app.py
from the project root directory.
"""

import sys, pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "personC_app"

# Ensure personC_app modules (vision_engine, fusion_engine, etc.) can be imported
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

target_file = APP_DIR / "app.py"
with open(target_file, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(target_file), "exec")

exec(code, {"__file__": str(target_file), "__name__": "__main__"})
