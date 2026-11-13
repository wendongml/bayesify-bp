"""Render the shared three-implementation website from executed notebooks."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "site/build.py"), run_name="__main__")
