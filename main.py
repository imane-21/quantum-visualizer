"""Entry point for the Qtum Calculator app.

This file supports being executed both as a module (package context) and as a
standalone script. When run as a script, it ensures the local `src` directory is
on sys.path and imports the GUI accordingly to avoid "attempted relative import
with no known parent package" errors.
"""
import sys
import os

try:
    # normal package relative import (works when using `python -m src.main`)
    from .gui import run_app
except Exception:
    # fallback when running the file directly: add src dir to path and import
    sys.path.insert(0, os.path.dirname(__file__))
    from gui import run_app


if __name__ == '__main__':
    run_app()
