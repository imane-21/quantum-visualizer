"""Entry point for the Qtum Calculator app.

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
