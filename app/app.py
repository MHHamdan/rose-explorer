"""ROSE explorer: run locally with `python app/app.py`, or deploy app/ as a Hugging Face Space (sdk: gradio).

The same interface runs with no server on the project's GitHub Pages site (site/index.html, Gradio-Lite).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from genai_explorer.ui import build, launch_kwargs  # noqa: E402

demo = build()

if __name__ == "__main__":
    demo.launch(**launch_kwargs())
