"""Entry point: python -m semantic_grain"""

import os
from pathlib import Path

# Set Gradio temp dir to a project-local path BEFORE importing gradio.
# Avoids Windows PermissionError when antivirus/indexer locks the default
# temp directory while Gradio's HTTP server tries to serve uploaded files.
_TEMP_DIR = Path(__file__).resolve().parent.parent / ".gradio_tmp"
_TEMP_DIR.mkdir(exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_TEMP_DIR)

import gradio as gr  # noqa: E402 — must come after GRADIO_TEMP_DIR is set

from semantic_grain.device import init as init_device  # noqa: E402
from semantic_grain.app import create_ui  # noqa: E402


def main():
    init_device()
    demo = create_ui()
    demo.launch(
        inbrowser=True,
        theme=gr.themes.Base(
            primary_hue="stone",
            neutral_hue="stone",
            font=gr.themes.GoogleFont("IBM Plex Mono"),
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .gr-button { border-radius: 2px !important; }
        """,
    )


if __name__ == "__main__":
    main()
