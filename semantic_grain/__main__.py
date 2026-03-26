"""Entry point: python -m semantic_grain"""

import gradio as gr

from semantic_grain.device import init as init_device
from semantic_grain.app import create_ui


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
