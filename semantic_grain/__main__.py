"""Entry point: python -m semantic_grain"""

from semantic_grain.app import create_ui


def main():
    demo = create_ui()
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
