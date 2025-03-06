import sys

import matplotlib.pyplot as plt


def safe_savefig_pdf(target):
    try:
        plt.savefig(f"{target}.pdf")
    except SyntaxError as e:
        # Sometimes, and only on the a compute cluster, I get "SyntaxError: not a PNG file"
        message = f"Got SyntaxError, and could not save figure to {target} as pdf. Defaulting to png format"
        print(message)
        print(message, file=sys.stderr)
        try:
            plt.savefig(f"{target}.png")
        except Exception as e:
            message = f"Got error {e} when trying to save as png as well. Not saving this summary."
            print(message)
            print(message, file=sys.stderr)
