import sys

from .cli import cli

if __name__ == "__main__":
    args = []
    for idx, val in enumerate(sys.argv):
        if ".py" in val:
            args = sys.argv[slice(idx + 1, len(sys.argv))]

    cli(args)
