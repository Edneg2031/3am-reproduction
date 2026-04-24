from __future__ import annotations

import argparse

from . import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="3AM unofficial reproduction utilities")
    parser.add_argument("--version", action="store_true", help="print package version")
    args = parser.parse_args()
    if args.version:
        print(__version__)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
