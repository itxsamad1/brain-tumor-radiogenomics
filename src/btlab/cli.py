"""Command-line entry point."""

from __future__ import annotations

import argparse

from btlab import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="btlab", description="Glioma radiogenomics toolkit")
    parser.add_argument("--version", action="version", version=f"btlab {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("info", help="Print environment / version info")

    args = parser.parse_args(argv)
    if args.cmd == "info" or args.cmd is None:
        print(f"btlab {__version__}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
