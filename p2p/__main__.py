"""Allow P2P to run with `python -m p2p`."""

from p2p.cli import main

if __name__ == "__main__":
    raise SystemExit(main())