"""CLI entry point for ``python -m bayescatrack``."""

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bayescatrack.core.bridge import main

raise SystemExit(main())
