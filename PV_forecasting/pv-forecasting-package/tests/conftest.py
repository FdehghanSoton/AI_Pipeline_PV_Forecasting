from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT_PATHS = (
    ROOT,
    ROOT / "analysis",
    ROOT / "analysis" / "Previous versions",
    ROOT / "baselines",
    ROOT / "config",
    ROOT / "tests",
)
for path in reversed(IMPORT_PATHS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
