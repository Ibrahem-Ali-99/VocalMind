"""Deprecated: use ``supabase_seed_audio.py`` (same CLI). Kept as a stable path for scripts/bookmarks."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "supabase_seed_audio.py"), run_name="__main__")
