"""Prepare the WhisperX speaker-role classifier from exported zip artifacts.

Usage:
    python infra/scripts/prepare_speaker_role_model.py
    python infra/scripts/prepare_speaker_role_model.py --zip-path "D:/path/to/speaker_classifier_export.zip"
    python infra/scripts/prepare_speaker_role_model.py --delete-zip
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP_PATH = REPO_ROOT / "speaker_classifier_export.zip"
TARGET_DIR = REPO_ROOT / "services" / "whisperx" / "models" / "speaker_role" / "distilbert"


def prepare(zip_path: Path, delete_zip: bool) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        distilbert_members = [m for m in members if m.startswith("distilbert/") and not m.endswith("/")]
        if not distilbert_members:
            raise RuntimeError("No distilbert model files found in archive.")

        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

        for member in distilbert_members:
            relative_path = Path(member).relative_to("distilbert")
            destination = TARGET_DIR / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as out:
                out.write(source.read())

    if delete_zip:
        zip_path.unlink(missing_ok=True)

    print(f"[OK] Prepared speaker-role model at: {TARGET_DIR}")
    if delete_zip:
        print(f"[OK] Removed archive: {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract speaker-role model for WhisperX.")
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH, help="Path to speaker model zip archive.")
    parser.add_argument("--delete-zip", action="store_true", help="Delete the zip archive after extraction.")
    args = parser.parse_args()

    prepare(args.zip_path.resolve(), args.delete_zip)


if __name__ == "__main__":
    main()
