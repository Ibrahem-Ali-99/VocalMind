from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MigrationStats:
    moved_policy: int = 0
    moved_sop: int = 0
    skipped: int = 0


def _collect_stems(pdf_dir: Path) -> set[str]:
    if not pdf_dir.exists():
        return set()
    stems: set[str] = set()
    for pattern in ("*.pdf", "*.PDF"):
        for pdf in pdf_dir.glob(pattern):
            stems.add(pdf.stem)
    return stems


def migrate_org_parsed_docs(org_dir: Path, dry_run: bool = False) -> MigrationStats:
    parsed_root = org_dir / "parsed-docs"
    if not parsed_root.exists() or not parsed_root.is_dir():
        return MigrationStats()

    policy_stems = _collect_stems(org_dir / "policy-docs")
    sop_stems = _collect_stems(org_dir / "sop-procedures")

    policies_dir = parsed_root / "policies"
    sops_dir = parsed_root / "sops"

    if not dry_run:
        policies_dir.mkdir(parents=True, exist_ok=True)
        sops_dir.mkdir(parents=True, exist_ok=True)

    stats = MigrationStats()

    for item in parsed_root.iterdir():
        if item.name in {"policies", "sops"}:
            continue
        if not item.is_file() or item.suffix.lower() != ".md":
            stats.skipped += 1
            continue

        stem = item.stem
        destination = None
        if stem in sop_stems:
            destination = sops_dir / item.name
            stats.moved_sop += 1
        elif stem in policy_stems:
            destination = policies_dir / item.name
            stats.moved_policy += 1
        else:
            stats.skipped += 1
            continue

        if dry_run:
            print(f"[DRY-RUN] {item} -> {destination}")
        else:
            shutil.move(str(item), str(destination))
            print(f"[MOVED] {item} -> {destination}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy parsed markdown files into parsed-docs/policies and parsed-docs/sops.",
    )
    parser.add_argument(
        "--root",
        default="sop-standards",
        help="Repository path that contains org folders (default: sop-standards).",
    )
    parser.add_argument(
        "--org",
        default=None,
        help="Optional organization slug to migrate only one org.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned file moves without modifying files.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Invalid root directory: {root}")

    org_dirs: list[Path]
    if args.org:
        org_dir = root / args.org
        if not org_dir.exists() or not org_dir.is_dir():
            raise SystemExit(f"Organization directory not found: {org_dir}")
        org_dirs = [org_dir]
    else:
        org_dirs = [d for d in root.iterdir() if d.is_dir()]

    total = MigrationStats()
    for org_dir in org_dirs:
        stats = migrate_org_parsed_docs(org_dir, dry_run=args.dry_run)
        total.moved_policy += stats.moved_policy
        total.moved_sop += stats.moved_sop
        total.skipped += stats.skipped
        print(
            f"[{org_dir.name}] moved_policy={stats.moved_policy} moved_sop={stats.moved_sop} skipped={stats.skipped}"
        )

    print(
        f"TOTAL moved_policy={total.moved_policy} moved_sop={total.moved_sop} skipped={total.skipped}"
    )


if __name__ == "__main__":
    main()
