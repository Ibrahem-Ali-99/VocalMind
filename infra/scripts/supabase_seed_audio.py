#!/usr/bin/env python3
"""
Upload local audio into Supabase Storage and register interactions via the backend API.

Typical (NileTech benchmark set under AudioData/):
  cd backend && uv run python ../infra/scripts/supabase_seed_audio.py

All .mp3 in a folder:
  cd backend && uv run python ../infra/scripts/supabase_seed_audio.py --glob "*.mp3"

Explicit paths (anywhere on disk):
  cd backend && uv run python ../infra/scripts/supabase_seed_audio.py \\
    --file ../AudioData/foo.mp3 --file D:/calls/bar.wav

Add uploads without wiping org DB / storage prefix (append-only):
  ... supabase_seed_audio.py --no-reset --glob "*.mp3" --storage-subdir my-batch-2026-04

Requires backend/.env: SUPABASE_URL, SUPABASE_SERVICE_KEY (or SUPABASE_SERVICE_ROLE_KEY).
Docker Compose should be up for the API + pipeline.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DEFAULT_BENCHMARK_FILES = (
    "easy_no_overlap.mp3",
    "easy_overlap.mp3",
    "hard_no_overlap.mp3",
    "hard_overlap.mp3",
    "medium_no_overlap.mp3",
    "medium_overlap.mp3",
    "telecom_call.mp3",
)

_AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac"})


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = REPO / "backend" / ".env"
    if p.is_file():
        load_dotenv(p, override=False)


def _supabase_client():
    _load_dotenv()
    url = (os.getenv("SUPABASE_URL") or "").strip()
    key = (os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_SERVICE_ROLE_KEY).", file=sys.stderr)
        raise SystemExit(1)
    from supabase import create_client

    return create_client(url, key)


def _http_json(method: str, url: str, body: dict | None, headers: dict[str, str], timeout: float = 120.0) -> tuple[int, dict | list | str]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            parsed: dict | list | str = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return e.code, parsed


def _login(base: str, email: str, password: str) -> str:
    form = urllib.parse.urlencode({"username": email, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/auth/login/access-token",
        data=form,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def _wait_health(base: str) -> None:
    health = base.replace("/api/v1", "").rstrip("/") + "/health"
    for _ in range(60):
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    print("BACKEND_HEALTH_TIMEOUT", health, file=sys.stderr)
    raise SystemExit(1)


def _delete_org_session_data(sb, org_id: str) -> None:
    for table in ("assistant_queries", "agent_performance_snapshots"):
        sb.table(table).delete().eq("organization_id", org_id).execute()

    cache_ids: list[str] = []
    try:
        rows = sb.table("interactions").select("id").eq("organization_id", org_id).execute()
        cache_ids = [r["id"] for r in (rows.data or [])]
    except Exception as exc:
        print("WARN list interactions for cache cleanup:", exc, file=sys.stderr)

    for i in range(0, len(cache_ids), 100):
        batch = cache_ids[i : i + 100]
        try:
            sb.table("interaction_llm_trigger_cache").delete().in_("interaction_id", batch).execute()
        except Exception:
            pass

    sb.table("interactions").delete().eq("organization_id", org_id).execute()


def _storage_delete_prefix(sb, bucket: str, prefix: str) -> None:
    prefix = prefix.strip("/")
    try:
        listed = sb.storage.from_(bucket).list(prefix)
    except Exception as exc:
        print("WARN storage list:", prefix, exc, file=sys.stderr)
        return
    names: list[str] = []
    for item in listed or []:
        name = item.get("name")
        if name:
            names.append(f"{prefix}/{name}")
    if not names:
        return
    try:
        sb.storage.from_(bucket).remove(names)
    except Exception as exc:
        print("WARN storage remove:", exc, file=sys.stderr)


def _upload_file(sb, bucket: str, object_path: str, local_path: Path) -> None:
    body = local_path.read_bytes()
    ctype = mimetypes.guess_type(local_path.name)[0] or "audio/mpeg"
    sb.storage.from_(bucket).upload(
        object_path,
        body,
        file_options={"content-type": ctype, "upsert": "true"},
    )


def _is_audio_path(p: Path) -> bool:
    return p.suffix.lower() in _AUDIO_SUFFIXES


def _resolve_uploads(args: argparse.Namespace) -> list[tuple[Path, str]]:
    """Return (local_path, object_basename) preserving stable order."""
    seen: set[str] = set()
    out: list[tuple[Path, str]] = []

    def add(local: Path, remote_name: str) -> None:
        key = f"{local.resolve()}|{remote_name}"
        if key in seen:
            return
        seen.add(key)
        out.append((local, remote_name))

    for raw in args.file or []:
        p = Path(raw).expanduser()
        if not p.is_file():
            print("ERROR: --file not found:", p, file=sys.stderr)
            raise SystemExit(1)
        if not _is_audio_path(p):
            print("WARN: suffix may be unsupported by API:", p, file=sys.stderr)
        add(p, p.name)

    if args.glob:
        if not args.audio_dir.is_dir():
            print("ERROR: --audio-dir must exist when using --glob:", args.audio_dir, file=sys.stderr)
            raise SystemExit(1)
        hits = sorted(args.audio_dir.glob(args.glob), key=lambda x: x.as_posix().lower())
        for p in hits:
            if p.is_file() and _is_audio_path(p):
                add(p, p.name)
    elif args.names is not None:
        if not args.audio_dir.is_dir():
            print("ERROR: --audio-dir must exist when using --names:", args.audio_dir, file=sys.stderr)
            raise SystemExit(1)
        for name in args.names:
            local = args.audio_dir / name
            if not local.is_file():
                print("ERROR: missing under --audio-dir:", local, file=sys.stderr)
                raise SystemExit(1)
            add(local, name)
    elif not out:
        if not args.audio_dir.is_dir():
            print("ERROR: --audio-dir not found:", args.audio_dir, file=sys.stderr)
            raise SystemExit(1)
        for name in DEFAULT_BENCHMARK_FILES:
            local = args.audio_dir / name
            if not local.is_file():
                print("ERROR: missing default benchmark file:", local, file=sys.stderr)
                raise SystemExit(1)
            add(local, name)

    if not out:
        print("ERROR: no audio files matched (use --file, --glob, or --names).", file=sys.stderr)
        raise SystemExit(1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload local audio to Supabase Storage and POST /interactions/from-storage.",
    )
    parser.add_argument("--email", default="manager@niletech.com")
    parser.add_argument("--password", default="password")
    parser.add_argument("--api-base", default="http://localhost:8000/api/v1")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=REPO / "AudioData",
        help="Base directory for --glob and --names (default: repo AudioData/)",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help="Local audio file (repeatable). Can be outside --audio-dir.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        metavar="PATTERN",
        help="Glob under --audio-dir, e.g. *.mp3 or recordings/**/*.wav",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Basenames under --audio-dir (overrides default benchmark list when set)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not delete org DB rows or clear the storage prefix (append uploads only)",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("SUPABASE_AUDIO_BUCKET", "recordings"),
        help="Supabase Storage bucket (must exist)",
    )
    parser.add_argument(
        "--storage-subdir",
        default="managed-audio",
        help="Folder under org slug in the bucket (e.g. managed-audio).",
    )
    parser.add_argument(
        "--agent-index",
        type=int,
        default=0,
        help="Index into GET /agents list (default: 0)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.names is not None and args.glob:
        print("ERROR: use either --names or --glob, not both.", file=sys.stderr)
        return 1
    if (args.names is not None or args.glob) and args.file:
        print("NOTE: combining --file with --names/--glob; all matching paths will upload.", file=sys.stderr)

    uploads = _resolve_uploads(args)

    sb = _supabase_client()
    user_rows = sb.table("users").select("id, organization_id").eq("email", args.email).limit(1).execute()
    if not user_rows.data:
        print("ERROR: user not found:", args.email, file=sys.stderr)
        return 1
    org_id = str(user_rows.data[0]["organization_id"])

    org_rows = sb.table("organizations").select("slug").eq("id", org_id).limit(1).execute()
    slug = (org_rows.data[0].get("slug") if org_rows.data else None) or "org"
    prefix = f"{slug}/{args.storage_subdir.strip('/')}"

    print("organization_id", org_id, "slug", slug, "storage_prefix", f"{args.bucket}/{prefix}")
    print("files", len(uploads), [f"{p.name}" for p, _ in uploads])

    if args.dry_run:
        print("dry-run: no changes")
        return 0

    if not args.no_reset:
        _storage_delete_prefix(sb, args.bucket, prefix)
        _delete_org_session_data(sb, org_id)
        print("cleared org session data + storage prefix")
    else:
        print("skip reset (--no-reset)")

    for local, remote_name in uploads:
        object_path = f"{prefix}/{remote_name}"
        print("upload", local, "->", f"{args.bucket}/{object_path}")
        _upload_file(sb, args.bucket, object_path, local)

    base = args.api_base.rstrip("/")
    _wait_health(base)
    token = _login(base, args.email, args.password)
    auth = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    st, agents = _http_json("GET", f"{base}/agents", None, auth)
    if st != 200 or not isinstance(agents, list) or not agents:
        print("ERROR: GET /agents", st, agents, file=sys.stderr)
        return 1
    idx = max(0, min(args.agent_index, len(agents) - 1))
    agent_id = agents[idx]["id"]

    for _local, remote_name in uploads:
        storage_path = f"{args.bucket}/{prefix}/{remote_name}"
        payload = {"storage_path": storage_path, "agent_id": agent_id, "verify_exists": True}
        st, body = _http_json("POST", f"{base}/interactions/from-storage", payload, auth, timeout=120.0)
        if st != 200:
            print("ERROR from-storage", storage_path, st, body, file=sys.stderr)
            return 1
        print("registered", (body or {}).get("interactionId") if isinstance(body, dict) else body)

    print("SUPABASE_SEED_OK", len(uploads), "interaction(s) queued.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
