#!/usr/bin/env python3
"""Full local E2E: login → from-storage → poll → detail + optional LLM triggers. Exit 1 on failure."""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def request(
    method: str,
    url: str,
    data: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> tuple[int, str]:
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local E2E verification for one audio path.")
    parser.add_argument("--base", default="http://localhost:8000/api/v1", help="API base URL")
    parser.add_argument("--email", default="manager@niletech.com")
    parser.add_argument("--password", default="password")
    parser.add_argument(
        "--storage-path",
        default="/AudioData/easy_no_overlap.mp3",
        help="Path as stored in interaction (mounted in backend container)",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--poll-max", type=int, default=120)
    parser.add_argument("--include-llm", action="store_true", help="Request LLM/RAG trigger payload on detail")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    health_url = base.replace("/api/v1", "").rstrip("/") + "/health"
    for _ in range(90):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("BACKEND_HEALTH_TIMEOUT", health_url, file=sys.stderr)
        return 1

    form = urllib.parse.urlencode(
        {"username": args.email, "password": args.password}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/auth/login/access-token",
        data=form,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    status, text = 0, ""
    _transient = (
        urllib.error.URLError,
        ConnectionResetError,
        OSError,
        http.client.RemoteDisconnected,
        TimeoutError,
    )
    for _attempt in range(60):
        req = urllib.request.Request(
            f"{base}/auth/login/access-token",
            data=form,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                status, text = resp.status, resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            status, text = e.code, e.read().decode("utf-8")
            break
        except _transient:
            time.sleep(2)
    else:
        print("LOGIN_UNREACHABLE", file=sys.stderr)
        return 1

    if status != 200:
        print("LOGIN_FAILED", status, text, file=sys.stderr)
        return 1
    token = json.loads(text)["access_token"]
    auth = {"Authorization": f"Bearer {token}"}

    status, agents_body = request("GET", f"{base}/agents", headers=auth, timeout=60)
    if status != 200:
        print("AGENTS_FAILED", status, agents_body, file=sys.stderr)
        return 1
    agents = json.loads(agents_body)
    if not agents:
        print("NO_AGENTS", file=sys.stderr)
        return 1
    agent_id = agents[0]["id"]

    payload = {
        "storage_path": args.storage_path,
        "agent_id": agent_id,
        "verify_exists": False,
    }
    status, create_body = request(
        "POST",
        f"{base}/interactions/from-storage",
        data=payload,
        headers={**auth, "Content-Type": "application/json"},
        timeout=60,
    )
    if status != 200:
        print("CREATE_FAILED", status, create_body, file=sys.stderr)
        return 1
    interaction_id = json.loads(create_body)["interactionId"]
    print("interaction_id", interaction_id)

    final_status = None
    for i in range(args.poll_max):
        ps, pt = request(
            "GET",
            f"{base}/interactions/{interaction_id}/processing-status",
            headers=auth,
            timeout=60,
        )
        if ps != 200:
            print("POLL_HTTP", ps, pt, file=sys.stderr)
            time.sleep(args.poll_interval)
            continue
        d = json.loads(pt)
        final_status = d.get("status")
        if final_status in ("completed", "failed"):
            print("poll", i, final_status)
            break
        time.sleep(args.poll_interval)
    else:
        print("POLL_TIMEOUT", file=sys.stderr)
        return 1

    if final_status != "completed":
        print("PROCESSING_NOT_COMPLETED", final_status, file=sys.stderr)
        return 1

    detail_url = f"{base}/interactions/{interaction_id}"
    if args.include_llm:
        detail_url += "?include_llm_triggers=true"
    ds, dt = request("GET", detail_url, headers=auth, timeout=120)
    if ds != 200:
        print("DETAIL_FAILED", ds, dt, file=sys.stderr)
        return 1
    detail = json.loads(dt)

    failures = detail.get("processingFailures") or []
    if failures:
        print("PROCESSING_FAILURES", json.dumps(failures, indent=2), file=sys.stderr)
        return 1

    utt = detail.get("utterances") or []
    if len(utt) < 5:
        print("TOO_FEW_UTTERANCES", len(utt), file=sys.stderr)
        return 1

    empty = sum(1 for u in utt if not (u.get("text") or "").strip())
    if empty:
        print("EMPTY_UTTERANCES", empty, file=sys.stderr)
        return 1

    inter = detail.get("interaction") or {}
    if (inter.get("language") or "").lower() in ("unknown", ""):
        print("LANGUAGE_UNKNOWN", file=sys.stderr)
        return 1

    print("--- utterances (speaker, text) ---")
    for u in utt:
        print(u.get("timestamp"), u.get("speaker"), u.get("text"))

    print("--- emotion_events count ---", len(detail.get("emotionEvents") or []))
    print("--- policy_violations count ---", len(detail.get("policyViolations") or []))
    print(
        "--- scores ---",
        "overall",
        inter.get("overallScore"),
        "resolved",
        inter.get("resolved"),
    )

    if args.include_llm:
        llm = detail.get("llmTriggers")
        rag = detail.get("ragCompliance")
        et = detail.get("emotionTriggers")
        for name, block in (("llmTriggers", llm), ("ragCompliance", rag), ("emotionTriggers", et)):
            if isinstance(block, dict) and block.get("available") is False:
                err = block.get("error", "")
                print(f"WARNING_{name}", err, file=sys.stderr)

    print("E2E_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
