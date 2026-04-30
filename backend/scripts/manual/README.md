Manual backend probes and one-off debugging helpers.

These scripts are intentionally outside `backend/tests` so `pytest` does not
collect them. They may depend on local audio files, ngrok URLs, seeded database
state, or running external services.

Run them from the `backend` directory unless a script says otherwise, for
example:

```powershell
uv run python scripts/manual/rag_smoke_probe.py
```
