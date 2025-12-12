"""FastAPI entry point placeholder for VocalMind backend."""
from fastapi import FastAPI

app = FastAPI(title="VocalMind")


@app.get("/health")
def health_check() -> dict:
    """Lightweight health probe."""
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
