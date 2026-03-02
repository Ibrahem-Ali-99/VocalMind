"""
VocalMind Schema Migration & Setup
==================================
Reads docker/init/01_schema.sql and applies it to the database.
Uses DATABASE_URL from .env.

Usage:
    python scripts/migrate.py
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg not found. Please install it.")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
ENV_FILE = ROOT / ".env"
SCHEMA_FILE = ROOT / "docker" / "init" / "01_schema.sql"

def load_config() -> str:
    """Load DATABASE_URL from .env."""
    if not ENV_FILE.exists():
        print(f"ERROR: .env not found at {ENV_FILE}")
        sys.exit(1)

    load_dotenv(ENV_FILE)
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        print("ERROR: DATABASE_URL is not set in .env")
        sys.exit(1)

    # asyncpg expects postgresql://
    if "+asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "")

    return db_url

async def main():
    if not SCHEMA_FILE.exists():
        print(f"ERROR: Schema file not found at {SCHEMA_FILE}")
        sys.exit(1)

    db_url = load_config()
    
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        sql = f.read()

    print("Connecting to Supabase...")
    conn = None
    try:
        conn = await asyncpg.connect(db_url)
        print("Connected.")
        
        print(f"Applying schema from {SCHEMA_FILE.name}...")
        await conn.execute(sql)
        print("Notifying PostgREST to reload schema...")
        await conn.execute("NOTIFY pgrst, 'reload schema';")
        print("✅  Schema applied successfully!")

    except Exception as e:
        print(f"❌  Error: {e}")
        sys.exit(1)
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
