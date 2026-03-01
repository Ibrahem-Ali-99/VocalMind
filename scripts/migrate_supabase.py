import asyncio
import os
from pathlib import Path
import asyncpg
from dotenv import load_dotenv

# Load from root .env
load_dotenv(Path(__file__).parent.parent / ".env")

async def migrate():
    # Use standard postgres driver URL for asyncpg
    # We need to replace postgresql+asyncpg with postgresql
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not found in .env")
        return
    
    # asyncpg expects postgresql:// but the user might use postgresql+asyncpg://
    if "asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "")

    schema_path = Path(__file__).parent.parent / "docker" / "init" / "01_schema.sql"
    if not schema_path.exists():
        print(f"Error: Schema file not found at {schema_path}")
        return

    print(f"Reading schema from {schema_path}...")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    print(f"Connecting to Supabase...")
    try:
        conn = await asyncpg.connect(db_url)
        print("Connection established.")
        
        print("Executing schema...")
        # asyncpg.execute can run multiple statements separated by semicolons
        await conn.execute(schema_sql)
        print("Schema applied successfully!")
        
        await conn.close()
    except Exception as e:
        print(f"Error applying schema: {e}")

if __name__ == "__main__":
    asyncio.run(migrate())
