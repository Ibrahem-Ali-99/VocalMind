import os
import sys
import psycopg
from dotenv import load_dotenv
from pathlib import Path

import importlib.util
from pathlib import Path

# Mock Supabase env vars so seed_database can be loaded for its constants
os.environ.setdefault("SUPABASE_URL", "http://mock")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "mock")

# Fix path to import seed_database
seed_db_path = Path(__file__).parent / "seed_database.py"
spec = importlib.util.spec_from_file_location("seed_database", seed_db_path)
seed_database = importlib.util.module_from_spec(spec)
# Prevent the module from exiting if it can't find keys (we already mocked them but just in case)
spec.loader.exec_module(seed_database)

load_dotenv(Path(__file__).parent.parent.parent / "backend" / ".env")

def seed():
    print("Seeding database via direct PostgreSQL connection (psycopg)...")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in .env")
        return
    
    if "+asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "")
    
    try:
        with psycopg.connect(db_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                print("Wiping existing data...")
                # Tables in reverse order of creation
                for table_name, _ in reversed(seed_database.TABLES_IN_ORDER):
                    cur.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
                print("  Wipe successful.")
                
                print("Inserting mock data...")
                for table_name, rows in seed_database.TABLES_IN_ORDER:
                    print(f"  Inserting into {table_name} ({len(rows)} rows)...")
                    if not rows: continue
                    
                    columns = rows[0].keys()
                    col_names = ", ".join(columns)
                    placeholders = ", ".join(["%s"] * len(columns))
                    
                    sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"
                    
                    data = [tuple(row[c] for c in columns) for row in rows]
                    cur.executemany(sql, data)
                
                conn.commit()
                print("\n✅  Seeding complete via direct connection!")

    except Exception as e:
        print(f"\n❌  Seeding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    seed()
