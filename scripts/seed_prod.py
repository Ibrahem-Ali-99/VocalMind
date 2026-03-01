import os
import sys
import time
from pathlib import Path
import psycopg
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv(Path(__file__).parent.parent / ".env")

PROD_URL = os.getenv("PROD_URL")
PROD_KEY = os.getenv("PROD_KEY")
PROD_HOST = os.getenv("PROD_HOST")
PROD_REF = os.getenv("PROD_REF")
PROD_PW = os.getenv("PROD_PW")

def seed_prod():
    print(f"Seeding Production database: {PROD_URL}")
    
    # Wipe data first via psycopg for speed and reliability
    print("Wiping existing data for clean seed...")
    conn_str = f"host={PROD_HOST} port=6543 user=postgres.{PROD_REF} password={PROD_PW} dbname=postgres sslmode=require"
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
                tables = [row[0] for row in cur.fetchall()]
                if tables:
                    tables_str = ", ".join([f'"{t}"' for t in tables])
                    cur.execute(f"TRUNCATE TABLE {tables_str} CASCADE")
                    conn.commit()
                    print("  Wipe successful.")
    except Exception as e:
        print(f"  Wipe failed (skipping): {e}")

    # Set environment variables
    os.environ["SUPABASE_URL"] = PROD_URL
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = PROD_KEY
    
    print("Waiting 45 seconds for Supabase schema cache to refresh (Production is slow)...")
    time.sleep(45)
    
    sys.path.append(str(Path(__file__).parent))
    import seed_database
    seed_database.seed()

if __name__ == "__main__":
    seed_prod()
