import os
import sys
import time
from pathlib import Path
import psycopg
from dotenv import load_dotenv
from postgrest.exceptions import APIError

# Load credentials from .env
load_dotenv(Path(__file__).parent.parent / ".env")

PROD_URL = os.getenv("PROD_URL")
PROD_KEY = os.getenv("PROD_KEY")
PROD_HOST = os.getenv("PROD_HOST")
PROD_REF = os.getenv("PROD_REF")
PROD_PW = os.getenv("PROD_PW")

def seed_prod_robust():
    print(f"Seeding Production database: {PROD_URL}")
    
    # Wipe data
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
        print(f"  Wipe failed: {e}")

    # Set environment variables
    os.environ["SUPABASE_URL"] = PROD_URL
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = PROD_KEY
    
    sys.path.append(str(Path(__file__).parent))
    import seed_database
    from supabase import create_client

    print("Starting robust seeder...")
    supabase = create_client(PROD_URL, PROD_KEY)
    
    for table_name, rows in seed_database.TABLES_IN_ORDER:
        retries = 10
        while retries > 0:
            try:
                print(f"  Inserting into {table_name}...")
                result = supabase.table(table_name).insert(rows).execute()
                print(f"    [OK] {table_name}: {len(result.data)} rows")
                break
            except APIError as e:
                # Check for schema cache error (PGRST205)
                if 'PGRST205' in str(e) or 'schema cache' in str(e):
                    print(f"    Schema cache lag detected for {table_name}. Retrying in 10s... ({retries} left)")
                    time.sleep(10)
                    retries -= 1
                else:
                    print(f"    CRITICAL ERROR seeding {table_name}: {e}")
                    raise e
            except Exception as e:
                print(f"    Unexpected error seeding {table_name}: {e}")
                raise e

    print("\nProduction seeding complete!")

if __name__ == "__main__":
    seed_prod_robust()
