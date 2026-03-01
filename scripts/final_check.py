import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

def list_tables():
    db_url = os.getenv("DATABASE_URL")
    if "asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "")
    
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
                tables = [row[0] for row in cur.fetchall()]
                print(f"Final Table List: {sorted(tables)}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    list_tables()
