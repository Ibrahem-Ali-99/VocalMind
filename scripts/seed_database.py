"""
VocalMind Database Seeder v5.2
Populates or cleans up sample data in the Supabase database.
Matches the schema with 16 domain tables + assistant_queries.

Usage:
    uv run python scripts/seed_database.py           # Seed data
    uv run python scripts/seed_database.py --cleanup  # Remove seed data
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===========================================================
# Deterministic UUIDs
# ===========================================================

ORG_IDS = [f"a0000000-0000-0000-0000-00000000000{i}" for i in range(1, 4)]
USER_IDS = [f"b0000000-0000-0000-0000-00000000000{i}" for i in range(1, 7)]
INTERACTION_IDS = [f"d0000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 6)]
TRANSCRIPT_IDS = [f"e0000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 6)]
UTTERANCE_IDS = [f"12000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 10)]
POLICY_IDS = [f"20000000-0000-0000-0000-00000000000{i}" for i in range(1, 4)]
FAQ_IDS = [f"f0000000-0000-0000-0000-00000000000{i}" for i in range(1, 4)]
EMOTION_EVENT_IDS = [f"11000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 3)]
SNAPSHOT_IDS = [f"22000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 2)]
QUERY_IDS = [f"33000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 2)]

# ===========================================================
# Sample Data
# ===========================================================

ORGANIZATIONS = [
    {"id": ORG_IDS[0], "name": "NileTech", "slug": "nile-tech", "status": "active"},
    {"id": ORG_IDS[1], "name": "CairoConnect", "slug": "cairo-connect", "status": "active"},
]

USERS = [
    {
        "id": USER_IDS[0],
        "organization_id": ORG_IDS[0],
        "email": "manager@niletech.com",
        "password_hash": "$2b$12$seedhashplaceholder",
        "name": "Galal Manager",
        "role": "manager",
        "is_active": True,
    },
    {
        "id": USER_IDS[1],
        "organization_id": ORG_IDS[0],
        "email": "agent@niletech.com",
        "password_hash": "$2b$12$seedhashplaceholder",
        "name": "Mohsen Agent",
        "role": "agent",
        "agent_type": "human",
        "is_active": True,
    },
    {
        "id": USER_IDS[2],
        "organization_id": ORG_IDS[1],
        "email": "manager@cairoconnect.com",
        "password_hash": "$2b$12$seedhashplaceholder",
        "name": "Ibrahem Manager",
        "role": "manager",
        "is_active": True,
    },
]

INTERACTIONS = [
    {
        "id": INTERACTION_IDS[0],
        "organization_id": ORG_IDS[0],
        "agent_id": USER_IDS[1],
        "uploaded_by": USER_IDS[0],
        "audio_file_path": "/audio/call_001.wav",
        "file_size_bytes": 2457600,
        "duration_seconds": 180,
        "file_format": "wav",
        "interaction_date": "2026-02-01T09:15:00Z",
        "processing_status": "completed",
    }
]

PROCESSING_JOBS = [
    {"interaction_id": INTERACTION_IDS[0], "stage": "diarization", "status": "completed"},
    {"interaction_id": INTERACTION_IDS[0], "stage": "stt", "status": "completed"},
    {"interaction_id": INTERACTION_IDS[0], "stage": "emotion", "status": "completed"},
    {"interaction_id": INTERACTION_IDS[0], "stage": "reasoning", "status": "completed"},
    {"interaction_id": INTERACTION_IDS[0], "stage": "scoring", "status": "completed"},
    {"interaction_id": INTERACTION_IDS[0], "stage": "rag_eval", "status": "completed"},
]

TRANSCRIPTS = [
    {"id": TRANSCRIPT_IDS[0], "interaction_id": INTERACTION_IDS[0], "full_text": "Agent: Hello. Customer: Hi, I need help.", "overall_confidence": 0.95},
]

UTTERANCES = [
    {
        "id": UTTERANCE_IDS[0],
        "interaction_id": INTERACTION_IDS[0],
        "transcript_id": TRANSCRIPT_IDS[0],
        "speaker_role": "agent",
        "user_id": USER_IDS[1],
        "sequence_index": 0,
        "start_time_seconds": 0.0,
        "end_time_seconds": 2.0,
        "text": "Hello.",
        "emotion": "neutral",
        "emotion_confidence": 0.9
    },
    {
        "id": UTTERANCE_IDS[1],
        "interaction_id": INTERACTION_IDS[0],
        "transcript_id": TRANSCRIPT_IDS[0],
        "speaker_role": "customer",
        "user_id": None,
        "sequence_index": 1,
        "start_time_seconds": 2.5,
        "end_time_seconds": 4.5,
        "text": "Hi, I need help.",
        "emotion": "frustrated",
        "emotion_confidence": 0.8
    }
]

EMOTION_EVENTS = [
    {
        "id": EMOTION_EVENT_IDS[0],
        "interaction_id": INTERACTION_IDS[0],
        "utterance_id": UTTERANCE_IDS[1],
        "previous_emotion": "neutral",
        "new_emotion": "frustrated",
        "emotion_delta": 0.5,
        "speaker_role": "customer",
        "llm_justification": "Customer expressed dissatisfaction.",
        "jump_to_seconds": 2.5,
        "is_flagged": False
    }
]

COMPANY_POLICIES = [
    {"id": POLICY_IDS[0], "policy_title": "Greeting Policy", "policy_category": "Communication", "policy_text": "Agents must greet customers warmly.", "is_active": True},
]

ORGANIZATION_POLICIES = [
    {"organization_id": ORG_IDS[0], "policy_id": POLICY_IDS[0], "is_active": True},
]

FAQ_ARTICLES = [
    {"id": FAQ_IDS[0], "question": "How to reset password?", "answer": "Go to settings.", "category": "Account", "is_active": True},
]

ORGANIZATION_FAQ_ARTICLES = [
    {"organization_id": ORG_IDS[0], "article_id": FAQ_IDS[0], "is_active": True},
]

INTERACTION_SCORES = [
    {
        "interaction_id": INTERACTION_IDS[0],
        "overall_score": 8.5,
        "empathy_score": 9.0,
        "policy_score": 10.0,
        "resolution_score": 8.0,
        "was_resolved": True,
        "total_silence_seconds": 5.0,
        "avg_response_time_seconds": 2.5
    }
]

AGENT_PERFORMANCE_SNAPSHOTS = [
    {
        "id": SNAPSHOT_IDS[0],
        "organization_id": ORG_IDS[0],
        "agent_id": USER_IDS[1],
        "period_type": "daily",
        "period_start": "2026-02-01",
        "period_end": "2026-02-01",
        "total_interactions": 1,
        "avg_overall_score": 8.5,
        "avg_empathy_score": 9.0,
        "avg_policy_score": 10.0,
        "avg_resolution_score": 8.0,
        "resolution_rate": 1.0
    }
]

ASSISTANT_QUERIES = [
    {
        "id": QUERY_IDS[0],
        "user_id": USER_IDS[0],
        "organization_id": ORG_IDS[0],
        "query_mode": "chat",
        "query_text": "How many calls today?",
        "response_text": "There was 1 call handled today."
    }
]

# Order matters for FK constraints
TABLES_IN_ORDER = [
    ("organizations", ORGANIZATIONS),
    ("users", USERS),
    ("company_policies", COMPANY_POLICIES),
    ("faq_articles", FAQ_ARTICLES),
    ("interactions", INTERACTIONS),
    ("processing_jobs", PROCESSING_JOBS),
    ("transcripts", TRANSCRIPTS),
    ("utterances", UTTERANCES),
    ("interaction_scores", INTERACTION_SCORES),
    ("policy_compliance", []),  # Optional placeholder
    ("emotion_events", EMOTION_EVENTS),
    ("emotion_feedback", []),   # Optional placeholder
    ("compliance_feedback", []),# Optional placeholder
    ("organization_policies", ORGANIZATION_POLICIES),
    ("organization_faq_articles", ORGANIZATION_FAQ_ARTICLES),
    ("agent_performance_snapshots", AGENT_PERFORMANCE_SNAPSHOTS),
    ("assistant_queries", ASSISTANT_QUERIES),
]

def seed():
    print("Seeding VocalMind database v5.2 ...")
    for table_name, rows in TABLES_IN_ORDER:
        try:
            result = supabase.table(table_name).insert(rows).execute()
            print(f"  [OK] {table_name}: {len(result.data)} rows")
        except Exception as e:
            print(f"  [ERROR] {table_name}: {e}")
            raise e
    print("\nDeep seeding complete.")

def cleanup():
    print("Cleaning up seed data...")
    try:
        supabase.table("organizations").delete().in_("id", ORG_IDS).execute()
        print("  ✓ Deleted seed data (cascaded from organizations)")
    except Exception as e:
        print(f"  [ERROR] Cleanup failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="VocalMind DB Seeder")
    parser.add_argument("--cleanup", action="store_true", help="Remove all seeded data")
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
    else:
        seed()

if __name__ == "__main__":
    main()
