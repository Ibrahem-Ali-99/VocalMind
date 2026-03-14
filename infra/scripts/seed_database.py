"""
VocalMind Database Seeder v5.2 — Comprehensive
Populates or cleans up sample data in the Supabase database.
Matches the schema with 16 domain tables + assistant_queries.

Usage:
    uv run python scripts/seed_database.py           # Seed data
    uv run python scripts/seed_database.py --cleanup  # Remove seed data
"""

import argparse
import os
import sys
import random
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
# Helpers
# ===========================================================
def uid(prefix, n):
    """Generate deterministic UUID: prefix + zero-padded n.
    First group is exactly 8 hex chars (prefix padded with zeros)."""
    first_group = (prefix + "000000")[:8]
    return f"{first_group}-0000-0000-0000-{str(n).zfill(12)}"

# bcrypt hash of "password"
HASH = "$2b$12$q8lyq/NpKlA80YMdzrKtPuHkg1pG4HIk1zIDPpKu78TPFy3zw6NW6"

EMOTIONS = ["neutral", "happy", "frustrated", "angry", "sad", "empathetic", "fearful"]

# ===========================================================
# 1. Organizations (keep original 2)
# ===========================================================
ORG_IDS = [uid("a0", 1), uid("a0", 2)]

ORGANIZATIONS = [
    {"id": ORG_IDS[0], "name": "NileTech", "slug": "nile-tech", "status": "active"},
    {"id": ORG_IDS[1], "name": "CairoConnect", "slug": "cairo-connect", "status": "active"},
]

# ===========================================================
# 2. Users — 2 managers + 8 agents
# ===========================================================
USER_IDS = [uid("b0", i) for i in range(1, 11)]

USERS = [
    # ── NileTech: 1 manager + 4 agents ──
    {"id": USER_IDS[0], "organization_id": ORG_IDS[0], "email": "manager@niletech.com",
     "password_hash": HASH, "name": "Galal Manager", "role": "manager", "is_active": True},
    {"id": USER_IDS[1], "organization_id": ORG_IDS[0], "email": "mohsen@niletech.com",
     "password_hash": HASH, "name": "Mohsen Agent", "role": "agent", "agent_type": "human", "is_active": True},
    {"id": USER_IDS[2], "organization_id": ORG_IDS[0], "email": "sara@niletech.com",
     "password_hash": HASH, "name": "Sara Agent", "role": "agent", "agent_type": "human", "is_active": True},
    {"id": USER_IDS[3], "organization_id": ORG_IDS[0], "email": "omar@niletech.com",
     "password_hash": HASH, "name": "Omar Agent", "role": "agent", "agent_type": "human", "is_active": True},
    {"id": USER_IDS[4], "organization_id": ORG_IDS[0], "email": "nourbot@niletech.com",
     "password_hash": HASH, "name": "Nour AI Bot", "role": "agent", "agent_type": "ai", "is_active": True},
    # ── CairoConnect: 1 manager + 4 agents ──
    {"id": USER_IDS[5], "organization_id": ORG_IDS[1], "email": "manager@cairoconnect.com",
     "password_hash": HASH, "name": "Ibrahem Manager", "role": "manager", "is_active": True},
    {"id": USER_IDS[6], "organization_id": ORG_IDS[1], "email": "yasmin@cairoconnect.com",
     "password_hash": HASH, "name": "Yasmin Agent", "role": "agent", "agent_type": "human", "is_active": True},
    {"id": USER_IDS[7], "organization_id": ORG_IDS[1], "email": "khaled@cairoconnect.com",
     "password_hash": HASH, "name": "Khaled Agent", "role": "agent", "agent_type": "human", "is_active": True},
    {"id": USER_IDS[8], "organization_id": ORG_IDS[1], "email": "dina@cairoconnect.com",
     "password_hash": HASH, "name": "Dina Agent", "role": "agent", "agent_type": "human", "is_active": True},
    {"id": USER_IDS[9], "organization_id": ORG_IDS[1], "email": "aibot@cairoconnect.com",
     "password_hash": HASH, "name": "Cairo AI Bot", "role": "agent", "agent_type": "ai", "is_active": True},
]

# Agent user IDs per org (for assigning interactions)
NILE_AGENTS = USER_IDS[1:5]   # indices 1-4
CAIRO_AGENTS = USER_IDS[6:10] # indices 6-9
NILE_MANAGER = USER_IDS[0]
CAIRO_MANAGER = USER_IDS[5]

# ===========================================================
# 3. Company Policies
# ===========================================================
POLICY_IDS = [uid("20", i) for i in range(1, 7)]

COMPANY_POLICIES = [
    {"id": POLICY_IDS[0], "policy_title": "Greeting Policy", "policy_category": "Communication",
     "policy_text": "Agents must greet customers warmly and professionally within the first 5 seconds of the call.", "is_active": True},
    {"id": POLICY_IDS[1], "policy_title": "Escalation Protocol", "policy_category": "Escalation",
     "policy_text": "If a customer expresses extreme frustration or anger, the agent must offer to escalate to a supervisor within 60 seconds.", "is_active": True},
    {"id": POLICY_IDS[2], "policy_title": "Hold Time Limit", "policy_category": "Communication",
     "policy_text": "Customers must not be placed on hold for more than 2 minutes without a status update.", "is_active": True},
    {"id": POLICY_IDS[3], "policy_title": "Data Privacy Compliance", "policy_category": "Privacy",
     "policy_text": "Agents must never ask for full credit card numbers or passwords over the phone.", "is_active": True},
    {"id": POLICY_IDS[4], "policy_title": "Refund Authorization", "policy_category": "Finance",
     "policy_text": "Agents may authorize refunds up to $50 without supervisor approval. Amounts above $50 require escalation.", "is_active": True},
    {"id": POLICY_IDS[5], "policy_title": "Closing Script", "policy_category": "Communication",
     "policy_text": "Agents must summarize the resolution and ask if there is anything else before ending the call.", "is_active": True},
]

# ===========================================================
# 4. FAQ Articles
# ===========================================================
FAQ_IDS = [uid("f0", i) for i in range(1, 9)]

FAQ_ARTICLES = [
    {"id": FAQ_IDS[0], "question": "How do I reset my password?", "answer": "Go to Settings > Security > Reset Password, or click 'Forgot password' on the login page.", "category": "Account", "is_active": True},
    {"id": FAQ_IDS[1], "question": "How do I update my billing information?", "answer": "Navigate to Settings > Billing > Payment Methods and update your card details.", "category": "Billing", "is_active": True},
    {"id": FAQ_IDS[2], "question": "What is the refund policy?", "answer": "Refunds are processed within 5-7 business days. Contact support for amounts over $50.", "category": "Billing", "is_active": True},
    {"id": FAQ_IDS[3], "question": "How do I contact technical support?", "answer": "Use the in-app chat, call +1-800-555-0199, or email support@vocalmind.com.", "category": "Technical", "is_active": True},
    {"id": FAQ_IDS[4], "question": "How to enable two-factor authentication?", "answer": "Go to Settings > Security > 2FA and follow the setup wizard with your authenticator app.", "category": "Account", "is_active": True},
    {"id": FAQ_IDS[5], "question": "What are the service hours?", "answer": "Our support team is available 24/7 for critical issues. General inquiries: Sun-Thu 9AM-6PM EET.", "category": "General", "is_active": True},
    {"id": FAQ_IDS[6], "question": "How do I cancel my subscription?", "answer": "Go to Settings > Subscription > Cancel. Note: cancellation takes effect at the end of the billing cycle.", "category": "Billing", "is_active": True},
    {"id": FAQ_IDS[7], "question": "How to export my data?", "answer": "Navigate to Settings > Data Management > Export. You can download CSV or JSON formats.", "category": "Technical", "is_active": True},
]

# ===========================================================
# 5. Interactions — 30 total (18 NileTech + 12 CairoConnect)
# ===========================================================
INTERACTION_IDS = [uid("d0", i) for i in range(1, 31)]
TRANSCRIPT_IDS = [uid("e0", i) for i in range(1, 31)]

random.seed(42)  # reproducible

# Conversation templates
CONVERSATIONS = [
    ("Agent: Good morning, how can I help you today?\nCustomer: Hi, I need help with my account.\nAgent: Sure, I can help with that. What seems to be the issue?\nCustomer: I cannot log in.\nAgent: Let me reset your password for you.\nCustomer: Thank you so much!", ["neutral","neutral","neutral","frustrated","empathetic","happy"]),
    ("Agent: Hello, welcome to support.\nCustomer: I want a refund for my last purchase.\nAgent: I understand. Can you provide the order number?\nCustomer: It is order 45892.\nAgent: I see the order. Let me process that refund for you right away.\nCustomer: Great, how long will it take?\nAgent: You should see it within 5 business days.\nCustomer: Okay, thanks.", ["neutral","frustrated","neutral","neutral","empathetic","neutral","neutral","happy"]),
    ("Agent: Thank you for calling, how may I assist you?\nCustomer: Your service has been terrible lately!\nAgent: I am really sorry to hear that. Can you tell me what happened?\nCustomer: My internet has been down for three days!\nAgent: That is unacceptable. Let me escalate this immediately.\nCustomer: Please do. This is ruining my work.", ["neutral","angry","empathetic","angry","empathetic","frustrated"]),
    ("Agent: Hi there, what can I do for you?\nCustomer: I need to update my billing address.\nAgent: Of course. What is the new address?\nCustomer: 123 Nile Street, Cairo.\nAgent: Updated! Anything else?\nCustomer: No, that is all. Thank you.", ["neutral","neutral","neutral","neutral","neutral","happy"]),
    ("Agent: Welcome to support, how can I help?\nCustomer: I am scared my account has been hacked.\nAgent: I understand your concern. Let me check the recent activity on your account.\nCustomer: Please hurry, I see charges I did not make.\nAgent: I have secured your account and reversed the unauthorized charges.\nCustomer: Oh thank goodness. Thank you so much.", ["neutral","fearful","empathetic","fearful","empathetic","happy"]),
    ("Agent: Good afternoon! How can I assist you today?\nCustomer: I have a question about my subscription plan.\nAgent: Happy to help! Which plan are you on?\nCustomer: The premium plan, but I want to downgrade.\nAgent: I can process that for you. The change will take effect next billing cycle.\nCustomer: Perfect, thanks for the quick help.", ["happy","neutral","happy","neutral","neutral","happy"]),
    ("Agent: Hello, thank you for reaching out.\nCustomer: I am having trouble with the mobile app.\nAgent: I am sorry about that. What exactly is happening?\nCustomer: It keeps crashing when I open it.\nAgent: Try clearing the cache and reinstalling. I will also report this to our tech team.\nCustomer: Okay, I will try that.", ["neutral","frustrated","empathetic","frustrated","neutral","neutral"]),
    ("Agent: Hi, how can I help you?\nCustomer: I am really upset about the service I received last time.\nAgent: I sincerely apologize. What happened?\nCustomer: The agent was rude and did not solve my problem.\nAgent: That is not acceptable. I will escalate this and personally ensure your issue is resolved today.\nCustomer: I appreciate that.", ["neutral","angry","empathetic","angry","empathetic","neutral"]),
    ("Agent: Good morning, support team here.\nCustomer: Can you help me set up two-factor authentication?\nAgent: Absolutely! Go to Settings, then Security, then enable 2FA.\nCustomer: Got it, what app should I use?\nAgent: Google Authenticator or Authy both work great.\nCustomer: Done! Thanks for walking me through it.", ["neutral","neutral","happy","neutral","neutral","happy"]),
    ("Agent: Thank you for calling. How may I help?\nCustomer: I want to cancel my subscription.\nAgent: I am sorry to hear that. May I ask why?\nCustomer: It is too expensive for what I get.\nAgent: I understand. Would you be interested in our discounted plan at 40% off?\nCustomer: Hmm, that sounds interesting. Tell me more.\nAgent: It includes all core features at a reduced price.\nCustomer: Okay, switch me to that plan instead.", ["neutral","sad","empathetic","frustrated","neutral","neutral","neutral","happy"]),
]

DURATIONS = [180, 245, 320, 150, 280, 190, 210, 360, 175, 300, 220, 260, 140, 310, 195,
             230, 270, 165, 340, 200, 250, 185, 290, 215, 330, 170, 240, 205, 275, 160]
FORMATS = ["wav", "wav", "mp3", "wav", "wav", "mp3", "wav", "wav", "mp3", "wav"]

# Build interactions
INTERACTIONS = []
TRANSCRIPTS = []
ALL_UTTERANCES = []
ALL_EMOTION_EVENTS = []
INTERACTION_SCORES = []
PROCESSING_JOBS = []

utt_counter = 0
ee_counter = 0

# Base date for spreading interactions
base_date = datetime(2026, 2, 1, tzinfo=timezone.utc)

for idx in range(30):
    iid = INTERACTION_IDS[idx]
    tid = TRANSCRIPT_IDS[idx]
    conv_idx = idx % len(CONVERSATIONS)
    full_text, emotions = CONVERSATIONS[conv_idx]
    lines = full_text.strip().split("\n")
    dur = DURATIONS[idx % len(DURATIONS)]
    fmt = FORMATS[idx % len(FORMATS)]
    file_size = dur * 16000 * 2  # rough WAV estimate

    # Assign to org/agent
    if idx < 18:
        org_id = ORG_IDS[0]
        agent_id = NILE_AGENTS[idx % len(NILE_AGENTS)]
        uploader = NILE_MANAGER
    else:
        org_id = ORG_IDS[1]
        agent_id = CAIRO_AGENTS[(idx - 18) % len(CAIRO_AGENTS)]
        uploader = CAIRO_MANAGER

    # Spread dates across weekdays
    day_offset = idx * 1.4
    int_date = base_date + timedelta(days=day_offset, hours=9 + (idx % 8), minutes=(idx * 7) % 60)

    status = "completed"
    lang = random.choice(["en", "ar", "en", "en"])

    INTERACTIONS.append({
        "id": iid, "organization_id": org_id, "agent_id": agent_id,
        "uploaded_by": uploader, "audio_file_path": f"/audio/call_{idx+1:03d}.{fmt}",
        "file_size_bytes": file_size, "duration_seconds": dur,
        "file_format": fmt, "interaction_date": int_date.isoformat(),
        "processing_status": status, "language_detected": lang,
        "has_overlap": idx % 7 == 0, "channel_count": 1
    })

    # Processing jobs (6 stages per interaction)
    for stage in ["diarization", "stt", "emotion", "reasoning", "scoring", "rag_eval"]:
        PROCESSING_JOBS.append({
            "interaction_id": iid, "stage": stage, "status": "completed"
        })

    # Transcript
    TRANSCRIPTS.append({
        "id": tid, "interaction_id": iid,
        "full_text": full_text,
        "overall_confidence": round(random.uniform(0.82, 0.98), 2)
    })

    # Utterances
    time_pos = 0.0
    for u_idx, line in enumerate(lines):
        utt_counter += 1
        speaker = "agent" if line.startswith("Agent:") else "customer"
        text = line.split(": ", 1)[1] if ": " in line else line
        emo = emotions[u_idx] if u_idx < len(emotions) else "neutral"
        seg_dur = round(random.uniform(1.5, 4.0), 1)

        ALL_UTTERANCES.append({
            "id": uid("12", utt_counter),
            "interaction_id": iid, "transcript_id": tid,
            "speaker_role": speaker,
            "user_id": agent_id if speaker == "agent" else None,
            "sequence_index": u_idx,
            "start_time_seconds": round(time_pos, 1),
            "end_time_seconds": round(time_pos + seg_dur, 1),
            "text": text, "emotion": emo,
            "emotion_confidence": round(random.uniform(0.65, 0.97), 2)
        })

        # Emotion event if emotion changed from previous
        if u_idx > 0 and u_idx < len(emotions):
            prev_emo = emotions[u_idx - 1]
            if emo != prev_emo:
                ee_counter += 1
                ALL_EMOTION_EVENTS.append({
                    "id": uid("11", ee_counter),
                    "interaction_id": iid,
                    "utterance_id": uid("12", utt_counter),
                    "previous_emotion": prev_emo,
                    "new_emotion": emo,
                    "emotion_delta": round(random.uniform(0.2, 0.8), 2),
                    "speaker_role": speaker,
                    "llm_justification": f"Speaker shifted from {prev_emo} to {emo}.",
                    "jump_to_seconds": round(time_pos, 1),
                    "is_flagged": ee_counter % 8 == 0  # ~12% flagged
                })

        time_pos += seg_dur + round(random.uniform(0.3, 1.5), 1)

    # Interaction scores
    base_score = round(random.uniform(5.0, 9.8), 1)
    INTERACTION_SCORES.append({
        "interaction_id": iid,
        "overall_score": base_score,
        "empathy_score": round(min(10.0, base_score + random.uniform(-1.5, 1.5)), 1),
        "policy_score": round(min(10.0, base_score + random.uniform(-2.0, 1.0)), 1),
        "resolution_score": round(min(10.0, base_score + random.uniform(-1.0, 2.0)), 1),
        "was_resolved": idx % 5 != 0,  # ~80% resolved
        "total_silence_seconds": round(random.uniform(2.0, 15.0), 1),
        "avg_response_time_seconds": round(random.uniform(1.0, 5.0), 1)
    })

# ===========================================================
# 6. Policy Compliance (~2 checks per interaction)
# ===========================================================
POLICY_COMPLIANCE = []
pc_counter = 0
for idx, iid in enumerate(INTERACTION_IDS[:30]):
    num_checks = 2 if idx % 3 != 0 else 3
    for p in range(num_checks):
        pc_counter += 1
        pol_id = POLICY_IDS[p % len(POLICY_IDS)]
        compliant = not (pc_counter % 5 == 0)  # ~20% non-compliant
        score = round(random.uniform(0.85, 1.0), 2) if compliant else round(random.uniform(0.2, 0.6), 2)
        POLICY_COMPLIANCE.append({
            "id": uid("30", pc_counter),
            "interaction_id": iid,
            "policy_id": pol_id,
            "is_compliant": compliant,
            "compliance_score": score,
            "llm_reasoning": f"{'Compliant' if compliant else 'Violation detected'} with {COMPANY_POLICIES[p % len(COMPANY_POLICIES)]['policy_title']}.",
            "evidence_text": "Reviewed relevant transcript segments.",
            "retrieved_policy_text": COMPANY_POLICIES[p % len(COMPANY_POLICIES)]["policy_text"]
        })

# ===========================================================
# 7. Organization Policies (assign policies to orgs)
# ===========================================================
ORGANIZATION_POLICIES = []
for org_id in ORG_IDS:
    for pol_id in POLICY_IDS:
        ORGANIZATION_POLICIES.append({
            "organization_id": org_id, "policy_id": pol_id, "is_active": True
        })

# ===========================================================
# 8. Organization FAQ Articles
# ===========================================================
ORGANIZATION_FAQ_ARTICLES = []
for org_id in ORG_IDS:
    for faq_id in FAQ_IDS:
        ORGANIZATION_FAQ_ARTICLES.append({
            "organization_id": org_id, "article_id": faq_id, "is_active": True
        })

# ===========================================================
# 9. Agent Performance Snapshots
# ===========================================================
AGENT_PERFORMANCE_SNAPSHOTS = []
snap_counter = 0
all_agents = NILE_AGENTS + CAIRO_AGENTS
for agent_id in all_agents:
    org_id = ORG_IDS[0] if agent_id in NILE_AGENTS else ORG_IDS[1]
    for period in [("daily", "2026-02-28", "2026-02-28"), ("weekly", "2026-02-24", "2026-02-28"),
                   ("monthly", "2026-02-01", "2026-02-28")]:
        snap_counter += 1
        AGENT_PERFORMANCE_SNAPSHOTS.append({
            "id": uid("22", snap_counter),
            "organization_id": org_id, "agent_id": agent_id,
            "period_type": period[0], "period_start": period[1], "period_end": period[2],
            "total_interactions": random.randint(3, 10),
            "avg_overall_score": round(random.uniform(6.0, 9.5), 1),
            "avg_empathy_score": round(random.uniform(6.0, 9.5), 1),
            "avg_policy_score": round(random.uniform(6.0, 9.5), 1),
            "avg_resolution_score": round(random.uniform(6.0, 9.5), 1),
            "resolution_rate": round(random.uniform(0.6, 1.0), 2)
        })

# ===========================================================
# 10. Emotion Feedback (8 records)
# ===========================================================
EMOTION_FEEDBACK = []
for i in range(min(8, len(ALL_EMOTION_EVENTS))):
    ee = ALL_EMOTION_EVENTS[i]
    reviewer = USER_IDS[0] if i < 4 else USER_IDS[5]  # managers review
    EMOTION_FEEDBACK.append({
        "emotion_event_id": ee["id"],
        "provided_by_user_id": reviewer,
        "llm_justification": ee["llm_justification"],
        "corrected_emotion": random.choice(EMOTIONS),
        "corrected_justification": "Manual review suggests different emotion.",
        "correction_reason": "LLM misclassified the tone.",
        "feedback_status": random.choice(["pending", "reviewed", "applied"]),
        "is_used_in_training": i % 3 == 0
    })

# ===========================================================
# 11. Compliance Feedback (5 records)
# ===========================================================
COMPLIANCE_FEEDBACK = []
non_compliant = [pc for pc in POLICY_COMPLIANCE if not pc["is_compliant"]]
for i in range(min(5, len(non_compliant))):
    pc = non_compliant[i]
    reviewer = USER_IDS[0] if i < 3 else USER_IDS[5]
    COMPLIANCE_FEEDBACK.append({
        "policy_compliance_id": pc["id"],
        "provided_by_user_id": reviewer,
        "original_is_compliant": False,
        "corrected_is_compliant": i % 2 == 0,
        "original_score": pc["compliance_score"],
        "corrected_score": round(random.uniform(0.7, 1.0), 2) if i % 2 == 0 else pc["compliance_score"],
        "correction_reason": "Re-evaluated transcript context.",
        "feedback_status": random.choice(["pending", "reviewed"]),
        "is_used_in_training": False
    })

# ===========================================================
# 12. Assistant Queries
# ===========================================================
QUERY_IDS_LIST = [uid("33", i) for i in range(1, 9)]

ASSISTANT_QUERIES = [
    {"id": QUERY_IDS_LIST[0], "user_id": USER_IDS[0], "organization_id": ORG_IDS[0],
     "query_mode": "chat", "query_text": "How many calls were handled today?", "response_text": "There were 5 calls handled today with an average score of 8.2."},
    {"id": QUERY_IDS_LIST[1], "user_id": USER_IDS[0], "organization_id": ORG_IDS[0],
     "query_mode": "chat", "query_text": "Which agent has the highest score this week?", "response_text": "Sara Agent has the highest average score of 92 this week."},
    {"id": QUERY_IDS_LIST[2], "user_id": USER_IDS[5], "organization_id": ORG_IDS[1],
     "query_mode": "chat", "query_text": "Show me policy violations from last week.", "response_text": "There were 3 policy violations detected: 2 greeting policy and 1 hold time violation."},
    {"id": QUERY_IDS_LIST[3], "user_id": USER_IDS[0], "organization_id": ORG_IDS[0],
     "query_mode": "voice", "query_text": "What is the resolution rate?", "response_text": "The current resolution rate is 80% across all agents."},
    {"id": QUERY_IDS_LIST[4], "user_id": USER_IDS[5], "organization_id": ORG_IDS[1],
     "query_mode": "chat", "query_text": "How is Yasmin performing?", "response_text": "Yasmin has handled 4 calls this month with an average score of 8.7 and 100% resolution rate."},
    {"id": QUERY_IDS_LIST[5], "user_id": USER_IDS[0], "organization_id": ORG_IDS[0],
     "query_mode": "chat", "query_text": "List the most common customer emotions.", "response_text": "The most common emotions are: neutral (35%), frustrated (25%), happy (20%), angry (10%), and others (10%)."},
    {"id": QUERY_IDS_LIST[6], "user_id": USER_IDS[5], "organization_id": ORG_IDS[1],
     "query_mode": "voice", "query_text": "Any escalations needed today?", "response_text": "There are 2 interactions flagged for escalation review based on emotion analysis."},
    {"id": QUERY_IDS_LIST[7], "user_id": USER_IDS[0], "organization_id": ORG_IDS[0],
     "query_mode": "chat", "query_text": "Compare agent performance this month.", "response_text": "Mohsen: 8.5 avg, Sara: 9.2 avg, Omar: 7.8 avg, Nour AI: 8.9 avg."},
]

# ===========================================================
# Table insertion order (FK-safe)
# ===========================================================
TABLES_IN_ORDER = [
    ("organizations", ORGANIZATIONS),
    ("users", USERS),
    ("company_policies", COMPANY_POLICIES),
    ("faq_articles", FAQ_ARTICLES),
    ("interactions", INTERACTIONS),
    ("processing_jobs", PROCESSING_JOBS),
    ("transcripts", TRANSCRIPTS),
    ("utterances", ALL_UTTERANCES),
    ("interaction_scores", INTERACTION_SCORES),
    ("emotion_events", ALL_EMOTION_EVENTS),
    ("policy_compliance", POLICY_COMPLIANCE),
    ("emotion_feedback", EMOTION_FEEDBACK),
    ("compliance_feedback", COMPLIANCE_FEEDBACK),
    ("organization_policies", ORGANIZATION_POLICIES),
    ("organization_faq_articles", ORGANIZATION_FAQ_ARTICLES),
    ("agent_performance_snapshots", AGENT_PERFORMANCE_SNAPSHOTS),
    ("assistant_queries", ASSISTANT_QUERIES),
]


def seed():
    print("Seeding VocalMind database v5.2 (comprehensive) ...")
    for table_name, rows in TABLES_IN_ORDER:
        if not rows:
            print(f"  [SKIP] {table_name}: 0 rows")
            continue
        try:
            result = supabase.table(table_name).insert(rows).execute()
            print(f"  [OK] {table_name}: {len(result.data)} rows")
        except Exception as e:
            print(f"  [ERROR] {table_name}: {e}")
            raise e
    print("\nComprehensive seeding complete.")


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
