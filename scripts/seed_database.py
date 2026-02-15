"""
VocalMind Database Seeder
Populates or cleans up sample data in the Supabase database.

Usage:
    uv run python scripts/seed_database.py           # Seed data
    uv run python scripts/seed_database.py --cleanup  # Remove seed data

Requires:
    SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env or environment.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===========================================================
# Deterministic UUIDs for easy reference and cleanup
# ===========================================================

ORG_IDS = [f"a0000000-0000-0000-0000-00000000000{i}" for i in range(1, 6)]
USER_IDS = [f"b0000000-0000-0000-0000-00000000000{i}" for i in range(1, 6)]
AGENT_IDS = [f"c0000000-0000-0000-0000-00000000000{i}" for i in range(1, 6)]
INTERACTION_IDS = [
    f"d0000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 16)
]
TRANSCRIPT_IDS = [
    f"e0000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 16)
]
UTTERANCE_IDS = [
    f"f0000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 16)
]
EMOTION_EVENT_IDS = [
    f"10000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 11)
]
POLICY_IDS = [
    f"20000000-0000-0000-0000-00000000000{i}" for i in range(1, 6)
]
COMPLIANCE_IDS = [
    f"30000000-0000-0000-0000-{str(i).zfill(12)}" for i in range(1, 11)
]
FEEDBACK_IDS = [
    f"40000000-0000-0000-0000-00000000000{i}" for i in range(1, 6)
]
QUERY_IDS = [
    f"50000000-0000-0000-0000-00000000000{i}" for i in range(1, 6)
]

# ===========================================================
# Sample Data
# ===========================================================

ORGANIZATIONS = [
    {"id": ORG_IDS[0], "name": "NileTech", "status": "active"},
    {"id": ORG_IDS[1], "name": "CairoConnect", "status": "active"},
    {"id": ORG_IDS[2], "name": "PyramidSupport", "status": "active"},
    {"id": ORG_IDS[3], "name": "DeltaServices", "status": "inactive"},
    {"id": ORG_IDS[4], "name": "SphinxTelecom", "status": "active"},
]

USERS = [
    {
        "id": USER_IDS[0],
        "organization_id": ORG_IDS[0],
        "email": "galal@niletech.com",
        "password_hash": "$2b$12$seedhashplaceholder001",
        "name": "Galal",
        "role": "admin",
        "is_active": True,
    },
    {
        "id": USER_IDS[1],
        "organization_id": ORG_IDS[1],
        "email": "ibrahem@cairoconnect.com",
        "password_hash": "$2b$12$seedhashplaceholder002",
        "name": "Ibrahem",
        "role": "manager",
        "is_active": True,
    },
    {
        "id": USER_IDS[2],
        "organization_id": ORG_IDS[2],
        "email": "mohamed@pyramidsupport.com",
        "password_hash": "$2b$12$seedhashplaceholder003",
        "name": "Mohamed",
        "role": "admin",
        "is_active": True,
    },
    {
        "id": USER_IDS[3],
        "organization_id": ORG_IDS[3],
        "email": "hassan@deltaservices.com",
        "password_hash": "$2b$12$seedhashplaceholder004",
        "name": "Hassan",
        "role": "manager",
        "is_active": True,
    },
    {
        "id": USER_IDS[4],
        "organization_id": ORG_IDS[4],
        "email": "ahmed@sphinxtelecom.com",
        "password_hash": "$2b$12$seedhashplaceholder005",
        "name": "Ahmed",
        "role": "manager",
        "is_active": True,
    },
]

AGENTS = [
    {
        "id": AGENT_IDS[0],
        "organization_id": ORG_IDS[0],
        "agent_code": "NT-101",
        "agent_type": "human",
        "department": "Sales",
        "is_active": True,
    },
    {
        "id": AGENT_IDS[1],
        "organization_id": ORG_IDS[1],
        "agent_code": "CC-201",
        "agent_type": "human",
        "department": "Support",
        "is_active": True,
    },
    {
        "id": AGENT_IDS[2],
        "organization_id": ORG_IDS[2],
        "agent_code": "PS-301",
        "agent_type": "bot",
        "department": "Technical",
        "is_active": True,
    },
    {
        "id": AGENT_IDS[3],
        "organization_id": ORG_IDS[3],
        "agent_code": "DS-401",
        "agent_type": "human",
        "department": "Billing",
        "is_active": True,
    },
    {
        "id": AGENT_IDS[4],
        "organization_id": ORG_IDS[4],
        "agent_code": "ST-501",
        "agent_type": "human",
        "department": "Retention",
        "is_active": True,
    },
]

INTERACTIONS = [
    {
        "id": INTERACTION_IDS[i],
        "organization_id": ORG_IDS[i // 3],
        "agent_id": AGENT_IDS[i // 3],
        "audio_file_path": path,
        "file_size_bytes": size,
        "duration_seconds": dur,
        "file_format": fmt,
        "interaction_date": date,
        "processing_status": status,
        "language_detected": "en",
        "has_overlap": overlap,
    }
    for i, (path, size, dur, fmt, date, status, overlap) in enumerate(
        [
            ("/audio/NT-101_call_001.wav", 2457600, 180, "wav", "2026-02-01T09:15:00+02:00", "completed", False),
            ("/audio/NT-101_call_002.mp3", 1843200, 420, "mp3", "2026-02-02T14:30:00+02:00", "completed", True),
            ("/audio/NT-101_call_003.wav", 3072000, 600, "wav", "2026-02-03T11:00:00+02:00", "completed", False),
            ("/audio/CC-201_call_001.wav", 1536000, 240, "wav", "2026-02-01T10:00:00+02:00", "completed", False),
            ("/audio/CC-201_call_002.mp3", 921600, 150, "mp3", "2026-02-04T16:45:00+02:00", "completed", False),
            ("/audio/CC-201_call_003.wav", 4096000, 900, "wav", "2026-02-05T08:20:00+02:00", "processing", True),
            ("/audio/PS-301_call_001.wav", 768000, 120, "wav", "2026-02-02T13:00:00+02:00", "completed", False),
            ("/audio/PS-301_call_002.mp3", 1228800, 300, "mp3", "2026-02-03T15:30:00+02:00", "completed", False),
            ("/audio/PS-301_call_003.wav", 614400, 90, "wav", "2026-02-06T09:45:00+02:00", "pending", False),
            ("/audio/DS-401_call_001.wav", 2048000, 360, "wav", "2026-02-01T11:30:00+02:00", "completed", False),
            ("/audio/DS-401_call_002.mp3", 3686400, 720, "mp3", "2026-02-04T09:00:00+02:00", "completed", True),
            ("/audio/DS-401_call_003.wav", 1024000, 200, "wav", "2026-02-07T14:00:00+02:00", "failed", False),
            ("/audio/ST-501_call_001.wav", 1843200, 480, "wav", "2026-02-02T10:15:00+02:00", "completed", False),
            ("/audio/ST-501_call_002.mp3", 2560000, 540, "mp3", "2026-02-05T13:00:00+02:00", "completed", False),
            ("/audio/ST-501_call_003.wav", 921600, 160, "wav", "2026-02-08T08:00:00+02:00", "processing", False),
        ]
    )
]

TRANSCRIPTS = [
    {"id": TRANSCRIPT_IDS[0], "interaction_id": INTERACTION_IDS[0], "confidence_score": 0.94, "full_text": "Agent: Thank you for calling NileTech Sales, my name is Mohsen. How can I help you today? Customer: Hi Mohsen, I am interested in upgrading my current plan. Agent: Of course! Let me pull up your account. I can see you are on our Basic plan. We have a Premium option that includes unlimited data. Customer: That sounds great, how much would that cost? Agent: It would be 299 EGP per month. Customer: That works for me, let us go ahead with the upgrade. Agent: Perfect, I have processed the upgrade. Is there anything else I can help with? Customer: No, that is all. Thank you! Agent: Thank you for choosing NileTech. Have a great day!"},
    {"id": TRANSCRIPT_IDS[1], "interaction_id": INTERACTION_IDS[1], "confidence_score": 0.91, "full_text": "Agent: NileTech Sales, Mohsen speaking. Customer: Yes, I have been waiting for 20 minutes already! I want to cancel my subscription. Agent: I am sorry about the wait. Let me see what I can do for you. May I ask why you want to cancel? Customer: The service has been terrible. My internet keeps dropping every hour. Agent: I completely understand your frustration. Let me check if there is a technical issue on our end. Customer: I have already called three times about this! Agent: I see the previous tickets. Let me escalate this to our technical team right away and offer you a 50% discount for the next 3 months. Customer: Fine, but if it does not improve I am switching providers. Agent: I understand. I will personally follow up within 48 hours."},
    {"id": TRANSCRIPT_IDS[2], "interaction_id": INTERACTION_IDS[2], "confidence_score": 0.96, "full_text": "Agent: Welcome to NileTech, this is Mohsen. How may I assist you? Customer: I need help understanding my bill. There are some charges I do not recognize. Agent: I would be happy to help you with that. Can you tell me which charges look unfamiliar? Customer: There is a charge for 150 EGP labeled Premium Add-on. I never signed up for any add-on. Agent: Let me investigate that for you. I can see it was added on January 15th. It appears it was added during a previous call. Customer: I did not authorize that. Agent: I understand. I will remove the charge and issue a full refund. You should see it within 3-5 business days. Customer: Thank you, I appreciate that. Agent: You are welcome. Is there anything else? Customer: No, that is all."},
    {"id": TRANSCRIPT_IDS[3], "interaction_id": INTERACTION_IDS[3], "confidence_score": 0.95, "full_text": "Agent: CairoConnect Support, Galal here. How can I help? Customer: My email has not been working since yesterday. Agent: I am sorry to hear that. Let me check your account status. Can you provide your account number? Customer: It is CC-4421. Agent: Thank you. I can see there was a server migration last night that may have affected some accounts. Let me reset your email configuration. Customer: How long will it take? Agent: It should be working within the next 10 minutes. I will stay on the line if you would like to verify. Customer: Yes please. Agent: Great, try logging in now. Customer: It works! Thank you so much. Agent: Happy to help!"},
    {"id": TRANSCRIPT_IDS[4], "interaction_id": INTERACTION_IDS[4], "confidence_score": 0.97, "full_text": "Agent: CairoConnect, Galal speaking. Customer: Hi, I just want to check when my contract expires. Agent: Sure, let me look that up. Your current contract ends on March 15, 2026. Customer: Great, and what are my renewal options? Agent: You can renew at the same rate or upgrade to our business plan which gives you priority support. Customer: I will think about it. Thank you. Agent: No problem, feel free to call back anytime."},
    {"id": TRANSCRIPT_IDS[5], "interaction_id": INTERACTION_IDS[5], "confidence_score": 0.89, "full_text": "Agent: CairoConnect Support, this is Galal. Customer: I am extremely upset. You charged my card twice for the same invoice! Agent: I sincerely apologize for that. Let me look into this immediately. Customer: This is the second time this has happened! Agent: I understand this is very frustrating. I can see the duplicate charge. I am initiating a refund right now. Customer: When will I get my money back? Agent: The refund will be processed within 24 hours. I am also adding a credit to your account as compensation. Customer: It should not have happened in the first place. Agent: You are absolutely right, and I apologize again. I am flagging this to our billing team to prevent it from happening again."},
    {"id": TRANSCRIPT_IDS[6], "interaction_id": INTERACTION_IDS[6], "confidence_score": 0.93, "full_text": "Agent: Thank you for contacting PyramidSupport. I am an automated assistant. How can I help you today? Customer: My router keeps disconnecting. Agent: I can help troubleshoot that. Have you tried restarting your router? Customer: Yes, multiple times. Agent: Let me run a remote diagnostic on your connection. Please wait a moment. I have detected an issue with your DNS settings. I will push an automatic fix now. Customer: Okay, let me check. It seems to be working now. Agent: Great! Is there anything else you need help with? Customer: No, thanks."},
    {"id": TRANSCRIPT_IDS[7], "interaction_id": INTERACTION_IDS[7], "confidence_score": 0.98, "full_text": "Agent: PyramidSupport automated service. How may I assist you? Customer: I need to change my Wi-Fi password. Agent: I can guide you through that. Please open your browser and navigate to 192.168.1.1. Customer: Done. Agent: Enter your admin credentials. The default username is admin and the password is on the sticker on your router. Customer: I am in. Agent: Navigate to Wireless Settings, then Security. You can change your password there. Customer: Got it, changed it. Thank you! Agent: You are welcome. For security, I recommend changing it every 90 days."},
    {"id": TRANSCRIPT_IDS[8], "interaction_id": INTERACTION_IDS[8], "confidence_score": 0.96, "full_text": "Agent: PyramidSupport, automated assistant ready to help. Customer: I want to know about your fiber optic plans. Agent: We offer three fiber plans: Basic at 200 Mbps, Standard at 500 Mbps, and Premium at 1 Gbps. Customer: What is the price for Premium? Agent: The Premium plan is 599 EGP per month with a 12-month contract. Customer: I will consider it. Agent: Would you like me to transfer you to a sales representative for more details? Customer: No, that is fine for now. Thank you."},
    {"id": TRANSCRIPT_IDS[9], "interaction_id": INTERACTION_IDS[9], "confidence_score": 0.95, "full_text": "Agent: DeltaServices Billing, Hassan speaking. How can I assist you? Customer: I am calling about a late payment notice. I already paid last week. Agent: Let me check your payment history. Can you provide the transaction reference? Customer: It is TXN-88234. Agent: I found it. The payment was received but not applied to your account due to an incorrect reference number. I will fix that now. Customer: So I do not owe anything? Agent: Correct, your account is now current. I apologize for the confusion. Customer: Thank you for sorting it out. Agent: You are welcome. Have a good day."},
    {"id": TRANSCRIPT_IDS[10], "interaction_id": INTERACTION_IDS[10], "confidence_score": 0.88, "full_text": "Agent: DeltaServices, Hassan here. Customer: I want to dispute a charge on my account. I was charged 500 EGP for a service I never used. Agent: I understand your concern. Let me review the charge. I can see it is for our Premium Support package. Customer: I never signed up for that! This is outrageous! Agent: I can see it was added through our app on January 20th. Customer: Someone must have done it by accident. I want a refund immediately! Agent: I understand. However, I need to file a dispute form which takes 5-7 business days. Customer: That is unacceptable! I need this resolved today! Agent: Let me speak with my supervisor. Please hold. I have been authorized to issue an immediate refund. Customer: Finally, thank you. Agent: I apologize for the inconvenience."},
    {"id": TRANSCRIPT_IDS[11], "interaction_id": INTERACTION_IDS[11], "confidence_score": 0.94, "full_text": "Agent: DeltaServices Billing, Hassan speaking. Customer: Hi, I just want to update my payment method. Agent: Sure, I can help with that. For security purposes I will need to verify your identity first. Can you confirm your date of birth? Customer: July 15, 1990. Agent: Thank you. What payment method would you like to add? Customer: A new credit card. Agent: Please provide the card number. Customer: 4532 actually, wait, can I do this online instead? Agent: Absolutely, you can update it through our app under Settings then Payment Methods. Customer: Perfect, I will do that. Thanks. Agent: You are welcome."},
    {"id": TRANSCRIPT_IDS[12], "interaction_id": INTERACTION_IDS[12], "confidence_score": 0.93, "full_text": "Agent: SphinxTelecom, Ahmed from Retention. How can I help you today? Customer: I am thinking about canceling my service. Agent: I am sorry to hear that. May I ask what prompted this decision? Customer: I found a better deal with another provider. Agent: I understand. Can you tell me what they are offering? Customer: 500 Mbps for 199 EGP per month. Agent: That is competitive. Let me see what we can offer. I can match that price and add free installation for a new router. Customer: Really? That changes things. Agent: Absolutely. I value your loyalty as a customer for 3 years. Customer: Okay, I will stay then. Thank you! Agent: Wonderful! I will process the new rate immediately."},
    {"id": TRANSCRIPT_IDS[13], "interaction_id": INTERACTION_IDS[13], "confidence_score": 0.92, "full_text": "Agent: SphinxTelecom Retention, Ahmed speaking. Customer: I have been having issues with my service quality. The speed test shows half of what I am paying for. Agent: I apologize for that. Let me run a diagnostic. I can see your line is performing below expected levels. There appears to be congestion in your area. Customer: So when will it be fixed? Agent: We have a network upgrade scheduled for your area next week. In the meantime, I can offer you a temporary speed boost at no extra charge. Customer: That would help. How long until the upgrade is done? Agent: The upgrade should be completed by February 15th. I will also credit your account for the affected period. Customer: That sounds fair. Thank you for being proactive about it. Agent: Of course, we want to make sure you are getting the service you are paying for."},
    {"id": TRANSCRIPT_IDS[14], "interaction_id": INTERACTION_IDS[14], "confidence_score": 0.97, "full_text": "Agent: SphinxTelecom, Ahmed here. How can I help? Customer: I just moved to a new apartment and I need to transfer my service. Agent: Congratulations on the move! I can help with that. What is your new address? Customer: 45 El-Nour Street, Nasr City. Agent: Let me check coverage in that area. Great news, we have full coverage there. The transfer will take 2-3 business days. Customer: Will my plan stay the same? Agent: Yes, everything transfers as-is. We just need to schedule a technician visit. Customer: Can they come on Saturday? Agent: Let me check availability. Yes, we have a slot on Saturday between 10 AM and 12 PM. Customer: Perfect, book it. Agent: Done! You will receive a confirmation SMS shortly."},
]

UTTERANCES = [
    {"id": UTTERANCE_IDS[i], "interaction_id": INTERACTION_IDS[i], "speaker_role": role, "start_time_seconds": st, "end_time_seconds": et, "emotion_label": emo, "emotion_confidence": conf, "text": txt}
    for i, (role, st, et, emo, conf, txt) in enumerate([
        ("customer", 5.0, 12.3, "neutral", 0.88, "Hi Mohsen, I am interested in upgrading my current plan."),
        ("customer", 3.2, 15.7, "angry", 0.92, "Yes, I have been waiting for 20 minutes already! I want to cancel my subscription."),
        ("customer", 8.0, 18.5, "frustrated", 0.85, "My internet keeps dropping every hour. I have already called three times about this!"),
        ("customer", 4.1, 10.0, "worried", 0.78, "I received a notification about a rate increase. Can you explain what is changing?"),
        ("customer", 2.5, 8.9, "neutral", 0.91, "I just want to check the status of my recent order."),
        ("customer", 3.0, 14.2, "angry", 0.95, "I have been overcharged for the third month in a row. This is completely unacceptable!"),
        ("customer", 6.0, 11.4, "neutral", 0.82, "Hi, I am calling to inquire about your business plans."),
        ("agent", 10.5, 22.0, "neutral", 0.90, "Thank you for calling. I can help you with your account settings today."),
        ("customer", 4.0, 9.8, "neutral", 0.87, "I need to update my billing address and payment method."),
        ("customer", 5.5, 16.0, "worried", 0.80, "I noticed some unauthorized charges on my account. I am concerned about security."),
        ("customer", 7.0, 20.3, "angry", 0.94, "Your service has been down all day! I am losing business because of this!"),
        ("customer", 3.8, 10.5, "neutral", 0.89, "Can you walk me through setting up the VPN on my device?"),
        ("customer", 4.5, 12.0, "disappointed", 0.76, "I was promised a callback within 24 hours, but nobody called me back."),
        ("customer", 6.2, 15.8, "frustrated", 0.83, "The new interface is confusing. I cannot find any of the features I used to use."),
        ("customer", 2.0, 8.0, "happy", 0.86, "Everything was resolved quickly. Thank you so much for your help!"),
    ])
]

EMOTION_EVENTS = [
    {"id": EMOTION_EVENT_IDS[0], "interaction_id": INTERACTION_IDS[1], "utterance_id": UTTERANCE_IDS[1], "event_type": "escalation", "previous_emotion": "frustrated", "new_emotion": "angry", "emotion_delta": 0.35, "trigger_category": "Long Wait", "timestamp_seconds": 15.7, "speaker_role": "customer", "verified_by_user_id": None},
    {"id": EMOTION_EVENT_IDS[1], "interaction_id": INTERACTION_IDS[1], "utterance_id": UTTERANCE_IDS[1], "event_type": "de_escalation", "previous_emotion": "angry", "new_emotion": "neutral", "emotion_delta": -0.40, "trigger_category": "Empathy", "timestamp_seconds": 180.0, "speaker_role": "customer", "verified_by_user_id": USER_IDS[0]},
    {"id": EMOTION_EVENT_IDS[2], "interaction_id": INTERACTION_IDS[2], "utterance_id": UTTERANCE_IDS[2], "event_type": "sentiment_drop", "previous_emotion": "neutral", "new_emotion": "frustrated", "emotion_delta": 0.30, "trigger_category": "Billing Error", "timestamp_seconds": 18.5, "speaker_role": "customer", "verified_by_user_id": None},
    {"id": EMOTION_EVENT_IDS[3], "interaction_id": INTERACTION_IDS[5], "utterance_id": UTTERANCE_IDS[5], "event_type": "escalation", "previous_emotion": "frustrated", "new_emotion": "angry", "emotion_delta": 0.45, "trigger_category": "Billing Error", "timestamp_seconds": 14.2, "speaker_role": "customer", "verified_by_user_id": None},
    {"id": EMOTION_EVENT_IDS[4], "interaction_id": INTERACTION_IDS[5], "utterance_id": UTTERANCE_IDS[5], "event_type": "de_escalation", "previous_emotion": "angry", "new_emotion": "calm", "emotion_delta": -0.50, "trigger_category": "Empathy", "timestamp_seconds": 300.0, "speaker_role": "customer", "verified_by_user_id": USER_IDS[1]},
    {"id": EMOTION_EVENT_IDS[5], "interaction_id": INTERACTION_IDS[10], "utterance_id": UTTERANCE_IDS[10], "event_type": "escalation", "previous_emotion": "neutral", "new_emotion": "angry", "emotion_delta": 0.60, "trigger_category": "Unauthorized Charge", "timestamp_seconds": 20.3, "speaker_role": "customer", "verified_by_user_id": None},
    {"id": EMOTION_EVENT_IDS[6], "interaction_id": INTERACTION_IDS[10], "utterance_id": UTTERANCE_IDS[10], "event_type": "emotion_shift", "previous_emotion": "angry", "new_emotion": "relieved", "emotion_delta": -0.55, "trigger_category": "Resolution", "timestamp_seconds": 600.0, "speaker_role": "customer", "verified_by_user_id": USER_IDS[3]},
    {"id": EMOTION_EVENT_IDS[7], "interaction_id": INTERACTION_IDS[12], "utterance_id": UTTERANCE_IDS[12], "event_type": "emotion_shift", "previous_emotion": "disappointed", "new_emotion": "happy", "emotion_delta": -0.40, "trigger_category": "Counter Offer", "timestamp_seconds": 300.0, "speaker_role": "customer", "verified_by_user_id": None},
    {"id": EMOTION_EVENT_IDS[8], "interaction_id": INTERACTION_IDS[13], "utterance_id": UTTERANCE_IDS[13], "event_type": "sentiment_drop", "previous_emotion": "neutral", "new_emotion": "frustrated", "emotion_delta": 0.25, "trigger_category": "Service Quality", "timestamp_seconds": 15.8, "speaker_role": "customer", "verified_by_user_id": None},
    {"id": EMOTION_EVENT_IDS[9], "interaction_id": INTERACTION_IDS[13], "utterance_id": UTTERANCE_IDS[13], "event_type": "de_escalation", "previous_emotion": "frustrated", "new_emotion": "satisfied", "emotion_delta": -0.35, "trigger_category": "Proactive Solution", "timestamp_seconds": 400.0, "speaker_role": "customer", "verified_by_user_id": USER_IDS[4]},
]

INTERACTION_SCORES = [
    {"interaction_id": INTERACTION_IDS[i], "overall_score": o, "policy_score": p, "total_silence_duration_seconds": s, "average_response_time_seconds": r, "was_resolved": w}
    for i, (o, p, s, r, w) in enumerate([
        (92.5, 95.0, 8.0, 2.1, True), (68.0, 60.0, 25.0, 5.8, True),
        (88.0, 90.0, 12.0, 3.2, True), (95.0, 98.0, 5.0, 1.5, True),
        (90.0, 92.0, 3.0, 1.8, True), (62.0, 55.0, 30.0, 6.5, True),
        (85.0, 88.0, 6.0, 2.0, True), (91.0, 93.0, 4.0, 1.2, True),
        (78.0, 80.0, 10.0, 3.5, False), (87.0, 85.0, 14.0, 3.0, True),
        (60.0, 50.0, 45.0, 8.2, True), (82.0, 88.0, 8.0, 2.5, False),
        (94.0, 96.0, 5.0, 1.6, True), (80.0, 78.0, 18.0, 4.0, True),
        (93.0, 95.0, 4.0, 1.4, True),
    ])
]

COMPANY_POLICIES = [
    {"id": POLICY_IDS[0], "organization_id": ORG_IDS[0], "policy_code": "POL-GREET-01", "category": "Communication", "policy_text": "Agents must greet the customer by name within the first 15 seconds of the call and introduce themselves with their full name and department.", "pinecone_id": "vec_pol_001"},
    {"id": POLICY_IDS[1], "organization_id": ORG_IDS[1], "policy_code": "POL-HOLD-01", "category": "Service Level", "policy_text": "Customers must not be placed on hold for more than 2 minutes without an update. Total hold time per call must not exceed 5 minutes.", "pinecone_id": "vec_pol_002"},
    {"id": POLICY_IDS[2], "organization_id": ORG_IDS[2], "policy_code": "POL-EMPATH-01", "category": "Communication", "policy_text": "Agents must acknowledge customer emotions and use empathetic language such as I understand and I apologize when the customer expresses frustration or dissatisfaction.", "pinecone_id": "vec_pol_003"},
    {"id": POLICY_IDS[3], "organization_id": ORG_IDS[3], "policy_code": "POL-ESCAL-01", "category": "Escalation", "policy_text": "If a customer requests to speak with a supervisor or the issue cannot be resolved within 10 minutes, the agent must escalate the call to a senior representative immediately.", "pinecone_id": "vec_pol_004"},
    {"id": POLICY_IDS[4], "organization_id": ORG_IDS[4], "policy_code": "POL-PRIV-01", "category": "Data Privacy", "policy_text": "Agents must verify customer identity using at least two data points before disclosing any account information. Credit card numbers must never be read back in full.", "pinecone_id": "vec_pol_005"},
]

POLICY_COMPLIANCE = [
    {"id": COMPLIANCE_IDS[0], "interaction_id": INTERACTION_IDS[0], "policy_id": POLICY_IDS[0], "is_compliant": True, "compliance_score": 95.0, "violation_severity": None, "confidence_score": 0.93, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Agent greeted customer by name within 10 seconds", "evidence_text": "Thank you for calling NileTech Sales, my name is Mohsen. How can I help you today?", "llm_reasoning": "The agent introduced themselves with their name and department (Sales) within the first sentence, meeting the 15-second greeting requirement.", "is_human_verified": True, "human_feedback_text": None},
    {"id": COMPLIANCE_IDS[1], "interaction_id": INTERACTION_IDS[1], "policy_id": POLICY_IDS[0], "is_compliant": False, "compliance_score": 40.0, "violation_severity": "minor", "confidence_score": 0.88, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Agent did not greet customer by name", "evidence_text": "NileTech Sales, Mohsen speaking.", "llm_reasoning": "The agent introduced themselves but did not greet the customer by name. The greeting was abbreviated.", "is_human_verified": False, "human_feedback_text": None},
    {"id": COMPLIANCE_IDS[2], "interaction_id": INTERACTION_IDS[3], "policy_id": POLICY_IDS[1], "is_compliant": True, "compliance_score": 90.0, "violation_severity": None, "confidence_score": 0.91, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Hold time within acceptable limits", "evidence_text": "I will stay on the line if you would like to verify.", "llm_reasoning": "No hold was used. The agent offered to stay on the line while the customer tested.", "is_human_verified": True, "human_feedback_text": None},
    {"id": COMPLIANCE_IDS[3], "interaction_id": INTERACTION_IDS[5], "policy_id": POLICY_IDS[1], "is_compliant": False, "compliance_score": 30.0, "violation_severity": "critical", "confidence_score": 0.95, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Excessive customer wait before agent response", "evidence_text": "I am extremely upset. You charged my card twice!", "llm_reasoning": "The interaction showed signs of long processing delays before resolution.", "is_human_verified": True, "human_feedback_text": "Confirmed: customer experienced extended wait"},
    {"id": COMPLIANCE_IDS[4], "interaction_id": INTERACTION_IDS[5], "policy_id": POLICY_IDS[2], "is_compliant": True, "compliance_score": 85.0, "violation_severity": None, "confidence_score": 0.90, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Agent used empathetic language during escalation", "evidence_text": "I sincerely apologize for that. Let me look into this immediately.", "llm_reasoning": "The agent used empathetic phrases like sincerely apologize and immediately.", "is_human_verified": False, "human_feedback_text": None},
    {"id": COMPLIANCE_IDS[5], "interaction_id": INTERACTION_IDS[10], "policy_id": POLICY_IDS[3], "is_compliant": False, "compliance_score": 35.0, "violation_severity": "critical", "confidence_score": 0.92, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Escalation delayed beyond 10-minute threshold", "evidence_text": "That is unacceptable! I need this resolved today!", "llm_reasoning": "The agent initially tried to process a 5-7 day dispute form instead of immediately escalating.", "is_human_verified": True, "human_feedback_text": "Agent should have escalated sooner"},
    {"id": COMPLIANCE_IDS[6], "interaction_id": INTERACTION_IDS[11], "policy_id": POLICY_IDS[4], "is_compliant": True, "compliance_score": 92.0, "violation_severity": None, "confidence_score": 0.96, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Identity verification performed correctly", "evidence_text": "For security purposes I will need to verify your identity first. Can you confirm your date of birth?", "llm_reasoning": "The agent correctly requested identity verification before proceeding.", "is_human_verified": True, "human_feedback_text": None},
    {"id": COMPLIANCE_IDS[7], "interaction_id": INTERACTION_IDS[12], "policy_id": POLICY_IDS[2], "is_compliant": True, "compliance_score": 98.0, "violation_severity": None, "confidence_score": 0.97, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Excellent empathy and retention approach", "evidence_text": "I understand. Can you tell me what they are offering?", "llm_reasoning": "The agent showed strong empathy and offered a competitive counter-offer.", "is_human_verified": False, "human_feedback_text": None},
    {"id": COMPLIANCE_IDS[8], "interaction_id": INTERACTION_IDS[13], "policy_id": POLICY_IDS[2], "is_compliant": True, "compliance_score": 88.0, "violation_severity": None, "confidence_score": 0.89, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Proactive empathy with service quality issue", "evidence_text": "I apologize for that. Let me run a diagnostic.", "llm_reasoning": "The agent immediately acknowledged the service issue and took proactive steps.", "is_human_verified": False, "human_feedback_text": None},
    {"id": COMPLIANCE_IDS[9], "interaction_id": INTERACTION_IDS[2], "policy_id": POLICY_IDS[2], "is_compliant": True, "compliance_score": 90.0, "violation_severity": None, "confidence_score": 0.91, "analyzed_by_model": "llama3.1-70b", "trigger_description": "Agent acknowledged billing frustration empathetically", "evidence_text": "I understand. I will remove the charge and issue a full refund.", "llm_reasoning": "The agent used empathetic language and took immediate corrective action.", "is_human_verified": True, "human_feedback_text": None},
]

HUMAN_FEEDBACK = [
    {"id": FEEDBACK_IDS[0], "interaction_id": INTERACTION_IDS[1], "provided_by_user_id": USER_IDS[0], "feedback_type": "emotion_label", "ai_output": {"emotion": "frustrated", "confidence": 0.78}, "corrected_output": {"emotion": "angry", "confidence": 0.95}, "correction_reason": "Customer was clearly angry, not just frustrated."},
    {"id": FEEDBACK_IDS[1], "interaction_id": INTERACTION_IDS[5], "provided_by_user_id": USER_IDS[1], "feedback_type": "score", "ai_output": {"overall_score": 75.0}, "corrected_output": {"overall_score": 62.0}, "correction_reason": "The duplicate billing error should lower the score more."},
    {"id": FEEDBACK_IDS[2], "interaction_id": INTERACTION_IDS[10], "provided_by_user_id": USER_IDS[3], "feedback_type": "compliance", "ai_output": {"is_compliant": True, "policy": "POL-ESCAL-01"}, "corrected_output": {"is_compliant": False, "policy": "POL-ESCAL-01"}, "correction_reason": "The agent delayed escalation."},
    {"id": FEEDBACK_IDS[3], "interaction_id": INTERACTION_IDS[6], "provided_by_user_id": USER_IDS[2], "feedback_type": "transcription", "ai_output": {"text": "I have detected an issue with your DSN settings"}, "corrected_output": {"text": "I have detected an issue with your DNS settings"}, "correction_reason": "ASR misrecognized DNS as DSN."},
    {"id": FEEDBACK_IDS[4], "interaction_id": INTERACTION_IDS[12], "provided_by_user_id": USER_IDS[4], "feedback_type": "emotion_label", "ai_output": {"emotion": "neutral", "confidence": 0.72}, "corrected_output": {"emotion": "disappointed", "confidence": 0.80}, "correction_reason": "Customer was disappointed, not neutral."},
]

MANAGER_QUERIES = [
    {"id": QUERY_IDS[0], "user_id": USER_IDS[0], "organization_id": ORG_IDS[0], "query_text": "Show me all calls with angry customers this week", "query_mode": "chat", "ai_query_understanding": "Filter interactions by emotion_label=angry and date within current week", "sql_code": "SELECT i.id, i.interaction_date, u.emotion_label FROM interactions i JOIN utterances u ON i.id = u.interaction_id WHERE u.emotion_label = 'angry'", "response_text": "Found 2 calls with angry customers this week.", "retrieved_policy_id": None},
    {"id": QUERY_IDS[1], "user_id": USER_IDS[1], "organization_id": ORG_IDS[1], "query_text": "What is the average score for our support team?", "query_mode": "chat", "ai_query_understanding": "Calculate average overall_score for CairoConnect", "sql_code": "SELECT AVG(s.overall_score) FROM interaction_scores s JOIN interactions i ON s.interaction_id = i.id WHERE i.organization_id = 'a0000000-0000-0000-0000-000000000002'", "response_text": "The average overall score for CairoConnect Support is 82.3.", "retrieved_policy_id": None},
    {"id": QUERY_IDS[2], "user_id": USER_IDS[3], "organization_id": ORG_IDS[3], "query_text": "Which agents violated the escalation policy?", "query_mode": "voice", "ai_query_understanding": "Find agents with non-compliant escalation policy records", "sql_code": "SELECT a.agent_code FROM policy_compliance pc JOIN interactions i ON pc.interaction_id = i.id JOIN agents a ON i.agent_id = a.id WHERE pc.is_compliant = false", "response_text": "Agent DS-401 had 1 escalation policy violation.", "retrieved_policy_id": POLICY_IDS[3]},
    {"id": QUERY_IDS[3], "user_id": USER_IDS[2], "organization_id": ORG_IDS[2], "query_text": "How many calls were resolved this month?", "query_mode": "chat", "ai_query_understanding": "Count interactions where was_resolved=true", "sql_code": "SELECT COUNT(*) FROM interaction_scores WHERE was_resolved = true", "response_text": "12 out of 15 calls were resolved this month (80%).", "retrieved_policy_id": None},
    {"id": QUERY_IDS[4], "user_id": USER_IDS[4], "organization_id": ORG_IDS[4], "query_text": "Are our agents following the data privacy policy?", "query_mode": "chat", "ai_query_understanding": "Check compliance records for data privacy policy", "sql_code": "SELECT pc.is_compliant, pc.compliance_score FROM policy_compliance pc WHERE pc.policy_id = '20000000-0000-0000-0000-000000000005'", "response_text": "1 out of 1 checked interaction was fully compliant (92%).", "retrieved_policy_id": POLICY_IDS[4]},
]

# ===========================================================
# Insertion order (FK dependencies)
# ===========================================================

TABLES_IN_ORDER = [
    ("organizations", ORGANIZATIONS),
    ("users", USERS),
    ("agents", AGENTS),
    ("interactions", INTERACTIONS),
    ("transcripts", TRANSCRIPTS),
    ("utterances", UTTERANCES),
    ("emotion_events", EMOTION_EVENTS),
    ("interaction_scores", INTERACTION_SCORES),
    ("company_policies", COMPANY_POLICIES),
    ("policy_compliance", POLICY_COMPLIANCE),
    ("human_feedback", HUMAN_FEEDBACK),
    ("manager_queries", MANAGER_QUERIES),
]

# Reverse order for cleanup (delete children first)
TABLES_REVERSED = list(reversed([t[0] for t in TABLES_IN_ORDER]))


def seed():
    """Insert all sample data."""
    print("Seeding VocalMind database...")
    for table_name, rows in TABLES_IN_ORDER:
        result = supabase.table(table_name).insert(rows).execute()
        print(f"  ✓ {table_name}: {len(result.data)} rows")
    print(f"\nDone! Inserted {sum(len(r) for _, r in TABLES_IN_ORDER)} rows.")


def cleanup():
    """Delete all seeded data by known IDs."""
    print("Cleaning up seed data...")
    # Deleting organizations cascades to everything
    supabase.table("organizations").delete().in_(
        "id", ORG_IDS
    ).execute()
    print("  ✓ Deleted all seed data (cascaded from organizations)")


def main():
    parser = argparse.ArgumentParser(description="VocalMind DB Seeder")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove all seeded data",
    )
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
    else:
        seed()


if __name__ == "__main__":
    main()
