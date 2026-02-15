-- ============================================================
-- VocalMind Seed Data (110 rows across 12 tables)
-- ============================================================

-- Organizations (5)
INSERT INTO
    organizations (id, name, status)
VALUES (
        'a0000000-0000-0000-0000-000000000001',
        'NileTech',
        'active'
    ),
    (
        'a0000000-0000-0000-0000-000000000002',
        'CairoConnect',
        'active'
    ),
    (
        'a0000000-0000-0000-0000-000000000003',
        'PyramidSupport',
        'active'
    ),
    (
        'a0000000-0000-0000-0000-000000000004',
        'DeltaServices',
        'inactive'
    ),
    (
        'a0000000-0000-0000-0000-000000000005',
        'SphinxTelecom',
        'active'
    );

-- Users (5)
INSERT INTO
    users (
        id,
        organization_id,
        email,
        password_hash,
        name,
        role,
        is_active
    )
VALUES (
        'b0000000-0000-0000-0000-000000000001',
        'a0000000-0000-0000-0000-000000000001',
        'galal@niletech.com',
        '$2b$12$seedhashplaceholder001',
        'Galal',
        'admin',
        true
    ),
    (
        'b0000000-0000-0000-0000-000000000002',
        'a0000000-0000-0000-0000-000000000002',
        'ibrahem@cairoconnect.com',
        '$2b$12$seedhashplaceholder002',
        'Ibrahem',
        'manager',
        true
    ),
    (
        'b0000000-0000-0000-0000-000000000003',
        'a0000000-0000-0000-0000-000000000003',
        'mohamed@pyramidsupport.com',
        '$2b$12$seedhashplaceholder003',
        'Mohamed',
        'admin',
        true
    ),
    (
        'b0000000-0000-0000-0000-000000000004',
        'a0000000-0000-0000-0000-000000000004',
        'hassan@deltaservices.com',
        '$2b$12$seedhashplaceholder004',
        'Hassan',
        'manager',
        true
    ),
    (
        'b0000000-0000-0000-0000-000000000005',
        'a0000000-0000-0000-0000-000000000005',
        'ahmed@sphinxtelecom.com',
        '$2b$12$seedhashplaceholder005',
        'Ahmed',
        'manager',
        true
    );

-- Agents (5)
INSERT INTO
    agents (
        id,
        organization_id,
        agent_code,
        agent_type,
        department,
        is_active
    )
VALUES (
        'c0000000-0000-0000-0000-000000000001',
        'a0000000-0000-0000-0000-000000000001',
        'NT-101',
        'human',
        'Sales',
        true
    ),
    (
        'c0000000-0000-0000-0000-000000000002',
        'a0000000-0000-0000-0000-000000000002',
        'CC-201',
        'human',
        'Support',
        true
    ),
    (
        'c0000000-0000-0000-0000-000000000003',
        'a0000000-0000-0000-0000-000000000003',
        'PS-301',
        'bot',
        'Technical',
        true
    ),
    (
        'c0000000-0000-0000-0000-000000000004',
        'a0000000-0000-0000-0000-000000000004',
        'DS-401',
        'human',
        'Billing',
        true
    ),
    (
        'c0000000-0000-0000-0000-000000000005',
        'a0000000-0000-0000-0000-000000000005',
        'ST-501',
        'human',
        'Retention',
        true
    );

-- Interactions (15)
INSERT INTO
    interactions (
        id,
        organization_id,
        agent_id,
        audio_file_path,
        file_size_bytes,
        duration_seconds,
        file_format,
        interaction_date,
        processing_status,
        language_detected,
        has_overlap
    )
VALUES (
        'd0000000-0000-0000-0000-000000000001',
        'a0000000-0000-0000-0000-000000000001',
        'c0000000-0000-0000-0000-000000000001',
        '/audio/NT-101_call_001.wav',
        2457600,
        180,
        'wav',
        '2026-02-01 09:15:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000002',
        'a0000000-0000-0000-0000-000000000001',
        'c0000000-0000-0000-0000-000000000001',
        '/audio/NT-101_call_002.mp3',
        1843200,
        420,
        'mp3',
        '2026-02-02 14:30:00+02',
        'completed',
        'en',
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000003',
        'a0000000-0000-0000-0000-000000000001',
        'c0000000-0000-0000-0000-000000000001',
        '/audio/NT-101_call_003.wav',
        3072000,
        600,
        'wav',
        '2026-02-03 11:00:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000004',
        'a0000000-0000-0000-0000-000000000002',
        'c0000000-0000-0000-0000-000000000002',
        '/audio/CC-201_call_001.wav',
        1536000,
        240,
        'wav',
        '2026-02-01 10:00:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000005',
        'a0000000-0000-0000-0000-000000000002',
        'c0000000-0000-0000-0000-000000000002',
        '/audio/CC-201_call_002.mp3',
        921600,
        150,
        'mp3',
        '2026-02-04 16:45:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000006',
        'a0000000-0000-0000-0000-000000000002',
        'c0000000-0000-0000-0000-000000000002',
        '/audio/CC-201_call_003.wav',
        4096000,
        900,
        'wav',
        '2026-02-05 08:20:00+02',
        'processing',
        'en',
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000007',
        'a0000000-0000-0000-0000-000000000003',
        'c0000000-0000-0000-0000-000000000003',
        '/audio/PS-301_call_001.wav',
        768000,
        120,
        'wav',
        '2026-02-02 13:00:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000008',
        'a0000000-0000-0000-0000-000000000003',
        'c0000000-0000-0000-0000-000000000003',
        '/audio/PS-301_call_002.mp3',
        1228800,
        300,
        'mp3',
        '2026-02-03 15:30:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000009',
        'a0000000-0000-0000-0000-000000000003',
        'c0000000-0000-0000-0000-000000000003',
        '/audio/PS-301_call_003.wav',
        614400,
        90,
        'wav',
        '2026-02-06 09:45:00+02',
        'pending',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000010',
        'a0000000-0000-0000-0000-000000000004',
        'c0000000-0000-0000-0000-000000000004',
        '/audio/DS-401_call_001.wav',
        2048000,
        360,
        'wav',
        '2026-02-01 11:30:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000011',
        'a0000000-0000-0000-0000-000000000004',
        'c0000000-0000-0000-0000-000000000004',
        '/audio/DS-401_call_002.mp3',
        3686400,
        720,
        'mp3',
        '2026-02-04 09:00:00+02',
        'completed',
        'en',
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000012',
        'a0000000-0000-0000-0000-000000000004',
        'c0000000-0000-0000-0000-000000000004',
        '/audio/DS-401_call_003.wav',
        1024000,
        200,
        'wav',
        '2026-02-07 14:00:00+02',
        'failed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000013',
        'a0000000-0000-0000-0000-000000000005',
        'c0000000-0000-0000-0000-000000000005',
        '/audio/ST-501_call_001.wav',
        1843200,
        480,
        'wav',
        '2026-02-02 10:15:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000014',
        'a0000000-0000-0000-0000-000000000005',
        'c0000000-0000-0000-0000-000000000005',
        '/audio/ST-501_call_002.mp3',
        2560000,
        540,
        'mp3',
        '2026-02-05 13:00:00+02',
        'completed',
        'en',
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000015',
        'a0000000-0000-0000-0000-000000000005',
        'c0000000-0000-0000-0000-000000000005',
        '/audio/ST-501_call_003.wav',
        921600,
        160,
        'wav',
        '2026-02-08 08:00:00+02',
        'processing',
        'en',
        false
    );

-- Transcripts (15)
INSERT INTO
    transcripts (
        id,
        interaction_id,
        full_text,
        confidence_score
    )
VALUES (
        'e0000000-0000-0000-0000-000000000001',
        'd0000000-0000-0000-0000-000000000001',
        'Agent: Thank you for calling NileTech Sales, my name is Mohsen. How can I help you today? Customer: Hi Mohsen, I am interested in upgrading my current plan. Agent: Of course! Let me pull up your account. I can see you are on our Basic plan. We have a Premium option that includes unlimited data. Customer: That sounds great, how much would that cost? Agent: It would be 299 EGP per month. Customer: That works for me, let us go ahead with the upgrade. Agent: Perfect, I have processed the upgrade. Is there anything else I can help with? Customer: No, that is all. Thank you! Agent: Thank you for choosing NileTech. Have a great day!',
        0.94
    ),
    (
        'e0000000-0000-0000-0000-000000000002',
        'd0000000-0000-0000-0000-000000000002',
        'Agent: NileTech Sales, Mohsen speaking. Customer: Yes, I have been waiting for 20 minutes already! I want to cancel my subscription. Agent: I am sorry about the wait. Let me see what I can do for you. May I ask why you want to cancel? Customer: The service has been terrible. My internet keeps dropping every hour. Agent: I completely understand your frustration. Let me check if there is a technical issue on our end. Customer: I have already called three times about this! Agent: I see the previous tickets. Let me escalate this to our technical team right away and offer you a 50% discount for the next 3 months. Customer: Fine, but if it does not improve I am switching providers. Agent: I understand. I will personally follow up within 48 hours.',
        0.91
    ),
    (
        'e0000000-0000-0000-0000-000000000003',
        'd0000000-0000-0000-0000-000000000003',
        'Agent: Welcome to NileTech, this is Mohsen. How may I assist you? Customer: I need help understanding my bill. There are some charges I do not recognize. Agent: I would be happy to help you with that. Can you tell me which charges look unfamiliar? Customer: There is a charge for 150 EGP labeled Premium Add-on. I never signed up for any add-on. Agent: Let me investigate that for you. I can see it was added on January 15th. It appears it was added during a previous call. Customer: I did not authorize that. Agent: I understand. I will remove the charge and issue a full refund. You should see it within 3-5 business days. Customer: Thank you, I appreciate that. Agent: You are welcome. Is there anything else? Customer: No, that is all.',
        0.96
    ),
    (
        'e0000000-0000-0000-0000-000000000004',
        'd0000000-0000-0000-0000-000000000004',
        'Agent: CairoConnect Support, Galal here. How can I help? Customer: My email has not been working since yesterday. Agent: I am sorry to hear that. Let me check your account status. Can you provide your account number? Customer: It is CC-4421. Agent: Thank you. I can see there was a server migration last night that may have affected some accounts. Let me reset your email configuration. Customer: How long will it take? Agent: It should be working within the next 10 minutes. I will stay on the line if you would like to verify. Customer: Yes please. Agent: Great, try logging in now. Customer: It works! Thank you so much. Agent: Happy to help!',
        0.95
    ),
    (
        'e0000000-0000-0000-0000-000000000005',
        'd0000000-0000-0000-0000-000000000005',
        'Agent: CairoConnect, Galal speaking. Customer: Hi, I just want to check when my contract expires. Agent: Sure, let me look that up. Your current contract ends on March 15, 2026. Customer: Great, and what are my renewal options? Agent: You can renew at the same rate or upgrade to our business plan which gives you priority support. Customer: I will think about it. Thank you. Agent: No problem, feel free to call back anytime.',
        0.97
    ),
    (
        'e0000000-0000-0000-0000-000000000006',
        'd0000000-0000-0000-0000-000000000006',
        'Agent: CairoConnect Support, this is Galal. Customer: I am extremely upset. You charged my card twice for the same invoice! Agent: I sincerely apologize for that. Let me look into this immediately. Customer: This is the second time this has happened! Agent: I understand this is very frustrating. I can see the duplicate charge. I am initiating a refund right now. Customer: When will I get my money back? Agent: The refund will be processed within 24 hours. I am also adding a credit to your account as compensation. Customer: It should not have happened in the first place. Agent: You are absolutely right, and I apologize again. I am flagging this to our billing team to prevent it from happening again.',
        0.89
    ),
    (
        'e0000000-0000-0000-0000-000000000007',
        'd0000000-0000-0000-0000-000000000007',
        'Agent: Thank you for contacting PyramidSupport. I am an automated assistant. How can I help you today? Customer: My router keeps disconnecting. Agent: I can help troubleshoot that. Have you tried restarting your router? Customer: Yes, multiple times. Agent: Let me run a remote diagnostic on your connection. Please wait a moment. I have detected an issue with your DNS settings. I will push an automatic fix now. Customer: Okay, let me check. It seems to be working now. Agent: Great! Is there anything else you need help with? Customer: No, thanks.',
        0.93
    ),
    (
        'e0000000-0000-0000-0000-000000000008',
        'd0000000-0000-0000-0000-000000000008',
        'Agent: PyramidSupport automated service. How may I assist you? Customer: I need to change my Wi-Fi password. Agent: I can guide you through that. Please open your browser and navigate to 192.168.1.1. Customer: Done. Agent: Enter your admin credentials. The default username is admin and the password is on the sticker on your router. Customer: I am in. Agent: Navigate to Wireless Settings, then Security. You can change your password there. Customer: Got it, changed it. Thank you! Agent: You are welcome. For security, I recommend changing it every 90 days.',
        0.98
    ),
    (
        'e0000000-0000-0000-0000-000000000009',
        'd0000000-0000-0000-0000-000000000009',
        'Agent: PyramidSupport, automated assistant ready to help. Customer: I want to know about your fiber optic plans. Agent: We offer three fiber plans: Basic at 200 Mbps, Standard at 500 Mbps, and Premium at 1 Gbps. Customer: What is the price for Premium? Agent: The Premium plan is 599 EGP per month with a 12-month contract. Customer: I will consider it. Agent: Would you like me to transfer you to a sales representative for more details? Customer: No, that is fine for now. Thank you.',
        0.96
    ),
    (
        'e0000000-0000-0000-0000-000000000010',
        'd0000000-0000-0000-0000-000000000010',
        'Agent: DeltaServices Billing, Hassan speaking. How can I assist you? Customer: I am calling about a late payment notice. I already paid last week. Agent: Let me check your payment history. Can you provide the transaction reference? Customer: It is TXN-88234. Agent: I found it. The payment was received but not applied to your account due to an incorrect reference number. I will fix that now. Customer: So I do not owe anything? Agent: Correct, your account is now current. I apologize for the confusion. Customer: Thank you for sorting it out. Agent: You are welcome. Have a good day.',
        0.95
    ),
    (
        'e0000000-0000-0000-0000-000000000011',
        'd0000000-0000-0000-0000-000000000011',
        'Agent: DeltaServices, Hassan here. Customer: I want to dispute a charge on my account. I was charged 500 EGP for a service I never used. Agent: I understand your concern. Let me review the charge. I can see it is for our Premium Support package. Customer: I never signed up for that! This is outrageous! Agent: I can see it was added through our app on January 20th. Customer: Someone must have done it by accident. I want a refund immediately! Agent: I understand. However, I need to file a dispute form which takes 5-7 business days. Customer: That is unacceptable! I need this resolved today! Agent: Let me speak with my supervisor. Please hold. I have been authorized to issue an immediate refund. Customer: Finally, thank you. Agent: I apologize for the inconvenience.',
        0.88
    ),
    (
        'e0000000-0000-0000-0000-000000000012',
        'd0000000-0000-0000-0000-000000000012',
        'Agent: DeltaServices Billing, Hassan speaking. Customer: Hi, I just want to update my payment method. Agent: Sure, I can help with that. For security purposes I will need to verify your identity first. Can you confirm your date of birth? Customer: July 15, 1990. Agent: Thank you. What payment method would you like to add? Customer: A new credit card. Agent: Please provide the card number. Customer: 4532 actually, wait, can I do this online instead? Agent: Absolutely, you can update it through our app under Settings then Payment Methods. Customer: Perfect, I will do that. Thanks. Agent: You are welcome.',
        0.94
    ),
    (
        'e0000000-0000-0000-0000-000000000013',
        'd0000000-0000-0000-0000-000000000013',
        'Agent: SphinxTelecom, Ahmed from Retention. How can I help you today? Customer: I am thinking about canceling my service. Agent: I am sorry to hear that. May I ask what prompted this decision? Customer: I found a better deal with another provider. Agent: I understand. Can you tell me what they are offering? Customer: 500 Mbps for 199 EGP per month. Agent: That is competitive. Let me see what we can offer. I can match that price and add free installation for a new router. Customer: Really? That changes things. Agent: Absolutely. I value your loyalty as a customer for 3 years. Customer: Okay, I will stay then. Thank you! Agent: Wonderful! I will process the new rate immediately.',
        0.93
    ),
    (
        'e0000000-0000-0000-0000-000000000014',
        'd0000000-0000-0000-0000-000000000014',
        'Agent: SphinxTelecom Retention, Ahmed speaking. Customer: I have been having issues with my service quality. The speed test shows half of what I am paying for. Agent: I apologize for that. Let me run a diagnostic. I can see your line is performing below expected levels. There appears to be congestion in your area. Customer: So when will it be fixed? Agent: We have a network upgrade scheduled for your area next week. In the meantime, I can offer you a temporary speed boost at no extra charge. Customer: That would help. How long until the upgrade is done? Agent: The upgrade should be completed by February 15th. I will also credit your account for the affected period. Customer: That sounds fair. Thank you for being proactive about it. Agent: Of course, we want to make sure you are getting the service you are paying for.',
        0.92
    ),
    (
        'e0000000-0000-0000-0000-000000000015',
        'd0000000-0000-0000-0000-000000000015',
        'Agent: SphinxTelecom, Ahmed here. How can I help? Customer: I just moved to a new apartment and I need to transfer my service. Agent: Congratulations on the move! I can help with that. What is your new address? Customer: 45 El-Nour Street, Nasr City. Agent: Let me check coverage in that area. Great news, we have full coverage there. The transfer will take 2-3 business days. Customer: Will my plan stay the same? Agent: Yes, everything transfers as-is. We just need to schedule a technician visit. Customer: Can they come on Saturday? Agent: Let me check availability. Yes, we have a slot on Saturday between 10 AM and 12 PM. Customer: Perfect, book it. Agent: Done! You will receive a confirmation SMS shortly.',
        0.97
    );

-- Utterances (15)
INSERT INTO
    utterances (
        id,
        interaction_id,
        speaker_role,
        start_time_seconds,
        end_time_seconds,
        emotion_label,
        emotion_confidence,
        text
    )
VALUES (
        'f0000000-0000-0000-0000-000000000001',
        'd0000000-0000-0000-0000-000000000001',
        'customer',
        5.0,
        12.3,
        'neutral',
        0.88,
        'Hi Mohsen, I am interested in upgrading my current plan.'
    ),
    (
        'f0000000-0000-0000-0000-000000000002',
        'd0000000-0000-0000-0000-000000000002',
        'customer',
        3.2,
        15.7,
        'angry',
        0.92,
        'Yes, I have been waiting for 20 minutes already! I want to cancel my subscription.'
    ),
    (
        'f0000000-0000-0000-0000-000000000003',
        'd0000000-0000-0000-0000-000000000003',
        'customer',
        8.0,
        18.5,
        'frustrated',
        0.85,
        'My internet keeps dropping every hour. I have already called three times about this!'
    ),
    (
        'f0000000-0000-0000-0000-000000000004',
        'd0000000-0000-0000-0000-000000000004',
        'customer',
        4.1,
        10.0,
        'worried',
        0.78,
        'I received a notification about a rate increase. Can you explain what is changing?'
    ),
    (
        'f0000000-0000-0000-0000-000000000005',
        'd0000000-0000-0000-0000-000000000005',
        'customer',
        2.5,
        8.9,
        'neutral',
        0.91,
        'I just want to check the status of my recent order.'
    ),
    (
        'f0000000-0000-0000-0000-000000000006',
        'd0000000-0000-0000-0000-000000000006',
        'customer',
        3.0,
        14.2,
        'angry',
        0.95,
        'I have been overcharged for the third month in a row. This is completely unacceptable!'
    ),
    (
        'f0000000-0000-0000-0000-000000000007',
        'd0000000-0000-0000-0000-000000000007',
        'customer',
        6.0,
        11.4,
        'neutral',
        0.82,
        'Hi, I am calling to inquire about your business plans.'
    ),
    (
        'f0000000-0000-0000-0000-000000000008',
        'd0000000-0000-0000-0000-000000000008',
        'agent',
        10.5,
        22.0,
        'neutral',
        0.90,
        'Thank you for calling. I can help you with your account settings today.'
    ),
    (
        'f0000000-0000-0000-0000-000000000009',
        'd0000000-0000-0000-0000-000000000009',
        'customer',
        4.0,
        9.8,
        'neutral',
        0.87,
        'I need to update my billing address and payment method.'
    ),
    (
        'f0000000-0000-0000-0000-000000000010',
        'd0000000-0000-0000-0000-000000000010',
        'customer',
        5.5,
        16.0,
        'worried',
        0.80,
        'I noticed some unauthorized charges on my account. I am concerned about security.'
    ),
    (
        'f0000000-0000-0000-0000-000000000011',
        'd0000000-0000-0000-0000-000000000011',
        'customer',
        7.0,
        20.3,
        'angry',
        0.94,
        'Your service has been down all day! I am losing business because of this!'
    ),
    (
        'f0000000-0000-0000-0000-000000000012',
        'd0000000-0000-0000-0000-000000000012',
        'customer',
        3.8,
        10.5,
        'neutral',
        0.89,
        'Can you walk me through setting up the VPN on my device?'
    ),
    (
        'f0000000-0000-0000-0000-000000000013',
        'd0000000-0000-0000-0000-000000000013',
        'customer',
        4.5,
        12.0,
        'disappointed',
        0.76,
        'I was promised a callback within 24 hours, but nobody called me back.'
    ),
    (
        'f0000000-0000-0000-0000-000000000014',
        'd0000000-0000-0000-0000-000000000014',
        'customer',
        6.2,
        15.8,
        'frustrated',
        0.83,
        'The new interface is confusing. I cannot find any of the features I used to use.'
    ),
    (
        'f0000000-0000-0000-0000-000000000015',
        'd0000000-0000-0000-0000-000000000015',
        'customer',
        2.0,
        8.0,
        'happy',
        0.86,
        'Everything was resolved quickly. Thank you so much for your help!'
    );

-- Emotion Events (10)
INSERT INTO
    emotion_events (
        id,
        interaction_id,
        utterance_id,
        event_type,
        previous_emotion,
        new_emotion,
        emotion_delta,
        trigger_category,
        timestamp_seconds,
        speaker_role,
        verified_by_user_id
    )
VALUES (
        '10000000-0000-0000-0000-000000000001',
        'd0000000-0000-0000-0000-000000000002',
        'f0000000-0000-0000-0000-000000000002',
        'escalation',
        'frustrated',
        'angry',
        0.35,
        'Long Wait',
        15.7,
        'customer',
        NULL
    ),
    (
        '10000000-0000-0000-0000-000000000002',
        'd0000000-0000-0000-0000-000000000002',
        'f0000000-0000-0000-0000-000000000002',
        'de_escalation',
        'angry',
        'neutral',
        -0.40,
        'Empathy',
        180.0,
        'customer',
        'b0000000-0000-0000-0000-000000000001'
    ),
    (
        '10000000-0000-0000-0000-000000000003',
        'd0000000-0000-0000-0000-000000000003',
        'f0000000-0000-0000-0000-000000000003',
        'sentiment_drop',
        'neutral',
        'frustrated',
        0.30,
        'Billing Error',
        18.5,
        'customer',
        NULL
    ),
    (
        '10000000-0000-0000-0000-000000000004',
        'd0000000-0000-0000-0000-000000000006',
        'f0000000-0000-0000-0000-000000000006',
        'escalation',
        'frustrated',
        'angry',
        0.45,
        'Billing Error',
        14.2,
        'customer',
        NULL
    ),
    (
        '10000000-0000-0000-0000-000000000005',
        'd0000000-0000-0000-0000-000000000006',
        'f0000000-0000-0000-0000-000000000006',
        'de_escalation',
        'angry',
        'calm',
        -0.50,
        'Empathy',
        300.0,
        'customer',
        'b0000000-0000-0000-0000-000000000002'
    ),
    (
        '10000000-0000-0000-0000-000000000006',
        'd0000000-0000-0000-0000-000000000011',
        'f0000000-0000-0000-0000-000000000011',
        'escalation',
        'neutral',
        'angry',
        0.60,
        'Unauthorized Charge',
        20.3,
        'customer',
        NULL
    ),
    (
        '10000000-0000-0000-0000-000000000007',
        'd0000000-0000-0000-0000-000000000011',
        'f0000000-0000-0000-0000-000000000011',
        'emotion_shift',
        'angry',
        'relieved',
        -0.55,
        'Resolution',
        600.0,
        'customer',
        'b0000000-0000-0000-0000-000000000004'
    ),
    (
        '10000000-0000-0000-0000-000000000008',
        'd0000000-0000-0000-0000-000000000013',
        'f0000000-0000-0000-0000-000000000013',
        'emotion_shift',
        'disappointed',
        'happy',
        -0.40,
        'Counter Offer',
        300.0,
        'customer',
        NULL
    ),
    (
        '10000000-0000-0000-0000-000000000009',
        'd0000000-0000-0000-0000-000000000014',
        'f0000000-0000-0000-0000-000000000014',
        'sentiment_drop',
        'neutral',
        'frustrated',
        0.25,
        'Service Quality',
        15.8,
        'customer',
        NULL
    ),
    (
        '10000000-0000-0000-0000-000000000010',
        'd0000000-0000-0000-0000-000000000014',
        'f0000000-0000-0000-0000-000000000014',
        'de_escalation',
        'frustrated',
        'satisfied',
        -0.35,
        'Proactive Solution',
        400.0,
        'customer',
        'b0000000-0000-0000-0000-000000000005'
    );

-- Interaction Scores (15)
INSERT INTO
    interaction_scores (
        interaction_id,
        overall_score,
        policy_score,
        total_silence_duration_seconds,
        average_response_time_seconds,
        was_resolved
    )
VALUES (
        'd0000000-0000-0000-0000-000000000001',
        92.5,
        95.0,
        8.0,
        2.1,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000002',
        68.0,
        60.0,
        25.0,
        5.8,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000003',
        88.0,
        90.0,
        12.0,
        3.2,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000004',
        95.0,
        98.0,
        5.0,
        1.5,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000005',
        90.0,
        92.0,
        3.0,
        1.8,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000006',
        62.0,
        55.0,
        30.0,
        6.5,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000007',
        85.0,
        88.0,
        6.0,
        2.0,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000008',
        91.0,
        93.0,
        4.0,
        1.2,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000009',
        78.0,
        80.0,
        10.0,
        3.5,
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000010',
        87.0,
        85.0,
        14.0,
        3.0,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000011',
        60.0,
        50.0,
        45.0,
        8.2,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000012',
        82.0,
        88.0,
        8.0,
        2.5,
        false
    ),
    (
        'd0000000-0000-0000-0000-000000000013',
        94.0,
        96.0,
        5.0,
        1.6,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000014',
        80.0,
        78.0,
        18.0,
        4.0,
        true
    ),
    (
        'd0000000-0000-0000-0000-000000000015',
        93.0,
        95.0,
        4.0,
        1.4,
        true
    );

-- Company Policies (5)
INSERT INTO
    company_policies (
        id,
        organization_id,
        policy_code,
        category,
        policy_text,
        pinecone_id
    )
VALUES (
        '20000000-0000-0000-0000-000000000001',
        'a0000000-0000-0000-0000-000000000001',
        'POL-GREET-01',
        'Communication',
        'Agents must greet the customer by name within the first 15 seconds of the call and introduce themselves with their full name and department.',
        'vec_pol_001'
    ),
    (
        '20000000-0000-0000-0000-000000000002',
        'a0000000-0000-0000-0000-000000000002',
        'POL-HOLD-01',
        'Service Level',
        'Customers must not be placed on hold for more than 2 minutes without an update. Total hold time per call must not exceed 5 minutes.',
        'vec_pol_002'
    ),
    (
        '20000000-0000-0000-0000-000000000003',
        'a0000000-0000-0000-0000-000000000003',
        'POL-EMPATH-01',
        'Communication',
        'Agents must acknowledge customer emotions and use empathetic language such as I understand and I apologize when the customer expresses frustration or dissatisfaction.',
        'vec_pol_003'
    ),
    (
        '20000000-0000-0000-0000-000000000004',
        'a0000000-0000-0000-0000-000000000004',
        'POL-ESCAL-01',
        'Escalation',
        'If a customer requests to speak with a supervisor or the issue cannot be resolved within 10 minutes, the agent must escalate the call to a senior representative immediately.',
        'vec_pol_004'
    ),
    (
        '20000000-0000-0000-0000-000000000005',
        'a0000000-0000-0000-0000-000000000005',
        'POL-PRIV-01',
        'Data Privacy',
        'Agents must verify customer identity using at least two data points before disclosing any account information. Credit card numbers must never be read back in full.',
        'vec_pol_005'
    );

-- Policy Compliance (10)
INSERT INTO
    policy_compliance (
        id,
        interaction_id,
        policy_id,
        is_compliant,
        compliance_score,
        violation_severity,
        confidence_score,
        analyzed_by_model,
        trigger_description,
        evidence_text,
        llm_reasoning,
        is_human_verified,
        human_feedback_text
    )
VALUES (
        '30000000-0000-0000-0000-000000000001',
        'd0000000-0000-0000-0000-000000000001',
        '20000000-0000-0000-0000-000000000001',
        true,
        95.0,
        NULL,
        0.93,
        'llama3.1-70b',
        'Agent greeted customer by name within 10 seconds',
        'Thank you for calling NileTech Sales, my name is Mohsen. How can I help you today?',
        'The agent introduced themselves with their name and department (Sales) within the first sentence, meeting the 15-second greeting requirement.',
        true,
        NULL
    ),
    (
        '30000000-0000-0000-0000-000000000002',
        'd0000000-0000-0000-0000-000000000002',
        '20000000-0000-0000-0000-000000000001',
        false,
        40.0,
        'minor',
        0.88,
        'llama3.1-70b',
        'Agent did not greet customer by name',
        'NileTech Sales, Mohsen speaking.',
        'The agent introduced themselves but did not greet the customer by name. The greeting was abbreviated.',
        false,
        NULL
    ),
    (
        '30000000-0000-0000-0000-000000000003',
        'd0000000-0000-0000-0000-000000000004',
        '20000000-0000-0000-0000-000000000002',
        true,
        90.0,
        NULL,
        0.91,
        'llama3.1-70b',
        'Hold time within acceptable limits',
        'I will stay on the line if you would like to verify.',
        'No hold was used. The agent offered to stay on the line while the customer tested.',
        true,
        NULL
    ),
    (
        '30000000-0000-0000-0000-000000000004',
        'd0000000-0000-0000-0000-000000000006',
        '20000000-0000-0000-0000-000000000002',
        false,
        30.0,
        'critical',
        0.95,
        'llama3.1-70b',
        'Excessive customer wait before agent response',
        'I am extremely upset. You charged my card twice!',
        'The interaction showed signs of long processing delays before resolution.',
        true,
        'Confirmed: customer experienced extended wait'
    ),
    (
        '30000000-0000-0000-0000-000000000005',
        'd0000000-0000-0000-0000-000000000006',
        '20000000-0000-0000-0000-000000000003',
        true,
        85.0,
        NULL,
        0.90,
        'llama3.1-70b',
        'Agent used empathetic language during escalation',
        'I sincerely apologize for that. Let me look into this immediately.',
        'The agent used empathetic phrases like sincerely apologize and immediately.',
        false,
        NULL
    ),
    (
        '30000000-0000-0000-0000-000000000006',
        'd0000000-0000-0000-0000-000000000011',
        '20000000-0000-0000-0000-000000000004',
        false,
        35.0,
        'critical',
        0.92,
        'llama3.1-70b',
        'Escalation delayed beyond 10-minute threshold',
        'That is unacceptable! I need this resolved today!',
        'The agent initially tried to process a 5-7 day dispute form instead of immediately escalating.',
        true,
        'Agent should have escalated sooner'
    ),
    (
        '30000000-0000-0000-0000-000000000007',
        'd0000000-0000-0000-0000-000000000012',
        '20000000-0000-0000-0000-000000000005',
        true,
        92.0,
        NULL,
        0.96,
        'llama3.1-70b',
        'Identity verification performed correctly',
        'For security purposes I will need to verify your identity first. Can you confirm your date of birth?',
        'The agent correctly requested identity verification before proceeding.',
        true,
        NULL
    ),
    (
        '30000000-0000-0000-0000-000000000008',
        'd0000000-0000-0000-0000-000000000013',
        '20000000-0000-0000-0000-000000000003',
        true,
        98.0,
        NULL,
        0.97,
        'llama3.1-70b',
        'Excellent empathy and retention approach',
        'I understand. Can you tell me what they are offering?',
        'The agent showed strong empathy and offered a competitive counter-offer.',
        false,
        NULL
    ),
    (
        '30000000-0000-0000-0000-000000000009',
        'd0000000-0000-0000-0000-000000000014',
        '20000000-0000-0000-0000-000000000003',
        true,
        88.0,
        NULL,
        0.89,
        'llama3.1-70b',
        'Proactive empathy with service quality issue',
        'I apologize for that. Let me run a diagnostic.',
        'The agent immediately acknowledged the service issue and took proactive steps.',
        false,
        NULL
    ),
    (
        '30000000-0000-0000-0000-000000000010',
        'd0000000-0000-0000-0000-000000000003',
        '20000000-0000-0000-0000-000000000003',
        true,
        90.0,
        NULL,
        0.91,
        'llama3.1-70b',
        'Agent acknowledged billing frustration empathetically',
        'I understand. I will remove the charge and issue a full refund.',
        'The agent used empathetic language and took immediate corrective action.',
        true,
        NULL
    );

-- Human Feedback (5)
INSERT INTO
    human_feedback (
        id,
        interaction_id,
        provided_by_user_id,
        feedback_type,
        ai_output,
        corrected_output,
        correction_reason
    )
VALUES (
        '40000000-0000-0000-0000-000000000001',
        'd0000000-0000-0000-0000-000000000002',
        'b0000000-0000-0000-0000-000000000001',
        'emotion_label',
        '{"emotion": "frustrated", "confidence": 0.78}',
        '{"emotion": "angry", "confidence": 0.95}',
        'Customer was clearly angry, not just frustrated.'
    ),
    (
        '40000000-0000-0000-0000-000000000002',
        'd0000000-0000-0000-0000-000000000006',
        'b0000000-0000-0000-0000-000000000002',
        'score',
        '{"overall_score": 75.0}',
        '{"overall_score": 62.0}',
        'The duplicate billing error should lower the score more.'
    ),
    (
        '40000000-0000-0000-0000-000000000003',
        'd0000000-0000-0000-0000-000000000011',
        'b0000000-0000-0000-0000-000000000004',
        'compliance',
        '{"is_compliant": true, "policy": "POL-ESCAL-01"}',
        '{"is_compliant": false, "policy": "POL-ESCAL-01"}',
        'The agent delayed escalation.'
    ),
    (
        '40000000-0000-0000-0000-000000000004',
        'd0000000-0000-0000-0000-000000000007',
        'b0000000-0000-0000-0000-000000000003',
        'transcription',
        '{"text": "I have detected an issue with your DSN settings"}',
        '{"text": "I have detected an issue with your DNS settings"}',
        'ASR misrecognized DNS as DSN.'
    ),
    (
        '40000000-0000-0000-0000-000000000005',
        'd0000000-0000-0000-0000-000000000013',
        'b0000000-0000-0000-0000-000000000005',
        'emotion_label',
        '{"text": "neutral", "confidence": 0.72}',
        '{"text": "disappointed", "confidence": 0.80}',
        'Customer was disappointed, not neutral.'
    );

-- Manager Queries (5)
INSERT INTO
    manager_queries (
        id,
        user_id,
        organization_id,
        query_text,
        query_mode,
        ai_query_understanding,
        sql_code,
        response_text,
        retrieved_policy_id
    )
VALUES (
        '50000000-0000-0000-0000-000000000001',
        'b0000000-0000-0000-0000-000000000001',
        'a0000000-0000-0000-0000-000000000001',
        'Show me all calls with angry customers this week',
        'chat',
        'Filter interactions by emotion_label=angry and date within current week',
        'SELECT i.id, i.interaction_date, u.emotion_label FROM interactions i JOIN utterances u ON i.id = u.interaction_id WHERE u.emotion_label = ''angry''',
        'Found 2 calls with angry customers this week.',
        NULL
    ),
    (
        '50000000-0000-0000-0000-000000000002',
        'b0000000-0000-0000-0000-000000000002',
        'a0000000-0000-0000-0000-000000000002',
        'What is the average score for our support team?',
        'chat',
        'Calculate average overall_score for CairoConnect',
        'SELECT AVG(s.overall_score) FROM interaction_scores s JOIN interactions i ON s.interaction_id = i.id WHERE i.organization_id = ''a0000000-0000-0000-0000-000000000002''',
        'The average overall score for CairoConnect Support is 82.3.',
        NULL
    ),
    (
        '50000000-0000-0000-0000-000000000003',
        'b0000000-0000-0000-0000-000000000004',
        'a0000000-0000-0000-0000-000000000004',
        'Which agents violated the escalation policy?',
        'voice',
        'Find agents with non-compliant escalation policy records',
        'SELECT a.agent_code FROM policy_compliance pc JOIN interactions i ON pc.interaction_id = i.id JOIN agents a ON i.agent_id = a.id WHERE pc.is_compliant = false',
        'Agent DS-401 had 1 escalation policy violation.',
        '20000000-0000-0000-0000-000000000004'
    ),
    (
        '50000000-0000-0000-0000-000000000004',
        'b0000000-0000-0000-0000-000000000003',
        'a0000000-0000-0000-0000-000000000003',
        'How many calls were resolved this month?',
        'chat',
        'Count interactions where was_resolved=true',
        'SELECT COUNT(*) FROM interaction_scores WHERE was_resolved = true',
        '12 out of 15 calls were resolved this month (80%).',
        NULL
    ),
    (
        '50000000-0000-0000-0000-000000000005',
        'b0000000-0000-0000-0000-000000000005',
        'a0000000-0000-0000-0000-000000000005',
        'Are our agents following the data privacy policy?',
        'chat',
        'Check compliance records for data privacy policy',
        'SELECT pc.is_compliant, pc.compliance_score FROM policy_compliance pc WHERE pc.policy_id = ''20000000-0000-0000-0000-000000000005''',
        '1 out of 1 checked interaction was fully compliant (92%).',
        '20000000-0000-0000-0000-000000000005'
    );