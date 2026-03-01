-- ============================================================
-- VocalMind Seed Data (v5.2)
-- Matches 16-table Schema
-- ============================================================

-- 1. Organizations
INSERT INTO organizations (id, name, slug, status) VALUES
  ('a0000000-0000-0000-0000-000000000001', 'NileTech', 'nile-tech', 'active'),
  ('a0000000-0000-0000-0000-000000000002', 'CairoConnect', 'cairo-connect', 'active');

-- 2. Users (Managers and Agents)
-- Passwords are set to 'password' hash using $2b$12$seedhashplaceholder...
INSERT INTO users (id, organization_id, email, password_hash, name, role, agent_type) VALUES
  ('b0000000-0000-0000-0000-000000000001', 'a0000000-0000-0000-0000-000000000001', 'manager@niletech.com', '$2b$12$seedhashplaceholder', 'Galal Manager', 'manager', NULL),
  ('b0000000-0000-0000-0000-000000000002', 'a0000000-0000-0000-0000-000000000001', 'agent@niletech.com', '$2b$12$seedhashplaceholder', 'Mohsen Agent', 'agent', 'human'),
  ('b0000000-0000-0000-0000-000000000003', 'a0000000-0000-0000-0000-000000000002', 'manager@cairoconnect.com', '$2b$12$seedhashplaceholder', 'Ibrahem Manager', 'manager', NULL);

-- 3. Interactions
INSERT INTO interactions (id, organization_id, agent_id, uploaded_by, audio_file_path, file_size_bytes, duration_seconds, file_format, interaction_date, processing_status) VALUES
  ('d0000000-0000-0000-0000-000000000001', 'a0000000-0000-0000-0000-000000000001', 'b0000000-0000-0000-0000-000000000002', 'b0000000-0000-0000-0000-000000000001', '/audio/call_001.wav', 2457600, 180, 'wav', '2026-02-01 09:15:00Z', 'completed');

-- 4. Processing Jobs
INSERT INTO processing_jobs (interaction_id, stage, status, completed_at) VALUES
  ('d0000000-0000-0000-0000-000000000001', 'diarization', 'completed', now()),
  ('d0000000-0000-0000-0000-000000000001', 'stt', 'completed', now()),
  ('d0000000-0000-0000-0000-000000000001', 'emotion', 'completed', now()),
  ('d0000000-0000-0000-0000-000000000001', 'scoring', 'completed', now());

-- 5. Transcripts
INSERT INTO transcripts (id, interaction_id, full_text, overall_confidence) VALUES
  ('e0000000-0000-0000-0000-000000000001', 'd0000000-0000-0000-0000-000000000001', 'Agent: Hello. Customer: Hi, I need help.', 0.95);

-- 6. Utterances
INSERT INTO utterances (id, interaction_id, transcript_id, speaker_role, user_id, sequence_index, start_time_seconds, end_time_seconds, text, emotion) VALUES
  ('f0000000-0000-0000-0000-000000000001', 'd0000000-0000-0000-0000-000000000001', 'e0000000-0000-0000-0000-000000000001', 'agent', 'b0000000-0000-0000-0000-000000000002', 0, 0.0, 2.0, 'Hello.', 'neutral'),
  ('f0000000-0000-0000-0000-000000000002', 'd0000000-0000-0000-0000-000000000001', 'e0000000-0000-0000-0000-000000000001', 'customer', NULL, 1, 2.5, 4.5, 'Hi, I need help.', 'frustrated');

-- 7. Emotion Events (with dispute fields)
INSERT INTO emotion_events (id, interaction_id, utterance_id, previous_emotion, new_emotion, emotion_delta, speaker_role, llm_justification, jump_to_seconds, is_flagged) VALUES
  ('11000000-0000-0000-0000-000000000001', 'd0000000-0000-0000-0000-000000000001', 'f0000000-0000-0000-0000-000000000002', 'neutral', 'frustrated', 0.5, 'customer', 'Customer expressed dissatisfaction.', 2.5, false);

-- 8. Interaction Scores
INSERT INTO interaction_scores (interaction_id, overall_score, empathy_score, policy_score, resolution_score, was_resolved, total_silence_seconds, avg_response_time_seconds) VALUES
  ('d0000000-0000-0000-0000-000000000001', 8.5, 9.0, 10.0, 8.0, true, 5.0, 2.5);

-- 9. Company Policies
INSERT INTO company_policies (id, policy_title, policy_category, policy_text) VALUES
  ('20000000-0000-0000-0000-000000000001', 'Greeting Policy', 'Communication', 'Agents must greet customers warmly and professionally.');

-- 10. Organization Policies
INSERT INTO organization_policies (organization_id, policy_id) VALUES
  ('a0000000-0000-0000-0000-000000000001', '20000000-0000-0000-0000-000000000001');

-- 11. FAQ Articles
INSERT INTO faq_articles (id, question, answer, category) VALUES
  ('f0000000-0000-0000-0000-000000000001', 'How to reset password?', 'Go to settings and click reset.', 'Account');

-- 12. Organization FAQ Articles
INSERT INTO organization_faq_articles (organization_id, article_id) VALUES
  ('a0000000-0000-0000-0000-000000000001', 'f0000000-0000-0000-0000-000000000001');

-- 13. Assistant Queries
INSERT INTO assistant_queries (id, user_id, organization_id, query_mode, query_text, response_text) VALUES
  ('33000000-0000-0000-0000-000000000001', 'b0000000-0000-0000-0000-000000000001', 'a0000000-0000-0000-0000-000000000001', 'chat', 'How many calls today?', 'There was 1 call handled today.');