-- ============================================================
-- VocalMind Schema v4.0 (PostgreSQL / Supabase)
-- 12 tables  |  Dropped: agents, human_feedback, rag_chunks, model_training_runs
-- Added: organization_policies  |  Split feedback into emotion_feedback + compliance_feedback
-- Skipped: knowledge_base_articles, organization_kb_articles (not needed)
-- ============================================================

-- 0. EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- for gen_random_uuid()

-- 1. ENUM TYPES
-- Drop old enums that no longer exist in v4
DO $$ BEGIN
    DROP TYPE IF EXISTS agent_type_enum CASCADE;
    DROP TYPE IF EXISTS event_type_enum CASCADE;
    DROP TYPE IF EXISTS violation_severity_enum CASCADE;
    DROP TYPE IF EXISTS feedback_type_enum CASCADE;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

CREATE TYPE org_status_enum   AS ENUM ('active', 'inactive', 'suspended');
CREATE TYPE user_role_enum    AS ENUM ('admin', 'manager', 'agent', 'ai_agent');
CREATE TYPE agent_type_enum   AS ENUM ('human', 'ai');
CREATE TYPE processing_status_enum AS ENUM ('pending', 'processing', 'completed', 'failed');
CREATE TYPE speaker_role_enum AS ENUM ('agent', 'customer');
CREATE TYPE query_mode_enum   AS ENUM ('voice', 'chat');


-- ============================================================
-- 2. TABLES
-- ============================================================

-- 1. organizations
CREATE TABLE organizations (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(255) NOT NULL,
    status     org_status_enum NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. users  (merged from old users + agents)
CREATE TABLE users (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email           VARCHAR(320) NOT NULL UNIQUE,
    password_hash   TEXT         NOT NULL,
    name            VARCHAR(255) NOT NULL,
    role            user_role_enum NOT NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    last_login_at   TIMESTAMPTZ  NULL,
    agent_type      agent_type_enum NULL   -- NULL for admin/manager roles
);

-- 3. interactions
CREATE TABLE interactions (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id   UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    agent_id          UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    audio_file_path   TEXT        NOT NULL,
    file_size_bytes   BIGINT      NOT NULL,
    duration_seconds  INTEGER     NOT NULL,
    file_format       VARCHAR(10) NOT NULL,
    interaction_date  TIMESTAMPTZ NOT NULL,
    processing_status processing_status_enum NOT NULL DEFAULT 'pending',
    language_detected VARCHAR(10) NULL,
    has_overlap       BOOLEAN     NOT NULL DEFAULT FALSE,
    channel_count     SMALLINT    NOT NULL DEFAULT 1
);

-- 4. transcripts  (1:1 with interactions)
CREATE TABLE transcripts (
    id                  UUID   PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id      UUID   NOT NULL UNIQUE REFERENCES interactions(id) ON DELETE CASCADE,
    full_text           TEXT   NULL,
    overall_confidence  FLOAT  NULL
);

-- 5. utterances
CREATE TABLE utterances (
    id                  UUID   PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id      UUID   NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    transcript_id       UUID   NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    speaker_role        speaker_role_enum NOT NULL,
    user_id             UUID   NULL REFERENCES users(id) ON DELETE SET NULL,
    sequence_index      INTEGER NOT NULL,
    start_time_seconds  FLOAT  NOT NULL,
    end_time_seconds    FLOAT  NOT NULL,
    text                TEXT   NOT NULL,
    emotion             VARCHAR(50) NULL,
    emotion_confidence  FLOAT  NULL
);

-- 6. emotion_events  (AI-generated)
CREATE TABLE emotion_events (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id    UUID        NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    utterance_id      UUID        NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    previous_emotion  VARCHAR(50) NULL,
    new_emotion       VARCHAR(50) NOT NULL,
    emotion_delta     FLOAT       NULL,
    speaker_role      speaker_role_enum NOT NULL,
    llm_justification TEXT        NULL,
    timestamp_seconds FLOAT       NOT NULL,
    confidence_score  FLOAT       NULL
);

-- 7. emotion_feedback
CREATE TABLE emotion_feedback (
    id                      UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    emotion_event_id        UUID    NOT NULL REFERENCES emotion_events(id) ON DELETE CASCADE,
    provided_by_user_id     UUID    NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    llm_justification       TEXT    NULL,
    corrected_emotion       VARCHAR(50) NOT NULL,
    corrected_justification TEXT    NULL,
    correction_reason       TEXT    NULL,
    is_used_in_training     BOOLEAN NOT NULL DEFAULT FALSE
);

-- 8. company_policies
CREATE TABLE company_policies (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_category VARCHAR(100) NOT NULL,
    policy_title    VARCHAR(255) NOT NULL,
    policy_text     TEXT         NOT NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 9. organization_policies  (junction table)
CREATE TABLE organization_policies (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    policy_id       UUID        NOT NULL REFERENCES company_policies(id) ON DELETE CASCADE,
    assigned_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    assigned_by     UUID        NULL REFERENCES users(id) ON DELETE SET NULL
);

-- 10. policy_compliance
CREATE TABLE policy_compliance (
    id                    UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id        UUID    NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    policy_id             UUID    NOT NULL REFERENCES company_policies(id) ON DELETE CASCADE,
    is_compliant          BOOLEAN NOT NULL,
    compliance_score      FLOAT   NOT NULL,
    llm_reasoning         TEXT    NULL,
    evidence_text         TEXT    NULL,
    retrieved_policy_text TEXT    NULL
);

-- 11. compliance_feedback
CREATE TABLE compliance_feedback (
    id                      UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_compliance_id    UUID    NOT NULL REFERENCES policy_compliance(id) ON DELETE CASCADE,
    provided_by_user_id     UUID    NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_is_compliant   BOOLEAN NOT NULL,
    corrected_is_compliant  BOOLEAN NOT NULL,
    original_score          FLOAT   NULL,
    corrected_score         FLOAT   NULL,
    correction_reason       TEXT    NULL,
    is_used_in_training     BOOLEAN NOT NULL DEFAULT FALSE
);

-- 12. interaction_scores  (1:1 with interactions)
CREATE TABLE interaction_scores (
    id                            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id                UUID        NOT NULL UNIQUE REFERENCES interactions(id) ON DELETE CASCADE,
    overall_score                 FLOAT       NULL,
    empathy_score                 FLOAT       NULL,
    policy_score                  FLOAT       NULL,
    resolution_score              FLOAT       NULL,
    total_silence_duration_seconds FLOAT      NULL,
    average_response_time_seconds  FLOAT      NULL,
    was_resolved                  BOOLEAN     NULL,
    scored_at                     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 13. manager_queries
CREATE TABLE manager_queries (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id   UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    query_text        TEXT        NOT NULL,
    query_mode        query_mode_enum NOT NULL,
    ai_understanding  TEXT        NULL,
    generated_sql     TEXT        NULL,
    response_text     TEXT        NULL,
    execution_time_ms INTEGER     NULL
);


-- ============================================================
-- 3. INDEXES
-- ============================================================
CREATE INDEX idx_users_organization_id            ON users(organization_id);
CREATE INDEX idx_interactions_organization_id      ON interactions(organization_id);
CREATE INDEX idx_interactions_agent_id             ON interactions(agent_id);
CREATE INDEX idx_interactions_date                 ON interactions(interaction_date);
CREATE INDEX idx_transcripts_interaction_id        ON transcripts(interaction_id);
CREATE INDEX idx_utterances_interaction_id         ON utterances(interaction_id);
CREATE INDEX idx_utterances_transcript_id          ON utterances(transcript_id);
CREATE INDEX idx_emotion_events_interaction_id     ON emotion_events(interaction_id);
CREATE INDEX idx_emotion_events_utterance_id       ON emotion_events(utterance_id);
CREATE INDEX idx_emotion_feedback_event_id         ON emotion_feedback(emotion_event_id);
CREATE INDEX idx_compliance_feedback_compliance_id ON compliance_feedback(policy_compliance_id);
CREATE INDEX idx_organization_policies_org_id      ON organization_policies(organization_id);
CREATE INDEX idx_organization_policies_policy_id   ON organization_policies(policy_id);
CREATE INDEX idx_policy_compliance_interaction_id  ON policy_compliance(interaction_id);
CREATE INDEX idx_policy_compliance_policy_id       ON policy_compliance(policy_id);
CREATE INDEX idx_manager_queries_user_id           ON manager_queries(user_id);
CREATE INDEX idx_manager_queries_org_id            ON manager_queries(organization_id);


-- ============================================================
-- 4. AUTO-UPDATE TRIGGER for updated_at columns
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_organizations_updated_at
    BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_company_policies_updated_at
    BEFORE UPDATE ON company_policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();