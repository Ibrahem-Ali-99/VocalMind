-- ============================================================
-- VocalMind Schema v5.2 (PostgreSQL / Supabase)
-- 16 Domain Tables + 1 Audit Table
-- Feature: Agent-Initiated Dispute Workflow
-- ============================================================

-- 0. EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. ENUM TYPES
DO $$ BEGIN
    DROP TYPE IF EXISTS org_status_enum CASCADE;
    DROP TYPE IF EXISTS user_role_enum CASCADE;
    DROP TYPE IF EXISTS agent_type_enum CASCADE;
    DROP TYPE IF EXISTS processing_status_enum CASCADE;
    DROP TYPE IF EXISTS job_stage_enum CASCADE;
    DROP TYPE IF EXISTS job_status_enum CASCADE;
    DROP TYPE IF EXISTS speaker_role_enum CASCADE;
    DROP TYPE IF EXISTS query_mode_enum CASCADE;
    DROP TYPE IF EXISTS feedback_status_enum CASCADE;
    DROP TYPE IF EXISTS period_type_enum CASCADE;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

CREATE TYPE org_status_enum   AS ENUM ('active', 'inactive', 'suspended');
CREATE TYPE user_role_enum    AS ENUM ('manager', 'agent');
CREATE TYPE agent_type_enum   AS ENUM ('human', 'ai');
CREATE TYPE processing_status_enum AS ENUM ('pending', 'processing', 'completed', 'failed');
CREATE TYPE job_stage_enum    AS ENUM ('diarization', 'stt', 'emotion', 'reasoning', 'scoring', 'rag_eval');
CREATE TYPE job_status_enum   AS ENUM ('pending', 'running', 'completed', 'failed');
CREATE TYPE speaker_role_enum AS ENUM ('agent', 'customer');
CREATE TYPE query_mode_enum   AS ENUM ('voice', 'chat');
CREATE TYPE feedback_status_enum AS ENUM ('pending', 'reviewed', 'applied');
CREATE TYPE period_type_enum   AS ENUM ('daily', 'weekly', 'monthly');


-- ============================================================
-- 2. TABLES
-- ============================================================

DROP TABLE IF EXISTS assistant_queries CASCADE;
DROP TABLE IF EXISTS agent_performance_snapshots CASCADE;
DROP TABLE IF EXISTS organization_faq_articles CASCADE;
DROP TABLE IF EXISTS faq_articles CASCADE;
DROP TABLE IF EXISTS organization_policies CASCADE;
DROP TABLE IF EXISTS company_policies CASCADE;
DROP TABLE IF EXISTS compliance_feedback CASCADE;
DROP TABLE IF EXISTS policy_compliance CASCADE;
DROP TABLE IF EXISTS interaction_scores CASCADE;
DROP TABLE IF EXISTS emotion_feedback CASCADE;
DROP TABLE IF EXISTS emotion_events CASCADE;
DROP TABLE IF EXISTS utterances CASCADE;
DROP TABLE IF EXISTS transcripts CASCADE;
DROP TABLE IF EXISTS processing_jobs CASCADE;
DROP TABLE IF EXISTS interactions CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS organizations CASCADE;

-- 1. organizations
CREATE TABLE organizations (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(255) NOT NULL,
    slug       VARCHAR(100) NOT NULL UNIQUE,
    status     org_status_enum NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. users
CREATE TABLE users (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email           VARCHAR(320) NOT NULL UNIQUE,
    password_hash   TEXT         NOT NULL,
    name            VARCHAR(255) NOT NULL,
    role            user_role_enum NOT NULL,
    agent_type      agent_type_enum NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    last_login_at   TIMESTAMPTZ  NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT users_agent_type_role_check CHECK (agent_type IS NULL OR role = 'agent')
);

-- 3. company_policies
CREATE TABLE company_policies (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_title    VARCHAR(255) NOT NULL,
    policy_category VARCHAR(100) NOT NULL,
    policy_text     TEXT         NOT NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 4. faq_articles
CREATE TABLE faq_articles (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    question    TEXT         NOT NULL,
    answer      TEXT         NOT NULL,
    category    VARCHAR(100) NOT NULL,
    is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 5. interactions
CREATE TABLE interactions (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id   UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    agent_id          UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    uploaded_by       UUID        NOT NULL REFERENCES users(id),
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

-- 6. processing_jobs
CREATE TABLE processing_jobs (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id  UUID         NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    stage           job_stage_enum   NOT NULL,
    status          job_status_enum  NOT NULL DEFAULT 'pending',
    started_at      TIMESTAMPTZ  NULL,
    completed_at    TIMESTAMPTZ  NULL,
    error_message   TEXT         NULL,
    retry_count     SMALLINT     NOT NULL DEFAULT 0
);

-- 7. transcripts
CREATE TABLE transcripts (
    id                  UUID   PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id      UUID   NOT NULL UNIQUE REFERENCES interactions(id) ON DELETE CASCADE,
    full_text           TEXT   NULL,
    overall_confidence  FLOAT  NULL
);

-- 8. utterances
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

-- 9. emotion_events
CREATE TABLE emotion_events (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id    UUID        NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    utterance_id      UUID        NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    previous_emotion  VARCHAR(50) NULL,
    new_emotion       VARCHAR(50) NOT NULL,
    emotion_delta     FLOAT       NULL,
    speaker_role      speaker_role_enum NOT NULL,
    llm_justification TEXT        NULL,
    jump_to_seconds   FLOAT       NOT NULL,
    confidence_score  FLOAT       NULL,
    is_flagged       BOOLEAN     NOT NULL DEFAULT FALSE,
    agent_flagged_by UUID        NULL REFERENCES users(id) ON DELETE SET NULL,
    agent_flagged_at TIMESTAMPTZ NULL,
    agent_flag_note  TEXT        NULL,
    CONSTRAINT emotion_events_agent_flag_consistency CHECK (
        (agent_flagged_by IS NULL AND agent_flagged_at IS NULL) OR
        (agent_flagged_by IS NOT NULL AND agent_flagged_at IS NOT NULL)
    )
);

-- 10. emotion_feedback
CREATE TABLE emotion_feedback (
    id                      UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    emotion_event_id        UUID    NOT NULL REFERENCES emotion_events(id) ON DELETE CASCADE,
    provided_by_user_id     UUID    NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    llm_justification       TEXT    NULL,
    corrected_emotion       VARCHAR(50) NOT NULL,
    corrected_justification TEXT    NULL,
    correction_reason       TEXT    NULL,
    feedback_status         feedback_status_enum NOT NULL DEFAULT 'pending',
    is_used_in_training     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 11. interaction_scores
CREATE TABLE interaction_scores (
    id                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id        UUID        NOT NULL UNIQUE REFERENCES interactions(id) ON DELETE CASCADE,
    overall_score         FLOAT       NULL,
    empathy_score         FLOAT       NULL,
    policy_score          FLOAT       NULL,
    resolution_score      FLOAT       NULL,
    was_resolved          BOOLEAN     NULL,
    total_silence_seconds  FLOAT      NULL,
    avg_response_time_seconds FLOAT      NULL,
    scored_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 12. policy_compliance
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

-- 13. compliance_feedback
CREATE TABLE compliance_feedback (
    id                      UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_compliance_id    UUID    NOT NULL REFERENCES policy_compliance(id) ON DELETE CASCADE,
    provided_by_user_id     UUID    NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_is_compliant   BOOLEAN NOT NULL,
    corrected_is_compliant  BOOLEAN NOT NULL,
    original_score          FLOAT   NULL,
    corrected_score         FLOAT   NULL,
    correction_reason       TEXT    NULL,
    feedback_status         feedback_status_enum NOT NULL DEFAULT 'pending',
    is_used_in_training     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 14. organization_policies
CREATE TABLE organization_policies (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    policy_id       UUID        NOT NULL REFERENCES company_policies(id) ON DELETE CASCADE,
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    assigned_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 15. organization_faq_articles
CREATE TABLE organization_faq_articles (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    article_id      UUID        NOT NULL REFERENCES faq_articles(id) ON DELETE CASCADE,
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    assigned_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 16. agent_performance_snapshots
CREATE TABLE agent_performance_snapshots (
    id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id     UUID         NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    agent_id            UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    period_type         period_type_enum NOT NULL,
    period_start        DATE         NOT NULL,
    period_end          DATE         NOT NULL,
    total_interactions  INTEGER      NOT NULL DEFAULT 0,
    avg_overall_score   FLOAT        NULL,
    avg_empathy_score   FLOAT        NULL,
    avg_policy_score    FLOAT        NULL,
    avg_resolution_score FLOAT       NULL,
    resolution_rate     FLOAT        NULL,
    computed_at         TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 17. assistant_queries
CREATE TABLE assistant_queries (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id   UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    query_mode        query_mode_enum NOT NULL,
    audio_input_path  TEXT        NULL,
    query_text        TEXT        NOT NULL,
    ai_understanding  TEXT        NULL,
    generated_sql     TEXT        NULL,
    response_text     TEXT        NULL,
    execution_time_ms INTEGER     NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 3. INDEXES
-- ============================================================
CREATE INDEX idx_users_organization_id            ON users(organization_id);
CREATE INDEX idx_interactions_organization_id      ON interactions(organization_id);
CREATE INDEX idx_interactions_agent_id             ON interactions(agent_id);
CREATE INDEX idx_interactions_date                 ON interactions(interaction_date);
CREATE INDEX idx_processing_jobs_interaction_id    ON processing_jobs(interaction_id);
CREATE INDEX idx_transcripts_interaction_id        ON transcripts(interaction_id);
CREATE INDEX idx_utterances_interaction_id         ON utterances(interaction_id);
CREATE INDEX idx_emotion_events_interaction_id     ON emotion_events(interaction_id);
CREATE INDEX idx_emotion_events_agent_flagged      ON emotion_events(agent_flagged_by) WHERE agent_flagged_by IS NOT NULL;
CREATE INDEX idx_organization_policies_org_id      ON organization_policies(organization_id);
CREATE INDEX idx_policy_compliance_interaction_id  ON policy_compliance(interaction_id);
CREATE INDEX idx_agent_snapshots_agent_id          ON agent_performance_snapshots(agent_id);
CREATE INDEX idx_assistant_queries_user_id         ON assistant_queries(user_id);

-- ============================================================
-- 4. AUTO-UPDATE TRIGGER
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

CREATE TRIGGER trg_faq_articles_updated_at
    BEFORE UPDATE ON faq_articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();