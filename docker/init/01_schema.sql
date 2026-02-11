-- ============================================================
-- VocalMind Schema (matches Supabase migration)
-- ============================================================

-- 1. ENUM TYPES
CREATE TYPE org_status_enum AS ENUM ('active', 'inactive', 'suspended');
CREATE TYPE user_role_enum AS ENUM ('admin', 'manager');
CREATE TYPE agent_type_enum AS ENUM ('human', 'bot');
CREATE TYPE processing_status_enum AS ENUM ('pending', 'processing', 'completed', 'failed');
CREATE TYPE speaker_role_enum AS ENUM ('agent', 'customer');
CREATE TYPE event_type_enum AS ENUM ('emotion_shift', 'sentiment_drop', 'escalation', 'de_escalation');
CREATE TYPE violation_severity_enum AS ENUM ('minor', 'critical');
CREATE TYPE feedback_type_enum AS ENUM ('emotion_label', 'score', 'transcription', 'compliance', 'other');
CREATE TYPE query_mode_enum AS ENUM ('voice', 'chat');

-- 2. TABLES

CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR NOT NULL,
    status org_status_enum NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email VARCHAR NOT NULL UNIQUE,
    password_hash VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    role user_role_enum NOT NULL DEFAULT 'manager',
    is_active BOOLEAN NOT NULL DEFAULT true,
    last_login_at TIMESTAMPTZ
);

CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    agent_code VARCHAR NOT NULL,
    agent_type agent_type_enum NOT NULL DEFAULT 'human',
    department VARCHAR,
    is_active BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    audio_file_path VARCHAR,
    file_size_bytes INTEGER,
    duration_seconds INTEGER,
    file_format VARCHAR,
    interaction_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    processing_status processing_status_enum NOT NULL DEFAULT 'pending',
    language_detected VARCHAR,
    has_overlap BOOLEAN DEFAULT false
);

CREATE TABLE transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id UUID NOT NULL UNIQUE REFERENCES interactions(id) ON DELETE CASCADE,
    full_text TEXT NOT NULL,
    confidence_score FLOAT
);

CREATE TABLE utterances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    speaker_role speaker_role_enum NOT NULL,
    start_time_seconds FLOAT NOT NULL,
    end_time_seconds FLOAT NOT NULL,
    emotion_label VARCHAR,
    emotion_confidence FLOAT
);

CREATE TABLE emotion_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    utterance_id UUID REFERENCES utterances(id) ON DELETE SET NULL,
    event_type event_type_enum NOT NULL,
    previous_emotion VARCHAR,
    new_emotion VARCHAR,
    emotion_delta FLOAT,
    trigger_category VARCHAR,
    timestamp_seconds FLOAT,
    speaker_role VARCHAR,
    verified_by_user_id UUID REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE interaction_scores (
    interaction_id UUID PRIMARY KEY REFERENCES interactions(id) ON DELETE CASCADE,
    overall_score FLOAT,
    policy_score FLOAT,
    total_silence_duration_seconds FLOAT,
    average_response_time_seconds FLOAT,
    was_resolved BOOLEAN
);

CREATE TABLE company_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    policy_code VARCHAR NOT NULL,
    category VARCHAR,
    policy_text TEXT NOT NULL,
    pinecone_id VARCHAR
);

CREATE TABLE policy_compliance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    policy_id UUID NOT NULL REFERENCES company_policies(id) ON DELETE CASCADE,
    is_compliant BOOLEAN,
    compliance_score FLOAT,
    violation_severity violation_severity_enum,
    confidence_score FLOAT,
    analyzed_by_model VARCHAR,
    trigger_description TEXT,
    evidence_text TEXT,
    llm_reasoning TEXT,
    is_human_verified BOOLEAN NOT NULL DEFAULT false,
    human_feedback_text TEXT
);

CREATE TABLE human_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    provided_by_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    feedback_type feedback_type_enum NOT NULL,
    ai_output JSONB,
    corrected_output JSONB,
    correction_reason TEXT
);

CREATE TABLE manager_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    query_mode query_mode_enum NOT NULL DEFAULT 'chat',
    ai_query_understanding VARCHAR,
    sql_code TEXT,
    response_text TEXT,
    retrieved_policy_id UUID REFERENCES company_policies(id) ON DELETE SET NULL
);

-- 3. INDEXES
CREATE INDEX idx_interactions_organization_id ON interactions(organization_id);
CREATE INDEX idx_interactions_agent_id ON interactions(agent_id);
CREATE INDEX idx_interactions_date ON interactions(interaction_date);
CREATE INDEX idx_utterances_interaction_id ON utterances(interaction_id);
CREATE INDEX idx_emotion_events_interaction_id ON emotion_events(interaction_id);
CREATE INDEX idx_policy_compliance_interaction_id ON policy_compliance(interaction_id);

-- 4. AUTO-UPDATE TRIGGER
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_organizations_updated_at
    BEFORE UPDATE ON organizations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
