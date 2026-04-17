-- Idempotent patch for existing databases (fresh installs get this column from 01_schema.sql).
ALTER TABLE assistant_queries ADD COLUMN IF NOT EXISTS result_rows JSONB;
