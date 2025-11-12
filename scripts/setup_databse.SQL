-- ============================================================
-- Drift-Aware Retraining Pipeline - Database Setup
-- Run this in Supabase SQL Editor
-- ============================================================

-- Enable pgvector extension for storing embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Table: embeddings_log
-- Stores all text embeddings (queries and documents)
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    embedding VECTOR(1536),  -- OpenAI ada-002 uses 1536 dimensions
    type TEXT CHECK (type IN ('query', 'document')) NOT NULL,
    text_content TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_embeddings_timestamp ON embeddings_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings_log(type);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings_log USING ivfflat (embedding vector_cosine_ops);

-- ============================================================
-- Table: interaction_log
-- Logs every AI interaction with flags
-- ============================================================
CREATE TABLE IF NOT EXISTS interaction_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_query TEXT NOT NULL,
    model_response TEXT,
    refusal_flag BOOLEAN DEFAULT FALSE,
    toxicity_flag BOOLEAN DEFAULT FALSE,
    user_feedback_score INTEGER CHECK (user_feedback_score BETWEEN 1 AND 5),
    model_version TEXT DEFAULT 'v1.0',
    tokens_used INTEGER DEFAULT 0,
    cost_usd DECIMAL(10,6) DEFAULT 0.0
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interaction_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_model_version ON interaction_log(model_version);
CREATE INDEX IF NOT EXISTS idx_interactions_refusal ON interaction_log(refusal_flag);
CREATE INDEX IF NOT EXISTS idx_interactions_toxicity ON interaction_log(toxicity_flag);

-- ============================================================
-- Table: drift_events
-- Records detected drift events and actions taken
-- ============================================================
CREATE TABLE IF NOT EXISTS drift_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    drift_type TEXT NOT NULL,  -- 'embedding', 'behavior', 'accuracy', 'action_triggered'
    drift_score DECIMAL(10,4) DEFAULT 0.0,
    threshold DECIMAL(10,4) DEFAULT 0.0,
    action_taken TEXT,  -- 'reindex_documents', 'fine_tune_model', etc.
    details JSONB DEFAULT '{}'::jsonb
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_drift_events_timestamp ON drift_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_drift_events_type ON drift_events(drift_type);
CREATE INDEX IF NOT EXISTS idx_drift_events_action ON drift_events(action_taken);

-- ============================================================
-- Table: model_versions
-- Tracks deployed model versions
-- ============================================================
CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    version_name TEXT UNIQUE NOT NULL,
    deployed_at TIMESTAMPTZ DEFAULT NOW(),
    accuracy DECIMAL(5,2),  -- 0.00 to 100.00
    avg_tokens INTEGER,
    is_active BOOLEAN DEFAULT FALSE,
    training_details JSONB DEFAULT '{}'::jsonb
);

-- Only one active version at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_active 
ON model_versions(is_active) 
WHERE is_active = TRUE;

-- ============================================================
-- Views for Analytics
-- ============================================================

-- View: Daily Metrics Summary
CREATE OR REPLACE VIEW daily_metrics AS
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_interactions,
    SUM(CASE WHEN refusal_flag THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as refusal_rate,
    SUM(CASE WHEN toxicity_flag THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as toxicity_rate,
    AVG(user_feedback_score) as avg_feedback,
    SUM(tokens_used) as total_tokens,
    SUM(cost_usd) as total_cost,
    model_version
FROM interaction_log
GROUP BY DATE(timestamp), model_version
ORDER BY date DESC;

-- View: Recent Drift Events
CREATE OR REPLACE VIEW recent_drifts AS
SELECT 
    timestamp,
    drift_type,
    drift_score,
    threshold,
    action_taken,
    CASE 
        WHEN drift_score > threshold * 2 THEN 'CRITICAL'
        WHEN drift_score > threshold THEN 'WARNING'
        ELSE 'NORMAL'
    END as severity
FROM drift_events
WHERE timestamp > NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC;

-- ============================================================
-- Functions for Common Queries
-- ============================================================

-- Function: Get embedding drift between two periods
CREATE OR REPLACE FUNCTION calculate_embedding_drift(
    days_recent INTEGER DEFAULT 7,
    days_baseline INTEGER DEFAULT 14
) RETURNS TABLE(
    recent_count BIGINT,
    baseline_count BIGINT,
    centroid_distance DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    WITH recent AS (
        SELECT AVG(embedding) as centroid
        FROM embeddings_log
        WHERE type = 'query' 
        AND timestamp > NOW() - INTERVAL '1 day' * days_recent
    ),
    baseline AS (
        SELECT AVG(embedding) as centroid
        FROM embeddings_log
        WHERE type = 'query'
        AND timestamp BETWEEN 
            NOW() - INTERVAL '1 day' * (days_baseline + days_recent)
            AND NOW() - INTERVAL '1 day' * days_recent
    )
    SELECT 
        (SELECT COUNT(*) FROM embeddings_log WHERE type='query' 
         AND timestamp > NOW() - INTERVAL '1 day' * days_recent),
        (SELECT COUNT(*) FROM embeddings_log WHERE type='query'
         AND timestamp BETWEEN NOW() - INTERVAL '1 day' * (days_baseline + days_recent)
         AND NOW() - INTERVAL '1 day' * days_recent),
        (SELECT recent.centroid <-> baseline.centroid FROM recent, baseline)::DECIMAL;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Initial Data
-- ============================================================

-- Insert default model version
INSERT INTO model_versions (version_name, accuracy, avg_tokens, is_active)
VALUES ('v1.0', 75.00, 200, TRUE)
ON CONFLICT (version_name) DO NOTHING;

-- ============================================================
-- Cleanup Functions (for testing)
-- ============================================================

-- Function to clear all test data (USE WITH CAUTION)
CREATE OR REPLACE FUNCTION reset_all_data() 
RETURNS void AS $$
BEGIN
    TRUNCATE embeddings_log, interaction_log, drift_events RESTART IDENTITY CASCADE;
    UPDATE model_versions SET is_active = FALSE;
    UPDATE model_versions SET is_active = TRUE WHERE version_name = 'v1.0';
    RAISE NOTICE 'All data cleared. Default model v1.0 reactivated.';
END;
$$ LANGUAGE plpgsql;

-- To use: SELECT reset_all_data();

-- ============================================================
-- Verify Setup
-- ============================================================

-- Check if everything is created
DO $$ 
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('embeddings_log', 'interaction_log', 'drift_events', 'model_versions');
    
    IF table_count = 4 THEN
        RAISE NOTICE '✅ All tables created successfully!';
    ELSE
        RAISE NOTICE '⚠️ Missing tables. Expected 4, found %', table_count;
    END IF;
END $$;

-- Check pgvector extension
SELECT 
    CASE 
        WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') 
        THEN '✅ pgvector extension is enabled'
        ELSE '⚠️ pgvector extension is NOT enabled'
    END as status;

-- Show table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN ('embeddings_log', 'interaction_log', 'drift_events', 'model_versions')
ORDER BY tablename;