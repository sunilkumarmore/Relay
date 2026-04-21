-- Sessions table
CREATE TABLE relay_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT UNIQUE NOT NULL,
  worker_id TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  task_goal TEXT NOT NULL,
  steps_total INTEGER NOT NULL DEFAULT 5,
  steps_completed INTEGER DEFAULT 0,
  status TEXT DEFAULT 'in_progress',
  final_report TEXT,
  current_machine TEXT,
  inference_node TEXT
);

-- Checkpoints table
CREATE TABLE relay_checkpoints (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT REFERENCES relay_sessions(session_id),
  worker_id TEXT NOT NULL,
  step_number INTEGER NOT NULL,
  problem TEXT NOT NULL,
  solution TEXT NOT NULL,
  reasoning TEXT NOT NULL,
  machine_id TEXT NOT NULL,
  inference_node TEXT NOT NULL,
  inference_latency_ms INTEGER,
  tokens_used INTEGER,
  completed_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(session_id, step_number)
);

-- Worker state table
CREATE TABLE relay_worker_state (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT REFERENCES relay_sessions(session_id) UNIQUE,
  worker_id TEXT NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  next_step_number INTEGER NOT NULL,
  next_problem TEXT NOT NULL,
  machine_id TEXT NOT NULL,
  inference_node TEXT NOT NULL,
  status TEXT DEFAULT 'active'
);

-- Inference log table (for dashboard metrics)
CREATE TABLE relay_inference_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  worker_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  inference_node TEXT NOT NULL,
  requested_at TIMESTAMPTZ DEFAULT NOW(),
  latency_ms INTEGER,
  tokens_used INTEGER,
  success BOOLEAN DEFAULT TRUE
);

-- Migration log table (for dashboard)
CREATE TABLE relay_migration_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT NOT NULL,
  worker_id TEXT NOT NULL,
  event TEXT NOT NULL,
  -- event values: started, evicted, migrated, resumed, completed
  from_machine TEXT,
  to_machine TEXT,
  step_at_event INTEGER,
  occurred_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_relay_checkpoints_session
  ON relay_checkpoints(session_id, step_number);
CREATE INDEX idx_relay_inference_log_time
  ON relay_inference_log(requested_at DESC);
CREATE INDEX idx_relay_migration_log_session
  ON relay_migration_log(session_id, occurred_at DESC);
