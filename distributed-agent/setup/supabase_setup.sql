-- Run this in Supabase SQL Editor before starting

CREATE TABLE agent_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  task_goal TEXT NOT NULL,
  steps_total INTEGER NOT NULL,
  steps_completed INTEGER DEFAULT 0,
  status TEXT DEFAULT 'in_progress',
  -- status values: in_progress, completed, failed
  final_report TEXT,
  machine_id TEXT
);

CREATE TABLE agent_checkpoints (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT REFERENCES agent_sessions(session_id),
  checkpoint_number INTEGER NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  machine_id TEXT NOT NULL,
  step_number INTEGER NOT NULL,
  problem TEXT NOT NULL,
  solution TEXT NOT NULL,
  reasoning TEXT NOT NULL,
  completed_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(session_id, step_number)
);

CREATE TABLE agent_state (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT REFERENCES agent_sessions(session_id) UNIQUE,
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  next_step_number INTEGER NOT NULL,
  next_problem TEXT NOT NULL,
  agent_scratchpad TEXT,
  machine_id TEXT NOT NULL
);

-- Index for fast lookups
CREATE INDEX idx_checkpoints_session
  ON agent_checkpoints(session_id, step_number);
CREATE INDEX idx_sessions_status
  ON agent_sessions(status);
