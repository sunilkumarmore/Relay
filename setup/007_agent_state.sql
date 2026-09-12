-- Relay migration 007: agent state.
--
-- Until now a resumed worker recovered its place in a list and nothing else,
-- because each step was an independent prompt with nothing to carry forward.
-- That made Relay a resumable queue. An agent accumulates — a conversation,
-- intermediate results, artifacts — and this is where that lives.

CREATE TABLE IF NOT EXISTS relay_agent_state (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT NOT NULL REFERENCES relay_sessions(session_id) ON DELETE CASCADE,
  step_number INTEGER NOT NULL,
  state_blob TEXT NOT NULL,        -- canonical JSON; see relay/agent/state.py
  state_hash TEXT NOT NULL,        -- sha256 of the blob, checked before rehydrating
  status TEXT NOT NULL DEFAULT 'complete',   -- complete | partial
  saved_at TIMESTAMPTZ DEFAULT NOW(),
  -- Append-only, one row per completed step, so any earlier point in a run is
  -- restorable rather than only the latest. A step has at most one complete row
  -- and at most one partial row.
  UNIQUE (session_id, step_number, status)
);

CREATE INDEX IF NOT EXISTS idx_relay_agent_state_session
  ON relay_agent_state(session_id, step_number DESC);

-- Steps can now name what they depend on and refer to earlier answers, so a
-- checkpoint records which step of the task it was, not just its number.
ALTER TABLE relay_checkpoints ADD COLUMN IF NOT EXISTS topic TEXT DEFAULT '';

-- `problem` now holds the full prompt that was sent, not the step template.
-- Dispute adjudication re-hashes and re-counts this text, so it has to be what
-- the provider actually received.
COMMENT ON COLUMN relay_checkpoints.problem IS
  'The complete prompt sent to the provider, including history. Disputes re-hash this.';
COMMENT ON COLUMN relay_checkpoints.reasoning IS
  'The model''s working, parsed out of the response. Distinct from solution.';

ALTER TABLE relay_agent_state ENABLE ROW LEVEL SECURITY;

-- Agent state is the consumer's own working memory: readable and writable by
-- any authenticated node that holds the session, and not public like the
-- directory tables.
DROP POLICY IF EXISTS relay_agent_state_node_read ON relay_agent_state;
CREATE POLICY relay_agent_state_node_read ON relay_agent_state
  FOR SELECT USING (relay_current_node() <> '');

DROP POLICY IF EXISTS relay_agent_state_node_write ON relay_agent_state;
CREATE POLICY relay_agent_state_node_write ON relay_agent_state
  FOR ALL USING (relay_current_node() <> '') WITH CHECK (relay_current_node() <> '');
