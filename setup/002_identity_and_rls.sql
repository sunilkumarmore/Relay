-- Relay migration 002: node identity, durable registry state, and row-level security.
--
-- Run this after setup/supabase_setup.sql on an existing project.
--
-- Two things change here. First, the registry stops keeping its state in process
-- memory. Second — and more importantly — the anon key stops being a master key.
-- Until now anyone holding it could read and rewrite every table in the project.

-- ---------------------------------------------------------------------------
-- Durable registry state
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS relay_nodes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  worker_id TEXT UNIQUE NOT NULL,
  node_id TEXT NOT NULL,              -- hex Ed25519 public key: the real identity
  session_id TEXT NOT NULL,
  machine_id TEXT NOT NULL,
  inference_node TEXT NOT NULL,
  steps_completed INTEGER DEFAULT 0,
  registered_at TIMESTAMPTZ DEFAULT NOW(),
  last_heartbeat TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS relay_registry_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  worker_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  inference_node TEXT NOT NULL,
  latency_ms INTEGER,
  success BOOLEAN DEFAULT TRUE,
  requested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relay_nodes_inference
  ON relay_nodes(inference_node);
CREATE INDEX IF NOT EXISTS idx_relay_registry_requests_node_time
  ON relay_registry_requests(inference_node, requested_at DESC);

-- ---------------------------------------------------------------------------
-- Row-level security
-- ---------------------------------------------------------------------------
-- Relay nodes authenticate by signing a challenge with their Ed25519 key and
-- exchanging it for a short-lived JWT carrying `relay_node_id`. Policies below
-- are written against that claim. The anon key keeps read access to the parts of
-- the system that are meant to be public (the directory of who is serving what)
-- and loses write access everywhere.

CREATE OR REPLACE FUNCTION relay_current_node() RETURNS TEXT AS $$
  SELECT COALESCE(
    NULLIF(current_setting('request.jwt.claims', TRUE)::json->>'relay_node_id', ''),
    ''
  );
$$ LANGUAGE SQL STABLE;

ALTER TABLE relay_sessions          ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_checkpoints       ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_worker_state      ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_inference_log     ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_migration_log     ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_nodes             ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_registry_requests ENABLE ROW LEVEL SECURITY;

-- Public directory: who is serving, and how they are performing. Read-only.
DROP POLICY IF EXISTS relay_nodes_public_read ON relay_nodes;
CREATE POLICY relay_nodes_public_read ON relay_nodes
  FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS relay_registry_requests_public_read ON relay_registry_requests;
CREATE POLICY relay_registry_requests_public_read ON relay_registry_requests
  FOR SELECT USING (TRUE);

-- A node writes only its own registration row.
DROP POLICY IF EXISTS relay_nodes_self_write ON relay_nodes;
CREATE POLICY relay_nodes_self_write ON relay_nodes
  FOR ALL
  USING (node_id = relay_current_node())
  WITH CHECK (node_id = relay_current_node());

DROP POLICY IF EXISTS relay_registry_requests_self_write ON relay_registry_requests;
CREATE POLICY relay_registry_requests_self_write ON relay_registry_requests
  FOR INSERT
  WITH CHECK (node_id = relay_current_node());

-- Session-scoped tables: readable by any authenticated node (the dashboard needs
-- the whole picture), writable only by a node that holds the session.
DO $$
DECLARE t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'relay_sessions', 'relay_checkpoints', 'relay_worker_state',
    'relay_inference_log', 'relay_migration_log'
  ] LOOP
    EXECUTE format('DROP POLICY IF EXISTS %I_node_read ON %I', t, t);
    EXECUTE format(
      'CREATE POLICY %I_node_read ON %I FOR SELECT USING (relay_current_node() <> '''')', t, t
    );
    EXECUTE format('DROP POLICY IF EXISTS %I_node_write ON %I', t, t);
    EXECUTE format(
      'CREATE POLICY %I_node_write ON %I FOR ALL '
      'USING (relay_current_node() <> '''') WITH CHECK (relay_current_node() <> '''')',
      t, t
    );
  END LOOP;
END $$;

-- Revoke the blanket grants the anon role receives by default.
REVOKE ALL ON relay_sessions, relay_checkpoints, relay_worker_state,
               relay_inference_log, relay_migration_log FROM anon;
GRANT SELECT ON relay_nodes, relay_registry_requests TO anon;
