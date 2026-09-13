-- Relay migration 008: make row-level security actually mean something.
--
-- Migration 002 turned RLS on and wrote policies against a `relay_node_id` JWT
-- claim. Two things were wrong with shipping it as it stood.
--
-- First, nothing produced that claim. Nodes connected with the anon key, which
-- carries no claim at all, so `relay_current_node()` returned '' and every
-- policy failed closed. Applying 002 locked the application out of its own
-- database. The token authority (relay/authority/) is the missing half: a node
-- proves it holds its Ed25519 key and receives a short-lived JWT carrying that
-- claim. It is the only component that needs the project's JWT secret.
--
-- Second, and worse, the write policies said only `relay_current_node() <> ''`.
-- Any node that could obtain a token — which is any node, by design — could
-- rewrite any other session's checkpoints. The comment claimed writes were
-- "writable only by a node that holds the session"; the SQL did not say that.
-- This migration makes the SQL say it.

-- ---------------------------------------------------------------------------
-- Ownership
-- ---------------------------------------------------------------------------
-- A session belongs to the consumer node that created it. Rows written before
-- this column existed have an empty owner and stay writable, so applying this
-- migration does not strand an existing deployment.

ALTER TABLE relay_sessions ADD COLUMN IF NOT EXISTS owner_node_id TEXT DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_relay_sessions_owner ON relay_sessions(owner_node_id);

CREATE OR REPLACE FUNCTION relay_owns_session(target TEXT) RETURNS BOOLEAN AS $$
  SELECT EXISTS (
    SELECT 1 FROM relay_sessions s
    WHERE s.session_id = target
      AND (s.owner_node_id = relay_current_node() OR COALESCE(s.owner_node_id, '') = '')
  );
$$ LANGUAGE SQL STABLE SECURITY DEFINER;

-- ---------------------------------------------------------------------------
-- Privileges
-- ---------------------------------------------------------------------------
-- 002 revoked the anon role and never granted the authenticated one. A policy
-- can pass and the request still fail, because RLS narrows privileges — it does
-- not confer them.

GRANT SELECT, INSERT, UPDATE, DELETE ON
  relay_sessions, relay_checkpoints, relay_worker_state,
  relay_inference_log, relay_migration_log, relay_agent_state,
  relay_nodes, relay_registry_requests, relay_offers, relay_provider_events,
  relay_provider_health, relay_receipts, relay_accounts, relay_ledger_tx,
  relay_ledger_entries, relay_reputation, relay_disputes
TO authenticated;

-- ---------------------------------------------------------------------------
-- Session-scoped tables
-- ---------------------------------------------------------------------------
-- Reads stay open to any authenticated node: the dashboard's job is to show the
-- whole market, and a session's existence is not the secret. Writes are the
-- owner's alone.

DROP POLICY IF EXISTS relay_sessions_node_read ON relay_sessions;
DROP POLICY IF EXISTS relay_sessions_node_write ON relay_sessions;

CREATE POLICY relay_sessions_read ON relay_sessions
  FOR SELECT USING (relay_current_node() <> '');

CREATE POLICY relay_sessions_insert ON relay_sessions
  FOR INSERT WITH CHECK (
    owner_node_id = relay_current_node() OR COALESCE(owner_node_id, '') = ''
  );

CREATE POLICY relay_sessions_update ON relay_sessions
  FOR UPDATE
  USING (owner_node_id = relay_current_node() OR COALESCE(owner_node_id, '') = '')
  WITH CHECK (owner_node_id = relay_current_node() OR COALESCE(owner_node_id, '') = '');

CREATE POLICY relay_sessions_delete ON relay_sessions
  FOR DELETE USING (owner_node_id = relay_current_node() OR COALESCE(owner_node_id, '') = '');

-- Children of a session follow the session.
DO $$
DECLARE t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'relay_checkpoints', 'relay_worker_state', 'relay_migration_log', 'relay_agent_state'
  ] LOOP
    EXECUTE format('DROP POLICY IF EXISTS %I_node_read ON %I', t, t);
    EXECUTE format('DROP POLICY IF EXISTS %I_node_write ON %I', t, t);
    EXECUTE format('DROP POLICY IF EXISTS %I_read ON %I', t, t);
    EXECUTE format('DROP POLICY IF EXISTS %I_owner_write ON %I', t, t);

    EXECUTE format(
      'CREATE POLICY %I_read ON %I FOR SELECT USING (relay_current_node() <> '''')', t, t
    );
    EXECUTE format(
      'CREATE POLICY %I_owner_write ON %I FOR ALL '
      'USING (relay_owns_session(session_id)) WITH CHECK (relay_owns_session(session_id))',
      t, t
    );
  END LOOP;
END $$;

-- relay_inference_log is the exception: the *provider* writes it, about a
-- session it does not own. It is append-only telemetry, so any authenticated
-- node may add to it and nobody may rewrite history.
DROP POLICY IF EXISTS relay_inference_log_node_read ON relay_inference_log;
DROP POLICY IF EXISTS relay_inference_log_node_write ON relay_inference_log;

CREATE POLICY relay_inference_log_read ON relay_inference_log
  FOR SELECT USING (relay_current_node() <> '');

CREATE POLICY relay_inference_log_append ON relay_inference_log
  FOR INSERT WITH CHECK (relay_current_node() <> '');

-- Deliberately no UPDATE or DELETE policy: with RLS on, their absence denies.

-- ---------------------------------------------------------------------------
-- Checking your own work
-- ---------------------------------------------------------------------------
-- Run as a node to confirm the claim is arriving. An empty result means the
-- token is missing its claim, or you are connected with the anon key.
COMMENT ON FUNCTION relay_current_node() IS
  'The relay_node_id claim from the request JWT. SELECT relay_current_node() to check yours.';
