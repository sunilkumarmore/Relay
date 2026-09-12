-- Relay migration 004: discovery and failover.
--
-- Consumers stop being configured with a provider and start choosing one. Two
-- things follow: they need somewhere to record what they observed, and a
-- session needs to remember which provider was serving it so a resumed worker
-- can go back to it.

CREATE TABLE IF NOT EXISTS relay_provider_health (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  observer_node_id TEXT NOT NULL,     -- who saw it; one consumer's bad day is not a verdict
  provider_node_id TEXT NOT NULL,
  ok BOOLEAN NOT NULL,
  latency_ms INTEGER,
  error TEXT DEFAULT '',
  observed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relay_provider_health_provider_time
  ON relay_provider_health(provider_node_id, observed_at DESC);

-- Which provider this session was last using. A resumed worker prefers it, and
-- re-selects if that provider is no longer offering terms the job accepts.
ALTER TABLE relay_worker_state ADD COLUMN IF NOT EXISTS provider_node_id TEXT DEFAULT '';

-- relay_migration_log gains a 'provider_switched' event. from_machine and
-- to_machine carry the provider node ids for that event kind.
COMMENT ON COLUMN relay_migration_log.event IS
  'started, evicted, migrated, resumed, completed, provider_switched';

ALTER TABLE relay_provider_health ENABLE ROW LEVEL SECURITY;

-- Health observations are public: that is what makes them useful to other
-- consumers. A node may only write observations under its own name, so nobody
-- can smear a competitor anonymously.
DROP POLICY IF EXISTS relay_provider_health_public_read ON relay_provider_health;
CREATE POLICY relay_provider_health_public_read ON relay_provider_health
  FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS relay_provider_health_self_write ON relay_provider_health;
CREATE POLICY relay_provider_health_self_write ON relay_provider_health
  FOR INSERT WITH CHECK (observer_node_id = relay_current_node());

GRANT SELECT ON relay_provider_health TO anon;
