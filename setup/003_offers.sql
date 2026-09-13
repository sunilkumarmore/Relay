-- Relay migration 003: the directory.
--
-- Providers stop being something a consumer is configured to know about and
-- start being something it can discover. An offer is a signed, expiring
-- statement of terms; a consumer checks the signature before trusting the price,
-- and checks the price again against its receipts in migration 004.

CREATE TABLE IF NOT EXISTS relay_offers (
  offer_id TEXT PRIMARY KEY,
  provider_node_id TEXT NOT NULL,
  endpoint_url TEXT NOT NULL,
  model TEXT NOT NULL,
  context_window INTEGER NOT NULL,
  price_in_per_1k DOUBLE PRECISION NOT NULL,
  price_out_per_1k DOUBLE PRECISION NOT NULL,
  max_concurrency INTEGER NOT NULL DEFAULT 1,
  region TEXT NOT NULL DEFAULT 'unknown',
  capabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
  published_at TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  signature TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relay_provider_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  provider_node_id TEXT NOT NULL,
  kind TEXT NOT NULL,           -- saturated, published, withdrawn, ...
  model TEXT DEFAULT '',
  detail JSONB NOT NULL DEFAULT '{}'::jsonb,
  occurred_at TIMESTAMPTZ DEFAULT NOW()
);

-- Consumers shop by model and price; that is the hot query.
CREATE INDEX IF NOT EXISTS idx_relay_offers_model_price
  ON relay_offers(model, price_out_per_1k);
CREATE INDEX IF NOT EXISTS idx_relay_offers_expiry
  ON relay_offers(expires_at);
CREATE INDEX IF NOT EXISTS idx_relay_provider_events_node_time
  ON relay_provider_events(provider_node_id, occurred_at DESC);

-- Input and output tokens are priced differently, so they have to be stored
-- separately. tokens_used stays as the sum for anything already reading it.
ALTER TABLE relay_inference_log ADD COLUMN IF NOT EXISTS tokens_in INTEGER DEFAULT 0;
ALTER TABLE relay_inference_log ADD COLUMN IF NOT EXISTS tokens_out INTEGER DEFAULT 0;
ALTER TABLE relay_checkpoints   ADD COLUMN IF NOT EXISTS tokens_in INTEGER DEFAULT 0;
ALTER TABLE relay_checkpoints   ADD COLUMN IF NOT EXISTS tokens_out INTEGER DEFAULT 0;

-- Directory tables follow the same rule as relay_nodes: public to read (that is
-- the point of a directory), writable only by the node the row belongs to.
ALTER TABLE relay_offers          ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_provider_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS relay_offers_public_read ON relay_offers;
CREATE POLICY relay_offers_public_read ON relay_offers FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS relay_offers_self_write ON relay_offers;
CREATE POLICY relay_offers_self_write ON relay_offers
  FOR ALL
  USING (provider_node_id = relay_current_node())
  WITH CHECK (provider_node_id = relay_current_node());

DROP POLICY IF EXISTS relay_provider_events_public_read ON relay_provider_events;
CREATE POLICY relay_provider_events_public_read ON relay_provider_events
  FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS relay_provider_events_self_write ON relay_provider_events;
CREATE POLICY relay_provider_events_self_write ON relay_provider_events
  FOR INSERT WITH CHECK (provider_node_id = relay_current_node());

GRANT SELECT ON relay_offers, relay_provider_events TO anon;
