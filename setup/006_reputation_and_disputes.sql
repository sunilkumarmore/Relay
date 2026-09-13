-- Relay migration 006: reputation, disputes, and stake.
--
-- Phase 4 made lying visible to the party being lied to. This makes it visible
-- to everyone, and expensive to the liar.

CREATE TABLE IF NOT EXISTS relay_reputation (
  node_id TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'provider',    -- provider | consumer
  score DOUBLE PRECISION NOT NULL,
  components JSONB NOT NULL DEFAULT '{}'::jsonb,
  computed_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (node_id, role)
);

-- The score is a deterministic function of public rows, so `components` is not
-- decoration: it is the working, and anyone can recompute it and compare.

CREATE TABLE IF NOT EXISTS relay_disputes (
  dispute_id TEXT PRIMARY KEY,
  receipt_id TEXT NOT NULL REFERENCES relay_receipts(receipt_id) ON DELETE CASCADE,
  opened_by TEXT NOT NULL,
  reason TEXT NOT NULL,                     -- hash_mismatch | token_overclaim | other
  status TEXT NOT NULL DEFAULT 'open',      -- open | upheld | rejected | unadjudicated
  evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
  opened_at TIMESTAMPTZ DEFAULT NOW(),
  resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_relay_disputes_receipt ON relay_disputes(receipt_id);
CREATE INDEX IF NOT EXISTS idx_relay_disputes_status ON relay_disputes(status, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_relay_reputation_role_score ON relay_reputation(role, score DESC);

-- relay_accounts.stake already exists from migration 005. It is what a slash
-- draws from: a provider with nothing at risk is not listed in the directory.
COMMENT ON COLUMN relay_accounts.stake IS
  'Credits the node has put at risk. Below the market minimum its offers are not listed.';

-- relay_provider_events gains a 'cross_checked' kind for sampled verification,
-- and 'understaked' for a provider that stopped advertising.
COMMENT ON COLUMN relay_provider_events.kind IS
  'saturated | cross_checked | understaked | published | withdrawn';

ALTER TABLE relay_reputation ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_disputes   ENABLE ROW LEVEL SECURITY;

-- Reputation is public in both directions: anyone may read it, and anyone may
-- recompute and write it, because the computation is deterministic over public
-- data. A wrong value is detectable by anyone who reruns it.
DROP POLICY IF EXISTS relay_reputation_public_read ON relay_reputation;
CREATE POLICY relay_reputation_public_read ON relay_reputation FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS relay_reputation_node_write ON relay_reputation;
CREATE POLICY relay_reputation_node_write ON relay_reputation
  FOR ALL USING (relay_current_node() <> '') WITH CHECK (relay_current_node() <> '');

-- A dispute is visible to the parties on the receipt, and may only be opened by
-- the node opening it under its own name.
DROP POLICY IF EXISTS relay_disputes_party_read ON relay_disputes;
CREATE POLICY relay_disputes_party_read ON relay_disputes
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM relay_receipts r
      WHERE r.receipt_id = relay_disputes.receipt_id
        AND (r.consumer_node_id = relay_current_node() OR r.provider_node_id = relay_current_node())
    )
  );

DROP POLICY IF EXISTS relay_disputes_open ON relay_disputes;
CREATE POLICY relay_disputes_open ON relay_disputes
  FOR INSERT WITH CHECK (opened_by = relay_current_node());

DROP POLICY IF EXISTS relay_disputes_resolve ON relay_disputes;
CREATE POLICY relay_disputes_resolve ON relay_disputes
  FOR UPDATE USING (relay_current_node() <> '') WITH CHECK (relay_current_node() <> '');

GRANT SELECT ON relay_reputation TO anon;
