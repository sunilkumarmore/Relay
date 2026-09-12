-- Relay migration 005: receipts and the ledger.
--
-- Metering existed from the start; nothing priced it, billed it, or proved it.
-- A receipt is the proof: signed by the provider that earned it, countersigned
-- by the consumer that owes it, and checkable by anyone afterwards. Credits move
-- only against receipts, in double entry, so a balance is something you can
-- verify rather than something you have to trust.

CREATE TABLE IF NOT EXISTS relay_receipts (
  receipt_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  step_number INTEGER NOT NULL,
  consumer_node_id TEXT NOT NULL,
  provider_node_id TEXT NOT NULL,
  offer_id TEXT NOT NULL,
  model TEXT DEFAULT '',
  request_hash TEXT NOT NULL,        -- sha256 over prompt + params
  response_hash TEXT NOT NULL,       -- sha256 over what came back
  tokens_in INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  latency_ms INTEGER NOT NULL,
  price_in_per_1k DOUBLE PRECISION NOT NULL,
  price_out_per_1k DOUBLE PRECISION NOT NULL,
  amount_credits DOUBLE PRECISION NOT NULL,
  issued_at TIMESTAMPTZ NOT NULL,
  provider_signature TEXT NOT NULL,
  consumer_signature TEXT DEFAULT '',
  status TEXT DEFAULT 'unacknowledged',   -- unacknowledged | acknowledged | disputed
  dispute_reason TEXT DEFAULT '',
  UNIQUE (job_id, step_number, provider_node_id)
);

CREATE INDEX IF NOT EXISTS idx_relay_receipts_job ON relay_receipts(job_id, step_number);
CREATE INDEX IF NOT EXISTS idx_relay_receipts_provider ON relay_receipts(provider_node_id, status);
CREATE INDEX IF NOT EXISTS idx_relay_receipts_consumer ON relay_receipts(consumer_node_id, status);

-- ---------------------------------------------------------------------------
-- Double-entry ledger
-- ---------------------------------------------------------------------------
-- Accounts are <node>:available and <node>:held:<job>. Holding per job matters:
-- one job completing must not release another job's committed budget. `world`
-- is the outside — deposits come from it, slashes go to it — so the invariant
-- (every entry together sums to zero) needs no special case.

CREATE TABLE IF NOT EXISTS relay_accounts (
  node_id TEXT PRIMARY KEY,
  balance_cached DOUBLE PRECISION NOT NULL DEFAULT 0,
  stake DOUBLE PRECISION NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- One row per transaction. The unique idempotency_key is what stops a retried
-- settlement paying twice: the claim fails, and no entries are written.
CREATE TABLE IF NOT EXISTS relay_ledger_tx (
  tx_id UUID PRIMARY KEY,
  idempotency_key TEXT UNIQUE NOT NULL,
  kind TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS relay_ledger_entries (
  entry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tx_id UUID NOT NULL REFERENCES relay_ledger_tx(tx_id) ON DELETE CASCADE,
  kind TEXT NOT NULL,            -- deposit | hold | release | settle | refund | slash
  account TEXT NOT NULL,
  debit DOUBLE PRECISION NOT NULL DEFAULT 0,
  credit DOUBLE PRECISION NOT NULL DEFAULT 0,
  ref_receipt_id TEXT DEFAULT '',
  ref_job_id TEXT DEFAULT '',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  CHECK (debit >= 0 AND credit >= 0),
  CHECK (debit = 0 OR credit = 0)
);

CREATE INDEX IF NOT EXISTS idx_relay_ledger_entries_account ON relay_ledger_entries(account);
CREATE INDEX IF NOT EXISTS idx_relay_ledger_entries_tx ON relay_ledger_entries(tx_id);
CREATE INDEX IF NOT EXISTS idx_relay_ledger_entries_receipt ON relay_ledger_entries(ref_receipt_id);

-- Enforce the invariant in the database, not only in the application. A bug in
-- one code path must not be able to mint credits.
CREATE OR REPLACE FUNCTION relay_check_tx_balances() RETURNS TRIGGER AS $$
DECLARE
  total_debit DOUBLE PRECISION;
  total_credit DOUBLE PRECISION;
BEGIN
  SELECT COALESCE(SUM(debit), 0), COALESCE(SUM(credit), 0)
    INTO total_debit, total_credit
    FROM relay_ledger_entries WHERE tx_id = NEW.tx_id;

  IF ABS(total_debit - total_credit) > 0.000001 THEN
    RAISE EXCEPTION 'Transaction % does not balance: debits % != credits %',
      NEW.tx_id, total_debit, total_credit;
  END IF;
  RETURN NULL;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS relay_ledger_balance_check ON relay_ledger_entries;
CREATE CONSTRAINT TRIGGER relay_ledger_balance_check
  AFTER INSERT ON relay_ledger_entries
  DEFERRABLE INITIALLY DEFERRED
  FOR EACH ROW EXECUTE FUNCTION relay_check_tx_balances();

-- ---------------------------------------------------------------------------
-- Security
-- ---------------------------------------------------------------------------
ALTER TABLE relay_receipts       ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_accounts       ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_ledger_tx      ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_ledger_entries ENABLE ROW LEVEL SECURITY;

-- A receipt is visible to its two parties, and writable by either — the
-- provider issues it, the consumer countersigns or disputes it.
DROP POLICY IF EXISTS relay_receipts_party_read ON relay_receipts;
CREATE POLICY relay_receipts_party_read ON relay_receipts
  FOR SELECT USING (
    consumer_node_id = relay_current_node() OR provider_node_id = relay_current_node()
  );

DROP POLICY IF EXISTS relay_receipts_party_write ON relay_receipts;
CREATE POLICY relay_receipts_party_write ON relay_receipts
  FOR ALL
  USING (consumer_node_id = relay_current_node() OR provider_node_id = relay_current_node())
  WITH CHECK (consumer_node_id = relay_current_node() OR provider_node_id = relay_current_node());

-- Balances are public so anyone can check the ledger sums to zero; entries are
-- append-only and never editable through the API.
DROP POLICY IF EXISTS relay_accounts_public_read ON relay_accounts;
CREATE POLICY relay_accounts_public_read ON relay_accounts FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS relay_ledger_entries_public_read ON relay_ledger_entries;
CREATE POLICY relay_ledger_entries_public_read ON relay_ledger_entries FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS relay_ledger_tx_public_read ON relay_ledger_tx;
CREATE POLICY relay_ledger_tx_public_read ON relay_ledger_tx FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS relay_ledger_entries_node_insert ON relay_ledger_entries;
CREATE POLICY relay_ledger_entries_node_insert ON relay_ledger_entries
  FOR INSERT WITH CHECK (relay_current_node() <> '');

DROP POLICY IF EXISTS relay_ledger_tx_node_insert ON relay_ledger_tx;
CREATE POLICY relay_ledger_tx_node_insert ON relay_ledger_tx
  FOR INSERT WITH CHECK (relay_current_node() <> '');

DROP POLICY IF EXISTS relay_accounts_node_write ON relay_accounts;
CREATE POLICY relay_accounts_node_write ON relay_accounts
  FOR ALL USING (relay_current_node() <> '') WITH CHECK (relay_current_node() <> '');

GRANT SELECT ON relay_accounts, relay_ledger_entries, relay_ledger_tx TO anon;
