-- Relay migration 009: distributed task dispatch.
--
-- Additive. Nothing here drops a column, rewrites a row, or changes the meaning
-- of an existing one. The inference path works exactly as it did before this
-- migration and exactly as it does after.
--
-- Two things happen. New tables hold tasks, their results, and the operands they
-- refer to. And four NOT NULL constraints that assumed every piece of paid work
-- was an inference call are relaxed, because a matrix multiplication has no
-- model, no context window, and no tokens.

-- ---------------------------------------------------------------------------
-- Relaxing the inference-shaped constraints
-- ---------------------------------------------------------------------------
-- relay_receipts.tokens_in/tokens_out and relay_offers.model/context_window/
-- price_*_per_1k were written when inference was the only kind of work. A task
-- receipt cannot fill them, so no task row could be inserted at all.
--
-- Defaults are supplied so existing readers that expect a number still get one,
-- and every existing row keeps the value it has.

ALTER TABLE relay_receipts ALTER COLUMN tokens_in DROP NOT NULL;
ALTER TABLE relay_receipts ALTER COLUMN tokens_out DROP NOT NULL;
ALTER TABLE relay_receipts ALTER COLUMN tokens_in SET DEFAULT 0;
ALTER TABLE relay_receipts ALTER COLUMN tokens_out SET DEFAULT 0;

ALTER TABLE relay_offers ALTER COLUMN model DROP NOT NULL;
ALTER TABLE relay_offers ALTER COLUMN context_window DROP NOT NULL;
ALTER TABLE relay_offers ALTER COLUMN price_in_per_1k DROP NOT NULL;
ALTER TABLE relay_offers ALTER COLUMN price_out_per_1k DROP NOT NULL;
ALTER TABLE relay_offers ALTER COLUMN model SET DEFAULT '';
ALTER TABLE relay_offers ALTER COLUMN context_window SET DEFAULT 0;
ALTER TABLE relay_offers ALTER COLUMN price_in_per_1k SET DEFAULT 0;
ALTER TABLE relay_offers ALTER COLUMN price_out_per_1k SET DEFAULT 0;

-- What a provider will run, and what it charges for it.
ALTER TABLE relay_offers ADD COLUMN IF NOT EXISTS task_types TEXT[] DEFAULT '{}';
ALTER TABLE relay_offers ADD COLUMN IF NOT EXISTS price_per_mega_unit DOUBLE PRECISION DEFAULT 0;
ALTER TABLE relay_offers ADD COLUMN IF NOT EXISTS price_per_task DOUBLE PRECISION DEFAULT 0;
ALTER TABLE relay_offers ADD COLUMN IF NOT EXISTS max_payload_bytes BIGINT DEFAULT 0;

-- A receipt can now describe either kind of work.
ALTER TABLE relay_receipts ADD COLUMN IF NOT EXISTS schema_version INTEGER DEFAULT 1;
ALTER TABLE relay_receipts ADD COLUMN IF NOT EXISTS work_kind TEXT DEFAULT 'inference';
ALTER TABLE relay_receipts ADD COLUMN IF NOT EXISTS work_units BIGINT DEFAULT 0;
ALTER TABLE relay_receipts ADD COLUMN IF NOT EXISTS task_id TEXT DEFAULT '';

-- ---------------------------------------------------------------------------
-- Operands
-- ---------------------------------------------------------------------------
-- Content-addressed: the primary key is the sha256 of the bytes together with
-- their shape. That makes the table's integrity self-evident rather than
-- something the database has to enforce — a provider re-hashes what it fetches
-- and refuses to compute on anything that does not match the address it asked
-- for, so a wrong operand is a failed task and never a wrong answer.
--
-- Insert is open to any authenticated node for that reason. The residual risk
-- is nuisance rather than corruption: a node could publish garbage under a hash
-- nobody has claimed yet and make tasks referring to it fail. Writers upsert,
-- so the honest publisher overwrites it.

CREATE TABLE IF NOT EXISTS relay_operands (
  operand_hash TEXT PRIMARY KEY,
  payload      JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Tasks
-- ---------------------------------------------------------------------------
-- The signature is the real protection here and the policies below are defence
-- in depth. Everything a provider relies on — who ordered the work, what the
-- work is, how much of it there is — is covered by the consumer's Ed25519
-- signature over `payload_hash` and `work_units`, and is checked in
-- relay/tasks/model.py before a second is spent on it. A node that got past RLS
-- and wrote a task row would still have to forge a signature for it to run.

CREATE TABLE IF NOT EXISTS relay_tasks (
  task_id           TEXT PRIMARY KEY,
  job_id            TEXT NOT NULL,
  schema_version    INTEGER NOT NULL DEFAULT 1,
  task_type         TEXT NOT NULL,
  payload           JSONB NOT NULL DEFAULT '{}'::jsonb,
  payload_hash      TEXT NOT NULL,
  deterministic     BOOLEAN NOT NULL DEFAULT FALSE,
  work_units        BIGINT NOT NULL DEFAULT 0,
  consumer_node_id  TEXT NOT NULL,
  max_seconds       INTEGER NOT NULL DEFAULT 300,
  max_price_credits DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at        TEXT NOT NULL DEFAULT '',
  expires_at        TEXT NOT NULL DEFAULT '',
  signature         TEXT NOT NULL DEFAULT '',

  -- Queue state. Outside the signature on purpose: the consumer signs what it
  -- wants done, not what has happened to the request since.
  status            TEXT NOT NULL DEFAULT 'queued',
  attempts          INTEGER NOT NULL DEFAULT 0,
  max_attempts      INTEGER NOT NULL DEFAULT 3,
  lease_holder      TEXT NOT NULL DEFAULT '',
  lease_expires_at  TEXT NOT NULL DEFAULT '',
  completed_by      TEXT NOT NULL DEFAULT '',
  output_hash       TEXT NOT NULL DEFAULT '',
  last_error        TEXT NOT NULL DEFAULT '',
  updated_at        TEXT NOT NULL DEFAULT '',

  -- Verification. An audit is the same work queued again and barred to the node
  -- whose answer it is checking, because a check the accused may answer is not
  -- one. `excluded_provider` is filtered client-side so pollers skip it cheaply
  -- and enforced in the update policy below, which is the half a dishonest
  -- client cannot route around.
  audit_of          TEXT NOT NULL DEFAULT '',
  excluded_provider TEXT NOT NULL DEFAULT ''
);

-- The claim query: queued work of a type I can run, oldest first. This is the
-- hot path — every provider runs it every poll.
CREATE INDEX IF NOT EXISTS idx_relay_tasks_claim
  ON relay_tasks(status, task_type, created_at);

-- The reaper's query: leases that have lapsed.
CREATE INDEX IF NOT EXISTS idx_relay_tasks_lease
  ON relay_tasks(status, lease_expires_at);

CREATE INDEX IF NOT EXISTS idx_relay_tasks_job ON relay_tasks(job_id);

CREATE TABLE IF NOT EXISTS relay_task_results (
  id                BIGSERIAL PRIMARY KEY,
  task_id           TEXT NOT NULL,
  job_id            TEXT NOT NULL,
  schema_version    INTEGER NOT NULL DEFAULT 1,
  provider_node_id  TEXT NOT NULL,
  consumer_node_id  TEXT NOT NULL,
  payload_hash      TEXT NOT NULL DEFAULT '',
  output            JSONB NOT NULL DEFAULT '{}'::jsonb,
  output_hash       TEXT NOT NULL DEFAULT '',
  work_units        BIGINT NOT NULL DEFAULT 0,
  status            TEXT NOT NULL DEFAULT 'ok',
  error             TEXT NOT NULL DEFAULT '',
  duration_ms       INTEGER NOT NULL DEFAULT 0,
  finished_at       TEXT NOT NULL DEFAULT '',
  signature         TEXT NOT NULL DEFAULT ''
);

-- Several results for one task is normal, not an error: a reissued block that
-- both providers finished leaves two, and comparing them is exactly what
-- verification wants.
CREATE INDEX IF NOT EXISTS idx_relay_task_results_task ON relay_task_results(task_id);
CREATE INDEX IF NOT EXISTS idx_relay_task_results_job ON relay_task_results(job_id);

-- ---------------------------------------------------------------------------
-- Row-level security
-- ---------------------------------------------------------------------------

ALTER TABLE relay_tasks        ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_task_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE relay_operands     ENABLE ROW LEVEL SECURITY;

GRANT SELECT, INSERT, UPDATE, DELETE ON
  relay_tasks, relay_task_results, relay_operands
TO authenticated;

GRANT USAGE, SELECT ON SEQUENCE relay_task_results_id_seq TO authenticated;

-- Reads are open to any authenticated node. A market where you cannot see the
-- work on offer is not one, and a provider has to read a task to decide whether
-- it can run it.
DROP POLICY IF EXISTS relay_tasks_read ON relay_tasks;
CREATE POLICY relay_tasks_read ON relay_tasks
  FOR SELECT USING (relay_current_node() <> '');

DROP POLICY IF EXISTS relay_tasks_insert ON relay_tasks;
CREATE POLICY relay_tasks_insert ON relay_tasks
  FOR INSERT WITH CHECK (consumer_node_id = relay_current_node());

-- Four legitimate writers, and no fifth:
--   the consumer that ordered the work (cancelling, re-pricing);
--   the node currently holding the lease (renewing, completing);
--   any node claiming work that is sitting in the queue;
--   any node returning a lease whose holder has stopped answering.
-- The last two are what let recovery happen without a coordinator: the
-- surviving devices reap for each other as a side effect of asking for work.
DROP POLICY IF EXISTS relay_tasks_update ON relay_tasks;
CREATE POLICY relay_tasks_update ON relay_tasks
  FOR UPDATE
  USING (
    consumer_node_id = relay_current_node()
    OR lease_holder = relay_current_node()
    OR status = 'queued'
    OR (status = 'leased' AND lease_expires_at <> '' AND lease_expires_at <= to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'))
  )
  WITH CHECK (
    -- Nobody takes a lease on an audit of their own work, whatever else is true.
    (lease_holder = '' OR excluded_provider = '' OR lease_holder <> excluded_provider)
    AND (
      consumer_node_id = relay_current_node()
      OR lease_holder = relay_current_node()
      OR lease_holder = ''
    )
  );

DROP POLICY IF EXISTS relay_tasks_delete ON relay_tasks;
CREATE POLICY relay_tasks_delete ON relay_tasks
  FOR DELETE USING (consumer_node_id = relay_current_node());

-- A result is a signed statement by the provider that made it. Nobody may file
-- one in another node's name, and nobody may revise one after the fact.
DROP POLICY IF EXISTS relay_task_results_read ON relay_task_results;
CREATE POLICY relay_task_results_read ON relay_task_results
  FOR SELECT USING (relay_current_node() <> '');

DROP POLICY IF EXISTS relay_task_results_insert ON relay_task_results;
CREATE POLICY relay_task_results_insert ON relay_task_results
  FOR INSERT WITH CHECK (provider_node_id = relay_current_node());

DROP POLICY IF EXISTS relay_operands_read ON relay_operands;
CREATE POLICY relay_operands_read ON relay_operands
  FOR SELECT USING (relay_current_node() <> '');

DROP POLICY IF EXISTS relay_operands_write ON relay_operands;
CREATE POLICY relay_operands_write ON relay_operands
  FOR INSERT WITH CHECK (relay_current_node() <> '');

DROP POLICY IF EXISTS relay_operands_update ON relay_operands;
CREATE POLICY relay_operands_update ON relay_operands
  FOR UPDATE USING (relay_current_node() <> '') WITH CHECK (relay_current_node() <> '');
