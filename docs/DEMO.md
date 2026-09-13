# The distributed matrix multiplication demo

What this shows: a multiplication too slow for one machine is split across
several devices, they do it in parallel, one of them is unplugged part-way
through, and the answer that comes back is still exactly right.

No device needs a GPU. No device needs a model. No device needs an open port.

---

## Thirty seconds, no setup

```bash
python -m relay.tasks demo
```

Three devices in one process, one of them killed mid-job. Ends with:

```
  assembly refuses, correctly: 2400 of 7200 cells never came back (first missing at row 0, column 0)
  device-3 never renewed. 2 lease(s) expired and requeued.
  bit-identical to the single-machine answer: True
```

The refusal matters as much as the success. A half-finished job does not
produce a half-right matrix — it produces no matrix, and says why.

---

## The real thing, across your actual devices

### 1. Database

Apply the migrations in order, in the Supabase SQL editor:

```
setup/supabase_setup.sql
setup/002_identity_and_rls.sql  ⚠ read the warning in README before this one
setup/003_offers.sql
setup/004_discovery.sql
setup/005_receipts_and_ledger.sql
setup/006_reputation_and_disputes.sql
setup/007_agent_state.sql
setup/008_token_authority.sql
setup/009_tasks.sql
```

009 is additive: new tables for tasks, results and operands, and it relaxes
four `NOT NULL` constraints that assumed all paid work was inference. No
existing row is rewritten and the inference path is untouched.

### 2. `.env` on every device

The whole file a provider needs:

```ini
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# If you applied 002/008, nodes need the token authority to reach the database:
RELAY_AUTHORITY_URL=http://<authority-host>:8790
```

That is it. Everything else has a working default. `.env.example` documents
the knobs — task types, concurrency, lease length, pausing.

On the machine running the token authority, and **nowhere else**:

```ini
RELAY_JWT_SECRET=<Supabase project JWT secret>
```

### 3. Start a provider on each device

```bash
python -m relay.tasks node
```

```
Relay provider node  57754c64b9992877…
  task types : matmul_block, text_transform, data_reduce
  lease      : 900s
  poll every : 5s
  Ctrl-C finishes the task in hand and exits.

  → matmul_block b5a4986c (2,304,000 units, job ec49c973)
  ✓ b5a4986c in 0.5s (4,592,905 units/s)
```

Laptops, desktops, a Raspberry Pi, an Android phone under Termux — anything
that runs Python 3.11 and can reach Supabase over HTTPS.

### 4. Order the work

From any machine — it does not have to be one of the providers:

```bash
python -m relay.tasks submit --rows 2400 --inner 2400 --cols 2400 --seed 7
```

```
job 4f2a…
  2400x2400 @ 2400x2400 = 13,824,000,000 multiply-accumulates in 32x12 tiles of 75x218
  seed 7 — keep it, `collect` needs it to check the answer
```

### 5. Watch, then collect

```bash
python -m relay.tasks status 4f2a…
python -m relay.tasks collect 4f2a… --seed 7
```

```
assembled 2400x2400
verified: bit-identical to the same multiplication on one machine
```

### 6. Pull the plug

While it runs, close a laptop lid or kill a node process. `status` shows the
tile going back to `queued`, another device picks it up, and `collect` still
verifies. Nothing detects the death — the device simply stops renewing its
lease, and the absence is the signal.

---

## Sizing it

Work is measured in multiply-accumulates: `rows x inner x cols`. Measured on
this machine, the pure-Python kernel runs at **~20 million per second per core**
for strips a few hundred wide (`19.1M/s` at 2400 wide, `22.9M/s` at 400). A
phone will be several times slower.

| Shape | Work | One laptop core |
|---|---|---|
| 600³ | 0.2B | ~11 seconds |
| 1000³ | 1.0B | ~50 seconds |
| 2400³ | 13.8B | ~11 minutes |
| 3600³ | 46.7B | ~39 minutes |
| 4200³ | 74.1B | **~62 minutes** |

So **4200³ is the hour-on-one-machine job.** Four laptops should return it in
roughly a quarter of that; a fleet of phones, rather more.

Memory: three 4200² float64 matrices are 141MB each, so about 420MB resident on
the machine that submits and collects. Comfortable inside 16GB.

### How it is cut up

Both operands are split, and every strip is stored once under the hash of its
bytes and fetched by hash. A strip of A is reused by every tile across its row,
a strip of B by every tile down its column, and a device that has fetched one
keeps it.

Splitting only by rows would be simpler and would not scale: every task would
need the whole of B, so the smallest thing a device must hold is the entire
141MB right operand. Splitting both ways bounds it — each strip is capped at
4MB, so 4200³ becomes 56x34 = 1904 tiles with strips of 2.6MB and 4.2MB.

Tiles are sized at about five seconds of assumed work each, capped by strip
size and by a quarter of the task deadline. Override with `--block-rows` and
`--block-cols`. Smaller tiles recover faster from a dead device and balance a
mixed fleet better; larger ones spend less on per-task overhead.

Start at 1000³ to confirm the plumbing before reaching for the full hour.

---

## Why it is slow on purpose

The kernel is pure Python, and it could be perhaps a hundred times faster with
numpy. It does not use numpy because BLAS reorders and blocks its arithmetic
for speed, so two numpy builds can disagree in the last bit of a float. Pure
Python floats are IEEE-754 doubles whose `*` and `+` are correctly rounded, so
a fixed accumulation order gives **identical bytes on every machine**.

That is what makes a provider checkable. An audit re-runs a block somewhere
else and compares hashes; if honest machines disagreed routinely, the check
would slash honest providers and would have to be abandoned. The speed is
traded for the ability to verify the work at all.

---

## What this demo does not show

- **No iOS app.** The node is Python. A phone running Termux works today; an
  iPhone does not. The protocol is plain HTTPS against PostgREST plus Ed25519
  signing, so an iOS client is a small app rather than a port — but it is not
  written.
- **No payment.** Credits move on an internal ledger. There is no rail to real
  money.
- **No sandbox for arbitrary code.** `python_exec` is off by default and is not
  a security boundary. WASM is the answer and is not built.
- **Only independent work fans out.** Matrix rows are independent, which is why
  this is an honest demonstration. A computation that is one long dependency
  chain uses one device however many are connected.
