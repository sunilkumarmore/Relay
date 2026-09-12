"""Reputation: making a provider's record visible, and lying expensive.

The score is a deterministic function of rows anyone can read. That matters more
than the exact formula: a reputation nobody can recompute is a reputation you
have to take on faith, which is the thing a marketplace is supposed to remove.
Run ``relay reputation recompute`` yourself and you should get the same number.

What it is built from — all of it evidence already in the store:

``reliability``
    The share of consumer observations where the provider actually answered.
``latency``
    How this provider's observed latency compares to the market's median.
``advertising``
    Whether it is currently keeping a live offer published.

Those three describe whether a provider *works*, and are combined as a weighted
mean. Honesty is not one of them, because it is not comparable to them: a
provider that bills fraudulently is not redeemed by being fast and always up —
being quick and available while overcharging you is worse, not better. So
integrity multiplies the result instead of averaging into it, and a proven
pattern of disputed receipts collapses the score no matter how good the service
was.

Integrity starts at 1 and only evidence moves it: a node with no history is
unaffected, one disputed receipt in fifty barely registers, and a node whose
every bill was refused falls to a quarter. Being new is not the same as being
bad; being caught is.

Service counts are Laplace-smoothed, so a provider with two good calls does not
outrank one with two hundred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Any

from relay.provider.offers import Offer, now_utc, parse_time
from relay.store import Store

PROVIDER = "provider"
CONSUMER = "consumer"

# Evidence older than this is history, not a description of the provider now.
DEFAULT_WINDOW = timedelta(days=7)

# A provider must be at least this trustworthy before the default policy will
# use it. Below the prior, which means: a provider with no record is fine, a
# provider with a bad record is not.
DEFAULT_MIN_REPUTATION = 0.4

# Weights for the service components only. Integrity is applied separately.
WEIGHTS = {
    "reliability": 0.55,
    "latency": 0.27,
    "advertising": 0.18,
}

# Pseudo-count for integrity. One, so that a single acknowledged receipt is
# already meaningful and a node with no record sits at 1.0 rather than at a
# prior that would quietly penalise newcomers.
INTEGRITY_PRIOR = 1.0

# Smoothing prior: two pseudo-observations, one good, one bad.
PRIOR_GOOD = 1.0
PRIOR_TOTAL = 2.0


def smoothed(good: float, total: float) -> float:
    return (good + PRIOR_GOOD) / (total + PRIOR_TOTAL)


@dataclass
class Score:
    node_id: str
    role: str
    score: float
    components: dict[str, Any] = field(default_factory=dict)
    computed_at: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "score": self.score,
            "components": self.components,
            "computed_at": self.computed_at,
        }


def _within(row: dict[str, Any], key: str, cutoff: datetime) -> bool:
    stamp = parse_time(row.get(key))
    return stamp is None or stamp >= cutoff


def score_provider(
    node_id: str,
    *,
    receipts: list[dict[str, Any]],
    health: list[dict[str, Any]],
    offers: list[dict[str, Any]],
    market_median_latency: float | None,
    at: datetime,
) -> Score:
    cutoff = at - DEFAULT_WINDOW
    components: dict[str, Any] = {}
    parts: list[tuple[float, float]] = []  # (weight, value)

    mine = [r for r in receipts if r.get("provider_node_id") == node_id]
    recent = [r for r in mine if _within(r, "issued_at", cutoff)]
    acknowledged = sum(1 for r in recent if r.get("status") == "acknowledged")
    disputed = sum(1 for r in recent if r.get("status") == "disputed")
    integrity = (acknowledged + INTEGRITY_PRIOR) / (acknowledged + disputed + INTEGRITY_PRIOR)
    components["integrity"] = {
        "value": round(integrity, 6),
        "acknowledged": acknowledged,
        "disputed": disputed,
        "note": "multiplies the service score; it is not averaged with it",
    }

    observations = [
        h for h in health if h.get("provider_node_id") == node_id and _within(h, "observed_at", cutoff)
    ]
    answered = sum(1 for h in observations if h.get("ok"))
    reliability = smoothed(answered, len(observations))
    components["reliability"] = {
        "value": round(reliability, 6),
        "ok": answered,
        "observations": len(observations),
    }
    parts.append((WEIGHTS["reliability"], reliability))

    latencies = [
        float(h["latency_ms"])
        for h in observations
        if h.get("ok") and h.get("latency_ms") is not None
    ]
    if latencies and market_median_latency:
        mine_median = median(latencies)
        # At the market median this is 0.5; twice as fast approaches 1, twice as
        # slow approaches 0.25. Relative, because "fast" only means anything
        # next to what else is on offer.
        ratio = market_median_latency / max(mine_median, 1.0)
        latency_score = max(0.0, min(1.0, ratio / (ratio + 1.0)))
        components["latency"] = {
            "value": round(latency_score, 6),
            "median_ms": round(mine_median, 1),
            "market_median_ms": round(market_median_latency, 1),
        }
        parts.append((WEIGHTS["latency"], latency_score))
    else:
        components["latency"] = {"value": None, "reason": "not enough observations"}

    live = [
        Offer.from_row(o)
        for o in offers
        if o.get("provider_node_id") == node_id
    ]
    advertising = 1.0 if any(o.is_usable(at) for o in live) else 0.0
    components["advertising"] = {"value": advertising, "offers": len(live)}
    parts.append((WEIGHTS["advertising"], advertising))

    # Renormalize over the components we could actually measure, so a missing
    # signal does not silently drag the score down.
    total_weight = sum(weight for weight, _ in parts)
    service = sum(weight * part for weight, part in parts) / total_weight if total_weight else 0.5
    components["service"] = {"value": round(service, 6)}

    # Dishonesty is not one bad attribute among several. It scales everything.
    value = service * integrity

    return Score(
        node_id=node_id,
        role=PROVIDER,
        score=round(value, 6),
        components=components,
        computed_at=at.isoformat(),
    )


def score_consumer(
    node_id: str, *, receipts: list[dict[str, Any]], disputes: list[dict[str, Any]], at: datetime
) -> Score:
    """Consumers have a record too: do they pay, and do they dispute honestly?

    A provider is entitled to refuse a consumer that disputes everything.
    """
    cutoff = at - DEFAULT_WINDOW
    mine = [
        r
        for r in receipts
        if r.get("consumer_node_id") == node_id and _within(r, "issued_at", cutoff)
    ]
    acknowledged = sum(1 for r in mine if r.get("status") == "acknowledged")
    disputed = sum(1 for r in mine if r.get("status") == "disputed")
    payment = smoothed(acknowledged, acknowledged + disputed)

    opened = [d for d in disputes if d.get("opened_by") == node_id]
    upheld = sum(1 for d in opened if d.get("status") == "upheld")
    rejected = sum(1 for d in opened if d.get("status") == "rejected")
    # Disputes that keep being rejected are the signal for abuse.
    honesty = smoothed(upheld, upheld + rejected)

    value = 0.6 * payment + 0.4 * honesty
    return Score(
        node_id=node_id,
        role=CONSUMER,
        score=round(value, 6),
        components={
            "payment": {"value": round(payment, 6), "acknowledged": acknowledged, "disputed": disputed},
            "dispute_honesty": {"value": round(honesty, 6), "upheld": upheld, "rejected": rejected},
        },
        computed_at=at.isoformat(),
    )


def recompute(store: Store, *, at: datetime | None = None) -> list[Score]:
    """Rescore every node the store knows about.

    Deliberately a pure read-then-write over public data: anyone can run this
    and check the result, which is what stops reputation being an oracle.
    """
    at = at or now_utc()
    receipts = store.list_receipts(limit=5000)
    health = store.list_provider_health(limit=5000)
    offers = store.list_offers()
    disputes = store.list_disputes(limit=5000)

    cutoff = at - DEFAULT_WINDOW
    all_latencies = [
        float(h["latency_ms"])
        for h in health
        if h.get("ok") and h.get("latency_ms") is not None and _within(h, "observed_at", cutoff)
    ]
    market_median = median(all_latencies) if all_latencies else None

    provider_ids = {str(r["provider_node_id"]) for r in receipts if r.get("provider_node_id")}
    provider_ids |= {str(h["provider_node_id"]) for h in health if h.get("provider_node_id")}
    provider_ids |= {str(o["provider_node_id"]) for o in offers if o.get("provider_node_id")}
    consumer_ids = {str(r["consumer_node_id"]) for r in receipts if r.get("consumer_node_id")}

    scores: list[Score] = []
    for node_id in sorted(provider_ids):
        scores.append(
            score_provider(
                node_id,
                receipts=receipts,
                health=health,
                offers=offers,
                market_median_latency=market_median,
                at=at,
            )
        )
    for node_id in sorted(consumer_ids):
        scores.append(score_consumer(node_id, receipts=receipts, disputes=disputes, at=at))

    for score in scores:
        store.upsert_reputation(
            score.node_id, score.role, score.score, score.components, score.computed_at
        )
    return scores


def reputations(store: Store, role: str = PROVIDER) -> dict[str, float]:
    """Current scores, for selection. Unscored nodes are simply absent."""
    try:
        rows = store.list_reputation(role)
    except Exception:
        return {}
    return {str(r["node_id"]): float(r.get("score") or 0.0) for r in rows}
