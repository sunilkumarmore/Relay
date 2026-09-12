"""Finding a provider and choosing between them.

Until now a consumer was told where to send its work. Here it looks: it reads
the directory, keeps the offers that meet its requirements and whose signatures
check out, and applies a policy to pick one. That is the difference between a
client with a configured server and a buyer in a market.

Selection is deliberately separate from execution. This module answers "who
should I ask?"; nothing here performs inference or knows how a request failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Any

from relay.provider.offers import Offer, now_utc, parse_time
from relay.store import Store

CHEAPEST = "cheapest"
FASTEST = "fastest"
ROUND_ROBIN = "round_robin"
PINNED = "pinned"
POLICIES = (CHEAPEST, FASTEST, ROUND_ROBIN, PINNED)

# How long a provider stays benched after failing us.
DEFAULT_FAILOVER_COOLDOWN_SECONDS = 120

# Observations older than this say nothing about how a provider is doing now.
HEALTH_WINDOW = timedelta(minutes=30)

# What a provider with no reputation row is treated as. Matches the smoothing
# prior in relay.reputation, so an unscored node and a freshly scored one with
# no history land in the same place.
UNSCORED = 0.5


class NoProviderAvailable(RuntimeError):
    """No offer in the directory meets the requirements — or every one that did
    has already failed us this run."""


@dataclass
class Requirements:
    """What a job needs. Anything left unset is not a constraint."""

    model: str | None = None
    max_price_in_per_1k: float | None = None
    max_price_out_per_1k: float | None = None
    min_context_window: int | None = None
    region: str | None = None
    min_reputation: float = 0.0
    budget_credits: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Requirements:
        data = data or {}
        return cls(
            model=data.get("model"),
            max_price_in_per_1k=_opt_float(data.get("max_price_in_per_1k")),
            max_price_out_per_1k=_opt_float(data.get("max_price_out_per_1k")),
            min_context_window=_opt_int(data.get("min_context_window")),
            region=data.get("region"),
            min_reputation=float(data.get("min_reputation") or 0.0),
            budget_credits=_opt_float(data.get("budget_credits")),
        )

    def accepts(self, offer: Offer, reputation: float = 0.0) -> bool:
        if self.model is not None and offer.model != self.model:
            return False
        if self.max_price_in_per_1k is not None and offer.price_in_per_1k > self.max_price_in_per_1k:
            return False
        if self.max_price_out_per_1k is not None and offer.price_out_per_1k > self.max_price_out_per_1k:
            return False
        if self.min_context_window is not None and offer.context_window < self.min_context_window:
            return False
        if self.min_reputation > 0.0 and reputation < self.min_reputation:
            return False
        # Region is a preference, not a filter — it orders results instead of
        # emptying them, so a job does not fail because one region is quiet.
        return True


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


class Directory:
    """Read-side view of the market."""

    def __init__(self, store: Store | None, *, min_stake: float = 0.0) -> None:
        self.store = store
        # Providers with nothing at risk are not listed: a slash has to be able
        # to cost them something, or the penalty is theatre.
        self.min_stake = min_stake

    def all_offers(self, model: str | None = None, at: datetime | None = None) -> list[Offer]:
        """Every offer that is currently a live, authentic commitment."""
        if self.store is None:
            return []
        try:
            rows = self.store.list_offers(model)
        except Exception:
            return []

        usable = []
        for row in rows:
            try:
                offer = Offer.from_row(row)
            except Exception:
                # A row we cannot even parse is not an offer we can rely on.
                continue
            if offer.is_usable(at):
                usable.append(offer)
        return usable

    def find_offers(
        self,
        requirements: Requirements,
        *,
        at: datetime | None = None,
        reputations: dict[str, float] | None = None,
    ) -> list[Offer]:
        if reputations is None:
            reputations = self.reputations()
        offers = self.all_offers(requirements.model, at)

        if self.min_stake > 0 and self.store is not None:
            from relay.disputes import staked_providers

            staked = staked_providers(self.store, self.min_stake)
            offers = [o for o in offers if o.provider_node_id in staked]

        return [
            offer
            for offer in offers
            # An unscored provider gets the prior, not zero: being new is not
            # the same as being bad.
            if requirements.accepts(offer, reputations.get(offer.provider_node_id, UNSCORED))
        ]

    def reputations(self) -> dict[str, float]:
        if self.store is None:
            return {}
        from relay.reputation import reputations as current

        return current(self.store)

    def latency_by_provider(self, at: datetime | None = None) -> dict[str, float]:
        """Mean recent latency per provider, from what consumers actually saw."""
        if self.store is None:
            return {}
        try:
            rows = self.store.list_provider_health(limit=500)
        except Exception:
            return {}

        cutoff = (at or now_utc()) - HEALTH_WINDOW
        samples: dict[str, list[float]] = {}
        for row in rows:
            if not row.get("ok"):
                continue
            observed = parse_time(row.get("observed_at"))
            if observed is not None and observed < cutoff:
                continue
            latency = row.get("latency_ms")
            if latency is None:
                continue
            samples.setdefault(str(row.get("provider_node_id", "")), []).append(float(latency))
        return {node: sum(values) / len(values) for node, values in samples.items() if values}


@dataclass
class Selector:
    """Applies a policy to a set of offers.

    Holds the round-robin cursor, which is the only piece of selection that
    needs memory between calls.
    """

    policy: str = CHEAPEST
    pinned_node_id: str | None = None
    _cursor: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown policy {self.policy!r}. Known: {', '.join(POLICIES)}")

    def order(
        self,
        offers: list[Offer],
        *,
        latencies: dict[str, float] | None = None,
        prefer_region: str | None = None,
    ) -> list[Offer]:
        """Rank offers best-first. Failover walks this list in order."""
        if not offers:
            return []

        if self.policy == PINNED:
            if self.pinned_node_id is None:
                return list(offers)
            return [o for o in offers if o.provider_node_id == self.pinned_node_id]

        def region_rank(offer: Offer) -> int:
            if prefer_region is None:
                return 0
            return 0 if offer.region == prefer_region else 1

        if self.policy == CHEAPEST:
            # Output dominates cost in practice, so it breaks the tie first.
            return sorted(
                offers,
                key=lambda o: (region_rank(o), o.price_out_per_1k, o.price_in_per_1k, o.offer_id),
            )

        if self.policy == FASTEST:
            latencies = latencies or {}
            known = [latencies[o.provider_node_id] for o in offers if o.provider_node_id in latencies]
            # An unmeasured provider sits mid-pack rather than last, and wins the
            # tie at that value, so a new entrant gets tried once and measured
            # instead of being frozen out for having no history.
            neutral = median(known) if known else 0.0
            return sorted(
                offers,
                key=lambda o: (
                    region_rank(o),
                    latencies.get(o.provider_node_id, neutral),
                    1 if o.provider_node_id in latencies else 0,
                    o.price_out_per_1k,
                    o.offer_id,
                ),
            )

        # Round robin: stable order, rotated by the cursor, so successive jobs
        # spread across providers instead of all piling onto the cheapest.
        ordered = sorted(offers, key=lambda o: (region_rank(o), o.offer_id))
        shift = self._cursor % len(ordered)
        self._cursor += 1
        return ordered[shift:] + ordered[:shift]

    def select(self, offers: list[Offer], **kwargs: Any) -> Offer | None:
        ranked = self.order(offers, **kwargs)
        return ranked[0] if ranked else None


class Bench:
    """Providers that have failed us recently, and are benched for a while.

    Kept per run rather than in the directory: one consumer's bad experience is
    not grounds for hiding a provider from everyone. Sharing that judgement is
    what reputation is for.
    """

    def __init__(self, cooldown_seconds: int = DEFAULT_FAILOVER_COOLDOWN_SECONDS) -> None:
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self._until: dict[str, datetime] = {}

    def penalize(self, provider_node_id: str, at: datetime | None = None) -> None:
        self._until[provider_node_id] = (at or now_utc()) + self.cooldown

    def is_benched(self, provider_node_id: str, at: datetime | None = None) -> bool:
        until = self._until.get(provider_node_id)
        if until is None:
            return False
        if (at or now_utc()) >= until:
            del self._until[provider_node_id]
            return False
        return True

    def filter(self, offers: list[Offer], at: datetime | None = None) -> list[Offer]:
        return [o for o in offers if not self.is_benched(o.provider_node_id, at)]
