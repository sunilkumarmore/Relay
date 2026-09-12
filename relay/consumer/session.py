"""A consumer's working relationship with a provider.

The worker used to hold a URL. It now holds a *binding*: an offer it selected, a
provider it registered with, and the terms it expects to be charged under. When
that provider stops working, the binding is replaced and the step is retried —
the job does not fail because one seller did.

What counts as "stops working" is deliberately narrow. A connection error, a
timeout, a 5xx, or a 429 are the provider's problem and trigger a switch. A 4xx
is ours and fails immediately: asking a second provider the same malformed
question will not help.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from relay import tokens as tokenizer
from relay.auth import RelayAuth
from relay.consumer.market import (
    DEFAULT_FAILOVER_COOLDOWN_SECONDS,
    Bench,
    Directory,
    NoProviderAvailable,
    Requirements,
    Selector,
)
from relay.disputes import REASON_HASH_MISMATCH, REASON_TOKEN_OVERCLAIM, open_dispute
from relay.identity import Identity
from relay.ledger import Ledger
from relay.provider.offers import Offer
from relay.receipts import Receipt, request_fingerprint, verify_receipt
from relay.store import Store

REGISTER_TIMEOUT_SECONDS = 10
HEARTBEAT_TIMEOUT_SECONDS = 5
INFERENCE_TIMEOUT_SECONDS = 240


@dataclass
class Binding:
    """Who we are currently buying from, and on what terms."""

    endpoint_url: str
    provider_node_id: str
    inference_node_id: str
    offer: Offer | None = None

    @property
    def is_direct(self) -> bool:
        """A pinned URL with no offer behind it — the pre-market arrangement,
        kept so an operator can still point a worker straight at a node."""
        return self.offer is None


class ProviderUnavailable(RuntimeError):
    """This provider failed in a way that is worth trying someone else for."""


class MarketSession:
    def __init__(
        self,
        identity: Identity,
        store: Store | None,
        *,
        worker_id: str,
        session_id: str,
        machine_id: str,
        requirements: Requirements | None = None,
        selector: Selector | None = None,
        pinned_url: str | None = None,
        cooldown_seconds: int = DEFAULT_FAILOVER_COOLDOWN_SECONDS,
        max_retries: int = 3,
        on_switch: Callable[[str, str, str], None] | None = None,
        prefer_provider: str | None = None,
        ledger: Ledger | None = None,
        min_stake: float = 0.0,
        verify_sample_rate: float = 0.0,
    ) -> None:
        self.identity = identity
        self.store = store
        self.worker_id = worker_id
        self.session_id = session_id
        self.machine_id = machine_id
        self.requirements = requirements or Requirements()
        self.selector = selector or Selector()
        self.pinned_url = pinned_url.rstrip("/") if pinned_url else None
        self.max_retries = max_retries
        self.on_switch = on_switch
        self.prefer_provider = prefer_provider or None

        self.directory = Directory(store, min_stake=min_stake)
        self.verify_sample_rate = max(0.0, min(1.0, verify_sample_rate))
        self.bench = Bench(cooldown_seconds)
        self.ledger = ledger
        self.binding: Binding | None = None
        self.switches = 0
        self.spent = 0.0
        self.disputed: list[Receipt] = []
        self.verifications: list[dict[str, Any]] = []

    @property
    def auth(self) -> RelayAuth:
        return RelayAuth(self.identity)

    # -- choosing ---------------------------------------------------------
    def candidates(self) -> list[Offer]:
        offers = self.directory.find_offers(self.requirements)
        available = self.bench.filter(offers)
        ranked = self.selector.order(
            available,
            latencies=self.directory.latency_by_provider(),
            prefer_region=self.requirements.region,
        )
        if self.prefer_provider:
            # A resumed worker goes back to the provider it was using, if that
            # provider is still offering terms we accept.
            preferred = [o for o in ranked if o.provider_node_id == self.prefer_provider]
            ranked = preferred + [o for o in ranked if o.provider_node_id != self.prefer_provider]
        return ranked

    def bind(self, reason: str = "initial") -> Binding:
        """Select a provider and register with it."""
        previous = self.binding.provider_node_id if self.binding else ""
        last_error: Exception | None = None

        for offer in self.candidates():
            try:
                node_label = self._register(offer.endpoint_url)
            except Exception as exc:  # unreachable, refusing us, or misbehaving
                last_error = exc
                self._observe(offer.provider_node_id, ok=False, error=str(exc))
                self.bench.penalize(offer.provider_node_id)
                continue

            self.binding = Binding(
                endpoint_url=offer.endpoint_url,
                provider_node_id=offer.provider_node_id,
                inference_node_id=node_label,
                offer=offer,
            )
            if previous and previous != offer.provider_node_id:
                self.switches += 1
                if self.on_switch:
                    self.on_switch(previous, offer.provider_node_id, reason)
            return self.binding

        if self.pinned_url:
            # No usable offers, but we were told exactly where to go.
            node_label = self._register(self.pinned_url)
            self.binding = Binding(
                endpoint_url=self.pinned_url,
                provider_node_id="",
                inference_node_id=node_label,
                offer=None,
            )
            return self.binding

        raise NoProviderAvailable(
            "No provider in the directory meets this job's requirements"
            + (f" (last error: {last_error})" if last_error else "")
        )

    def ensure_bound(self) -> Binding:
        return self.binding or self.bind()

    # -- talking to the provider -----------------------------------------
    def _register(self, endpoint_url: str) -> str:
        response = requests.post(
            f"{endpoint_url}/worker/register",
            json={
                "worker_id": self.worker_id,
                "session_id": self.session_id,
                "machine_id": self.machine_id,
            },
            timeout=REGISTER_TIMEOUT_SECONDS,
            auth=self.auth,
        )
        response.raise_for_status()
        return str(response.json().get("inference_node_id", "unknown"))

    def heartbeat(self, steps_completed: int) -> None:
        if self.binding is None:
            return
        try:
            requests.post(
                f"{self.binding.endpoint_url}/worker/heartbeat",
                json={
                    "worker_id": self.worker_id,
                    "last_checkpoint": steps_completed,
                    "steps_completed": steps_completed,
                },
                timeout=HEARTBEAT_TIMEOUT_SECONDS,
                auth=self.auth,
            )
        except Exception:
            # A missed heartbeat is not worth failing over for; the next
            # inference call will find out soon enough.
            pass

    def release(self) -> None:
        if self.binding is None:
            return
        try:
            requests.post(
                f"{self.binding.endpoint_url}/worker/deregister",
                json={"worker_id": self.worker_id},
                timeout=HEARTBEAT_TIMEOUT_SECONDS,
                auth=self.auth,
            )
        except Exception:
            pass

    def _attempt(
        self, binding: Binding, prompt: str, max_tokens: int, step_number: int = 0
    ) -> dict[str, Any]:
        """One provider, with retries. Raises ProviderUnavailable to fail over."""
        payload: dict[str, Any] = {
            "worker_id": self.worker_id,
            "session_id": self.session_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "step_number": step_number,
        }
        if binding.offer is not None:
            payload["model"] = binding.offer.model
        url = f"{binding.endpoint_url}/inference/complete"
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url, json=payload, timeout=INFERENCE_TIMEOUT_SECONDS, auth=self.auth
                )
                if response.status_code == 401:
                    # The provider forgot us — it restarted, or pruned us as stale.
                    self._register(binding.endpoint_url)
                    response = requests.post(
                        url, json=payload, timeout=INFERENCE_TIMEOUT_SECONDS, auth=self.auth
                    )
                if response.status_code == 429:
                    # Advertised capacity is gone. Someone else may have some.
                    retry_after = response.headers.get("Retry-After", "?")
                    raise ProviderUnavailable(f"saturated (retry after {retry_after}s)")
                response.raise_for_status()
                return response.json()
            except ProviderUnavailable:
                raise
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 500
                if status < 500:
                    # Our mistake. Another provider will reject it identically.
                    raise
                last_exc = exc
            if attempt < self.max_retries:
                wait = 2**attempt
                print(
                    f"Inference attempt {attempt + 1}/{self.max_retries + 1} failed, "
                    f"retrying in {wait}s: {last_exc}"
                )
                time.sleep(wait)

        raise ProviderUnavailable(f"failed after {self.max_retries + 1} attempts: {last_exc}")

    def complete(self, prompt: str, max_tokens: int, step_number: int = 0) -> dict[str, Any]:
        """Run one step, changing provider as many times as it takes."""
        binding = self.ensure_bound()
        # One shot at every provider that qualifies, plus the one we start on.
        budget = max(1, len(self.candidates())) + 1

        for _ in range(budget):
            started = time.perf_counter()
            try:
                result = self._attempt(binding, prompt, max_tokens, step_number)
            except ProviderUnavailable as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                self._observe(binding.provider_node_id, ok=False, latency_ms=latency_ms, error=str(exc))
                if binding.provider_node_id:
                    self.bench.penalize(binding.provider_node_id)
                print(f"Provider {binding.provider_node_id[:12] or binding.endpoint_url} unavailable: {exc}")
                try:
                    binding = self.bind(reason=str(exc)[:120])
                except NoProviderAvailable:
                    raise RuntimeError(f"No provider could serve this step: {exc}") from exc
                continue

            self._observe(
                binding.provider_node_id,
                ok=True,
                latency_ms=int(result.get("latency_ms") or 0),
            )
            self._handle_receipt(binding, result, prompt, max_tokens, step_number)
            return result

        raise RuntimeError("Exhausted every provider that met this job's requirements")

    # -- paying -----------------------------------------------------------
    def _handle_receipt(
        self,
        binding: Binding,
        result: dict[str, Any],
        prompt: str,
        max_tokens: int,
        step_number: int,
    ) -> None:
        """Check the bill before paying it, then pay it.

        A receipt that does not survive checking is recorded as disputed rather
        than quietly dropped: the work was still done, and Phase 5 decides who
        was right. What we do not do is countersign it.
        """
        if binding.offer is None or "receipt" not in result:
            # A pinned endpoint with no offer behind it bills nothing.
            return

        try:
            receipt = Receipt.from_row(result["receipt"])
        except Exception as exc:
            print(f"Provider returned an unreadable receipt: {exc}")
            return

        response_text = str(result.get("response", "")).strip()
        problems = verify_receipt(
            receipt,
            offer=binding.offer,
            consumer_node_id=self.identity.node_id,
            request_hash=request_fingerprint(prompt, max_tokens, binding.offer.model),
            response_text=response_text,
            job_id=self.session_id,
            step_number=step_number,
        )
        if self._should_sample():
            # An audit, not a verdict — run it regardless of how the bill checks
            # out, since the point is to find patterns over many calls.
            self._cross_check(binding, prompt, max_tokens, result)

        problems += self._token_objections(receipt, prompt, response_text, binding.offer.model)

        if problems:
            reason = "; ".join(problems)
            print(f"Refusing to acknowledge receipt {receipt.receipt_id[:8]}: {reason}")
            disputed = receipt.disputed(reason)
            self.disputed.append(disputed)
            self._save_receipt(disputed)
            self._open_dispute(receipt, problems, max_tokens)
            return

        acknowledged = receipt.acknowledged_by(self.identity)
        self._save_receipt(acknowledged)
        self.spent = round(self.spent + acknowledged.amount_credits, 6)

        if self.ledger is not None:
            try:
                self.ledger.settle(
                    self.identity.node_id,
                    acknowledged.provider_node_id,
                    acknowledged.amount_credits,
                    receipt_id=acknowledged.receipt_id,
                    job_id=self.session_id,
                )
            except Exception as exc:
                print(f"Could not settle receipt {acknowledged.receipt_id[:8]}: {exc}")

    def _token_objections(
        self, receipt: Receipt, prompt: str, response_text: str, model: str
    ) -> list[str]:
        """Is the provider's token claim credible against our own count?

        Only over-claiming is an objection. Under-claiming means the provider
        charged us less than it could have, which is not a problem we have.
        """
        objections = []
        for label, text, claimed in (
            ("input", prompt, receipt.tokens_in),
            ("output", response_text, receipt.tokens_out),
        ):
            measured = tokenizer.estimate(text, model)
            if not measured.permits(claimed):
                objections.append(
                    f"{label} tokens claimed {claimed}, we measured about {measured.tokens} "
                    f"(tolerance {int(measured.tolerance * 100)}%)"
                )
        return objections

    def _open_dispute(self, receipt: Receipt, problems: list[str], max_tokens: int) -> None:
        """Raise it formally, but only for the kinds a program can rule on."""
        if self.store is None:
            return
        joined = " ".join(problems)
        if "tokens claimed" in joined:
            reason = REASON_TOKEN_OVERCLAIM
        elif "hash does not match" in joined:
            reason = REASON_HASH_MISMATCH
        else:
            # Price or identity mismatches are real, but they are settled by
            # simply not paying; there is nothing for an adjudicator to weigh.
            return
        try:
            open_dispute(
                self.store,
                receipt_id=receipt.receipt_id,
                opened_by=self.identity.node_id,
                reason=reason,
                evidence={"problems": problems, "max_tokens": max_tokens},
            )
        except Exception:
            pass

    def _should_sample(self) -> bool:
        if self.verify_sample_rate <= 0:
            return False
        import random

        return random.random() < self.verify_sample_rate

    def _cross_check(
        self, binding: Binding, prompt: str, max_tokens: int, result: dict[str, Any]
    ) -> None:
        """Ask a second provider the same question and compare.

        For a deterministic model this is an exact comparison. Otherwise all it
        can honestly check is plausibility — that the token counts and latency
        are in the same world. Either way the outcome feeds reputation rather
        than triggering a payment: sampling finds patterns, not verdicts.
        """
        if binding.offer is None:
            return
        others = [
            o
            for o in self.candidates()
            if o.provider_node_id != binding.provider_node_id and o.model == binding.offer.model
        ]
        if not others:
            return

        second = others[0]
        try:
            node_label = self._register(second.endpoint_url)
            other_binding = Binding(second.endpoint_url, second.provider_node_id, node_label, second)
            other = self._attempt(other_binding, prompt, max_tokens)
        except Exception:
            # The cross-check failing says nothing about the provider we bought
            # from, so it is not evidence against anyone.
            return

        ours = str(result.get("response", "")).strip()
        theirs = str(other.get("response", "")).strip()
        our_tokens = int(result.get("tokens_out") or 0)
        their_tokens = int(other.get("tokens_out") or 0)
        ceiling = max(their_tokens * 3, their_tokens + 32)

        finding = {
            "provider_node_id": binding.provider_node_id,
            "compared_with": second.provider_node_id,
            "identical": ours == theirs,
            "tokens_out": our_tokens,
            "other_tokens_out": their_tokens,
            "plausible": our_tokens <= ceiling,
        }
        self.verifications.append(finding)

        if self.store is not None:
            try:
                self.store.insert_provider_event(
                    binding.provider_node_id, "cross_checked", binding.offer.model, finding
                )
            except Exception:
                pass

    def _save_receipt(self, receipt: Receipt) -> None:
        if self.store is None:
            return
        try:
            self.store.upsert_receipt(receipt.to_row())
        except Exception:
            pass

    # -- telemetry --------------------------------------------------------
    def _observe(
        self, provider_node_id: str, *, ok: bool, latency_ms: int | None = None, error: str = ""
    ) -> None:
        """Record what we saw. Shared so other consumers can learn from it."""
        if self.store is None or not provider_node_id:
            return
        try:
            self.store.insert_provider_health(
                observer_node_id=self.identity.node_id,
                provider_node_id=provider_node_id,
                ok=ok,
                latency_ms=latency_ms,
                error=error[:500],
            )
        except Exception:
            pass
