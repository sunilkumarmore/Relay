"""The token exchange: proving a key, getting database access.

This is the piece that makes migration 002's policies mean anything, so the
tests that matter here are the refusals.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from relay.authority.server import Authority, TokenRequest, create_app, signing_message
from relay.authority.tokens import ALGORITHM, TokenError, mint, node_id_of, read
from relay.identity import Identity

SECRET = "a-supabase-jwt-secret"


@pytest.fixture
def authority():
    return Authority(SECRET, ttl_seconds=3600)


@pytest.fixture
def client(authority):
    with TestClient(create_app(authority)) as c:
        yield c


def exchange(client, identity, *, nonce=None, signer=None, node_id=None):
    """Walk the two-round exchange, with hooks to break each half."""
    claimed = node_id or identity.node_id
    if nonce is None:
        challenge = client.post("/auth/challenge", json={"node_id": claimed})
        assert challenge.status_code == 200, challenge.text
        body = challenge.json()
        nonce, audience = body["nonce"], body["audience"]
    else:
        audience = "authenticated"

    signature = (signer or identity).sign(signing_message(claimed, nonce, audience)).hex()
    return client.post(
        "/auth/token", json={"node_id": claimed, "nonce": nonce, "signature": signature}
    )


# -- minting ---------------------------------------------------------------


def test_a_minted_token_carries_the_node_id_claim():
    """Every RLS policy in migration 002 is written against this one claim."""
    identity = Identity.generate()
    issued = mint(identity.node_id, SECRET)
    claims = read(issued.access_token, SECRET)

    assert claims["relay_node_id"] == identity.node_id
    assert claims["role"] == "authenticated", "not service_role, which would bypass RLS entirely"
    assert claims["exp"] > claims["iat"]


def test_a_token_is_a_standard_jwt_postgrest_can_verify():
    issued = mint("ab" * 32, SECRET)
    header = jwt.get_unverified_header(issued.access_token)
    assert header["alg"] == ALGORITHM
    assert header["typ"] == "JWT"


def test_a_token_signed_with_another_secret_does_not_verify():
    issued = mint("ab" * 32, SECRET)
    with pytest.raises(TokenError):
        read(issued.access_token, "not-the-projects-secret")


def test_minting_without_a_secret_says_what_is_missing():
    with pytest.raises(TokenError, match="RELAY_JWT_SECRET"):
        mint("ab" * 32, "")


def test_tokens_expire():
    issued = mint("ab" * 32, SECRET, ttl_seconds=1, now=int(time.time()) - 10)
    assert issued.expires_in == 0
    with pytest.raises(TokenError):
        read(issued.access_token, SECRET)


def test_peeking_at_a_token_never_raises():
    assert node_id_of("not-a-token") == ""
    assert node_id_of(mint("cd" * 32, SECRET).access_token) == "cd" * 32


# -- the exchange ----------------------------------------------------------


def test_a_node_that_holds_its_key_gets_a_token(client):
    identity = Identity.generate()
    response = exchange(client, identity)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["node_id"] == identity.node_id
    assert read(body["access_token"], SECRET)["relay_node_id"] == identity.node_id


def test_a_signature_from_another_key_is_refused(client):
    """The whole point: holding the key is what proves the identity."""
    victim, impostor = Identity.generate(), Identity.generate()
    response = exchange(client, victim, signer=impostor)

    assert response.status_code == 401
    assert response.json()["detail"] == "bad_signature"


def test_a_nonce_works_exactly_once(client):
    """A token request replayed inside a clock-skew window would hand the
    replayer a bearer credential, which is why this is a nonce and not a
    timestamp like the provider-call signing uses."""
    identity = Identity.generate()
    nonce = client.post("/auth/challenge", json={"node_id": identity.node_id}).json()["nonce"]

    assert exchange(client, identity, nonce=nonce).status_code == 200
    replay = exchange(client, identity, nonce=nonce)
    assert replay.status_code == 401
    assert replay.json()["detail"] == "unknown_or_used_challenge"


def test_a_wrong_guess_burns_the_challenge(client):
    """Otherwise a nonce is an unlimited number of attempts."""
    identity, impostor = Identity.generate(), Identity.generate()
    nonce = client.post("/auth/challenge", json={"node_id": identity.node_id}).json()["nonce"]

    assert exchange(client, identity, nonce=nonce, signer=impostor).status_code == 401
    assert exchange(client, identity, nonce=nonce).status_code == 401


def test_an_invented_nonce_is_refused(client):
    identity = Identity.generate()
    response = exchange(client, identity, nonce="a-nonce-nobody-issued")
    assert response.status_code == 401


def test_a_challenge_cannot_be_redeemed_by_a_different_node(client):
    alice, mallory = Identity.generate(), Identity.generate()
    nonce = client.post("/auth/challenge", json={"node_id": alice.node_id}).json()["nonce"]

    # Mallory signs correctly, but for a challenge issued to Alice.
    signature = mallory.sign(signing_message(mallory.node_id, nonce, "authenticated")).hex()
    response = client.post(
        "/auth/token",
        json={"node_id": mallory.node_id, "nonce": nonce, "signature": signature},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "challenge_belongs_to_another_node"


def test_an_expired_challenge_is_refused(authority):
    identity = Identity.generate()
    issued = authority.challenge(identity.node_id, now=1000.0)
    signature = identity.sign(signing_message(identity.node_id, issued["nonce"], "authenticated")).hex()

    with pytest.raises(Exception) as exc:
        authority.token(
            TokenRequest(node_id=identity.node_id, nonce=issued["nonce"], signature=signature),
            now=1000.0 + authority.challenge_ttl_seconds + 1,
        )
    assert exc.value.status_code == 401


def test_a_signature_for_another_context_does_not_transfer(client):
    """The signed message names this exchange, so a signature gathered
    elsewhere cannot be presented as proof here."""
    identity = Identity.generate()
    nonce = client.post("/auth/challenge", json={"node_id": identity.node_id}).json()["nonce"]

    wrong = identity.sign(b"some other message entirely").hex()
    response = client.post(
        "/auth/token", json={"node_id": identity.node_id, "nonce": nonce, "signature": wrong}
    )
    assert response.status_code == 401


def test_a_malformed_node_id_is_rejected_at_the_challenge(client):
    response = client.post("/auth/challenge", json={"node_id": "not-a-key"})
    assert response.status_code == 400
    assert response.json()["detail"] == "not_a_node_id"


def test_a_malformed_signature_is_rejected_cleanly(client):
    identity = Identity.generate()
    nonce = client.post("/auth/challenge", json={"node_id": identity.node_id}).json()["nonce"]
    response = client.post(
        "/auth/token", json={"node_id": identity.node_id, "nonce": nonce, "signature": "zzz"}
    )
    assert response.status_code == 401


def test_expired_challenges_do_not_accumulate(authority):
    for _ in range(5):
        authority.challenge(Identity.generate().node_id, now=1000.0)
    assert authority.outstanding_challenges == 0, "pruned once past their TTL"


def test_anyone_with_a_key_may_get_a_token(client):
    """Deliberate. A token says which node you are; what that node may do is
    the database's business, not this service's."""
    for _ in range(3):
        assert exchange(client, Identity.generate()).status_code == 200


def test_health_reports_whether_a_secret_is_configured():
    with TestClient(create_app(Authority(""))) as c:
        assert c.get("/health").json()["configured"] is False
    with TestClient(create_app(Authority(SECRET))) as c:
        assert c.get("/health").json()["configured"] is True


def test_an_unconfigured_authority_refuses_rather_than_issues(client):
    unconfigured = Authority("")
    with TestClient(create_app(unconfigured)) as c:
        identity = Identity.generate()
        assert exchange(c, identity).status_code == 503
