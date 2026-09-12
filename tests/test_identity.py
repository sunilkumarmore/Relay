from __future__ import annotations

import stat

import pytest

from relay.identity import Identity, IdentityError, node_id_for, public_key_from_node_id, verify


def test_node_id_is_the_public_key(tmp_path):
    identity = Identity.generate()
    assert identity.node_id == node_id_for(identity.public_key)
    assert len(identity.node_id) == 64  # 32 raw bytes, hex
    assert public_key_from_node_id(identity.node_id)


def test_sign_and_verify_round_trip():
    identity = Identity.generate()
    sig = identity.sign(b"a message")
    assert verify(identity.node_id, b"a message", sig) is True
    assert verify(identity.node_id, b"a different message", sig) is False


def test_another_nodes_signature_does_not_verify():
    alice, mallory = Identity.generate(), Identity.generate()
    sig = alice.sign(b"m")
    assert verify(mallory.node_id, b"m", sig) is False


def test_garbage_node_id_is_rejected_not_crashed():
    assert verify("not-hex", b"m", b"sig") is False
    with pytest.raises(IdentityError):
        public_key_from_node_id("zz")


def test_key_is_persisted_and_reloaded(tmp_path):
    path = tmp_path / "node.key"
    first = Identity.load_or_create(path)
    second = Identity.load_or_create(path)
    assert first.node_id == second.node_id, "a node must keep its identity across restarts"


def test_key_file_is_not_readable_by_others(tmp_path):
    path = tmp_path / "node.key"
    Identity.load_or_create(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"private key is {oct(mode)}"


def test_separate_paths_are_separate_nodes(tmp_path):
    a = Identity.load_or_create(tmp_path / "a.key")
    b = Identity.load_or_create(tmp_path / "b.key")
    assert a.node_id != b.node_id


def test_unreadable_key_is_a_clear_error(tmp_path):
    path = tmp_path / "bad.key"
    path.write_text("not a pem file", encoding="utf-8")
    with pytest.raises(IdentityError, match="Could not read node key"):
        Identity.load(path)
