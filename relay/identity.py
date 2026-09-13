"""Node identity.

Every Relay node — provider or consumer — is an Ed25519 keypair. The public key,
hex-encoded, *is* the node's id: it is not assigned by a registry, cannot be
claimed by anyone else, and survives the node moving between machines.

``WORKER_ID`` and ``MACHINE_ID`` remain human labels for dashboards and logs.
They are not identity, and nothing may be authorized on the strength of them.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from relay import config

DEFAULT_KEY_PATH = "~/.relay/node.key"


class IdentityError(RuntimeError):
    pass


class Identity:
    """A node's keypair. Only the holder of the private key can act as this node."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private = private_key
        self.public_key = private_key.public_key()

    @property
    def node_id(self) -> str:
        return node_id_for(self.public_key)

    def sign(self, message: bytes) -> bytes:
        return self._private.sign(message)

    # -- persistence ------------------------------------------------------
    @classmethod
    def generate(cls) -> Identity:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: str | os.PathLike[str] | None = None) -> Identity:
        key_path = Path(path or config.get("RELAY_KEY_PATH", DEFAULT_KEY_PATH)).expanduser()
        if key_path.exists():
            return cls.load(key_path)
        identity = cls.generate()
        identity.save(key_path)
        return identity

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Identity:
        raw = Path(path).expanduser().read_bytes()
        try:
            key = serialization.load_pem_private_key(raw, password=None)
        except Exception as exc:
            raise IdentityError(f"Could not read node key at {path}: {exc}") from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise IdentityError(f"Node key at {path} is not an Ed25519 key")
        return cls(key)

    def save(self, path: str | os.PathLike[str]) -> None:
        key_path = Path(path).expanduser()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        pem = self._private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        # Create with 0600 from the start — never briefly world-readable.
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as handle:
            handle.write(pem)
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)


def node_id_for(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw.hex()


def public_key_from_node_id(node_id: str) -> Ed25519PublicKey:
    try:
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(node_id))
    except Exception as exc:
        raise IdentityError(f"Not a valid node id: {node_id!r}") from exc


def verify(node_id: str, message: bytes, signature: bytes) -> bool:
    try:
        public_key_from_node_id(node_id).verify(signature, message)
        return True
    except (InvalidSignature, IdentityError):
        return False


def identity_from_env(env_path: str | None = None) -> Identity:
    config.load_env(env_path)
    return Identity.load_or_create()
