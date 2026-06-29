"""
Cryptographic utilities for drone security.

Provides command signing, message encryption, and secure hashing.
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import hmac
import secrets
import base64
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


class CryptoError(Exception):
    """Cryptographic operation failed."""
    pass


class SignatureError(CryptoError):
    """Signature verification failed."""
    pass


class EncryptionError(CryptoError):
    """Encryption/decryption failed."""
    pass


@dataclass
class SignedCommand:
    """A signed drone command."""
    command_type: str
    parameters: Dict[str, Any]
    timestamp: float
    nonce: str
    signature: str
    signer_id: str

    def is_expired(self, max_age: float = 300) -> bool:
        """Check if command is expired (default 5 minutes)."""
        return (time.time() - self.timestamp) > max_age

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'command_type': self.command_type,
            'parameters': self.parameters,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'signature': self.signature,
            'signer_id': self.signer_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignedCommand':
        """Create from dictionary."""
        return cls(
            command_type=data['command_type'],
            parameters=data.get('parameters', {}),
            timestamp=data['timestamp'],
            nonce=data['nonce'],
            signature=data['signature'],
            signer_id=data['signer_id'],
        )


class CommandSigner:
    """
    Signs and verifies drone commands.

    Uses HMAC-SHA256 for command authentication.
    Prevents command replay and tampering.
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize command signer.

        Args:
            secret_key: Shared secret for signing
        """
        self._secret = secret_key or os.environ.get(
            'DRONE_COMMAND_SECRET',
            secrets.token_hex(32)
        )
        self._used_nonces: set = set()
        self._nonce_max_age = 600  # 10 minutes

    def sign_command(
        self,
        command_type: str,
        parameters: Dict[str, Any],
        signer_id: str,
    ) -> SignedCommand:
        """
        Sign a drone command.

        Args:
            command_type: Type of command (e.g., 'takeoff', 'land')
            parameters: Command parameters
            signer_id: ID of the signer

        Returns:
            SignedCommand with signature
        """
        timestamp = time.time()
        nonce = secrets.token_hex(16)

        # Create message to sign
        message = self._create_message(
            command_type, parameters, timestamp, nonce, signer_id
        )

        # Create signature
        signature = hmac.new(
            self._secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return SignedCommand(
            command_type=command_type,
            parameters=parameters,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            signer_id=signer_id,
        )

    def verify_command(
        self,
        command: SignedCommand,
        max_age: float = 300,
    ) -> bool:
        """
        Verify a signed command.

        Args:
            command: Signed command to verify
            max_age: Maximum age in seconds

        Returns:
            True if valid

        Raises:
            SignatureError: If verification fails
        """
        # Check expiration
        if command.is_expired(max_age):
            raise SignatureError("Command has expired")

        # Check nonce (prevent replay)
        if command.nonce in self._used_nonces:
            raise SignatureError("Command nonce already used (replay attack)")

        # Recreate message
        message = self._create_message(
            command.command_type,
            command.parameters,
            command.timestamp,
            command.nonce,
            command.signer_id,
        )

        # Verify signature
        expected_sig = hmac.new(
            self._secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(command.signature, expected_sig):
            raise SignatureError("Invalid signature")

        # Mark nonce as used
        self._used_nonces.add(command.nonce)
        self._cleanup_old_nonces()

        return True

    def _create_message(
        self,
        command_type: str,
        parameters: Dict[str, Any],
        timestamp: float,
        nonce: str,
        signer_id: str,
    ) -> str:
        """Create canonical message for signing."""
        # Sort parameters for consistent ordering
        sorted_params = json.dumps(parameters, sort_keys=True)
        return f"{command_type}|{sorted_params}|{timestamp}|{nonce}|{signer_id}"

    def _cleanup_old_nonces(self):
        """Remove old nonces to prevent memory growth."""
        # Simple cleanup - in production, use time-based expiration
        if len(self._used_nonces) > 10000:
            self._used_nonces.clear()


class MessageEncryptor:
    """
    Authenticated message encryption.

    Encrypt-then-MAC over a stream cipher built from PBKDF2(key, iv). The
    HMAC-SHA256 tag is verified *before* the plaintext is decoded, so a
    wrong key (or tampered ciphertext) always raises ``EncryptionError``
    deterministically instead of returning garbage that may or may not
    decode as UTF-8.

    Wire format (base64-encoded):
        iv (16 B) || tag (32 B) || ciphertext (variable)

    For high-assurance deployments still prefer AES-GCM from a vetted
    library; this construction is stdlib-only for portability.
    """

    _IV_LEN = 16
    _TAG_LEN = 32

    def __init__(self, key: Optional[str] = None):
        """
        Initialize encryptor.

        Args:
            key: Encryption key (32+ characters recommended)
        """
        self._key = key or os.environ.get(
            'DRONE_ENCRYPTION_KEY',
            secrets.token_hex(32)
        )

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a message and authenticate it. Returns base64."""
        iv = secrets.token_bytes(self._IV_LEN)
        enc_key, mac_key = self._derive_keys(iv)
        pt_bytes = plaintext.encode('utf-8')
        ct = self._xor_stream(pt_bytes, enc_key)
        tag = hmac.new(mac_key, iv + ct, hashlib.sha256).digest()
        return base64.b64encode(iv + tag + ct).decode('ascii')

    def decrypt(self, ciphertext: str) -> str:
        """Verify MAC and decrypt. Raises ``EncryptionError`` on any failure."""
        try:
            blob = base64.b64decode(ciphertext.encode('ascii'))
        except Exception as exc:
            raise EncryptionError(f"Decryption failed: invalid base64 ({exc})")
        if len(blob) < self._IV_LEN + self._TAG_LEN:
            raise EncryptionError("Decryption failed: ciphertext too short")
        iv = blob[:self._IV_LEN]
        tag = blob[self._IV_LEN:self._IV_LEN + self._TAG_LEN]
        ct = blob[self._IV_LEN + self._TAG_LEN:]
        enc_key, mac_key = self._derive_keys(iv)
        expected = hmac.new(mac_key, iv + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise EncryptionError("Decryption failed: authentication tag mismatch")
        pt_bytes = self._xor_stream(ct, enc_key)
        try:
            return pt_bytes.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise EncryptionError(f"Decryption failed: invalid utf-8 ({exc})")

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt a dictionary as JSON."""
        return self.encrypt(json.dumps(data))

    def decrypt_dict(self, ciphertext: str) -> Dict[str, Any]:
        """Decrypt to dictionary."""
        return json.loads(self.decrypt(ciphertext))

    def _derive_keys(self, iv: bytes) -> Tuple[bytes, bytes]:
        """Derive (encryption key, MAC key) from the master key and IV."""
        material = hashlib.pbkdf2_hmac(
            'sha256',
            self._key.encode('utf-8'),
            iv,
            10000,
            dklen=64,
        )
        return material[:32], material[32:]

    @staticmethod
    def _xor_stream(data: bytes, key: bytes) -> bytes:
        """XOR ``data`` against a keystream derived by counter-mode hashing."""
        out = bytearray(len(data))
        keystream = bytearray()
        counter = 0
        while len(keystream) < len(data):
            keystream += hashlib.sha256(key + counter.to_bytes(8, 'big')).digest()
            counter += 1
        for i, b in enumerate(data):
            out[i] = b ^ keystream[i]
        return bytes(out)

    # Backwards-compat shim: some old call sites may still use _derive_key.
    def _derive_key(self, iv: bytes) -> bytes:
        return self._derive_keys(iv)[0]


class SecureHash:
    """
    Secure hashing utilities.

    Provides password hashing and verification.
    """

    def __init__(self, iterations: int = 100000):
        """
        Initialize hasher.

        Args:
            iterations: PBKDF2 iterations
        """
        self._iterations = iterations

    def hash_password(self, password: str) -> str:
        """
        Hash a password.

        Args:
            password: Plain text password

        Returns:
            Salted hash string
        """
        salt = secrets.token_bytes(32)
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt,
            self._iterations
        )

        # Combine salt and hash
        combined = salt + hash_bytes
        return base64.b64encode(combined).decode('utf-8')

    def verify_password(self, password: str, hash_string: str) -> bool:
        """
        Verify a password against hash.

        Args:
            password: Plain text password
            hash_string: Stored hash string

        Returns:
            True if password matches
        """
        try:
            combined = base64.b64decode(hash_string.encode('utf-8'))
            salt = combined[:32]
            stored_hash = combined[32:]

            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt,
                self._iterations
            )

            return hmac.compare_digest(stored_hash, computed_hash)

        except Exception:
            return False

    def hash_data(self, data: bytes) -> str:
        """Create SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()

    def hash_file(self, filepath: str) -> str:
        """Create SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


def generate_secure_key(length: int = 32) -> str:
    """Generate a secure random key."""
    return secrets.token_hex(length)


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time."""
    return hmac.compare_digest(a.encode(), b.encode())
