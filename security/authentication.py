"""
Authentication System for Drone API.

Provides API key management and JWT token authentication.
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import secrets
import base64
import hmac
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime, timedelta


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


class TokenExpiredError(AuthenticationError):
    """Token has expired."""
    pass


@dataclass
class APIKey:
    """API key with metadata."""
    key_id: str
    key_hash: str  # Store hash, not plain key
    name: str
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    roles: List[str] = field(default_factory=lambda: ['viewer'])
    enabled: bool = True
    last_used: Optional[float] = None
    usage_count: int = 0

    def is_valid(self) -> bool:
        """Check if key is valid (enabled and not expired)."""
        if not self.enabled:
            return False
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True

    def record_usage(self):
        """Record key usage."""
        self.last_used = time.time()
        self.usage_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (safe, no key_hash)."""
        return {
            'key_id': self.key_id,
            'name': self.name,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'roles': self.roles,
            'enabled': self.enabled,
            'last_used': self.last_used,
            'usage_count': self.usage_count,
        }


@dataclass
class AuthToken:
    """JWT-like authentication token."""
    token_id: str
    user_id: str
    roles: List[str]
    issued_at: float
    expires_at: float
    claims: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'token_id': self.token_id,
            'user_id': self.user_id,
            'roles': self.roles,
            'issued_at': self.issued_at,
            'expires_at': self.expires_at,
            'claims': self.claims,
        }


class APIKeyManager:
    """
    Manages API keys for authentication.

    API keys are used for service-to-service authentication
    and programmatic access to the drone API.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize API key manager.

        Args:
            storage_path: Path to store API keys (JSON file)
        """
        self._keys: Dict[str, APIKey] = {}
        self._storage_path = storage_path
        self._secret = os.environ.get('DRONE_API_SECRET', secrets.token_hex(32))

        if storage_path and os.path.exists(storage_path):
            self._load_keys()

    def generate_key(
        self,
        name: str,
        roles: Optional[List[str]] = None,
        expires_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.

        Args:
            name: Descriptive name for the key
            roles: Roles to assign to the key
            expires_days: Days until expiration (None = never)

        Returns:
            Tuple of (plain_key, APIKey object)
        """
        # Generate secure random key
        plain_key = f"drone_{secrets.token_urlsafe(32)}"
        key_id = secrets.token_hex(8)
        key_hash = self._hash_key(plain_key)

        expires_at = None
        if expires_days:
            expires_at = time.time() + (expires_days * 86400)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            roles=roles or ['viewer'],
            expires_at=expires_at,
        )

        self._keys[key_id] = api_key
        self._save_keys()

        return plain_key, api_key

    def validate_key(self, plain_key: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Args:
            plain_key: The plain text API key

        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = self._hash_key(plain_key)

        for api_key in self._keys.values():
            if hmac.compare_digest(api_key.key_hash, key_hash):
                if api_key.is_valid():
                    api_key.record_usage()
                    self._save_keys()
                    return api_key
                return None

        return None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].enabled = False
            self._save_keys()
            return True
        return False

    def delete_key(self, key_id: str) -> bool:
        """Delete an API key."""
        if key_id in self._keys:
            del self._keys[key_id]
            self._save_keys()
            return True
        return False

    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self._keys.get(key_id)

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (safe, no hashes)."""
        return [key.to_dict() for key in self._keys.values()]

    def _hash_key(self, plain_key: str) -> str:
        """Hash an API key with secret."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            plain_key.encode(),
            self._secret.encode(),
            100000
        ).hex()

    def _save_keys(self):
        """Save keys to storage."""
        if not self._storage_path:
            return

        data = {}
        for key_id, api_key in self._keys.items():
            data[key_id] = {
                'key_id': api_key.key_id,
                'key_hash': api_key.key_hash,
                'name': api_key.name,
                'created_at': api_key.created_at,
                'expires_at': api_key.expires_at,
                'roles': api_key.roles,
                'enabled': api_key.enabled,
                'last_used': api_key.last_used,
                'usage_count': api_key.usage_count,
            }

        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        with open(self._storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_keys(self):
        """Load keys from storage."""
        try:
            with open(self._storage_path, 'r') as f:
                data = json.load(f)

            for key_id, key_data in data.items():
                self._keys[key_id] = APIKey(
                    key_id=key_data['key_id'],
                    key_hash=key_data['key_hash'],
                    name=key_data['name'],
                    created_at=key_data.get('created_at', time.time()),
                    expires_at=key_data.get('expires_at'),
                    roles=key_data.get('roles', ['viewer']),
                    enabled=key_data.get('enabled', True),
                    last_used=key_data.get('last_used'),
                    usage_count=key_data.get('usage_count', 0),
                )
        except Exception:
            pass


class TokenManager:
    """
    Manages JWT-like authentication tokens.

    Tokens are used for session-based authentication
    after initial login with API key or credentials.
    """

    def __init__(self, secret: Optional[str] = None, token_lifetime: int = 3600):
        """
        Initialize token manager.

        Args:
            secret: Secret key for signing tokens
            token_lifetime: Token lifetime in seconds (default 1 hour)
        """
        self._secret = secret or os.environ.get('DRONE_TOKEN_SECRET', secrets.token_hex(32))
        self._token_lifetime = token_lifetime
        self._active_tokens: Dict[str, AuthToken] = {}
        self._revoked_tokens: set = set()

    def create_token(
        self,
        user_id: str,
        roles: List[str],
        claims: Optional[Dict[str, Any]] = None,
        lifetime: Optional[int] = None,
    ) -> str:
        """
        Create a new authentication token.

        Args:
            user_id: User identifier
            roles: User roles
            claims: Additional claims
            lifetime: Custom lifetime in seconds

        Returns:
            Encoded token string
        """
        token_id = secrets.token_hex(16)
        issued_at = time.time()
        expires_at = issued_at + (lifetime or self._token_lifetime)

        token = AuthToken(
            token_id=token_id,
            user_id=user_id,
            roles=roles,
            issued_at=issued_at,
            expires_at=expires_at,
            claims=claims or {},
        )

        self._active_tokens[token_id] = token

        # Encode token
        return self._encode_token(token)

    def validate_token(self, token_string: str) -> AuthToken:
        """
        Validate and decode a token.

        Args:
            token_string: Encoded token string

        Returns:
            AuthToken if valid

        Raises:
            AuthenticationError: If token is invalid
            TokenExpiredError: If token is expired
        """
        try:
            token = self._decode_token(token_string)
        except Exception as e:
            raise AuthenticationError(f"Invalid token: {e}")

        if token.token_id in self._revoked_tokens:
            raise AuthenticationError("Token has been revoked")

        if token.is_expired():
            raise TokenExpiredError("Token has expired")

        return token

    def revoke_token(self, token_string: str) -> bool:
        """Revoke a token."""
        try:
            token = self._decode_token(token_string)
            self._revoked_tokens.add(token.token_id)
            if token.token_id in self._active_tokens:
                del self._active_tokens[token.token_id]
            return True
        except Exception:
            return False

    def refresh_token(self, token_string: str) -> str:
        """
        Refresh a token (create new with same claims).

        Args:
            token_string: Current token

        Returns:
            New token string
        """
        token = self.validate_token(token_string)

        # Revoke old token
        self._revoked_tokens.add(token.token_id)

        # Create new token
        return self.create_token(
            user_id=token.user_id,
            roles=token.roles,
            claims=token.claims,
        )

    def cleanup_expired(self):
        """Remove expired tokens from memory."""
        current_time = time.time()
        expired = [
            tid for tid, token in self._active_tokens.items()
            if token.expires_at < current_time
        ]
        for tid in expired:
            del self._active_tokens[tid]

    def _encode_token(self, token: AuthToken) -> str:
        """Encode token to string."""
        payload = json.dumps(token.to_dict()).encode()
        signature = hmac.new(
            self._secret.encode(),
            payload,
            hashlib.sha256
        ).digest()

        # Encode payload and signature as base64, then join with '.'
        # This avoids issues with signature bytes containing '.'
        payload_b64 = base64.urlsafe_b64encode(payload).decode()
        signature_b64 = base64.urlsafe_b64encode(signature).decode()
        return f"{payload_b64}.{signature_b64}"

    def _decode_token(self, token_string: str) -> AuthToken:
        """Decode token from string."""
        try:
            # Split on '.' to get base64-encoded payload and signature
            parts = token_string.split('.')
            if len(parts) != 2:
                raise AuthenticationError("Invalid token format")

            payload_b64, signature_b64 = parts
            payload = base64.urlsafe_b64decode(payload_b64.encode())
            signature = base64.urlsafe_b64decode(signature_b64.encode())

            # Verify signature
            expected_sig = hmac.new(
                self._secret.encode(),
                payload,
                hashlib.sha256
            ).digest()

            if not hmac.compare_digest(signature, expected_sig):
                raise AuthenticationError("Invalid signature")

            data = json.loads(payload.decode())

            return AuthToken(
                token_id=data['token_id'],
                user_id=data['user_id'],
                roles=data['roles'],
                issued_at=data['issued_at'],
                expires_at=data['expires_at'],
                claims=data.get('claims', {}),
            )

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            raise AuthenticationError(f"Token decode failed: {e}")


def authenticate_request(
    api_key_manager: APIKeyManager,
    token_manager: TokenManager,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[bool, Optional[List[str]], str]:
    """
    Authenticate a request using API key or token.

    Args:
        api_key_manager: API key manager
        token_manager: Token manager
        api_key: API key from request
        token: Bearer token from request

    Returns:
        Tuple of (authenticated, roles, user_id)
    """
    # Try API key first
    if api_key:
        key = api_key_manager.validate_key(api_key)
        if key:
            return True, key.roles, f"apikey:{key.key_id}"

    # Try token
    if token:
        try:
            auth_token = token_manager.validate_token(token)
            return True, auth_token.roles, auth_token.user_id
        except (AuthenticationError, TokenExpiredError):
            pass

    return False, None, ""
