"""
Tests for security module.
"""

import pytest
import time
import os
import tempfile
from security import (
    # Authentication
    APIKeyManager,
    TokenManager,
    AuthToken,
    APIKey,
    AuthenticationError,
    TokenExpiredError,
    # Authorization
    Permission,
    Role,
    RoleManager,
    AccessControl,
    AuthorizationError,
    # Crypto
    CommandSigner,
    SignedCommand,
    MessageEncryptor,
    SecureHash,
    CryptoError,
    # Config
    SecureConfigManager,
    ConfigEncryption,
)


class TestAPIKeyManager:
    """Tests for API key management."""

    def test_generate_key(self):
        manager = APIKeyManager()
        plain_key, api_key = manager.generate_key("Test Key", roles=['operator'])

        assert plain_key.startswith("drone_")
        assert api_key.name == "Test Key"
        assert 'operator' in api_key.roles

    def test_validate_key(self):
        manager = APIKeyManager()
        plain_key, api_key = manager.generate_key("Test Key")

        validated = manager.validate_key(plain_key)

        assert validated is not None
        assert validated.key_id == api_key.key_id

    def test_invalid_key(self):
        manager = APIKeyManager()
        manager.generate_key("Test Key")

        validated = manager.validate_key("invalid_key_12345")

        assert validated is None

    def test_revoke_key(self):
        manager = APIKeyManager()
        plain_key, api_key = manager.generate_key("Test Key")

        manager.revoke_key(api_key.key_id)
        validated = manager.validate_key(plain_key)

        assert validated is None

    def test_key_expiration(self):
        manager = APIKeyManager()
        plain_key, api_key = manager.generate_key("Test Key", expires_days=0)
        # Key should be expired immediately (0 days)

        # Actually set expiration to past
        api_key.expires_at = time.time() - 1

        assert not api_key.is_valid()

    def test_list_keys(self):
        manager = APIKeyManager()
        manager.generate_key("Key 1")
        manager.generate_key("Key 2")

        keys = manager.list_keys()

        assert len(keys) == 2
        assert any(k['name'] == "Key 1" for k in keys)

    def test_usage_tracking(self):
        manager = APIKeyManager()
        plain_key, api_key = manager.generate_key("Test Key")

        manager.validate_key(plain_key)
        manager.validate_key(plain_key)

        assert api_key.usage_count == 2
        assert api_key.last_used is not None


class TestTokenManager:
    """Tests for token management."""

    def test_create_token(self):
        manager = TokenManager()
        token = manager.create_token("user123", roles=['operator'])

        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_token(self):
        manager = TokenManager()
        token_str = manager.create_token("user123", roles=['operator'])

        token = manager.validate_token(token_str)

        assert token.user_id == "user123"
        assert 'operator' in token.roles

    def test_invalid_token(self):
        manager = TokenManager()

        with pytest.raises(AuthenticationError):
            manager.validate_token("invalid_token")

    def test_expired_token(self):
        manager = TokenManager(token_lifetime=0)  # Immediate expiration
        token_str = manager.create_token("user123", roles=['operator'])

        time.sleep(0.1)

        with pytest.raises(TokenExpiredError):
            manager.validate_token(token_str)

    def test_revoke_token(self):
        manager = TokenManager()
        token_str = manager.create_token("user123", roles=['operator'])

        manager.revoke_token(token_str)

        with pytest.raises(AuthenticationError):
            manager.validate_token(token_str)

    def test_refresh_token(self):
        manager = TokenManager()
        old_token = manager.create_token("user123", roles=['operator'])

        new_token = manager.refresh_token(old_token)

        # Old token should be revoked
        with pytest.raises(AuthenticationError):
            manager.validate_token(old_token)

        # New token should work
        validated = manager.validate_token(new_token)
        assert validated.user_id == "user123"

    def test_custom_claims(self):
        manager = TokenManager()
        token_str = manager.create_token(
            "user123",
            roles=['operator'],
            claims={'drone_id': 'drone_001'}
        )

        token = manager.validate_token(token_str)

        assert token.claims.get('drone_id') == 'drone_001'


class TestRoleManager:
    """Tests for role management."""

    def test_default_roles(self):
        manager = RoleManager()

        assert manager.get_role('viewer') is not None
        assert manager.get_role('operator') is not None
        assert manager.get_role('pilot') is not None
        assert manager.get_role('admin') is not None

    def test_viewer_permissions(self):
        manager = RoleManager()
        viewer = manager.get_role('viewer')

        assert viewer.has_permission(Permission.VIEW_TELEMETRY)
        assert not viewer.has_permission(Permission.ARM)

    def test_operator_permissions(self):
        manager = RoleManager()
        operator = manager.get_role('operator')

        assert operator.has_permission(Permission.VIEW_TELEMETRY)
        assert operator.has_permission(Permission.ARM)
        assert operator.has_permission(Permission.TAKEOFF)
        assert not operator.has_permission(Permission.KILL_SWITCH)

    def test_admin_has_all_permissions(self):
        manager = RoleManager()
        admin = manager.get_role('admin')

        for permission in Permission:
            assert admin.has_permission(permission)

    def test_create_custom_role(self):
        manager = RoleManager()
        role = manager.create_role(
            "custom",
            "Custom role",
            {Permission.VIEW_TELEMETRY, Permission.ARM}
        )

        assert role.has_permission(Permission.VIEW_TELEMETRY)
        assert role.has_permission(Permission.ARM)
        assert not role.has_permission(Permission.TAKEOFF)

    def test_get_permissions_for_roles(self):
        manager = RoleManager()
        permissions = manager.get_permissions_for_roles(['viewer', 'operator'])

        assert Permission.VIEW_TELEMETRY in permissions
        assert Permission.ARM in permissions


class TestAccessControl:
    """Tests for access control."""

    def test_check_permission(self):
        ac = AccessControl()

        assert ac.check_permission(['operator'], Permission.ARM)
        assert not ac.check_permission(['viewer'], Permission.ARM)

    def test_check_endpoint(self):
        ac = AccessControl()

        assert ac.check_endpoint(['operator'], '/api/arm')
        assert not ac.check_endpoint(['viewer'], '/api/arm')
        assert ac.check_endpoint(['viewer'], '/api/status')

    def test_check_event(self):
        ac = AccessControl()

        assert ac.check_event(['operator'], 'arm_drone')
        assert not ac.check_event(['viewer'], 'arm_drone')

    def test_require_permission(self):
        ac = AccessControl()

        # Should not raise
        ac.require_permission(['operator'], Permission.ARM)

        # Should raise
        with pytest.raises(AuthorizationError):
            ac.require_permission(['viewer'], Permission.ARM)

    def test_get_allowed_endpoints(self):
        ac = AccessControl()
        allowed = ac.get_allowed_endpoints(['operator'])

        assert '/api/arm' in allowed
        assert '/api/status' in allowed

    def test_get_allowed_events(self):
        ac = AccessControl()
        allowed = ac.get_allowed_events(['pilot'])

        assert 'arm_drone' in allowed
        assert 'kill_switch' in allowed


class TestCommandSigner:
    """Tests for command signing."""

    def test_sign_command(self):
        signer = CommandSigner()
        command = signer.sign_command(
            "takeoff",
            {"altitude": 30},
            "user123"
        )

        assert command.command_type == "takeoff"
        assert command.parameters["altitude"] == 30
        assert len(command.signature) == 64  # SHA-256 hex
        assert len(command.nonce) == 32

    def test_verify_command(self):
        signer = CommandSigner()
        command = signer.sign_command(
            "takeoff",
            {"altitude": 30},
            "user123"
        )

        assert signer.verify_command(command)

    def test_tampered_command(self):
        signer = CommandSigner()
        command = signer.sign_command(
            "takeoff",
            {"altitude": 30},
            "user123"
        )

        # Tamper with command
        command.parameters["altitude"] = 100

        from security.crypto import SignatureError
        with pytest.raises(SignatureError):
            signer.verify_command(command)

    def test_expired_command(self):
        signer = CommandSigner()
        command = signer.sign_command(
            "takeoff",
            {"altitude": 30},
            "user123"
        )

        # Make command old
        command.timestamp = time.time() - 400  # 6+ minutes ago

        from security.crypto import SignatureError
        with pytest.raises(SignatureError):
            signer.verify_command(command, max_age=300)

    def test_replay_prevention(self):
        signer = CommandSigner()
        command = signer.sign_command(
            "takeoff",
            {"altitude": 30},
            "user123"
        )

        # First verification should work
        assert signer.verify_command(command)

        # Second verification (replay) should fail
        from security.crypto import SignatureError
        with pytest.raises(SignatureError):
            signer.verify_command(command)


class TestMessageEncryptor:
    """Tests for message encryption."""

    def test_encrypt_decrypt(self):
        encryptor = MessageEncryptor()
        plaintext = "Hello, World!"

        ciphertext = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(ciphertext)

        assert decrypted == plaintext
        assert ciphertext != plaintext

    def test_encrypt_dict(self):
        encryptor = MessageEncryptor()
        data = {"api_key": "secret123", "setting": "value"}

        ciphertext = encryptor.encrypt_dict(data)
        decrypted = encryptor.decrypt_dict(ciphertext)

        assert decrypted == data

    def test_different_keys_fail(self):
        encryptor1 = MessageEncryptor("key1")
        encryptor2 = MessageEncryptor("key2")

        ciphertext = encryptor1.encrypt("secret")

        from security.crypto import EncryptionError
        with pytest.raises(EncryptionError):
            encryptor2.decrypt(ciphertext)


class TestSecureHash:
    """Tests for secure hashing."""

    def test_hash_password(self):
        hasher = SecureHash()
        password = "MySecurePassword123!"

        hash1 = hasher.hash_password(password)
        hash2 = hasher.hash_password(password)

        # Different hashes (different salts)
        assert hash1 != hash2

    def test_verify_password(self):
        hasher = SecureHash()
        password = "MySecurePassword123!"

        hash_str = hasher.hash_password(password)

        assert hasher.verify_password(password, hash_str)
        assert not hasher.verify_password("WrongPassword", hash_str)

    def test_hash_data(self):
        hasher = SecureHash()

        hash1 = hasher.hash_data(b"test data")
        hash2 = hasher.hash_data(b"test data")
        hash3 = hasher.hash_data(b"different data")

        assert hash1 == hash2
        assert hash1 != hash3


class TestSecureConfigManager:
    """Tests for secure configuration storage."""

    @pytest.fixture
    def config_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigEncryption(storage_path=tmpdir)
            manager = SecureConfigManager(config)
            yield manager

    def test_save_load(self, config_manager):
        data = {"setting": "value", "number": 42}
        config_manager.save("test", data)

        loaded = config_manager.load("test")

        assert loaded == data

    def test_encrypted_fields(self, config_manager):
        data = {"api_key": "secret123", "public_setting": "visible"}
        config_manager.save("test", data)

        loaded = config_manager.load("test")

        assert loaded["api_key"] == "secret123"
        assert loaded["public_setting"] == "visible"

    def test_get_set(self, config_manager):
        config_manager.set("test", "key1", "value1")
        config_manager.set("test", "key2", "value2")

        assert config_manager.get("test", "key1") == "value1"
        assert config_manager.get("test", "key2") == "value2"
        assert config_manager.get("test", "missing", "default") == "default"

    def test_delete(self, config_manager):
        config_manager.save("test", {"key": "value"})
        config_manager.delete("test", "key")

        loaded = config_manager.load("test")
        assert "key" not in loaded

    def test_delete_namespace(self, config_manager):
        config_manager.save("test", {"key": "value"})
        config_manager.delete("test")

        assert config_manager.load("test") is None

    def test_list_namespaces(self, config_manager):
        config_manager.save("ns1", {"data": 1})
        config_manager.save("ns2", {"data": 2})

        namespaces = config_manager.list_namespaces()

        assert "ns1" in namespaces
        assert "ns2" in namespaces

    def test_export_redacted(self, config_manager):
        data = {"api_key": "secret123", "public": "visible"}
        config_manager.save("test", data)

        exported = config_manager.export("test", include_encrypted=False)

        assert exported["api_key"] == "***REDACTED***"
        assert exported["public"] == "visible"

    def test_validate(self, config_manager):
        config_manager.save("test", {"key": "value"})

        valid, issues = config_manager.validate("test")

        assert valid
        assert len(issues) == 0

    def test_validate_missing(self, config_manager):
        valid, issues = config_manager.validate("nonexistent")

        assert not valid
        assert len(issues) > 0


class TestIntegration:
    """Integration tests for security components."""

    def test_full_auth_flow(self):
        # Setup
        api_key_manager = APIKeyManager()
        token_manager = TokenManager()
        role_manager = RoleManager()
        access_control = AccessControl(role_manager)

        # Generate API key for operator
        plain_key, api_key = api_key_manager.generate_key(
            "Operator Key",
            roles=['operator']
        )

        # Validate API key
        validated = api_key_manager.validate_key(plain_key)
        assert validated is not None

        # Create token from API key
        token_str = token_manager.create_token(
            user_id=f"apikey:{validated.key_id}",
            roles=validated.roles,
        )

        # Validate token
        token = token_manager.validate_token(token_str)

        # Check permissions
        assert access_control.check_permission(token.roles, Permission.ARM)
        assert not access_control.check_permission(token.roles, Permission.KILL_SWITCH)

    def test_signed_command_with_auth(self):
        # Setup
        api_key_manager = APIKeyManager()
        signer = CommandSigner()
        access_control = AccessControl()

        # Generate key
        plain_key, api_key = api_key_manager.generate_key(
            "Pilot Key",
            roles=['pilot']
        )

        # Validate key
        validated = api_key_manager.validate_key(plain_key)

        # Check permission for kill switch
        assert access_control.check_permission(validated.roles, Permission.KILL_SWITCH)

        # Sign command
        command = signer.sign_command(
            "kill",
            {"reason": "emergency"},
            f"apikey:{validated.key_id}"
        )

        # Verify command
        assert signer.verify_command(command)
