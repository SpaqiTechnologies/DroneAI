"""
Secure Configuration Storage.

Provides encrypted storage for sensitive configuration data.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

from .crypto import MessageEncryptor, SecureHash, generate_secure_key


@dataclass
class ConfigEncryption:
    """Configuration for secure storage."""
    encryption_key: Optional[str] = None
    storage_path: str = "config/secure"
    auto_backup: bool = True
    backup_count: int = 5


class SecureConfigManager:
    """
    Manages encrypted configuration storage.

    Stores sensitive data like API keys, credentials, and
    critical settings in encrypted form.
    """

    # Fields that should always be encrypted
    SENSITIVE_FIELDS = {
        'api_key', 'api_secret', 'password', 'token', 'secret',
        'private_key', 'credentials', 'auth_token', 'encryption_key',
    }

    def __init__(self, config: Optional[ConfigEncryption] = None):
        """
        Initialize secure config manager.

        Args:
            config: Configuration options
        """
        self._config = config or ConfigEncryption()
        self._encryptor = MessageEncryptor(self._config.encryption_key)
        self._hasher = SecureHash()
        self._storage_path = Path(self._config.storage_path)
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Ensure storage directory exists
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        namespace: str,
        data: Dict[str, Any],
        encrypt_all: bool = False,
    ) -> bool:
        """
        Save configuration data.

        Args:
            namespace: Configuration namespace (e.g., 'drone', 'api')
            data: Configuration data
            encrypt_all: Encrypt all fields, not just sensitive ones

        Returns:
            True if successful
        """
        try:
            # Backup existing config
            if self._config.auto_backup:
                self._backup(namespace)

            # Process data
            processed = {}
            for key, value in data.items():
                if encrypt_all or self._is_sensitive(key):
                    # Encrypt sensitive values
                    if isinstance(value, (dict, list)):
                        encrypted = self._encryptor.encrypt(json.dumps(value))
                    else:
                        encrypted = self._encryptor.encrypt(str(value))
                    processed[key] = {'_encrypted': encrypted}
                else:
                    processed[key] = value

            # Add metadata
            config_data = {
                'namespace': namespace,
                'version': 1,
                'updated_at': time.time(),
                'data': processed,
                'checksum': self._hasher.hash_data(
                    json.dumps(processed, sort_keys=True).encode()
                ),
            }

            # Save to file
            filepath = self._storage_path / f"{namespace}.json"
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)

            # Update cache
            self._cache[namespace] = data

            return True

        except Exception as e:
            print(f"[SecureConfig] Save failed: {e}")
            return False

    def load(self, namespace: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration data.

        Args:
            namespace: Configuration namespace

        Returns:
            Configuration data or None
        """
        # Check cache
        if namespace in self._cache:
            return self._cache[namespace].copy()

        filepath = self._storage_path / f"{namespace}.json"
        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)

            # Verify checksum
            stored_checksum = config_data.get('checksum')
            computed_checksum = self._hasher.hash_data(
                json.dumps(config_data['data'], sort_keys=True).encode()
            )
            if stored_checksum != computed_checksum:
                print(f"[SecureConfig] Checksum mismatch for {namespace}")
                return None

            # Decrypt data
            processed = config_data['data']
            data = {}
            for key, value in processed.items():
                if isinstance(value, dict) and '_encrypted' in value:
                    # Decrypt value
                    decrypted = self._encryptor.decrypt(value['_encrypted'])
                    # Try to parse as JSON
                    try:
                        data[key] = json.loads(decrypted)
                    except json.JSONDecodeError:
                        data[key] = decrypted
                else:
                    data[key] = value

            # Update cache
            self._cache[namespace] = data

            return data.copy()

        except Exception as e:
            print(f"[SecureConfig] Load failed: {e}")
            return None

    def get(
        self,
        namespace: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a specific configuration value.

        Args:
            namespace: Configuration namespace
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        data = self.load(namespace)
        if data is None:
            return default
        return data.get(key, default)

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        encrypt: Optional[bool] = None,
    ) -> bool:
        """
        Set a specific configuration value.

        Args:
            namespace: Configuration namespace
            key: Configuration key
            value: Value to set
            encrypt: Force encryption (None = auto)

        Returns:
            True if successful
        """
        data = self.load(namespace) or {}
        data[key] = value

        # Determine if should encrypt
        encrypt_all = encrypt if encrypt is not None else self._is_sensitive(key)

        return self.save(namespace, data, encrypt_all=False)

    def delete(self, namespace: str, key: Optional[str] = None) -> bool:
        """
        Delete configuration.

        Args:
            namespace: Configuration namespace
            key: Specific key to delete (None = delete entire namespace)

        Returns:
            True if successful
        """
        if key is None:
            # Delete entire namespace
            filepath = self._storage_path / f"{namespace}.json"
            if filepath.exists():
                filepath.unlink()
            if namespace in self._cache:
                del self._cache[namespace]
            return True
        else:
            # Delete specific key
            data = self.load(namespace)
            if data and key in data:
                del data[key]
                return self.save(namespace, data)
            return False

    def list_namespaces(self) -> List[str]:
        """List all configuration namespaces."""
        namespaces = []
        for filepath in self._storage_path.glob("*.json"):
            if not filepath.name.endswith('.backup.json'):
                namespaces.append(filepath.stem)
        return namespaces

    def export(
        self,
        namespace: str,
        include_encrypted: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Export configuration (optionally excluding encrypted values).

        Args:
            namespace: Configuration namespace
            include_encrypted: Include encrypted values

        Returns:
            Exportable configuration
        """
        data = self.load(namespace)
        if data is None:
            return None

        if include_encrypted:
            return data.copy()

        # Redact sensitive fields
        exported = {}
        for key, value in data.items():
            if self._is_sensitive(key):
                exported[key] = "***REDACTED***"
            else:
                exported[key] = value

        return exported

    def import_config(
        self,
        namespace: str,
        data: Dict[str, Any],
        merge: bool = False,
    ) -> bool:
        """
        Import configuration.

        Args:
            namespace: Configuration namespace
            data: Configuration data to import
            merge: Merge with existing (True) or replace (False)

        Returns:
            True if successful
        """
        if merge:
            existing = self.load(namespace) or {}
            existing.update(data)
            data = existing

        return self.save(namespace, data)

    def _is_sensitive(self, key: str) -> bool:
        """Check if key should be encrypted."""
        key_lower = key.lower()
        return any(
            sensitive in key_lower
            for sensitive in self.SENSITIVE_FIELDS
        )

    def _backup(self, namespace: str):
        """Create backup of configuration."""
        filepath = self._storage_path / f"{namespace}.json"
        if not filepath.exists():
            return

        # Rotate backups
        for i in range(self._config.backup_count - 1, 0, -1):
            old_backup = self._storage_path / f"{namespace}.{i}.backup.json"
            new_backup = self._storage_path / f"{namespace}.{i+1}.backup.json"
            if old_backup.exists():
                old_backup.rename(new_backup)

        # Create new backup
        backup_path = self._storage_path / f"{namespace}.1.backup.json"
        with open(filepath, 'r') as src:
            with open(backup_path, 'w') as dst:
                dst.write(src.read())

    def restore_backup(
        self,
        namespace: str,
        backup_number: int = 1,
    ) -> bool:
        """
        Restore configuration from backup.

        Args:
            namespace: Configuration namespace
            backup_number: Backup number (1 = most recent)

        Returns:
            True if successful
        """
        backup_path = self._storage_path / f"{namespace}.{backup_number}.backup.json"
        if not backup_path.exists():
            return False

        filepath = self._storage_path / f"{namespace}.json"

        try:
            with open(backup_path, 'r') as src:
                with open(filepath, 'w') as dst:
                    dst.write(src.read())

            # Clear cache
            if namespace in self._cache:
                del self._cache[namespace]

            return True

        except Exception:
            return False

    def clear_cache(self):
        """Clear configuration cache."""
        self._cache.clear()

    def validate(self, namespace: str) -> tuple[bool, List[str]]:
        """
        Validate configuration integrity.

        Args:
            namespace: Configuration namespace

        Returns:
            Tuple of (valid, list of issues)
        """
        issues = []

        filepath = self._storage_path / f"{namespace}.json"
        if not filepath.exists():
            return False, ["Configuration file not found"]

        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)

            # Check required fields
            if 'namespace' not in config_data:
                issues.append("Missing namespace field")
            if 'data' not in config_data:
                issues.append("Missing data field")
            if 'checksum' not in config_data:
                issues.append("Missing checksum field")

            # Verify checksum
            if 'checksum' in config_data and 'data' in config_data:
                computed = self._hasher.hash_data(
                    json.dumps(config_data['data'], sort_keys=True).encode()
                )
                if config_data['checksum'] != computed:
                    issues.append("Checksum verification failed")

            # Try to decrypt all encrypted values
            if 'data' in config_data:
                for key, value in config_data['data'].items():
                    if isinstance(value, dict) and '_encrypted' in value:
                        try:
                            self._encryptor.decrypt(value['_encrypted'])
                        except Exception:
                            issues.append(f"Cannot decrypt field: {key}")

        except json.JSONDecodeError:
            issues.append("Invalid JSON format")
        except Exception as e:
            issues.append(f"Validation error: {e}")

        return len(issues) == 0, issues
