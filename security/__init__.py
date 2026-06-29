"""
Security Module for Drone AI Application.

Provides authentication, authorization, command signing, and secure configuration.
"""

from .authentication import (
    APIKeyManager,
    TokenManager,
    AuthToken,
    APIKey,
    AuthenticationError,
    TokenExpiredError,
)

from .authorization import (
    Permission,
    Role,
    RoleManager,
    AccessControl,
    AuthorizationError,
)

from .crypto import (
    CommandSigner,
    SignedCommand,
    MessageEncryptor,
    SecureHash,
    CryptoError,
)

from .secure_config import (
    SecureConfigManager,
    ConfigEncryption,
)

from .middleware import (
    init_security,
    get_api_key_manager,
    get_token_manager,
    get_access_control,
    flask_auth_required,
    flask_optional_auth,
    socketio_auth_required,
    create_fastapi_auth_dependency,
    FastAPISecurityMiddleware,
    generate_api_key,
    create_session_token,
    validate_credentials,
)

__all__ = [
    # Authentication
    'APIKeyManager',
    'TokenManager',
    'AuthToken',
    'APIKey',
    'AuthenticationError',
    'TokenExpiredError',
    # Authorization
    'Permission',
    'Role',
    'RoleManager',
    'AccessControl',
    'AuthorizationError',
    # Crypto
    'CommandSigner',
    'SignedCommand',
    'MessageEncryptor',
    'SecureHash',
    'CryptoError',
    # Config
    'SecureConfigManager',
    'ConfigEncryption',
    # Middleware
    'init_security',
    'get_api_key_manager',
    'get_token_manager',
    'get_access_control',
    'flask_auth_required',
    'flask_optional_auth',
    'socketio_auth_required',
    'create_fastapi_auth_dependency',
    'FastAPISecurityMiddleware',
    'generate_api_key',
    'create_session_token',
    'validate_credentials',
]
