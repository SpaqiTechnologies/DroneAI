"""
Security middleware for web servers.

Provides authentication and authorization middleware for Flask and FastAPI.
"""

from __future__ import annotations

import functools
from typing import Optional, Callable, List, Any

from .authentication import APIKeyManager, TokenManager, AuthenticationError, TokenExpiredError
from .authorization import AccessControl, Permission, AuthorizationError


# Global instances (can be replaced with dependency injection)
_api_key_manager: Optional[APIKeyManager] = None
_token_manager: Optional[TokenManager] = None
_access_control: Optional[AccessControl] = None


def init_security(
    api_key_manager: Optional[APIKeyManager] = None,
    token_manager: Optional[TokenManager] = None,
    access_control: Optional[AccessControl] = None,
    force: bool = False,
):
    """
    Initialize security components.

    Args:
        api_key_manager: API key manager instance
        token_manager: Token manager instance
        access_control: Access control instance
        force: Force reinitialization even if already initialized
    """
    global _api_key_manager, _token_manager, _access_control

    # Only initialize if not already initialized (unless forced)
    if not force:
        if _api_key_manager is None:
            _api_key_manager = api_key_manager or APIKeyManager()
        if _token_manager is None:
            _token_manager = token_manager or TokenManager()
        if _access_control is None:
            _access_control = access_control or AccessControl()
    else:
        _api_key_manager = api_key_manager or APIKeyManager()
        _token_manager = token_manager or TokenManager()
        _access_control = access_control or AccessControl()


def get_api_key_manager() -> APIKeyManager:
    """Get API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def get_token_manager() -> TokenManager:
    """Get token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager


def get_access_control() -> AccessControl:
    """Get access control instance."""
    global _access_control
    if _access_control is None:
        _access_control = AccessControl()
    return _access_control


# ---------------------------------------------------------------------------
# Flask Integration
# ---------------------------------------------------------------------------

def flask_auth_required(permissions: Optional[List[Permission]] = None):
    """
    Flask decorator for authentication and authorization.

    Args:
        permissions: Required permissions (optional)

    Usage:
        @app.route('/api/arm', methods=['POST'])
        @flask_auth_required([Permission.ARM])
        def arm_drone():
            ...
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify

            # Get authorization header
            auth_header = request.headers.get('Authorization', '')
            api_key = request.headers.get('X-API-Key', '')

            roles = []
            user_id = None

            # Try API key first
            if api_key:
                api_key_mgr = get_api_key_manager()
                validated = api_key_mgr.validate_key(api_key)
                if validated:
                    roles = validated.roles
                    user_id = f"apikey:{validated.key_id}"
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid API key'
                    }), 401

            # Try Bearer token
            elif auth_header.startswith('Bearer '):
                token_str = auth_header[7:]
                try:
                    token_mgr = get_token_manager()
                    token = token_mgr.validate_token(token_str)
                    roles = token.roles
                    user_id = token.user_id
                except TokenExpiredError:
                    return jsonify({
                        'success': False,
                        'error': 'Token expired'
                    }), 401
                except AuthenticationError:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid token'
                    }), 401
            else:
                return jsonify({
                    'success': False,
                    'error': 'Authentication required'
                }), 401

            # Check permissions if specified
            if permissions:
                ac = get_access_control()
                for perm in permissions:
                    if not ac.check_permission(roles, perm):
                        return jsonify({
                            'success': False,
                            'error': f'Permission denied: {perm.value}'
                        }), 403

            # Store user info in request context
            request.user_id = user_id
            request.user_roles = roles

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def flask_optional_auth():
    """
    Flask decorator for optional authentication.

    Validates credentials if provided but doesn't require them.
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request

            auth_header = request.headers.get('Authorization', '')
            api_key = request.headers.get('X-API-Key', '')

            request.user_id = None
            request.user_roles = []

            if api_key:
                api_key_mgr = get_api_key_manager()
                validated = api_key_mgr.validate_key(api_key)
                if validated:
                    request.user_id = f"apikey:{validated.key_id}"
                    request.user_roles = validated.roles

            elif auth_header.startswith('Bearer '):
                token_str = auth_header[7:]
                try:
                    token_mgr = get_token_manager()
                    token = token_mgr.validate_token(token_str)
                    request.user_id = token.user_id
                    request.user_roles = token.roles
                except (TokenExpiredError, AuthenticationError):
                    pass

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def socketio_auth_required(permissions: Optional[List[Permission]] = None):
    """
    SocketIO decorator for authenticated events.

    Args:
        permissions: Required permissions

    Usage:
        @socketio.on('arm')
        @socketio_auth_required([Permission.ARM])
        def handle_arm(data):
            ...
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(data=None, *args, **kwargs):
            from flask_socketio import emit
            from flask import request

            # Get token from data or session
            token_str = None
            if isinstance(data, dict):
                token_str = data.get('token') or data.get('api_key')

            # Also check session for previously authenticated socket
            if hasattr(request, 'sid'):
                session_data = getattr(request, 'namespace', {})
                if hasattr(session_data, 'get'):
                    token_str = token_str or session_data.get('token')

            if not token_str:
                emit('error', {'message': 'Authentication required'})
                return

            roles = []
            user_id = None

            # Validate token or API key
            if token_str.startswith('drone_'):
                api_key_mgr = get_api_key_manager()
                validated = api_key_mgr.validate_key(token_str)
                if validated:
                    roles = validated.roles
                    user_id = f"apikey:{validated.key_id}"
            else:
                try:
                    token_mgr = get_token_manager()
                    token = token_mgr.validate_token(token_str)
                    roles = token.roles
                    user_id = token.user_id
                except (TokenExpiredError, AuthenticationError):
                    emit('error', {'message': 'Invalid or expired token'})
                    return

            # Check permissions
            if permissions:
                ac = get_access_control()
                for perm in permissions:
                    if not ac.check_permission(roles, perm):
                        emit('error', {'message': f'Permission denied: {perm.value}'})
                        return

            # Add auth info to data
            if isinstance(data, dict):
                data['_user_id'] = user_id
                data['_roles'] = roles

            return f(data, *args, **kwargs)
        return decorated_function
    return decorator


# ---------------------------------------------------------------------------
# FastAPI Integration
# ---------------------------------------------------------------------------

def create_fastapi_auth_dependency(permissions: Optional[List[Permission]] = None):
    """
    Create a FastAPI dependency for authentication.

    Args:
        permissions: Required permissions

    Usage:
        from fastapi import Depends

        @app.post("/api/arm")
        async def arm_drone(auth: dict = Depends(create_fastapi_auth_dependency([Permission.ARM]))):
            user_id = auth['user_id']
            ...
    """
    async def auth_dependency(
        authorization: str = None,
        x_api_key: str = None,
    ) -> dict:
        from fastapi import Header, HTTPException

        # Import here to avoid circular imports
        roles = []
        user_id = None

        # Try API key
        if x_api_key:
            api_key_mgr = get_api_key_manager()
            validated = api_key_mgr.validate_key(x_api_key)
            if validated:
                roles = validated.roles
                user_id = f"apikey:{validated.key_id}"
            else:
                raise HTTPException(status_code=401, detail="Invalid API key")

        # Try Bearer token
        elif authorization and authorization.startswith('Bearer '):
            token_str = authorization[7:]
            try:
                token_mgr = get_token_manager()
                token = token_mgr.validate_token(token_str)
                roles = token.roles
                user_id = token.user_id
            except TokenExpiredError:
                raise HTTPException(status_code=401, detail="Token expired")
            except AuthenticationError:
                raise HTTPException(status_code=401, detail="Invalid token")
        else:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Check permissions
        if permissions:
            ac = get_access_control()
            for perm in permissions:
                if not ac.check_permission(roles, perm):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Permission denied: {perm.value}"
                    )

        return {
            'user_id': user_id,
            'roles': roles,
        }

    return auth_dependency


class FastAPISecurityMiddleware:
    """
    FastAPI middleware for security logging and rate limiting.

    Usage:
        app.add_middleware(FastAPISecurityMiddleware)
    """

    def __init__(self, app, exclude_paths: Optional[List[str]] = None):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            exclude_paths: Paths to exclude from logging
        """
        self.app = app
        self.exclude_paths = exclude_paths or ['/docs', '/openapi.json', '/']
        self._request_counts: dict = {}

    async def __call__(self, scope, receive, send):
        """Process request."""
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        path = scope.get('path', '')

        # Skip excluded paths
        if any(path.startswith(ep) for ep in self.exclude_paths):
            await self.app(scope, receive, send)
            return

        # Log request (could add rate limiting here)
        client = scope.get('client', ('unknown', 0))
        # print(f"[Security] {scope['method']} {path} from {client[0]}")

        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def generate_api_key(name: str, roles: List[str] = None) -> tuple[str, dict]:
    """
    Generate a new API key.

    Args:
        name: Key name/description
        roles: Roles to assign

    Returns:
        Tuple of (plain_key, key_info)
    """
    api_key_mgr = get_api_key_manager()
    plain_key, api_key = api_key_mgr.generate_key(name, roles=roles or ['operator'])

    return plain_key, {
        'key_id': api_key.key_id,
        'name': api_key.name,
        'roles': api_key.roles,
        'created_at': api_key.created_at,
    }


def create_session_token(user_id: str, roles: List[str] = None) -> str:
    """
    Create a session token for a user.

    Args:
        user_id: User identifier
        roles: User roles

    Returns:
        Token string
    """
    token_mgr = get_token_manager()
    return token_mgr.create_token(user_id, roles=roles or ['viewer'])


def validate_credentials(api_key: str = None, token: str = None) -> Optional[dict]:
    """
    Validate API key or token.

    Args:
        api_key: API key to validate
        token: Token to validate

    Returns:
        User info dict or None
    """
    if api_key:
        api_key_mgr = get_api_key_manager()
        validated = api_key_mgr.validate_key(api_key)
        if validated:
            return {
                'user_id': f"apikey:{validated.key_id}",
                'roles': validated.roles,
                'type': 'api_key',
            }

    if token:
        try:
            token_mgr = get_token_manager()
            validated = token_mgr.validate_token(token)
            return {
                'user_id': validated.user_id,
                'roles': validated.roles,
                'type': 'token',
            }
        except (TokenExpiredError, AuthenticationError):
            pass

    return None
