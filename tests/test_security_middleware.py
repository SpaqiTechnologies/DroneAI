"""
Tests for security middleware.
"""

import asyncio
import pytest
from security import (
    init_security,
    get_api_key_manager,
    get_token_manager,
    get_access_control,
    generate_api_key,
    create_session_token,
    validate_credentials,
    Permission,
    AuthenticationError,
    TokenExpiredError,
)
from security.middleware import (
    flask_auth_required,
    flask_optional_auth,
    socketio_auth_required,
    create_fastapi_auth_dependency,
)


def reset_security():
    """Reset security state for testing."""
    from security import middleware
    middleware._api_key_manager = None
    middleware._token_manager = None
    middleware._access_control = None


class TestSecurityInitialization:
    """Tests for security initialization."""

    def test_init_security(self):
        reset_security()
        init_security()

        assert get_api_key_manager() is not None
        assert get_token_manager() is not None
        assert get_access_control() is not None

    def test_generate_api_key(self):
        reset_security()
        init_security()
        plain_key, info = generate_api_key("Test Key", ['operator'])

        assert plain_key.startswith("drone_")
        assert info['name'] == "Test Key"
        assert 'operator' in info['roles']

    def test_create_session_token(self):
        reset_security()
        init_security()
        token = create_session_token("user123", ['viewer'])

        assert token is not None
        assert len(token) > 0

    def test_validate_api_key(self):
        reset_security()
        init_security()
        plain_key, _ = generate_api_key("Validation Test", ['pilot'])

        result = validate_credentials(api_key=plain_key)

        assert result is not None
        assert 'pilot' in result['roles']
        assert result['type'] == 'api_key'

    def test_validate_token(self):
        reset_security()
        init_security()
        token = create_session_token("user123", ['operator'])

        result = validate_credentials(token=token)

        assert result is not None
        assert result['user_id'] == 'user123'
        assert result['type'] == 'token'

    def test_validate_invalid_credentials(self):
        reset_security()
        init_security()

        result = validate_credentials(api_key="invalid_key")
        assert result is None

        result = validate_credentials(token="invalid_token")
        assert result is None


class TestFlaskAuthDecorator:
    """Tests for Flask authentication decorator."""

    def test_decorator_creation(self):
        decorator = flask_auth_required([Permission.ARM])
        assert decorator is not None

    def test_optional_auth_decorator(self):
        decorator = flask_optional_auth()
        assert decorator is not None


class TestSocketIOAuthDecorator:
    """Tests for SocketIO authentication decorator."""

    def test_decorator_creation(self):
        decorator = socketio_auth_required([Permission.ARM])
        assert decorator is not None


try:
    from fastapi import HTTPException
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestFastAPIAuthDependency:
    """Tests for FastAPI authentication dependency."""

    def test_dependency_creation(self):
        dependency = create_fastapi_auth_dependency([Permission.VIEW_TELEMETRY])
        assert dependency is not None

    def test_dependency_with_valid_api_key(self):
        async def run_test():
            init_security()
            plain_key, _ = generate_api_key("FastAPI Test", ['operator'])

            dependency = create_fastapi_auth_dependency([Permission.ARM])

            # Simulate header values
            result = await dependency(
                authorization=None,
                x_api_key=plain_key,
            )

            assert result['roles'] == ['operator']
            assert 'apikey:' in result['user_id']

        asyncio.run(run_test())

    def test_dependency_with_valid_token(self):
        async def run_test():
            init_security()
            token = create_session_token("fastapi_user", ['pilot'])

            dependency = create_fastapi_auth_dependency([Permission.KILL_SWITCH])

            result = await dependency(
                authorization=f"Bearer {token}",
                x_api_key=None,
            )

            assert result['user_id'] == 'fastapi_user'
            assert 'pilot' in result['roles']

        asyncio.run(run_test())

    def test_dependency_missing_auth(self):
        async def run_test():
            dependency = create_fastapi_auth_dependency()

            with pytest.raises(HTTPException) as exc_info:
                await dependency(authorization=None, x_api_key=None)

            assert exc_info.value.status_code == 401

        asyncio.run(run_test())

    def test_dependency_invalid_api_key(self):
        async def run_test():
            init_security()

            dependency = create_fastapi_auth_dependency()

            with pytest.raises(HTTPException) as exc_info:
                await dependency(authorization=None, x_api_key="invalid_key")

            assert exc_info.value.status_code == 401

        asyncio.run(run_test())

    def test_dependency_permission_denied(self):
        async def run_test():
            init_security()

            # Viewer doesn't have KILL_SWITCH permission
            plain_key, _ = generate_api_key("Limited User", ['viewer'])

            dependency = create_fastapi_auth_dependency([Permission.KILL_SWITCH])

            with pytest.raises(HTTPException) as exc_info:
                await dependency(authorization=None, x_api_key=plain_key)

            assert exc_info.value.status_code == 403

        asyncio.run(run_test())


class TestEndToEndAuth:
    """End-to-end authentication tests."""

    def test_api_key_to_token_flow(self):
        reset_security()
        init_security()

        # 1. Generate API key
        plain_key, key_info = generate_api_key("E2E Test", ['operator'])

        # 2. Validate API key
        auth = validate_credentials(api_key=plain_key)
        assert auth is not None
        assert auth['type'] == 'api_key'

        # 3. Create token from user info
        user_id = f"apikey:{key_info['key_id']}"
        token = create_session_token(user_id, key_info['roles'])

        # 4. Validate token
        token_auth = validate_credentials(token=token)
        assert token_auth is not None
        assert token_auth['type'] == 'token'
        assert token_auth['user_id'] == user_id

    def test_permission_check_flow(self):
        reset_security()
        init_security()

        # Create keys with different roles
        viewer_key, _ = generate_api_key("Viewer", ['viewer'])
        operator_key, _ = generate_api_key("Operator", ['operator'])
        pilot_key, _ = generate_api_key("Pilot", ['pilot'])
        admin_key, _ = generate_api_key("Admin", ['admin'])

        ac = get_access_control()

        # Validate and check permissions
        viewer_auth = validate_credentials(api_key=viewer_key)
        assert not ac.check_permission(viewer_auth['roles'], Permission.ARM)
        assert ac.check_permission(viewer_auth['roles'], Permission.VIEW_TELEMETRY)

        operator_auth = validate_credentials(api_key=operator_key)
        assert ac.check_permission(operator_auth['roles'], Permission.ARM)
        assert not ac.check_permission(operator_auth['roles'], Permission.KILL_SWITCH)

        pilot_auth = validate_credentials(api_key=pilot_key)
        assert ac.check_permission(pilot_auth['roles'], Permission.KILL_SWITCH)

        admin_auth = validate_credentials(api_key=admin_key)
        assert ac.check_permission(admin_auth['roles'], Permission.MANAGE_KEYS)


class TestFastAPISecurityMiddleware:
    """Tests for FastAPI security middleware."""

    def test_middleware_creation(self):
        from security.middleware import FastAPISecurityMiddleware

        # Mock app
        class MockApp:
            pass

        middleware = FastAPISecurityMiddleware(MockApp())
        assert middleware is not None
        assert middleware.exclude_paths == ['/docs', '/openapi.json', '/']

    def test_middleware_custom_exclude(self):
        from security.middleware import FastAPISecurityMiddleware

        class MockApp:
            pass

        middleware = FastAPISecurityMiddleware(
            MockApp(),
            exclude_paths=['/health', '/metrics']
        )
        assert '/health' in middleware.exclude_paths
