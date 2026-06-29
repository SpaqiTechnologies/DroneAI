"""
Authorization System for Drone API.

Provides role-based access control (RBAC) for drone operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set
from enum import Enum


class AuthorizationError(Exception):
    """Authorization failed."""
    pass


class Permission(Enum):
    """Drone operation permissions."""
    # Read permissions
    VIEW_TELEMETRY = "view_telemetry"
    VIEW_STATUS = "view_status"
    VIEW_LOGS = "view_logs"
    VIEW_CONFIG = "view_config"
    VIEW_MISSIONS = "view_missions"

    # Control permissions
    ARM = "arm"
    DISARM = "disarm"
    TAKEOFF = "takeoff"
    LAND = "land"
    RTL = "rtl"
    HOVER = "hover"

    # Navigation permissions
    GOTO = "goto"
    FOLLOW = "follow"
    ORBIT = "orbit"
    GUIDED = "guided"

    # Mission permissions
    CREATE_MISSION = "create_mission"
    EDIT_MISSION = "edit_mission"
    DELETE_MISSION = "delete_mission"
    EXECUTE_MISSION = "execute_mission"
    ABORT_MISSION = "abort_mission"

    # Configuration permissions
    EDIT_CONFIG = "edit_config"
    EDIT_GEOFENCE = "edit_geofence"
    EDIT_FAILSAFE = "edit_failsafe"

    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_KEYS = "manage_keys"
    SYSTEM_ADMIN = "system_admin"

    # Emergency permissions
    EMERGENCY_STOP = "emergency_stop"
    KILL_SWITCH = "kill_switch"


@dataclass
class Role:
    """Role with associated permissions."""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_role: Optional[str] = None  # Inherit from parent

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission."""
        return permission in self.permissions

    def add_permission(self, permission: Permission):
        """Add permission to role."""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission):
        """Remove permission from role."""
        self.permissions.discard(permission)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'permissions': [p.value for p in self.permissions],
            'parent_role': self.parent_role,
        }


class RoleManager:
    """
    Manages roles and their permissions.

    Provides predefined roles for common use cases.
    """

    def __init__(self):
        """Initialize with default roles."""
        self._roles: Dict[str, Role] = {}
        self._setup_default_roles()

    def _setup_default_roles(self):
        """Setup default role hierarchy."""
        # Viewer - read only
        viewer = Role(
            name="viewer",
            description="Read-only access to telemetry and status",
            permissions={
                Permission.VIEW_TELEMETRY,
                Permission.VIEW_STATUS,
                Permission.VIEW_MISSIONS,
            }
        )

        # Operator - can fly the drone
        operator = Role(
            name="operator",
            description="Can control drone flight",
            permissions={
                # Inherit viewer permissions
                Permission.VIEW_TELEMETRY,
                Permission.VIEW_STATUS,
                Permission.VIEW_MISSIONS,
                Permission.VIEW_LOGS,
                # Flight control
                Permission.ARM,
                Permission.DISARM,
                Permission.TAKEOFF,
                Permission.LAND,
                Permission.RTL,
                Permission.HOVER,
                Permission.GOTO,
                Permission.FOLLOW,
                Permission.ORBIT,
                Permission.GUIDED,
                # Missions
                Permission.CREATE_MISSION,
                Permission.EXECUTE_MISSION,
                Permission.ABORT_MISSION,
                # Emergency
                Permission.EMERGENCY_STOP,
            },
            parent_role="viewer"
        )

        # Pilot - full flight authority
        pilot = Role(
            name="pilot",
            description="Full flight authority including kill switch",
            permissions={
                # All operator permissions
                *operator.permissions,
                # Additional permissions
                Permission.VIEW_CONFIG,
                Permission.EDIT_MISSION,
                Permission.DELETE_MISSION,
                Permission.KILL_SWITCH,
            },
            parent_role="operator"
        )

        # Maintainer - configuration access
        maintainer = Role(
            name="maintainer",
            description="Can configure drone settings",
            permissions={
                # All pilot permissions
                *pilot.permissions,
                # Config permissions
                Permission.EDIT_CONFIG,
                Permission.EDIT_GEOFENCE,
                Permission.EDIT_FAILSAFE,
            },
            parent_role="pilot"
        )

        # Admin - full access
        admin = Role(
            name="admin",
            description="Full system access",
            permissions={
                # All permissions
                *[p for p in Permission]
            },
            parent_role="maintainer"
        )

        self._roles = {
            'viewer': viewer,
            'operator': operator,
            'pilot': pilot,
            'maintainer': maintainer,
            'admin': admin,
        }

    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name."""
        return self._roles.get(name)

    def create_role(
        self,
        name: str,
        description: str,
        permissions: Set[Permission],
        parent_role: Optional[str] = None,
    ) -> Role:
        """Create a custom role."""
        role = Role(
            name=name,
            description=description,
            permissions=permissions,
            parent_role=parent_role,
        )

        # Inherit from parent if specified
        if parent_role and parent_role in self._roles:
            role.permissions.update(self._roles[parent_role].permissions)

        self._roles[name] = role
        return role

    def delete_role(self, name: str) -> bool:
        """Delete a custom role (cannot delete default roles)."""
        if name in ['viewer', 'operator', 'pilot', 'maintainer', 'admin']:
            return False
        if name in self._roles:
            del self._roles[name]
            return True
        return False

    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles."""
        return [role.to_dict() for role in self._roles.values()]

    def get_permissions_for_roles(self, role_names: List[str]) -> Set[Permission]:
        """Get combined permissions for multiple roles."""
        permissions = set()
        for name in role_names:
            role = self._roles.get(name)
            if role:
                permissions.update(role.permissions)
        return permissions


class AccessControl:
    """
    Access control enforcement.

    Checks permissions for API requests and drone operations.
    """

    def __init__(self, role_manager: Optional[RoleManager] = None):
        """Initialize access control."""
        self._role_manager = role_manager or RoleManager()

        # Map API endpoints to required permissions
        self._endpoint_permissions: Dict[str, Permission] = {
            # Telemetry
            '/api/status': Permission.VIEW_STATUS,
            '/api/telemetry': Permission.VIEW_TELEMETRY,
            '/api/logs': Permission.VIEW_LOGS,
            '/api/config': Permission.VIEW_CONFIG,

            # Control
            '/api/arm': Permission.ARM,
            '/api/disarm': Permission.DISARM,
            '/api/takeoff': Permission.TAKEOFF,
            '/api/land': Permission.LAND,
            '/api/rtl': Permission.RTL,
            '/api/hover': Permission.HOVER,
            '/api/goto': Permission.GOTO,
            '/api/follow': Permission.FOLLOW,

            # Missions
            '/api/mission/create': Permission.CREATE_MISSION,
            '/api/mission/start': Permission.EXECUTE_MISSION,
            '/api/mission/abort': Permission.ABORT_MISSION,
            '/api/mission/delete': Permission.DELETE_MISSION,

            # Config
            '/api/config/edit': Permission.EDIT_CONFIG,
            '/api/geofence/edit': Permission.EDIT_GEOFENCE,
            '/api/failsafe/edit': Permission.EDIT_FAILSAFE,

            # Admin
            '/api/users': Permission.MANAGE_USERS,
            '/api/keys': Permission.MANAGE_KEYS,

            # Emergency
            '/api/emergency_stop': Permission.EMERGENCY_STOP,
            '/api/kill': Permission.KILL_SWITCH,
        }

        # Map socket events to permissions
        self._event_permissions: Dict[str, Permission] = {
            'arm_drone': Permission.ARM,
            'disarm_drone': Permission.DISARM,
            'start_takeoff': Permission.TAKEOFF,
            'start_landing': Permission.LAND,
            'return_to_home': Permission.RTL,
            'emergency_stop': Permission.EMERGENCY_STOP,
            'kill_switch': Permission.KILL_SWITCH,
            'start_mission': Permission.EXECUTE_MISSION,
            'abort_mission': Permission.ABORT_MISSION,
            'follow_start': Permission.FOLLOW,
            'plan_path': Permission.GOTO,
        }

    def check_permission(
        self,
        roles: List[str],
        permission: Permission,
    ) -> bool:
        """
        Check if roles have required permission.

        Args:
            roles: User's roles
            permission: Required permission

        Returns:
            True if authorized
        """
        user_permissions = self._role_manager.get_permissions_for_roles(roles)
        return permission in user_permissions

    def check_endpoint(
        self,
        roles: List[str],
        endpoint: str,
    ) -> bool:
        """
        Check if roles can access endpoint.

        Args:
            roles: User's roles
            endpoint: API endpoint path

        Returns:
            True if authorized
        """
        # Find matching permission
        required_permission = None
        for path, permission in self._endpoint_permissions.items():
            if endpoint.startswith(path):
                required_permission = permission
                break

        if required_permission is None:
            # No permission required for endpoint
            return True

        return self.check_permission(roles, required_permission)

    def check_event(
        self,
        roles: List[str],
        event: str,
    ) -> bool:
        """
        Check if roles can emit socket event.

        Args:
            roles: User's roles
            event: Socket event name

        Returns:
            True if authorized
        """
        required_permission = self._event_permissions.get(event)

        if required_permission is None:
            # No permission required for event
            return True

        return self.check_permission(roles, required_permission)

    def require_permission(
        self,
        roles: List[str],
        permission: Permission,
    ):
        """
        Require permission or raise error.

        Args:
            roles: User's roles
            permission: Required permission

        Raises:
            AuthorizationError: If not authorized
        """
        if not self.check_permission(roles, permission):
            raise AuthorizationError(
                f"Permission denied: {permission.value} required"
            )

    def add_endpoint_permission(self, endpoint: str, permission: Permission):
        """Add permission requirement for endpoint."""
        self._endpoint_permissions[endpoint] = permission

    def add_event_permission(self, event: str, permission: Permission):
        """Add permission requirement for socket event."""
        self._event_permissions[event] = permission

    def get_allowed_endpoints(self, roles: List[str]) -> List[str]:
        """Get list of allowed endpoints for roles."""
        user_permissions = self._role_manager.get_permissions_for_roles(roles)
        allowed = []
        for endpoint, permission in self._endpoint_permissions.items():
            if permission in user_permissions:
                allowed.append(endpoint)
        return allowed

    def get_allowed_events(self, roles: List[str]) -> List[str]:
        """Get list of allowed socket events for roles."""
        user_permissions = self._role_manager.get_permissions_for_roles(roles)
        allowed = []
        for event, permission in self._event_permissions.items():
            if permission in user_permissions:
                allowed.append(event)
        return allowed
