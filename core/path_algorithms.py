"""
Advanced Path Planning Algorithms

Implements A* and RRT path planning for 3D drone navigation with
obstacle avoidance and optimization.

Algorithms:
- A* (A-Star): Optimal grid-based search
- RRT (Rapidly-exploring Random Tree): Sampling-based planning
- RRT*: Optimized RRT with rewiring
- Potential Fields: Reactive navigation
"""

from __future__ import annotations

import math
import heapq
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Any, Callable
from enum import Enum


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point3D:
    """3D point representation."""
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Point3D') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def distance_2d(self, other: 'Point3D') -> float:
        """Calculate 2D distance (ignoring z)."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2
        )

    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def normalized(self) -> 'Point3D':
        """Return normalized vector."""
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag < 1e-9:
            return Point3D(0, 0, 0)
        return Point3D(self.x / mag, self.y / mag, self.z / mag)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {'x': self.x, 'y': self.y, 'z': self.z}


@dataclass
class Obstacle:
    """Obstacle representation for path planning."""
    center: Point3D
    radius: float  # Bounding sphere radius
    height: float = 0.0  # Optional height for cylinder obstacles
    obstacle_type: str = "sphere"  # sphere, cylinder, box

    def contains(self, point: Point3D, margin: float = 0.0) -> bool:
        """Check if point is inside obstacle (with margin)."""
        if self.obstacle_type == "sphere":
            return point.distance_to(self.center) < (self.radius + margin)
        elif self.obstacle_type == "cylinder":
            # Check horizontal distance and height
            h_dist = point.distance_2d(self.center)
            z_diff = abs(point.z - self.center.z)
            return h_dist < (self.radius + margin) and z_diff < (self.height / 2 + margin)
        else:
            # Default sphere check
            return point.distance_to(self.center) < (self.radius + margin)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'center': self.center.to_dict(),
            'radius': self.radius,
            'height': self.height,
            'type': self.obstacle_type,
        }


@dataclass
class PathResult:
    """Result of path planning."""
    success: bool = False
    path: List[Point3D] = field(default_factory=list)
    cost: float = float('inf')
    nodes_explored: int = 0
    computation_time: float = 0.0
    algorithm: str = ""
    smoothed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'path': [p.to_dict() for p in self.path],
            'cost': self.cost,
            'nodes_explored': self.nodes_explored,
            'computation_time': self.computation_time,
            'algorithm': self.algorithm,
            'smoothed': self.smoothed,
            'waypoint_count': len(self.path),
        }


class PlannerConfig:
    """Configuration for path planners."""

    def __init__(
        self,
        resolution: float = 1.0,
        safety_margin: float = 1.5,
        max_iterations: int = 10000,
        timeout: float = 5.0,
        min_altitude: float = 5.0,
        max_altitude: float = 120.0,
        max_climb_angle: float = 45.0,
        max_speed: float = 10.0,
    ):
        self.resolution = resolution          # Grid resolution in meters
        self.safety_margin = safety_margin    # Safety margin around obstacles
        self.max_iterations = max_iterations  # Max planning iterations
        self.timeout = timeout                # Planning timeout in seconds
        self.min_altitude = min_altitude      # Minimum altitude AGL
        self.max_altitude = max_altitude      # Maximum altitude AGL
        self.max_climb_angle = max_climb_angle  # Max climb/descent angle degrees
        self.max_speed = max_speed            # Max speed for time calculations


# ---------------------------------------------------------------------------
# A* Algorithm
# ---------------------------------------------------------------------------

@dataclass(order=True)
class AStarNode:
    """Node for A* search."""
    f_score: float
    point: Point3D = field(compare=False)
    g_score: float = field(compare=False, default=float('inf'))
    parent: Optional['AStarNode'] = field(compare=False, default=None)


class AStarPlanner:
    """
    A* path planning algorithm for 3D grid-based search.

    Features:
    - 26-connected 3D grid (all diagonal neighbors)
    - Altitude constraints
    - Obstacle avoidance with safety margin
    - Path smoothing
    """

    # 26 neighbors in 3D
    NEIGHBORS_3D = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialize A* planner.

        Args:
            config: Planner configuration
        """
        self.config = config or PlannerConfig()
        self._obstacles: List[Obstacle] = []

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Set obstacles for planning."""
        self._obstacles = obstacles

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle."""
        self._obstacles.append(obstacle)

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self._obstacles.clear()

    def plan(
        self,
        start: Point3D,
        goal: Point3D,
        heuristic: Optional[Callable[[Point3D, Point3D], float]] = None,
    ) -> PathResult:
        """
        Find optimal path from start to goal using A*.

        Args:
            start: Start position
            goal: Goal position
            heuristic: Optional custom heuristic function

        Returns:
            PathResult with planned path
        """
        start_time = time.time()
        result = PathResult(algorithm="A*")

        # Default heuristic: Euclidean distance
        if heuristic is None:
            heuristic = lambda p1, p2: p1.distance_to(p2)

        # Discretize start and goal to grid
        start_grid = self._to_grid(start)
        goal_grid = self._to_grid(goal)

        # Check if start/goal are valid
        if self._is_collision(start_grid):
            result.success = False
            result.computation_time = time.time() - start_time
            return result

        if self._is_collision(goal_grid):
            result.success = False
            result.computation_time = time.time() - start_time
            return result

        # Initialize data structures
        open_set: List[AStarNode] = []
        closed_set: Set[Tuple[float, float, float]] = set()
        g_scores: Dict[Tuple[float, float, float], float] = {}

        # Start node
        start_node = AStarNode(
            f_score=heuristic(start_grid, goal_grid),
            point=start_grid,
            g_score=0,
        )
        heapq.heappush(open_set, start_node)
        g_scores[start_grid.to_tuple()] = 0

        iterations = 0
        timeout_time = start_time + self.config.timeout

        while open_set and iterations < self.config.max_iterations:
            if time.time() > timeout_time:
                break

            iterations += 1
            current = heapq.heappop(open_set)
            current_tuple = current.point.to_tuple()

            # Check if goal reached
            if current.point.distance_to(goal_grid) < self.config.resolution:
                # Reconstruct path
                path = self._reconstruct_path(current)
                path.append(goal)  # Add exact goal

                result.success = True
                result.path = path
                result.cost = current.g_score
                result.nodes_explored = iterations
                result.computation_time = time.time() - start_time
                return result

            if current_tuple in closed_set:
                continue

            closed_set.add(current_tuple)

            # Explore neighbors
            for dx, dy, dz in self.NEIGHBORS_3D:
                neighbor_point = Point3D(
                    current.point.x + dx * self.config.resolution,
                    current.point.y + dy * self.config.resolution,
                    current.point.z + dz * self.config.resolution,
                )

                neighbor_tuple = neighbor_point.to_tuple()

                # Skip if in closed set
                if neighbor_tuple in closed_set:
                    continue

                # Check altitude constraints
                if not self._is_valid_altitude(neighbor_point):
                    continue

                # Check collision
                if self._is_collision(neighbor_point):
                    continue

                # Check climb angle
                if not self._is_valid_climb(current.point, neighbor_point):
                    continue

                # Calculate cost
                move_cost = current.point.distance_to(neighbor_point)
                tentative_g = current.g_score + move_cost

                # Check if better path
                if tentative_g < g_scores.get(neighbor_tuple, float('inf')):
                    g_scores[neighbor_tuple] = tentative_g

                    neighbor_node = AStarNode(
                        f_score=tentative_g + heuristic(neighbor_point, goal_grid),
                        point=neighbor_point,
                        g_score=tentative_g,
                        parent=current,
                    )
                    heapq.heappush(open_set, neighbor_node)

        result.nodes_explored = iterations
        result.computation_time = time.time() - start_time
        return result

    def _to_grid(self, point: Point3D) -> Point3D:
        """Snap point to grid."""
        res = self.config.resolution
        return Point3D(
            round(point.x / res) * res,
            round(point.y / res) * res,
            round(point.z / res) * res,
        )

    def _is_collision(self, point: Point3D) -> bool:
        """Check if point collides with any obstacle."""
        for obstacle in self._obstacles:
            if obstacle.contains(point, self.config.safety_margin):
                return True
        return False

    def _is_valid_altitude(self, point: Point3D) -> bool:
        """Check altitude constraints."""
        return self.config.min_altitude <= point.z <= self.config.max_altitude

    def _is_valid_climb(self, from_point: Point3D, to_point: Point3D) -> bool:
        """Check if climb/descent angle is valid."""
        h_dist = from_point.distance_2d(to_point)
        if h_dist < 1e-6:
            return True  # Vertical movement ok

        v_dist = abs(to_point.z - from_point.z)
        angle = math.degrees(math.atan2(v_dist, h_dist))
        return angle <= self.config.max_climb_angle

    def _reconstruct_path(self, node: AStarNode) -> List[Point3D]:
        """Reconstruct path from goal to start."""
        path = []
        current = node
        while current is not None:
            path.append(current.point)
            current = current.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# RRT Algorithm
# ---------------------------------------------------------------------------

@dataclass
class RRTNode:
    """Node for RRT tree."""
    point: Point3D
    parent: Optional['RRTNode'] = None
    cost: float = 0.0
    children: List['RRTNode'] = field(default_factory=list)


class RRTPlanner:
    """
    Rapidly-exploring Random Tree (RRT) path planner.

    Features:
    - Sampling-based planning for high-dimensional spaces
    - Goal biasing for faster convergence
    - Path smoothing
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        step_size: float = 2.0,
        goal_bias: float = 0.1,
    ):
        """
        Initialize RRT planner.

        Args:
            config: Planner configuration
            step_size: Maximum step size for tree extension
            goal_bias: Probability of sampling goal
        """
        self.config = config or PlannerConfig()
        self.step_size = step_size
        self.goal_bias = goal_bias
        self._obstacles: List[Obstacle] = []

        # Workspace bounds (will be set based on start/goal)
        self._bounds_min = Point3D(-100, -100, 0)
        self._bounds_max = Point3D(100, 100, 120)

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Set obstacles for planning."""
        self._obstacles = obstacles

    def set_bounds(self, min_point: Point3D, max_point: Point3D) -> None:
        """Set workspace bounds."""
        self._bounds_min = min_point
        self._bounds_max = max_point

    def plan(self, start: Point3D, goal: Point3D) -> PathResult:
        """
        Find path from start to goal using RRT.

        Args:
            start: Start position
            goal: Goal position

        Returns:
            PathResult with planned path
        """
        start_time = time.time()
        result = PathResult(algorithm="RRT")

        # Update bounds based on start/goal
        self._update_bounds(start, goal)

        # Initialize tree
        root = RRTNode(point=start)
        nodes = [root]

        goal_node = None
        iterations = 0
        timeout_time = start_time + self.config.timeout

        while iterations < self.config.max_iterations:
            if time.time() > timeout_time:
                break

            iterations += 1

            # Sample random point (with goal bias)
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample()

            # Find nearest node
            nearest = self._nearest_node(nodes, sample)

            # Extend toward sample
            new_point = self._steer(nearest.point, sample)

            # Check collision along edge
            if self._is_collision_free_edge(nearest.point, new_point):
                # Add new node
                new_node = RRTNode(
                    point=new_point,
                    parent=nearest,
                    cost=nearest.cost + nearest.point.distance_to(new_point),
                )
                nearest.children.append(new_node)
                nodes.append(new_node)

                # Check if goal reached
                if new_point.distance_to(goal) < self.step_size:
                    if self._is_collision_free_edge(new_point, goal):
                        goal_node = RRTNode(
                            point=goal,
                            parent=new_node,
                            cost=new_node.cost + new_point.distance_to(goal),
                        )
                        break

        if goal_node is not None:
            path = self._extract_path(goal_node)
            result.success = True
            result.path = path
            result.cost = goal_node.cost
        else:
            # Return closest path to goal
            closest = self._nearest_node(nodes, goal)
            result.path = self._extract_path(closest)
            result.cost = closest.cost

        result.nodes_explored = iterations
        result.computation_time = time.time() - start_time
        return result

    def _update_bounds(self, start: Point3D, goal: Point3D) -> None:
        """Update bounds based on start and goal."""
        margin = 50.0

        self._bounds_min = Point3D(
            min(start.x, goal.x) - margin,
            min(start.y, goal.y) - margin,
            max(self.config.min_altitude, min(start.z, goal.z) - margin),
        )
        self._bounds_max = Point3D(
            max(start.x, goal.x) + margin,
            max(start.y, goal.y) + margin,
            min(self.config.max_altitude, max(start.z, goal.z) + margin),
        )

    def _random_sample(self) -> Point3D:
        """Generate random sample within bounds."""
        return Point3D(
            random.uniform(self._bounds_min.x, self._bounds_max.x),
            random.uniform(self._bounds_min.y, self._bounds_max.y),
            random.uniform(self._bounds_min.z, self._bounds_max.z),
        )

    def _nearest_node(self, nodes: List[RRTNode], point: Point3D) -> RRTNode:
        """Find nearest node to point."""
        nearest = nodes[0]
        min_dist = float('inf')

        for node in nodes:
            dist = node.point.distance_to(point)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def _steer(self, from_point: Point3D, to_point: Point3D) -> Point3D:
        """Steer from one point toward another, limited by step size."""
        direction = to_point - from_point
        dist = from_point.distance_to(to_point)

        if dist <= self.step_size:
            return to_point

        direction = direction.normalized()
        return from_point + direction * self.step_size

    def _is_collision_free_edge(
        self,
        from_point: Point3D,
        to_point: Point3D,
    ) -> bool:
        """Check if edge is collision-free."""
        # Check altitude
        if not (self.config.min_altitude <= to_point.z <= self.config.max_altitude):
            return False

        # Sample along edge
        dist = from_point.distance_to(to_point)
        num_samples = max(2, int(dist / (self.config.resolution * 0.5)))

        for i in range(num_samples + 1):
            t = i / num_samples
            point = Point3D(
                from_point.x + t * (to_point.x - from_point.x),
                from_point.y + t * (to_point.y - from_point.y),
                from_point.z + t * (to_point.z - from_point.z),
            )

            for obstacle in self._obstacles:
                if obstacle.contains(point, self.config.safety_margin):
                    return False

        return True

    def _extract_path(self, node: RRTNode) -> List[Point3D]:
        """Extract path from node to root."""
        path = []
        current = node
        while current is not None:
            path.append(current.point)
            current = current.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# RRT* Algorithm
# ---------------------------------------------------------------------------

class RRTStarPlanner(RRTPlanner):
    """
    RRT* (RRT-Star) path planner.

    Extends RRT with:
    - Near-optimal paths through rewiring
    - Converges to optimal solution given enough time
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        step_size: float = 2.0,
        goal_bias: float = 0.1,
        search_radius: float = 5.0,
    ):
        """
        Initialize RRT* planner.

        Args:
            config: Planner configuration
            step_size: Maximum step size
            goal_bias: Probability of sampling goal
            search_radius: Radius for neighbor search and rewiring
        """
        super().__init__(config, step_size, goal_bias)
        self.search_radius = search_radius

    def plan(self, start: Point3D, goal: Point3D) -> PathResult:
        """
        Find near-optimal path using RRT*.

        Args:
            start: Start position
            goal: Goal position

        Returns:
            PathResult with planned path
        """
        start_time = time.time()
        result = PathResult(algorithm="RRT*")

        self._update_bounds(start, goal)

        # Initialize tree
        root = RRTNode(point=start, cost=0.0)
        nodes = [root]

        best_goal_node = None
        best_cost = float('inf')

        iterations = 0
        timeout_time = start_time + self.config.timeout

        while iterations < self.config.max_iterations:
            if time.time() > timeout_time:
                break

            iterations += 1

            # Sample
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample()

            # Find nearest
            nearest = self._nearest_node(nodes, sample)

            # Steer
            new_point = self._steer(nearest.point, sample)

            if not self._is_collision_free_edge(nearest.point, new_point):
                continue

            # Find neighbors within search radius
            neighbors = self._find_neighbors(nodes, new_point)

            # Choose best parent
            best_parent = nearest
            best_parent_cost = nearest.cost + nearest.point.distance_to(new_point)

            for neighbor in neighbors:
                if neighbor == nearest:
                    continue

                cost = neighbor.cost + neighbor.point.distance_to(new_point)
                if cost < best_parent_cost:
                    if self._is_collision_free_edge(neighbor.point, new_point):
                        best_parent = neighbor
                        best_parent_cost = cost

            # Create new node
            new_node = RRTNode(
                point=new_point,
                parent=best_parent,
                cost=best_parent_cost,
            )
            best_parent.children.append(new_node)
            nodes.append(new_node)

            # Rewire neighbors
            for neighbor in neighbors:
                if neighbor == best_parent:
                    continue

                new_cost = new_node.cost + new_node.point.distance_to(neighbor.point)
                if new_cost < neighbor.cost:
                    if self._is_collision_free_edge(new_node.point, neighbor.point):
                        # Rewire
                        if neighbor.parent:
                            neighbor.parent.children.remove(neighbor)
                        neighbor.parent = new_node
                        neighbor.cost = new_cost
                        new_node.children.append(neighbor)
                        self._update_descendant_costs(neighbor)

            # Check goal
            if new_point.distance_to(goal) < self.step_size:
                if self._is_collision_free_edge(new_point, goal):
                    goal_cost = new_node.cost + new_point.distance_to(goal)
                    if goal_cost < best_cost:
                        best_cost = goal_cost
                        best_goal_node = RRTNode(
                            point=goal,
                            parent=new_node,
                            cost=goal_cost,
                        )

        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            result.success = True
            result.path = path
            result.cost = best_goal_node.cost
        else:
            closest = self._nearest_node(nodes, goal)
            result.path = self._extract_path(closest)
            result.cost = closest.cost

        result.nodes_explored = iterations
        result.computation_time = time.time() - start_time
        return result

    def _find_neighbors(
        self,
        nodes: List[RRTNode],
        point: Point3D,
    ) -> List[RRTNode]:
        """Find all neighbors within search radius."""
        neighbors = []
        for node in nodes:
            if node.point.distance_to(point) <= self.search_radius:
                neighbors.append(node)
        return neighbors

    def _update_descendant_costs(self, node: RRTNode) -> None:
        """Update costs of all descendants after rewiring."""
        for child in node.children:
            child.cost = node.cost + node.point.distance_to(child.point)
            self._update_descendant_costs(child)


# ---------------------------------------------------------------------------
# Potential Field Planner
# ---------------------------------------------------------------------------

class PotentialFieldPlanner:
    """
    Artificial Potential Field path planner.

    Uses attractive and repulsive forces for reactive navigation.
    Good for local obstacle avoidance, may get stuck in local minima.
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 100.0,
        influence_distance: float = 10.0,
        step_size: float = 0.5,
    ):
        """
        Initialize potential field planner.

        Args:
            config: Planner configuration
            attractive_gain: Gain for attractive force
            repulsive_gain: Gain for repulsive force
            influence_distance: Distance where obstacles start repelling
            step_size: Step size for gradient descent
        """
        self.config = config or PlannerConfig()
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance
        self.step_size = step_size
        self._obstacles: List[Obstacle] = []

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Set obstacles."""
        self._obstacles = obstacles

    def plan(self, start: Point3D, goal: Point3D) -> PathResult:
        """
        Find path using gradient descent on potential field.

        Args:
            start: Start position
            goal: Goal position

        Returns:
            PathResult with planned path
        """
        start_time = time.time()
        result = PathResult(algorithm="PotentialField")

        path = [start]
        current = start

        iterations = 0
        timeout_time = start_time + self.config.timeout
        goal_threshold = self.step_size * 2

        while iterations < self.config.max_iterations:
            if time.time() > timeout_time:
                break

            iterations += 1

            # Check if goal reached
            if current.distance_to(goal) < goal_threshold:
                path.append(goal)
                result.success = True
                break

            # Calculate forces
            force = self._calculate_total_force(current, goal)

            # Check for local minimum (force too small)
            force_magnitude = math.sqrt(force.x**2 + force.y**2 + force.z**2)
            if force_magnitude < 0.01:
                # Random perturbation to escape local minimum
                force = Point3D(
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-0.5, 0.5),
                )
                force_magnitude = 1.0

            # Normalize and step
            force = force.normalized()
            new_point = current + force * self.step_size

            # Check altitude
            new_point = Point3D(
                new_point.x,
                new_point.y,
                max(self.config.min_altitude,
                    min(self.config.max_altitude, new_point.z)),
            )

            # Add to path if significantly different
            if new_point.distance_to(current) > 0.01:
                path.append(new_point)
                current = new_point

        result.path = path
        result.cost = sum(
            path[i].distance_to(path[i + 1])
            for i in range(len(path) - 1)
        )
        result.nodes_explored = iterations
        result.computation_time = time.time() - start_time

        return result

    def _calculate_total_force(
        self,
        point: Point3D,
        goal: Point3D,
    ) -> Point3D:
        """Calculate total force at point."""
        # Attractive force toward goal
        attractive = self._attractive_force(point, goal)

        # Repulsive forces from obstacles
        repulsive = Point3D(0, 0, 0)
        for obstacle in self._obstacles:
            repulsive = repulsive + self._repulsive_force(point, obstacle)

        return attractive + repulsive

    def _attractive_force(self, point: Point3D, goal: Point3D) -> Point3D:
        """Calculate attractive force toward goal."""
        direction = goal - point
        return direction.normalized() * self.attractive_gain

    def _repulsive_force(self, point: Point3D, obstacle: Obstacle) -> Point3D:
        """Calculate repulsive force from obstacle."""
        # Distance to obstacle surface
        dist = point.distance_to(obstacle.center) - obstacle.radius

        if dist >= self.influence_distance:
            return Point3D(0, 0, 0)

        if dist <= 0:
            dist = 0.01  # Prevent division by zero

        # Force magnitude (inverse square law)
        magnitude = self.repulsive_gain * (1.0 / dist - 1.0 / self.influence_distance) / (dist ** 2)

        # Direction away from obstacle
        direction = point - obstacle.center
        direction = direction.normalized()

        return direction * magnitude


# ---------------------------------------------------------------------------
# Path Smoothing
# ---------------------------------------------------------------------------

def smooth_path(
    path: List[Point3D],
    obstacles: List[Obstacle],
    safety_margin: float = 1.5,
    iterations: int = 100,
) -> List[Point3D]:
    """
    Smooth a path by removing unnecessary waypoints.

    Args:
        path: Original path
        obstacles: Obstacles to avoid
        safety_margin: Safety margin around obstacles
        iterations: Number of smoothing iterations

    Returns:
        Smoothed path
    """
    if len(path) <= 2:
        return path

    smoothed = list(path)

    for _ in range(iterations):
        if len(smoothed) <= 2:
            break

        # Random shortcut attempt
        i = random.randint(0, len(smoothed) - 1)
        j = random.randint(i + 2, min(i + 10, len(smoothed) - 1)) if i + 2 < len(smoothed) else None

        if j is None:
            continue

        # Check if shortcut is collision-free
        if _is_edge_collision_free(smoothed[i], smoothed[j], obstacles, safety_margin):
            # Remove intermediate points
            smoothed = smoothed[:i + 1] + smoothed[j:]

    return smoothed


def _is_edge_collision_free(
    from_point: Point3D,
    to_point: Point3D,
    obstacles: List[Obstacle],
    safety_margin: float,
) -> bool:
    """Check if edge is collision-free."""
    dist = from_point.distance_to(to_point)
    num_samples = max(2, int(dist / 0.5))

    for i in range(num_samples + 1):
        t = i / num_samples
        point = Point3D(
            from_point.x + t * (to_point.x - from_point.x),
            from_point.y + t * (to_point.y - from_point.y),
            from_point.z + t * (to_point.z - from_point.z),
        )

        for obstacle in obstacles:
            if obstacle.contains(point, safety_margin):
                return False

    return True


# ---------------------------------------------------------------------------
# Unified Planner Interface
# ---------------------------------------------------------------------------

class PathPlannerType(Enum):
    """Available path planner types."""
    A_STAR = "a_star"
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    POTENTIAL_FIELD = "potential_field"


class UnifiedPathPlanner:
    """
    Unified interface for all path planning algorithms.

    Provides automatic algorithm selection and fallback.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        """Initialize unified planner."""
        self.config = config or PlannerConfig()
        self._obstacles: List[Obstacle] = []

        # Initialize all planners
        self._a_star = AStarPlanner(self.config)
        self._rrt = RRTPlanner(self.config)
        self._rrt_star = RRTStarPlanner(self.config)
        self._potential = PotentialFieldPlanner(self.config)

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Set obstacles for all planners."""
        self._obstacles = obstacles
        self._a_star.set_obstacles(obstacles)
        self._rrt.set_obstacles(obstacles)
        self._rrt_star.set_obstacles(obstacles)
        self._potential.set_obstacles(obstacles)

    def plan(
        self,
        start: Point3D,
        goal: Point3D,
        algorithm: PathPlannerType = PathPlannerType.A_STAR,
        smooth: bool = True,
    ) -> PathResult:
        """
        Plan path using specified algorithm.

        Args:
            start: Start position
            goal: Goal position
            algorithm: Algorithm to use
            smooth: Whether to smooth the path

        Returns:
            PathResult with planned path
        """
        if algorithm == PathPlannerType.A_STAR:
            result = self._a_star.plan(start, goal)
        elif algorithm == PathPlannerType.RRT:
            result = self._rrt.plan(start, goal)
        elif algorithm == PathPlannerType.RRT_STAR:
            result = self._rrt_star.plan(start, goal)
        elif algorithm == PathPlannerType.POTENTIAL_FIELD:
            result = self._potential.plan(start, goal)
        else:
            result = self._a_star.plan(start, goal)

        # Smooth path if requested
        if smooth and result.success and len(result.path) > 2:
            result.path = smooth_path(
                result.path,
                self._obstacles,
                self.config.safety_margin,
            )
            result.smoothed = True

            # Recalculate cost
            result.cost = sum(
                result.path[i].distance_to(result.path[i + 1])
                for i in range(len(result.path) - 1)
            )

        return result

    def plan_with_fallback(
        self,
        start: Point3D,
        goal: Point3D,
        primary: PathPlannerType = PathPlannerType.A_STAR,
        smooth: bool = True,
    ) -> PathResult:
        """
        Plan with automatic fallback to other algorithms.

        Args:
            start: Start position
            goal: Goal position
            primary: Primary algorithm to try first
            smooth: Whether to smooth the path

        Returns:
            PathResult from first successful algorithm
        """
        # Try algorithms in order
        algorithms = [primary]

        # Add fallbacks
        if primary != PathPlannerType.RRT_STAR:
            algorithms.append(PathPlannerType.RRT_STAR)
        if primary != PathPlannerType.RRT:
            algorithms.append(PathPlannerType.RRT)
        if primary != PathPlannerType.A_STAR:
            algorithms.append(PathPlannerType.A_STAR)

        for algo in algorithms:
            result = self.plan(start, goal, algo, smooth)
            if result.success:
                return result

        # Return last result even if failed
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get planner status."""
        return {
            'obstacle_count': len(self._obstacles),
            'config': {
                'resolution': self.config.resolution,
                'safety_margin': self.config.safety_margin,
                'max_iterations': self.config.max_iterations,
                'timeout': self.config.timeout,
            },
        }
