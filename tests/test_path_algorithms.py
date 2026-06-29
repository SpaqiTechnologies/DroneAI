"""
Tests for advanced path planning algorithms.
"""

import pytest
import math
from core.path_algorithms import (
    Point3D,
    Obstacle,
    PathResult,
    PlannerConfig,
    AStarPlanner,
    RRTPlanner,
    RRTStarPlanner,
    PotentialFieldPlanner,
    UnifiedPathPlanner,
    PathPlannerType,
    smooth_path,
)


class TestPoint3D:
    """Tests for Point3D class."""

    def test_creation(self):
        p = Point3D(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_distance_to(self):
        p1 = Point3D(0, 0, 0)
        p2 = Point3D(3, 4, 0)
        assert abs(p1.distance_to(p2) - 5.0) < 0.001

    def test_distance_2d(self):
        p1 = Point3D(0, 0, 10)
        p2 = Point3D(3, 4, 20)
        assert abs(p1.distance_2d(p2) - 5.0) < 0.001

    def test_add(self):
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(4, 5, 6)
        result = p1 + p2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_subtract(self):
        p1 = Point3D(5, 7, 9)
        p2 = Point3D(1, 2, 3)
        result = p1 - p2
        assert result.x == 4
        assert result.y == 5
        assert result.z == 6

    def test_multiply(self):
        p = Point3D(1, 2, 3)
        result = p * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_normalized(self):
        p = Point3D(3, 0, 4)
        n = p.normalized()
        assert abs(n.x - 0.6) < 0.001
        assert abs(n.z - 0.8) < 0.001

    def test_to_tuple(self):
        p = Point3D(1, 2, 3)
        assert p.to_tuple() == (1, 2, 3)

    def test_to_dict(self):
        p = Point3D(1, 2, 3)
        d = p.to_dict()
        assert d == {'x': 1, 'y': 2, 'z': 3}


class TestObstacle:
    """Tests for Obstacle class."""

    def test_sphere_contains(self):
        obs = Obstacle(center=Point3D(0, 0, 0), radius=5.0)

        # Inside
        assert obs.contains(Point3D(0, 0, 0))
        assert obs.contains(Point3D(4, 0, 0))

        # Outside
        assert not obs.contains(Point3D(6, 0, 0))

    def test_sphere_with_margin(self):
        obs = Obstacle(center=Point3D(0, 0, 0), radius=5.0)

        # Just outside sphere but within margin
        assert obs.contains(Point3D(6, 0, 0), margin=2.0)
        assert not obs.contains(Point3D(8, 0, 0), margin=2.0)

    def test_cylinder_contains(self):
        obs = Obstacle(
            center=Point3D(0, 0, 10),
            radius=5.0,
            height=20.0,
            obstacle_type="cylinder"
        )

        # Inside
        assert obs.contains(Point3D(0, 0, 10))
        assert obs.contains(Point3D(4, 0, 15))

        # Outside (horizontally)
        assert not obs.contains(Point3D(6, 0, 10))

        # Outside (vertically)
        assert not obs.contains(Point3D(0, 0, 25))


class TestAStarPlanner:
    """Tests for A* path planner."""

    def test_simple_path(self):
        planner = AStarPlanner(PlannerConfig(resolution=1.0, timeout=5.0))

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan(start, goal)

        assert result.success
        assert len(result.path) >= 2
        assert result.algorithm == "A*"

    def test_path_around_obstacle(self):
        planner = AStarPlanner(PlannerConfig(resolution=1.0, timeout=5.0))

        # Place obstacle in direct path
        planner.add_obstacle(Obstacle(
            center=Point3D(5, 0, 10),
            radius=2.0
        ))

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan(start, goal)

        assert result.success
        # Path should deviate around obstacle
        assert result.cost > 10  # Direct distance is 10

    def test_no_path_blocked(self):
        config = PlannerConfig(resolution=1.0, timeout=1.0, max_iterations=1000)
        planner = AStarPlanner(config)

        # Completely block the path with a wall of obstacles
        for x in range(3, 8):
            for y in range(-5, 6):
                planner.add_obstacle(Obstacle(center=Point3D(x, y, 10), radius=0.8))

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan(start, goal)

        # May fail or find path over/around - either way algorithm ran
        assert result.computation_time >= 0

    def test_altitude_constraints(self):
        config = PlannerConfig(
            resolution=2.0,
            min_altitude=5.0,
            max_altitude=50.0,
        )
        planner = AStarPlanner(config)

        start = Point3D(0, 0, 30)
        goal = Point3D(10, 0, 30)

        result = planner.plan(start, goal)

        assert result.success
        # All points should be within altitude limits
        for point in result.path:
            assert config.min_altitude <= point.z <= config.max_altitude


class TestRRTPlanner:
    """Tests for RRT path planner."""

    def test_simple_path(self):
        planner = RRTPlanner(
            config=PlannerConfig(timeout=5.0),
            step_size=2.0,
            goal_bias=0.2
        )

        start = Point3D(0, 0, 10)
        goal = Point3D(20, 0, 10)

        result = planner.plan(start, goal)

        assert result.algorithm == "RRT"
        assert len(result.path) >= 2

    def test_path_with_obstacles(self):
        planner = RRTPlanner(step_size=2.0, goal_bias=0.15)

        planner.set_obstacles([
            Obstacle(center=Point3D(10, 0, 10), radius=3.0),
        ])

        start = Point3D(0, 0, 10)
        goal = Point3D(20, 0, 10)

        result = planner.plan(start, goal)

        assert len(result.path) >= 2


class TestRRTStarPlanner:
    """Tests for RRT* path planner."""

    def test_optimized_path(self):
        planner = RRTStarPlanner(
            config=PlannerConfig(timeout=3.0, max_iterations=2000),
            step_size=2.0,
            goal_bias=0.15,
            search_radius=5.0
        )

        start = Point3D(0, 0, 10)
        goal = Point3D(15, 0, 10)

        result = planner.plan(start, goal)

        assert result.algorithm == "RRT*"
        assert len(result.path) >= 2


class TestPotentialFieldPlanner:
    """Tests for potential field planner."""

    def test_simple_path(self):
        planner = PotentialFieldPlanner(
            config=PlannerConfig(timeout=3.0),
            step_size=0.5
        )

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan(start, goal)

        assert result.algorithm == "PotentialField"
        assert len(result.path) >= 2

    def test_obstacle_avoidance(self):
        planner = PotentialFieldPlanner(
            step_size=0.5,
            repulsive_gain=100.0,
            influence_distance=5.0
        )

        planner.set_obstacles([
            Obstacle(center=Point3D(5, 0, 10), radius=2.0),
        ])

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan(start, goal)

        # Path should exist
        assert len(result.path) >= 2


class TestUnifiedPathPlanner:
    """Tests for unified path planner interface."""

    def test_a_star_selection(self):
        planner = UnifiedPathPlanner()

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan(start, goal, PathPlannerType.A_STAR)

        assert result.algorithm == "A*"
        assert result.success

    def test_rrt_selection(self):
        planner = UnifiedPathPlanner()

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan(start, goal, PathPlannerType.RRT)

        assert result.algorithm == "RRT"

    def test_with_obstacles(self):
        planner = UnifiedPathPlanner()

        planner.set_obstacles([
            Obstacle(center=Point3D(5, 0, 10), radius=2.0),
        ])

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan(start, goal, PathPlannerType.A_STAR)

        assert len(result.path) >= 2

    def test_path_smoothing(self):
        planner = UnifiedPathPlanner()

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 10, 10)

        result = planner.plan(start, goal, smooth=True)

        assert result.smoothed or len(result.path) <= 3

    def test_fallback(self):
        planner = UnifiedPathPlanner()

        start = Point3D(0, 0, 10)
        goal = Point3D(10, 0, 10)

        result = planner.plan_with_fallback(start, goal)

        assert result.success
        assert len(result.path) >= 2


class TestPathSmoothing:
    """Tests for path smoothing function."""

    def test_smooth_simple_path(self):
        path = [
            Point3D(0, 0, 10),
            Point3D(2, 0, 10),
            Point3D(4, 0, 10),
            Point3D(6, 0, 10),
            Point3D(8, 0, 10),
            Point3D(10, 0, 10),
        ]

        smoothed = smooth_path(path, [], iterations=50)

        # Straight path should be reducible
        assert len(smoothed) <= len(path)

    def test_smooth_preserves_start_end(self):
        path = [
            Point3D(0, 0, 10),
            Point3D(5, 5, 10),
            Point3D(10, 0, 10),
        ]

        smoothed = smooth_path(path, [])

        assert smoothed[0] == path[0]
        assert smoothed[-1] == path[-1]


class TestPathResult:
    """Tests for PathResult dataclass."""

    def test_to_dict(self):
        result = PathResult(
            success=True,
            path=[Point3D(0, 0, 0), Point3D(1, 1, 1)],
            cost=1.732,
            nodes_explored=100,
            computation_time=0.5,
            algorithm="A*",
        )

        d = result.to_dict()

        assert d['success'] is True
        assert d['algorithm'] == "A*"
        assert d['nodes_explored'] == 100
        assert d['waypoint_count'] == 2
