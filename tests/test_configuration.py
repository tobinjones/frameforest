"""Tests for the Configuration class."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skspatial.objects import Point, Points

from scenetree import Workspace


class TestConnectByTransform:
    """Tests for Configuration.connect_by_transform()."""

    def test_connect_with_identity(self) -> None:
        """Connecting scenes with identity transform works."""
        ws = Workspace()
        ws.create_scene("a")
        ws.create_scene("b")
        config = ws.create_configuration("test")

        identity = np.eye(4)
        config.connect_by_transform("a", "b", identity)

        result = config.get_transform("a", "b")
        assert_array_almost_equal(result, identity)

    def test_connect_with_translation(self) -> None:
        """Connecting scenes with translation transform works."""
        ws = Workspace()
        ws.create_scene("a")
        ws.create_scene("b")
        config = ws.create_configuration("test")

        transform = np.eye(4)
        transform[:3, 3] = [10, 20, 30]  # Translation
        config.connect_by_transform("a", "b", transform)

        result = config.get_transform("a", "b")
        assert_array_almost_equal(result, transform)

    def test_connect_nonexistent_from_scene_raises(self) -> None:
        """Connecting from nonexistent scene raises KeyError."""
        ws = Workspace()
        ws.create_scene("b")
        config = ws.create_configuration("test")

        with pytest.raises(KeyError, match="'nonexistent' does not exist"):
            config.connect_by_transform("nonexistent", "b", np.eye(4))

    def test_connect_nonexistent_to_scene_raises(self) -> None:
        """Connecting to nonexistent scene raises KeyError."""
        ws = Workspace()
        ws.create_scene("a")
        config = ws.create_configuration("test")

        with pytest.raises(KeyError, match="'nonexistent' does not exist"):
            config.connect_by_transform("a", "nonexistent", np.eye(4))


class TestGetTransform:
    """Tests for Configuration.get_transform()."""

    def test_get_inverse_transform(self) -> None:
        """Getting transform in reverse direction returns inverse."""
        ws = Workspace()
        ws.create_scene("a")
        ws.create_scene("b")
        config = ws.create_configuration("test")

        transform = np.eye(4)
        transform[:3, 3] = [10, 20, 30]
        config.connect_by_transform("a", "b", transform)

        # Get reverse transform
        inverse = config.get_transform("b", "a")
        assert_array_almost_equal(inverse[:3, 3], [-10, -20, -30])

    def test_get_chained_transform(self) -> None:
        """Transform through multiple scenes is computed correctly."""
        ws = Workspace()
        ws.create_scene("a")
        ws.create_scene("b")
        ws.create_scene("c")
        config = ws.create_configuration("test")

        # a -> b: translate by [10, 0, 0]
        t_ab = np.eye(4)
        t_ab[:3, 3] = [10, 0, 0]
        config.connect_by_transform("a", "b", t_ab)

        # b -> c: translate by [0, 20, 0]
        t_bc = np.eye(4)
        t_bc[:3, 3] = [0, 20, 0]
        config.connect_by_transform("b", "c", t_bc)

        # a -> c should be [10, 20, 0]
        t_ac = config.get_transform("a", "c")
        assert_array_almost_equal(t_ac[:3, 3], [10, 20, 0])

    def test_get_transform_no_path_raises(self) -> None:
        """Getting transform with no path raises KeyError."""
        ws = Workspace()
        ws.create_scene("a")
        ws.create_scene("b")
        config = ws.create_configuration("test")
        # No connection made

        with pytest.raises(KeyError):
            config.get_transform("a", "b")


class TestConnectByBestFitPoints:
    """Tests for Configuration.connect_by_best_fit_points()."""

    def test_best_fit_translation_only(self) -> None:
        """Best fit with translated point clouds finds correct translation."""
        ws = Workspace()
        # Scene A has points at origin region
        ws.create_scene(
            "a",
            {
                "p1": Point([0, 0, 0]),
                "p2": Point([1, 0, 0]),
                "p3": Point([0, 1, 0]),
            },
        )
        # Scene B has same points translated by [10, 20, 30]
        ws.create_scene(
            "b",
            {
                "p1": Point([10, 20, 30]),
                "p2": Point([11, 20, 30]),
                "p3": Point([10, 21, 30]),
            },
        )

        config = ws.create_configuration("test")
        transform = config.connect_by_best_fit_points("a", "b")

        # Transform should translate [0,0,0] -> [10,20,30]
        assert_array_almost_equal(transform[:3, 3], [10, 20, 30], decimal=5)

    def test_best_fit_rotation(self) -> None:
        """Best fit with rotated point clouds finds correct rotation."""
        ws = Workspace()
        # Scene A: points along X and Y axes
        ws.create_scene(
            "a",
            {
                "p1": Point([0, 0, 0]),
                "p2": Point([1, 0, 0]),
                "p3": Point([0, 1, 0]),
            },
        )
        # Scene B: 90 degree rotation around Z axis (X->Y, Y->-X)
        ws.create_scene(
            "b",
            {
                "p1": Point([0, 0, 0]),
                "p2": Point([0, 1, 0]),
                "p3": Point([-1, 0, 0]),
            },
        )

        config = ws.create_configuration("test")
        transform = config.connect_by_best_fit_points("a", "b")

        # Apply transform to a's p2 [1,0,0] -> should get b's p2 [0,1,0]
        p2_a = np.array([1, 0, 0, 1])
        p2_transformed = transform @ p2_a
        assert_array_almost_equal(p2_transformed[:3], [0, 1, 0], decimal=5)

    def test_best_fit_with_subset_of_points(self) -> None:
        """Best fit can use a subset of shared points."""
        ws = Workspace()
        ws.create_scene(
            "a",
            {
                "p1": Point([0, 0, 0]),
                "p2": Point([1, 0, 0]),
                "p3": Point([0, 1, 0]),
                "p4": Point([100, 100, 100]),  # Outlier - should be excluded
            },
        )
        ws.create_scene(
            "b",
            {
                "p1": Point([10, 20, 30]),
                "p2": Point([11, 20, 30]),
                "p3": Point([10, 21, 30]),
                "p4": Point([0, 0, 0]),  # Outlier with different position
            },
        )

        config = ws.create_configuration("test")
        # Only use p1, p2, p3 for fitting
        transform = config.connect_by_best_fit_points("a", "b", object_ids=["p1", "p2", "p3"])

        # Should get clean translation ignoring p4
        assert_array_almost_equal(transform[:3, 3], [10, 20, 30], decimal=5)

    def test_best_fit_uses_points_centroid(self) -> None:
        """Best fit uses centroid for Points objects with multiple observations."""
        ws = Workspace()
        ws.create_scene(
            "a",
            {
                "p1": Points([[0, 0, 0], [0.2, 0.2, 0.2]]),  # Centroid: [0.1, 0.1, 0.1]
                "p2": Point([1, 0, 0]),
                "p3": Point([0, 1, 0]),
            },
        )
        ws.create_scene(
            "b",
            {
                "p1": Point([10.1, 20.1, 30.1]),  # Matches centroid + translation
                "p2": Point([11, 20, 30]),
                "p3": Point([10, 21, 30]),
            },
        )

        config = ws.create_configuration("test")
        transform = config.connect_by_best_fit_points("a", "b")

        # Should handle mixed Point/Points correctly
        assert_array_almost_equal(transform[:3, 3], [10, 20, 30], decimal=5)

    def test_best_fit_insufficient_points_raises(self) -> None:
        """Best fit with fewer than 3 shared points raises ValueError."""
        ws = Workspace()
        ws.create_scene("a", {"p1": Point([0, 0, 0]), "p2": Point([1, 0, 0])})
        ws.create_scene("b", {"p1": Point([0, 0, 0]), "p2": Point([1, 0, 0])})

        config = ws.create_configuration("test")
        with pytest.raises(ValueError, match="at least 3 shared points"):
            config.connect_by_best_fit_points("a", "b")

    def test_best_fit_no_shared_points_raises(self) -> None:
        """Best fit with no shared points raises ValueError."""
        ws = Workspace()
        ws.create_scene("a", {"p1": Point([0, 0, 0])})
        ws.create_scene("b", {"p2": Point([0, 0, 0])})

        config = ws.create_configuration("test")
        with pytest.raises(ValueError, match="at least 3 shared points"):
            config.connect_by_best_fit_points("a", "b")

    def test_best_fit_nonexistent_scene_raises(self) -> None:
        """Best fit with nonexistent scene raises KeyError."""
        ws = Workspace()
        ws.create_scene("a", {"p1": Point([0, 0, 0])})

        config = ws.create_configuration("test")
        with pytest.raises(KeyError):
            config.connect_by_best_fit_points("a", "nonexistent")


class TestAsTransformManager:
    """Tests for Configuration.as_transform_manager()."""

    def test_returns_copy(self) -> None:
        """as_transform_manager() returns a copy, not the original."""
        ws = Workspace()
        ws.create_scene("a")
        ws.create_scene("b")
        config = ws.create_configuration("test")
        config.connect_by_transform("a", "b", np.eye(4))

        tm = config.as_transform_manager()

        # Modifying the copy shouldn't affect the original
        new_transform = np.eye(4)
        new_transform[:3, 3] = [999, 999, 999]
        tm.add_transform("a", "b", new_transform)

        # Original should still have identity
        original = config.get_transform("a", "b")
        assert_array_almost_equal(original[:3, 3], [0, 0, 0])
