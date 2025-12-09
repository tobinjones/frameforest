"""Tests for the View class."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skspatial.objects import Line, Point, Points

from scenetree import Workspace


class TestViewCreation:
    """Tests for Configuration.view_from()."""

    def test_create_view(self) -> None:
        """view_from() creates a View with correct reference scene."""
        ws = Workspace()
        ws.create_scene("ref")
        config = ws.create_configuration("test")

        view = config.view_from("ref")
        assert view.reference_scene == "ref"

    def test_view_from_nonexistent_raises(self) -> None:
        """view_from() with nonexistent scene raises KeyError."""
        ws = Workspace()
        config = ws.create_configuration("test")

        with pytest.raises(KeyError, match="does not exist"):
            config.view_from("nonexistent")


class TestViewGetObject:
    """Tests for View.get_object()."""

    def test_get_point_with_identity(self) -> None:
        """Getting Point through identity transform returns same coordinates."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("other", {"p": Point([1, 2, 3])})
        config = ws.create_configuration("test")
        config.connect_by_transform("other", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.get_object("other", "p")

        assert isinstance(result, Point)
        assert_array_almost_equal(np.asarray(result), [1, 2, 3])

    def test_get_point_with_translation(self) -> None:
        """Getting Point through translation transform applies offset."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("other", {"p": Point([1, 2, 3])})
        config = ws.create_configuration("test")

        transform = np.eye(4)
        transform[:3, 3] = [10, 20, 30]
        config.connect_by_transform("other", "ref", transform)

        view = config.view_from("ref")
        result = view.get_object("other", "p")

        assert_array_almost_equal(np.asarray(result), [11, 22, 33])

    def test_get_points_transformed(self) -> None:
        """Getting Points object transforms all points."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("other", {"pts": Points([[0, 0, 0], [1, 1, 1]])})
        config = ws.create_configuration("test")

        transform = np.eye(4)
        transform[:3, 3] = [10, 10, 10]
        config.connect_by_transform("other", "ref", transform)

        view = config.view_from("ref")
        result = view.get_object("other", "pts")

        assert isinstance(result, Points)
        coords = np.asarray(result)
        assert_array_almost_equal(coords[0], [10, 10, 10])
        assert_array_almost_equal(coords[1], [11, 11, 11])

    def test_get_unsupported_type_returns_not_implemented(self) -> None:
        """Getting unsupported object type returns NotImplemented."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("other", {"line": Line([0, 0, 0], [1, 0, 0])})
        config = ws.create_configuration("test")
        config.connect_by_transform("other", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.get_object("other", "line")

        assert result is NotImplemented

    def test_get_object_no_transform_path_raises(self) -> None:
        """Getting object with no transform path raises KeyError."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("other", {"p": Point([1, 2, 3])})
        config = ws.create_configuration("test")
        # No connection

        view = config.view_from("ref")
        with pytest.raises(KeyError):
            view.get_object("other", "p")

    def test_get_object_nonexistent_raises(self) -> None:
        """Getting nonexistent object raises KeyError."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("other")
        config = ws.create_configuration("test")
        config.connect_by_transform("other", "ref", np.eye(4))

        view = config.view_from("ref")
        with pytest.raises(KeyError):
            view.get_object("other", "nonexistent")


class TestViewQuery:
    """Tests for View.query()."""

    def test_query_all_objects(self) -> None:
        """query('*') returns all objects from connected scenes."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("a", {"p1": Point([1, 0, 0])})
        ws.create_scene("b", {"p2": Point([0, 1, 0])})
        config = ws.create_configuration("test")
        config.connect_by_transform("a", "ref", np.eye(4))
        config.connect_by_transform("b", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.query("*")

        assert set(result.keys()) == {"p1", "p2"}

    def test_query_with_pattern(self) -> None:
        """query() with pattern filters by object ID."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene(
            "other",
            {
                "QP.F1": Point([1, 0, 0]),
                "QP.F2": Point([2, 0, 0]),
                "SMR.1": Point([3, 0, 0]),
            },
        )
        config = ws.create_configuration("test")
        config.connect_by_transform("other", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.query("QP.*")

        assert set(result.keys()) == {"QP.F1", "QP.F2"}

    def test_query_consolidates_points_across_scenes(self) -> None:
        """query() consolidates same-ID Points from multiple scenes."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("a", {"p": Point([1, 0, 0])})
        ws.create_scene("b", {"p": Point([2, 0, 0])})
        config = ws.create_configuration("test")
        config.connect_by_transform("a", "ref", np.eye(4))
        config.connect_by_transform("b", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.query("p")

        # Both points should be consolidated into single Points object
        assert "p" in result
        assert isinstance(result["p"], Points)
        coords = np.asarray(result["p"])
        assert coords.shape == (2, 3)

    def test_query_consolidates_points_objects(self) -> None:
        """query() flattens Points objects when consolidating."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("a", {"p": Points([[1, 0, 0], [1.1, 0, 0]])})
        ws.create_scene("b", {"p": Point([2, 0, 0])})
        config = ws.create_configuration("test")
        config.connect_by_transform("a", "ref", np.eye(4))
        config.connect_by_transform("b", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.query("p")

        # Should have 3 points total (2 from 'a', 1 from 'b')
        coords = np.asarray(result["p"])
        assert coords.shape == (3, 3)

    def test_query_transforms_coordinates(self) -> None:
        """query() applies transforms to coordinates."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("other", {"p": Point([0, 0, 0])})
        config = ws.create_configuration("test")

        transform = np.eye(4)
        transform[:3, 3] = [10, 20, 30]
        config.connect_by_transform("other", "ref", transform)

        view = config.view_from("ref")
        result = view.query("p")

        coords = np.asarray(result["p"])
        assert_array_almost_equal(coords[0], [10, 20, 30])

    def test_query_specific_scenes(self) -> None:
        """query() with from_scenes limits to specified scenes."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("a", {"p": Point([1, 0, 0])})
        ws.create_scene("b", {"p": Point([2, 0, 0])})
        ws.create_scene("c", {"p": Point([3, 0, 0])})
        config = ws.create_configuration("test")
        config.connect_by_transform("a", "ref", np.eye(4))
        config.connect_by_transform("b", "ref", np.eye(4))
        config.connect_by_transform("c", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.query("p", from_scenes=["a", "b"])

        # Should only have 2 points (from a and b, not c)
        coords = np.asarray(result["p"])
        assert coords.shape == (2, 3)

    def test_query_skips_unsupported_types(self) -> None:
        """query() skips objects with unsupported types."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene(
            "other",
            {
                "p": Point([1, 0, 0]),
                "line": Line([0, 0, 0], [1, 0, 0]),
            },
        )
        config = ws.create_configuration("test")
        config.connect_by_transform("other", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.query("*")

        # Only point should be in result, line should be skipped
        assert "p" in result
        assert "line" not in result

    def test_query_excludes_disconnected_scenes(self) -> None:
        """query() excludes scenes not connected to reference."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("connected", {"p1": Point([1, 0, 0])})
        ws.create_scene("disconnected", {"p2": Point([2, 0, 0])})
        config = ws.create_configuration("test")
        config.connect_by_transform("connected", "ref", np.eye(4))
        # 'disconnected' is not connected

        view = config.view_from("ref")
        result = view.query("*")

        assert "p1" in result
        assert "p2" not in result

    def test_query_empty_result(self) -> None:
        """query() returns empty dict when no matches."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("other", {"p": Point([1, 0, 0])})
        config = ws.create_configuration("test")
        config.connect_by_transform("other", "ref", np.eye(4))

        view = config.view_from("ref")
        result = view.query("nonexistent.*")

        assert result == {}


class TestViewChainedTransforms:
    """Tests for View with multi-hop transform paths."""

    def test_get_object_through_chain(self) -> None:
        """get_object() works through chained transforms."""
        ws = Workspace()
        ws.create_scene("ref")
        ws.create_scene("mid")
        ws.create_scene("far", {"p": Point([0, 0, 0])})
        config = ws.create_configuration("test")

        # ref <- mid <- far
        t1 = np.eye(4)
        t1[:3, 3] = [10, 0, 0]
        config.connect_by_transform("mid", "ref", t1)

        t2 = np.eye(4)
        t2[:3, 3] = [0, 20, 0]
        config.connect_by_transform("far", "mid", t2)

        view = config.view_from("ref")
        result = view.get_object("far", "p")

        # Point should be translated by combined [10, 20, 0]
        assert_array_almost_equal(np.asarray(result), [10, 20, 0])
