"""Tests for the Workspace class."""

import pytest
from skspatial.objects import Point

from scenetree import Scene, Workspace


class TestWorkspaceCreation:
    """Tests for Workspace initialization."""

    def test_empty_workspace(self) -> None:
        """A new workspace has no scenes or configurations."""
        ws = Workspace()
        assert list(ws) == []

    def test_workspace_iteration(self) -> None:
        """Iterating over workspace yields scene names."""
        ws = Workspace()
        ws.create_scene("scene_a")
        ws.create_scene("scene_b")
        assert set(ws) == {"scene_a", "scene_b"}


class TestSceneCreation:
    """Tests for Workspace.create_scene()."""

    def test_create_empty_scene(self) -> None:
        """Creating a scene without objects returns empty Scene proxy."""
        ws = Workspace()
        scene = ws.create_scene("test")
        assert isinstance(scene, Scene)
        assert scene.name == "test"
        assert len(scene) == 0

    def test_create_scene_with_objects(self) -> None:
        """Creating a scene with objects populates the scene."""
        ws = Workspace()
        p1 = Point([1, 2, 3])
        scene = ws.create_scene("test", {"QP.F1": p1})
        assert len(scene) == 1
        assert "QP.F1" in scene

    def test_create_scene_copies_objects(self) -> None:
        """Objects dict is copied, not stored by reference."""
        ws = Workspace()
        objects = {"QP.F1": Point([1, 2, 3])}
        ws.create_scene("test", objects)

        # Modifying original dict doesn't affect scene
        objects["QP.F2"] = Point([4, 5, 6])
        assert "QP.F2" not in ws["test"]

    def test_create_duplicate_scene_raises(self) -> None:
        """Creating a scene with an existing name raises ValueError."""
        ws = Workspace()
        ws.create_scene("test")
        with pytest.raises(ValueError, match="already exists"):
            ws.create_scene("test")


class TestSceneAccess:
    """Tests for accessing scenes via Workspace.__getitem__."""

    def test_getitem_returns_scene_proxy(self) -> None:
        """Workspace['name'] returns a Scene proxy."""
        ws = Workspace()
        ws.create_scene("test")
        scene = ws["test"]
        assert isinstance(scene, Scene)
        assert scene.name == "test"

    def test_getitem_nonexistent_raises(self) -> None:
        """Accessing nonexistent scene raises KeyError."""
        ws = Workspace()
        with pytest.raises(KeyError, match="does not exist"):
            _ = ws["nonexistent"]

    def test_contains_existing_scene(self) -> None:
        """'name' in workspace returns True for existing scenes."""
        ws = Workspace()
        ws.create_scene("test")
        assert "test" in ws

    def test_contains_nonexistent_scene(self) -> None:
        """'name' in workspace returns False for nonexistent scenes."""
        ws = Workspace()
        assert "nonexistent" not in ws


class TestConfigurationCreation:
    """Tests for Workspace.create_configuration()."""

    def test_create_configuration(self) -> None:
        """Creating a configuration returns a Configuration proxy."""
        ws = Workspace()
        config = ws.create_configuration("test")
        assert config.name == "test"

    def test_create_duplicate_configuration_raises(self) -> None:
        """Creating a configuration with existing name raises ValueError."""
        ws = Workspace()
        ws.create_configuration("test")
        with pytest.raises(ValueError, match="already exists"):
            ws.create_configuration("test")

    def test_get_configuration(self) -> None:
        """Workspace.configuration() returns existing configuration."""
        ws = Workspace()
        ws.create_configuration("test")
        config = ws.configuration("test")
        assert config.name == "test"

    def test_get_nonexistent_configuration_raises(self) -> None:
        """Getting nonexistent configuration raises KeyError."""
        ws = Workspace()
        with pytest.raises(KeyError, match="does not exist"):
            ws.configuration("nonexistent")
