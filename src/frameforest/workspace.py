"""Core workspace for managing geometric objects across coordinate frames."""

from typing import Any

from pytransform3d.transform_manager import TransformManager


class Workspace:
    """Container for geometric objects organized by frames and configurations.

    The workspace manages:
    - Scenes: Collections of geometric objects with coordinates in specific frames
    - Configurations: Spatial arrangements defining how frames relate via transforms
    """

    def __init__(self) -> None:
        """Initialize an empty workspace."""
        self._scenes: dict[str, dict[str, Any]] = {}
        self._configurations: dict[str, TransformManager] = {}
