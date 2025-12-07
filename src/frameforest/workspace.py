"""Core workspace for managing geometric objects across coordinate frames."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Self

from pytransform3d.transform_manager import TransformManager

if TYPE_CHECKING:
    from collections.abc import ItemsView


class Frame:
    """Proxy object providing dict-like access to objects in a frame.

    Frame objects are lightweight proxies that reference data stored in the
    parent Workspace. They should not be stored long-term; retrieve a fresh
    proxy from the workspace when needed.
    """

    def __init__(self, workspace: "Workspace", name: str) -> None:
        """Initialize a frame proxy.

        Args:
            workspace: The parent workspace containing the frame data.
            name: The name of this frame.
        """
        self._workspace = workspace
        self._name = name

    @property
    def name(self) -> str:
        """The frame name."""
        return self._name

    def _get_data(self) -> dict[str, Any]:
        """Get the underlying data dict, raising KeyError if frame doesn't exist."""
        return self._workspace._scenes[self._name]

    def __getitem__(self, object_id: str) -> Any:
        """Get an object by ID: frame['QP.F1']"""
        return self._get_data()[object_id]

    def __setitem__(self, object_id: str, data: Any) -> None:
        """Set an object by ID: frame['QP.F1'] = point"""
        self._get_data()[object_id] = data

    def __delitem__(self, object_id: str) -> None:
        """Delete an object by ID: del frame['QP.F1']"""
        del self._get_data()[object_id]

    def __contains__(self, object_id: object) -> bool:
        """Check if object exists: 'QP.F1' in frame"""
        try:
            return object_id in self._get_data()
        except KeyError:
            return False

    def __iter__(self) -> Iterator[str]:
        """Iterate over object IDs: for obj_id in frame"""
        return iter(self._get_data())

    def __len__(self) -> int:
        """Number of objects in frame: len(frame)"""
        return len(self._get_data())

    def items(self) -> "ItemsView[str, Any]":
        """Dict-like items(): for obj_id, data in frame.items()"""
        return self._get_data().items()

    def update(self, objects: dict[str, Any]) -> None:
        """Batch add objects: frame.update({'QP.F1': p1, 'QP.F2': p2})"""
        self._get_data().update(objects)

    def __ior__(self, objects: dict[str, Any]) -> Self:
        """Batch add objects: frame |= {'QP.F1': p1, 'QP.F2': p2}"""
        self.update(objects)
        return self


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

    def create_frame(self, name: str) -> Frame:
        """Create a new frame and return a proxy to it.

        Args:
            name: The name for the new frame.

        Returns:
            A Frame proxy for the newly created frame.

        Raises:
            ValueError: If a frame with this name already exists.
        """
        if name in self._scenes:
            raise ValueError(f"Frame '{name}' already exists")
        self._scenes[name] = {}
        return Frame(self, name)

    def __getitem__(self, frame: str) -> Frame:
        """Get a frame proxy by name: ws['frame_A']

        Raises:
            KeyError: If the frame doesn't exist (use create_frame first).
        """
        if frame not in self._scenes:
            raise KeyError(f"Frame '{frame}' does not exist. Use create_frame() first.")
        return Frame(self, frame)

    def __contains__(self, frame: object) -> bool:
        """Check if frame exists: 'frame_A' in ws"""
        return frame in self._scenes

    def __iter__(self) -> Iterator[str]:
        """Iterate over frame names: for frame in ws"""
        return iter(self._scenes)
