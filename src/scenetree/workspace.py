"""Core workspace for managing geometric objects across coordinate frames."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation
from skspatial.objects import Point, Points

if TYPE_CHECKING:
    from collections.abc import ItemsView


class Scene:
    """Proxy object providing dict-like access to objects in a scene.

    Scene objects are lightweight proxies that reference data stored in the
    parent Workspace. They should not be stored long-term; retrieve a fresh
    proxy from the workspace when needed.
    """

    def __init__(self, workspace: "Workspace", name: str) -> None:
        """Initialize a scene proxy.

        Args:
            workspace: The parent workspace containing the scene data.
            name: The name of this scene.
        """
        self._workspace = workspace
        self._name = name

    @property
    def name(self) -> str:
        """The scene name."""
        return self._name

    def _get_data(self) -> dict[str, Any]:
        """Get the underlying data dict, raising KeyError if scene doesn't exist."""
        return self._workspace._scenes[self._name]

    def __getitem__(self, object_id: str) -> Any:
        """Get an object by ID: scene['QP.F1']"""
        return self._get_data()[object_id]

    def __setitem__(self, object_id: str, data: Any) -> None:
        """Set an object by ID: scene['QP.F1'] = point"""
        self._get_data()[object_id] = data

    def __delitem__(self, object_id: str) -> None:
        """Delete an object by ID: del scene['QP.F1']"""
        del self._get_data()[object_id]

    def __contains__(self, object_id: object) -> bool:
        """Check if object exists: 'QP.F1' in scene"""
        try:
            return object_id in self._get_data()
        except KeyError:
            return False

    def __iter__(self) -> Iterator[str]:
        """Iterate over object IDs: for obj_id in scene"""
        return iter(self._get_data())

    def __len__(self) -> int:
        """Number of objects in scene: len(scene)"""
        return len(self._get_data())

    def items(self) -> "ItemsView[str, Any]":
        """Dict-like items(): for obj_id, data in scene.items()"""
        return self._get_data().items()

    def update(self, objects: dict[str, Any]) -> None:
        """Batch add objects: scene.update({'QP.F1': p1, 'QP.F2': p2})"""
        self._get_data().update(objects)

    def __ior__(self, objects: dict[str, Any]) -> Self:
        """Batch add objects: scene |= {'QP.F1': p1, 'QP.F2': p2}"""
        self.update(objects)
        return self

    def add_points_from_observations(
        self, observations: Iterable[tuple[str, npt.ArrayLike]]
    ) -> None:
        """Add Points objects from an iterable of named point observations.

        Multiple observations with the same object_id are coalesced into a single
        Points object containing all observed coordinates.

        Args:
            observations: An iterable of (object_id, coordinates) tuples, where
                coordinates is an array-like of shape (3,) representing [x, y, z].

        Example:
            scene.points_from_observations([
                ("QP.F1", [1, 2, 3]),
                ("QP.F1", [1.1, 2.1, 3.1]),  # second observation of same point
                ("QP.F2", [4, 5, 6]),
            ])
            # scene["QP.F1"] is now Points([[1, 2, 3], [1.1, 2.1, 3.1]])
            # scene["QP.F2"] is now Points([[4, 5, 6]])
        """
        grouped: dict[str, list[npt.ArrayLike]] = defaultdict(list)
        for object_id, coords in observations:
            grouped[object_id].append(coords)

        data = self._get_data()
        for object_id, coords_list in grouped.items():
            data[object_id] = Points(coords_list)

    def get_point(self, object_id: str) -> npt.NDArray[np.floating[Any]]:
        """Get a single 3D position for an object.

        For Point objects: returns the point coordinates.
        For Points objects: returns the centroid of all points.

        Args:
            object_id: The object ID to look up.

        Returns:
            A numpy array of shape (3,) with the point coordinates.

        Raises:
            KeyError: If the object doesn't exist.
            TypeError: If the object is not a Point or Points.
        """
        obj = self._get_data()[object_id]
        if isinstance(obj, Point):
            return np.asarray(obj)
        elif isinstance(obj, Points):
            return np.asarray(obj.centroid())
        else:
            raise TypeError(f"Expected Point or Points, got {type(obj).__name__}")

    def get_mean_points(self) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Get mean positions for all Point/Points objects in the scene.

        Returns:
            A dict mapping object_id to a numpy array of shape (3,).
            Only includes objects that are Point or Points instances.
        """
        result: dict[str, npt.NDArray[np.floating[Any]]] = {}
        for object_id, obj in self._get_data().items():
            if isinstance(obj, Point):
                result[object_id] = np.asarray(obj)
            elif isinstance(obj, Points):
                result[object_id] = np.asarray(obj.centroid())
        return result


class Configuration:
    """Proxy object for managing transforms between scenes.

    Configuration objects are lightweight proxies that reference a TransformManager
    stored in the parent Workspace. They should not be stored long-term; retrieve
    a fresh proxy from the workspace when needed.
    """

    def __init__(self, workspace: "Workspace", name: str) -> None:
        """Initialize a configuration proxy.

        Args:
            workspace: The parent workspace containing the configuration data.
            name: The name of this configuration.
        """
        self._workspace = workspace
        self._name = name

    @property
    def name(self) -> str:
        """The configuration name."""
        return self._name

    def _get_tm(self) -> TransformManager:
        """Get the underlying TransformManager."""
        return self._workspace._configurations[self._name]

    def connect(
        self, from_scene: str, to_scene: str, transform: npt.NDArray[np.floating[Any]]
    ) -> None:
        """Add a transform connecting two scenes.

        Args:
            from_scene: The source scene name.
            to_scene: The destination scene name.
            transform: A 4x4 homogeneous transformation matrix.

        Raises:
            KeyError: If either scene doesn't exist in the workspace.
        """
        if from_scene not in self._workspace._scenes:
            raise KeyError(f"Scene '{from_scene}' does not exist")
        if to_scene not in self._workspace._scenes:
            raise KeyError(f"Scene '{to_scene}' does not exist")
        self._get_tm().add_transform(from_scene, to_scene, transform)

    def get_transform(self, from_scene: str, to_scene: str) -> npt.NDArray[np.floating[Any]]:
        """Get the transform between two scenes.

        If the scenes are not directly connected, the transform will be computed
        by following the path through the transform graph.

        Args:
            from_scene: The source scene name.
            to_scene: The destination scene name.

        Returns:
            A 4x4 homogeneous transformation matrix.

        Raises:
            KeyError: If no path exists between the scenes.
        """
        return self._get_tm().get_transform(from_scene, to_scene)

    def as_transform_manager(self) -> TransformManager:
        """Return a copy of the underlying TransformManager.

        Returns:
            A deep copy of the TransformManager, safe to modify without
            affecting the workspace.
        """
        return deepcopy(self._get_tm())

    def best_fit_points(
        self,
        from_scene: str,
        to_scene: str,
        object_ids: Iterable[str] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute and add a best-fit rigid transform between two scenes.

        Finds Point/Points objects shared between the scenes and computes
        the optimal rigid transform (rotation + translation) that aligns
        the from_scene points to the to_scene points using least-squares.

        The computed transform is added to this configuration.

        Args:
            from_scene: The source scene name.
            to_scene: The destination scene name.
            object_ids: Optional subset of object IDs to use for fitting.
                If None, uses all shared Point/Points objects.

        Returns:
            The 4x4 homogeneous transformation matrix.

        Raises:
            KeyError: If either scene doesn't exist.
            ValueError: If fewer than 3 shared points are found.
        """
        from_scene_proxy = self._workspace[from_scene]
        to_scene_proxy = self._workspace[to_scene]

        from_points_dict = from_scene_proxy.get_mean_points()
        to_points_dict = to_scene_proxy.get_mean_points()

        # Find shared object IDs
        if object_ids is not None:
            shared_ids = set(object_ids) & from_points_dict.keys() & to_points_dict.keys()
        else:
            shared_ids = from_points_dict.keys() & to_points_dict.keys()

        if len(shared_ids) < 3:
            raise ValueError(f"Need at least 3 shared points for best fit, found {len(shared_ids)}")

        # Build point arrays in consistent order
        shared_ids_list = list(shared_ids)
        from_points = np.array([from_points_dict[k] for k in shared_ids_list])
        to_points = np.array([to_points_dict[k] for k in shared_ids_list])

        # Compute centroids
        from_centroid = from_points.mean(axis=0)
        to_centroid = to_points.mean(axis=0)

        # Center the point clouds
        from_centered = from_points - from_centroid
        to_centered = to_points - to_centroid

        # Find optimal rotation using Kabsch algorithm
        rotation, _ = Rotation.align_vectors(to_centered, from_centered)

        # Build 4x4 homogeneous transform:
        # T = translate_to @ rotate @ translate_from_inv
        # Which transforms a point p as: T @ p = to_centroid + R @ (p - from_centroid)
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = to_centroid - rotation.apply(from_centroid)

        # Add to configuration
        self._get_tm().add_transform(from_scene, to_scene, transform)

        return transform


class Workspace:
    """Container for geometric objects organized by scenes and configurations.

    The workspace manages:
    - Scenes: Collections of geometric objects with coordinates in specific frames
    - Configurations: Spatial arrangements defining how frames relate via transforms
    """

    def __init__(self) -> None:
        """Initialize an empty workspace."""
        self._scenes: dict[str, dict[str, Any]] = {}
        self._configurations: dict[str, TransformManager] = {}

    def create_scene(self, name: str, objects: dict[str, Any] | None = None) -> Scene:
        """Create a new scene and return a proxy to it.

        Args:
            name: The name for the new scene.
            objects: Optional dict of objects to populate the scene with.

        Returns:
            A Scene proxy for the newly created scene.

        Raises:
            ValueError: If a scene with this name already exists.
        """
        if name in self._scenes:
            raise ValueError(f"Scene '{name}' already exists")
        self._scenes[name] = objects.copy() if objects else {}
        return Scene(self, name)

    def create_configuration(self, name: str) -> Configuration:
        """Create a new configuration and return a proxy to it.

        Args:
            name: The name for the new configuration.

        Returns:
            A Configuration proxy for the newly created configuration.

        Raises:
            ValueError: If a configuration with this name already exists.
        """
        if name in self._configurations:
            raise ValueError(f"Configuration '{name}' already exists")
        self._configurations[name] = TransformManager()
        return Configuration(self, name)

    def configuration(self, name: str) -> Configuration:
        """Get a configuration proxy by name.

        Args:
            name: The name of the configuration.

        Returns:
            A Configuration proxy.

        Raises:
            KeyError: If the configuration doesn't exist.
        """
        if name not in self._configurations:
            raise KeyError(
                f"Configuration '{name}' does not exist. Use create_configuration() first."
            )
        return Configuration(self, name)

    def __getitem__(self, scene: str) -> Scene:
        """Get a scene proxy by name: ws['scene_A']

        Raises:
            KeyError: If the scene doesn't exist (use create_scene first).
        """
        if scene not in self._scenes:
            raise KeyError(f"Scene '{scene}' does not exist. Use create_scene() first.")
        return Scene(self, scene)

    def __contains__(self, scene: object) -> bool:
        """Check if scene exists: 'scene_A' in ws"""
        return scene in self._scenes

    def __iter__(self) -> Iterator[str]:
        """Iterate over scene names: for scene in ws"""
        return iter(self._scenes)
