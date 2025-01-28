#Bounding boxes för allt, static=byggnader, dynamic=bilar

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import carla
import numpy as np
import numpy.typing as npt
from carla_utils import location_to_array, get_prioritized
import config

# Joins vertices from carla.BoundingBox.get_local_vertices()/get_world_vertices()
# into edges
BBOX_EDGE_PAIRS = np.array(
    [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [0, 4],
        [4, 5],
        [5, 1],
        [5, 7],
        [7, 6],
        [6, 4],
        [6, 2],
        [7, 3],
    ]
)


@dataclass(frozen=True)
class BboxCorners:
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]


class BboxHandler:
    # Gets, processes and stores bounding boxes from carla
    def __init__(self, world: carla.World) -> None:
        self.world = world
        (
            self.static_vertices,
            self.static_edges,
            self.vertices_per_bbox,
        ) = self.get_static_vertices_edges()

    def get_static_vertices_edges(
        self,
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], int]:
        # Gets the vertices and edges of bounding boxes for the static objects on the map in the world frame of reference

        vertices: List[List[int]] = []
        env_objs: List[carla.EnvironmentObject] = self.world.get_environment_objects(
            config.STATIC_BBOXES_TO_PLOT
        )

        # Get all the vertices and temporarily store as a list of numpy arrays
        for env_obj in env_objs:
            vertices_carla = env_obj.bounding_box.get_local_vertices()
            vertices_per_bbox = len(vertices_carla)
            vertices.extend(
                [[vertex.x, vertex.y, vertex.z] for vertex in vertices_carla]
            )
        n_bboxes = len(vertices) // vertices_per_bbox

        # Generate edge pairings for all the vertices
        edges = (
            np.tile(BBOX_EDGE_PAIRS, (n_bboxes, 1)).reshape(
                -1, BBOX_EDGE_PAIRS.shape[0], 2
            )
            + np.arange(0, len(vertices), vertices_per_bbox)[:, np.newaxis, np.newaxis]
        ).reshape(-1, 2)

        # Convert vertices to a numpy array before returning it
        return np.asarray(vertices).reshape(-1, 3), edges, vertices_per_bbox

    def get_dynamic_vertices(
        self, actor_ids: Set[int]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Return the vertices of the bounding boxes for the dynamic objects on the map
        vertices: List[List[int]] = []
        actors: List[carla.Vehicle] = self.world.get_actors(list(actor_ids))
        for actor in actors:
            tag = get_prioritized(actor.semantic_tags)
            if tag in config.DYNAMIC_BBOXES_TO_PLOT:
                vertices_carla = actor.bounding_box.get_world_vertices(
                    actor.get_transform()
                )
                vertices.extend(
                    [[vertex.x, vertex.y, vertex.z] for vertex in vertices_carla]
                )

        n_bboxes = len(vertices) // self.vertices_per_bbox
        edges = (
            np.tile(BBOX_EDGE_PAIRS, (n_bboxes, 1)).reshape(
                -1, BBOX_EDGE_PAIRS.shape[0], 2
            )
            + np.arange(
                self.static_vertices.shape[0],
                self.static_vertices.shape[0] + len(vertices),
                self.vertices_per_bbox,
            )[:, np.newaxis, np.newaxis]
        ).reshape(-1, 2)
        return np.asarray(vertices).reshape(-1, 3), edges

    def get_connected_dynamic_vertices(
        self, actor_ids: Set[int]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        # Return the vertices of the bounding boxes for the dynamic objects on the map
        vertices_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        actors: List[carla.Vehicle] = self.world.get_actors(list(actor_ids))
        for actor in actors:
            tag = get_prioritized(actor.semantic_tags)
            #if tag in config.DYNAMIC_BBOXES_TO_PLOT:
            vertices_carla = actor.bounding_box.get_world_vertices(
                actor.get_transform()
            )
            vertices = np.array([[vertex.x, vertex.y, vertex.z] for vertex in vertices_carla])
            n_bboxes = len(vertices) // self.vertices_per_bbox
            edges = (
                np.tile(BBOX_EDGE_PAIRS, (n_bboxes, 1)).reshape(
                    -1, BBOX_EDGE_PAIRS.shape[0], 2
                )
                + np.arange(
                    self.static_vertices.shape[0],
                    self.static_vertices.shape[0] + len(vertices),
                    self.vertices_per_bbox,
                )[:, np.newaxis, np.newaxis]
            ).reshape(-1, 2)
            vertices_dict[actor.id] = (vertices, edges)
        return vertices_dict


    def get_transformed_vertices_edges(
        self,
        actor_ids: Set[int],
        ego_pos: npt.NDArray[np.float64],
        world_to_ego_rot: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Transform all stored bounding boxes to the input frame of reference

        dynamic_vertices, dynamic_edges = self.get_dynamic_vertices(actor_ids)

        edges = np.vstack((self.static_edges, dynamic_edges))
        vertices = np.vstack((self.static_vertices, dynamic_vertices))
        vertices = vertices @ world_to_ego_rot.T
        vertices += ego_pos
        return vertices, edges
    
    def get_transformed_vertices_edges_dynamic(
        self,
        actor_ids: Set[int],
        ego_pos: npt.NDArray[np.float64],
        world_to_ego_rot: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Transform all stored bounding boxes to the input frame of reference

        dynamic_vertices, edges = self.get_dynamic_vertices(actor_ids)

        #edges = np.vstack((self.static_edges, dynamic_edges))
        #vertices = np.vstack((self.static_vertices, dynamic_vertices))
        vertices = dynamic_vertices @ world_to_ego_rot.T
        vertices += ego_pos
        return vertices, edges


def get_actor_bbox_corners(
    actor_ids: List[int], world: carla.World
) -> Dict[int, BboxCorners]:
    # Compute bounding box corners for all the actors on the map in the actors frame of reference
    actor_bboxes = {}
    actors: List[carla.Vehicle] = world.get_actors(actor_ids)
    for actor in actors:
        bbox = actor.bounding_box

        # Compute new bounding boxes for each scatterer
        bboxes: List[carla.BoundingBox] = []
        for scatterer in config.SCATTERERS:
            extent = carla.Vector3D(
                scatterer.extent_ratio.x * bbox.extent.x,
                scatterer.extent_ratio.y * bbox.extent.y,
                scatterer.extent_ratio.z * bbox.extent.z,
            )
            center = carla.Location(
                bbox.location.x + extent.x * scatterer.center_ratio.x,
                bbox.location.y + extent.y * scatterer.center_ratio.y,
                bbox.location.z + extent.z * scatterer.center_ratio.z,
            )
            bboxes.append(carla.BoundingBox(center, extent))

        vertices = np.array(
            [
                [location_to_array(loc) for loc in bbox.get_local_vertices()]
                for bbox in bboxes
            ]
        )

        # Compute upper and lower corners
        vertices_max = np.max(vertices, axis=1, keepdims=True)
        vertices_min = np.min(vertices, axis=1, keepdims=True)
        actor_bboxes[actor.id] = BboxCorners(lower=vertices_min, upper=vertices_max)

    return actor_bboxes
