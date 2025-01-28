from typing import Dict
import carla
from carla_utils import string_to_tag
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationParams:
    rcs: float
    clustering_distance: float


@dataclass(frozen=True)
class Scatterer:
    center_ratio: carla.Vector3D
    extent_ratio: carla.Vector3D


# Parameters used for the simulation of different types of objects.
# Adding a tag will enable it in the simulation, and vice versa.
# rcs is used for both static and dynamic objects, and determines
# probability of detections and signal strength in the simulation.
# clustering_distance is only used for static object, and determines the
# distance threshold for clustering
RADAR_PARAMS: Dict[int, SimulationParams] = {
    string_to_tag["Buildings"]: SimulationParams(rcs=15, clustering_distance=2),
    string_to_tag["Poles"]: SimulationParams(rcs=1, clustering_distance=2),
    string_to_tag["Car"]: SimulationParams(rcs=10, clustering_distance=1),
    string_to_tag["GuardRail"]: SimulationParams(rcs=1, clustering_distance=2),
    string_to_tag["Fences"]: SimulationParams(rcs=1, clustering_distance=2),
}

# Extent of scatterers on the sides of the vehicle as a fraction of the vehicles bounding box extent
extent_side = carla.Vector3D((1 / 6), (1 / 2), 1)

# Extent of scatterers on the front and back of the vehicle as a fraction of the vehicles bounding box extent
extent_front_back = carla.Vector3D((1 / 4), (1 / 2), 1)

# Scatterers on each vehicle
# The scatterer center is given by: center = bbox_center + (center_ratio * extent)
SCATTERERS = [
    Scatterer(center_ratio=carla.Vector3D(2, 1, 0), extent_ratio=extent_side),
    Scatterer(center_ratio=carla.Vector3D(2, -1, 0), extent_ratio=extent_side),
    Scatterer(center_ratio=carla.Vector3D(-2, 1, 0), extent_ratio=extent_side),
    Scatterer(center_ratio=carla.Vector3D(-2, -1, 0), extent_ratio=extent_side),
    Scatterer(center_ratio=carla.Vector3D(3, 0, 0), extent_ratio=extent_front_back),
    Scatterer(center_ratio=carla.Vector3D(-3, 0, 0), extent_ratio=extent_front_back),
]

# radar_sim will plot bounding boxes for static objects of the following type
# Currently, only a single type is supported
STATIC_BBOXES_TO_PLOT = string_to_tag["Buildings"]

# radar_sim will plot bounding boxes for the dynamic objects of the following type
DYNAMIC_BBOXES_TO_PLOT = [string_to_tag["Car"], string_to_tag["Motorcycle"]]

#string_to_tag["Truck"]
#string_to_tag["Bus"]