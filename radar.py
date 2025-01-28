import time
from typing import Dict, List, Optional, Set, Tuple
from bboxes import BboxCorners
import config
from carla_utils import get_prioritized
import numpy as np
import numpy.typing as npt
import scipy.cluster
import carla
import fastcluster

from radar_model import RadarModel

rng = np.random.default_rng()


def uniform_filter(
    data: npt.NDArray[np.float64],
    grid_length: float,
    property: Optional[npt.NDArray[np.int_]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    # Downsample point cloud to a voxel grid
    # An optional property can be provided, duplicate points with different properties
    # will be kept. A typical use case is to keep multiple points in the same voxel with different tags.
    if property is None:
        filtered_data = (data / grid_length).astype(int)
    else:
        filtered_data = np.hstack(
            ((data / grid_length).astype(int), property[:, None].astype(int))
        )
    view = np.ascontiguousarray(filtered_data).view(
        np.dtype((np.void, filtered_data.dtype.itemsize * filtered_data.shape[1]))
    )
    _, idx = np.unique(view, return_index=True)
    filtered_data = filtered_data.astype(float) + grid_length / 2
    return filtered_data[idx, : data.shape[1]] * grid_length, idx


class SemanticRaycastHandler:
    # Receives data from the semantic raycast sensor. The data is preprocessed and additional
    # information is fetched from Carla.
    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Vehicle,
        sensor_transform: carla.Transform,
        upper_fov: float,
        lower_fov: float,
        horizontal_fov: float,
        range_: float,
    ) -> None:
        # The point cloud is downsampled to a voxel grid with sides of length self.downsample_grid_size
        # to reduce computational burden
        self.downsample_grid_size = 0.03

        self.ego_vehicle = ego_vehicle
        self.world = world
        self.init_sensor(
            sensor_transform,
            upper_fov,
            lower_fov,
            horizontal_fov,
            range_,
        )

        self.point_cloud: npt.NDArray[np.float64]
        self.point_to_id: npt.NDArray[np.int_]
        self.point_to_tag: npt.NDArray[np.int_]
        self.actors: List[carla.Vehicle]
        self.ids: Set[int]
        self.tags: Set[int]
        self.id_to_tag: Dict[int, int]
        self.velocities: npt.NDArray[np.float64]
        self.snr = None
        self.rcs = None

        self.actor_id: int

        self.sensor_rot: npt.NDArray[np.float64]
        self.sensor_pos: npt.NDArray[np.float64]
        self.sensor_to_world_transform: npt.NDArray[np.float64]
        self.relative_transforms: Dict[int, npt.NDArray[np.float64]]

    def init_sensor(
        self,
        sensor_transform: carla.Transform,
        upper_fov: float,
        lower_fov: float,
        horizontal_fov: float,
        range_: float,
    ) -> None:
        # Init the raycast sensor and attach it to the ego vehicle
        channels = 64
        points_per_second = 500000
        delta = self.world.get_settings().fixed_delta_seconds
        rotation_frequency = 1 / delta

        sensor_bp = self.world.get_blueprint_library().find(
            "sensor.lidar.ray_cast_semantic"
        )

        sensor_bp.set_attribute("upper_fov", str(upper_fov))
        sensor_bp.set_attribute("lower_fov", str(lower_fov))
        sensor_bp.set_attribute("horizontal_fov", str(horizontal_fov))
        sensor_bp.set_attribute("channels", str(channels))
        sensor_bp.set_attribute("range", str(range_))
        sensor_bp.set_attribute("rotation_frequency", str(rotation_frequency))
        sensor_bp.set_attribute("points_per_second", str(points_per_second))

        self.sensor: carla.Sensor = self.world.spawn_actor(
            sensor_bp, sensor_transform, attach_to=self.ego_vehicle
        )
        self.actor_id = self.sensor.id

    def calc_velocities(self) -> None:
        # Calculate the relative velocity of every point
        ego_vel = self.ego_vehicle.get_velocity()
        ego_vel_np = self.sensor_rot.dot(np.array([ego_vel.x, ego_vel.y, ego_vel.z]))

        point_vels = np.zeros_like(self.point_cloud)
        for actor in self.actors:
            actor_vel = actor.get_velocity()
            actor_vel_np = self.sensor_rot.dot(
                np.array([actor_vel.x, actor_vel.y, actor_vel.z])
            )  # Rotate velocity vector to vehicle frame
            point_vels[self.point_to_id == actor.id] = actor_vel_np

        delta_vel = point_vels - ego_vel_np  # Compute velocity relative to vehicle
        target_vec_norm = self.point_cloud / (
            np.linalg.norm(self.point_cloud, 2, axis=1, keepdims=True) + 1e-16
        )  # Compute unit vectors (small delta to avoid issues where norm = 0)
        self.velocities = np.sum(
            delta_vel * target_vec_norm, axis=1
        )  # Row-wise dot product

    def parse_data(self, data: carla.SemanticLidarMeasurement) -> None:
        data = np.frombuffer(
            data.raw_data,
            dtype=np.dtype(
                [
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("CosAngle", np.float32),
                    ("ObjIdx", np.uint32),
                    ("ObjTag", np.uint32),
                ]
            ),
        )

        self.point_cloud = np.array([data["x"], data["y"], data["z"]]).T
        self.point_to_tag = np.array(data["ObjTag"])
        self.point_to_id = np.array(data["ObjIdx"])

    def get_ego_transforms(self) -> None:
        self.sensor_to_world_transform = np.array(
            self.sensor.get_transform().get_inverse_matrix()
        )
        self.sensor_pos = self.sensor_to_world_transform[0:3, 3]
        self.sensor_rot = self.sensor_to_world_transform[0:3, 0:3]

    def downsample(self) -> None:
        if self.downsample_grid_size > 0:
            self.point_cloud, idx = uniform_filter(
                self.point_cloud, self.downsample_grid_size
            )
            self.point_to_tag = self.point_to_tag[idx]
            self.point_to_id = self.point_to_id[idx]

    def get_actor_data(self) -> None:
        # Fetch and preprocess data for later use

        # Set of all obeserved actors
        self.ids = set(self.point_to_id)
        # Points that don't correspond to actors are set to zero, so it should be removed
        self.ids.discard(0)
        self.ids = {int(s) for s in self.ids}
        self.actors = self.world.get_actors(list(self.ids))
        self.relative_transforms = self.get_relative_transforms()
        # Map actor of actors to tag
        self.id_to_tag = {
            actor.id: get_prioritized(actor.semantic_tags) for actor in self.actors
        }
        # Set of all observed tags
        self.tags = set(self.point_to_tag)

    def get_relative_transforms(self) -> Dict[int, npt.NDArray[np.float64]]:
        # Compute transforms from sensor frame to actor frame
        transforms = {}
        for actor in self.actors:
            transform = np.array(
                self.sensor.get_transform().get_inverse_matrix()
            ) @ np.array(actor.get_transform().get_matrix())
            transforms[actor.id] = transform
        return transforms

    def update(self, data: carla.SemanticLidarMeasurement) -> None:
        # Update function, can be used as a callback
        self.parse_data(data)
        self.downsample()
        self.get_ego_transforms()
        self.get_actor_data()
        self.calc_velocities()


class RadarDataHandler:
    # Receives data from the semantic raycast sensor, and simulates radar data.
    # The data is preprocessed by SemanticRaycastHandler.
    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Vehicle,
        sensor_transform: carla.Transform,
        upper_fov: float,
        lower_fov: float,
        horizontal_fov: float,
        range_: float,
    ) -> None:
        self.working = False
        self.world = world
        self.range_ = range_
        self.horizontal_fov = horizontal_fov

        self.point_cloud: npt.NDArray[np.float64]
        self.point_to_id: npt.NDArray[np.int_]
        self.point_to_tag: npt.NDArray[np.int_]
        self.ids: Set[int]
        self.tags: Set[int]
        self.velocities: npt.NDArray[np.float64]
        self.snr: npt.NDArray[np.float64]
        self.rcs: npt.NDArray[np.float64]

        self.sensor_rot: npt.NDArray[np.float64]
        self.sensor_pos: npt.NDArray[np.float64]
        self.sensor_to_world_transform: npt.NDArray[np.float64]
        self.relative_transforms: Dict[int, npt.NDArray[np.float64]]

        self.radar_model = RadarModel()
        self.raycast_sensor = SemanticRaycastHandler(
            world,
            ego_vehicle,
            sensor_transform,
            upper_fov,
            lower_fov,
            horizontal_fov,
            range_,
        )
        self.actor_id = self.raycast_sensor.sensor.id
        self.raycast_sensor.sensor.listen(lambda data: self.callback(data))
        self.sim_radar = True
        self.vehicle_bbox_corners: Dict[int, BboxCorners]

        # The flattened point cloud is downsampled to a grid with sides of length self.downsample_grid_size
        self.downsample_grid_size = self.raycast_sensor.downsample_grid_size

    def callback(self, data: carla.SemanticLidarMeasurement) -> None:
        # Simulates radar detections from semantic raycast sensor data
        self.working = True
        self.raycast_sensor.update(data)  # Preprocess data
        self.sensor_rot = self.raycast_sensor.sensor_rot
        self.sensor_pos = self.raycast_sensor.sensor_pos
        self.sensor_to_world_transform = self.raycast_sensor.sensor_to_world_transform
        self.relative_transforms = self.raycast_sensor.relative_transforms
        self.update()
        self.working = False

    def random_pts(
        self,
        pts: npt.NDArray[np.float64],
        vels: npt.NDArray[np.float64],
        r: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        n_pts: int,
        weights: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        # Weighted random sampling of points, returns the sampled measurements

        idx = rng.choice(
            pts.shape[0],
            size=n_pts,
            replace=False,
            p=weights / np.sum(weights),
        )
        pts = pts[idx, :]
        r = r[idx]
        phi = phi[idx]

        # Use the mean velocity of the object
        vels = np.full(pts.shape[0], np.mean(vels))
        return pts, vels, r, phi

    def set_sim_radar(self, sim_radar: bool) -> None:
        # Enable and disable the radar simulation
        self.sim_radar = sim_radar

    def set_vehicle_bbox_corners(
        self, vehicle_bbox_corners: Dict[int, BboxCorners]
    ) -> None:
        # Save the bounding box corners of all vehicles in the vehicle frame of reference
        self.vehicle_bbox_corners = {
            id_: BboxCorners(
                lower=corners.lower[:, :, :2],
                upper=corners.upper[:, :, :2],
            )
            for id_, corners in vehicle_bbox_corners.items()
        }

    def sim_radar_pts(
        self,
        pts: npt.NDArray[np.float64],
        vels: npt.NDArray[np.float64],
        params: config.SimulationParams,
        bbox_corners: Optional[BboxCorners],
        transform: Optional[npt.NDArray[np.float64]],
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        # Return detections for all points if simulation should be skipped
        if not self.sim_radar:
            r = np.linalg.norm(pts, 2, axis=1)
            phi = np.arctan2(pts[:, 1] * -1, pts[:, 0])
            snr_db = 10.0 * np.log10(self.radar_model.compute_snr(r, phi, params.rcs))
            rcs = np.full(vels.shape, params.rcs)
            return pts, vels, r, phi, snr_db, rcs

        if pts.shape[0] <= 0:
            return (
                np.array([]).reshape(0, pts.shape[1]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            )

        r = np.linalg.norm(pts, 2, axis=1)
        phi = np.arctan2(pts[:, 1] * -1, pts[:, 0])

        # Ratio between max range and the points under test
        range_prop = ((self.range_ - r) / self.range_) ** 2
        # Heuristic that determines the max number of detections
        # Plot is available in plots.py
        max_pts = (
            int(
                pts.shape[0] ** (1 / 4)
                * (np.mean(range_prop) ** 2)
                * self.radar_model.gaussian_gain(np.mean(phi, keepdims=True))
            )
            + 1
        )
        # Heuristic that determines the weights for the random sampling
        weights = (range_prop**2) * self.radar_model.gaussian_gain(phi)
        if bbox_corners and transform is not None:
            # Make sure most points end up inside bounding boxes

            # Transform points to actor frame
            pts_tr = pts - transform[0:2, 3]
            pts_tr = pts_tr @ transform[0:2, 0:2]
            # Check if there's points in any bounding box
            in_bbox = (
                ((bbox_corners.lower <= pts_tr) & (pts_tr <= bbox_corners.upper))
                .all(2)
                .any(0)
            )
            # Greatly reduce weights of points that aren't in any bounding box
            weights = weights * (in_bbox + 1e-9)

        # Sample points
        pts, vels, r, phi = self.random_pts(pts, vels, r, phi, max_pts, weights)

        # Compute SNR
        snr = self.radar_model.compute_snr(r, phi, params.rcs)
        snr_db = 10.0 * np.log10(snr)

        # Simulate detector
        detected = self.detected(snr)

        if np.all(~detected):
            # No detections, return empty arrays
            return (
                np.array([]).reshape(0, pts.shape[1]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            )

        # Get detections
        pts = pts[detected]
        vels = vels[detected]
        r = r[detected]
        phi = phi[detected]
        snr_db = snr_db[detected]

        # Simulate RCS estimation
        rcs = self.radar_model.sim_rcs(r, params.rcs)
        return pts, vels, r, phi, snr_db, rcs

    def sim_objects(
        self,
        point_cloud: npt.NDArray[np.float64],
        source_velocities: npt.NDArray[np.float64],
        ids: Set[int],
        id_to_tag: Dict[int, int],
        source_point_to_id: npt.NDArray[np.int_],
    ) -> None:
        # Reset IDs and tags
        self.ids = set()
        self.tags = set()

        # Add an empty array to each list to avoid issues with
        # concatenation if we don't get any detections later on
        point_clouds_list = [np.array([]).reshape((0, point_cloud.shape[1]))]
        velocities_list = [np.array([])]
        point_to_id_list = [np.array([], dtype=int)]
        point_to_tag_list = [np.array([], dtype=int)]
        r_list = [np.array([])]
        phi_list = [np.array([])]
        snr_list = [np.array([])]
        rcs_list = [np.array([])]

        # This isn't as bad as it looks, we're dealing with quite small array sizes
        # and a few IDs
        for id_ in ids:
            tag = id_to_tag[id_]
            if not (tag in config.RADAR_PARAMS.keys()):
                continue

            idx = source_point_to_id == id_

            # Simulate radar detections
            pts_id, vels_id, r_id, phi_id, snr_id, rcs_id = self.sim_radar_pts(
                point_cloud[idx],
                source_velocities[idx],
                config.RADAR_PARAMS[tag],
                self.vehicle_bbox_corners.get(id_),
                self.relative_transforms.get(id_),
            )

            # Append detected points and all associated data to lists
            if pts_id.shape[0] > 0:
                velocities_list.append(vels_id)
                point_to_tag_list.append(np.full(pts_id.shape[0], tag))
                point_clouds_list.append(pts_id)
                r_list.append(r_id)
                phi_list.append(phi_id)
                snr_list.append(snr_id)
                rcs_list.append(rcs_id)
                point_to_id_list.append(np.full(pts_id.shape[0], id_))
                self.tags.add(tag)
                self.ids.add(id_)

        # Concatenate all sub-arrays into arrays
        self.point_cloud = np.vstack(point_clouds_list)
        self.velocities = np.hstack(velocities_list)
        self.r = np.hstack(r_list)
        self.phi = np.hstack(phi_list)
        self.snr = np.hstack(snr_list)
        self.rcs = np.hstack(rcs_list)
        self.point_to_tag = np.hstack(point_to_tag_list)
        self.point_to_id = np.hstack(point_to_id_list)

    def cluster_actorless(
        self,
        point_cloud: npt.NDArray[np.float64],
        point_to_actor: npt.NDArray[np.int_],
        point_to_tag: npt.NDArray[np.int_],
        tags: Set[int],
    ) -> Tuple[npt.NDArray[np.int_], Set[int], Dict[int, int]]:
        # Cluster points that don't have actor IDs, assign internal IDs instead

        # Use negative IDs to avoid collitions with Carlas actor IDs
        current_id = -1
        id_to_tag: Dict[int, int] = {}
        point_to_id = point_to_actor
        for tag in tags:
            params = config.RADAR_PARAMS[tag]

            # Points that don't have actor IDs are mapped to 0 in point_to_actor
            # Need use actual indices here instead of a boolean index array to
            # recover indices in point_to_actor later on
            idx = np.flatnonzero(
                np.logical_and(point_to_tag == tag, point_to_actor == 0)
            )

            # If all points already have an ID, continue
            if idx.size == 0:
                continue
            pts = point_cloud[idx]

            if pts.shape[0] > 1:
                # Cluster and label points
                linked = fastcluster.linkage_vector(
                    pts, method="single", metric="euclidean"
                )
                label_to_idx = scipy.cluster.hierarchy.fcluster(
                    linked, t=params.clustering_distance, criterion="distance"
                )
            else:
                # Handle single point case
                label_to_idx = [1]

            # Set of unique labels
            labels = set(label_to_idx)
            for label in labels:
                # Recover point index, assign ID to the cluster
                point_to_id[idx[label == label_to_idx]] = current_id
                # Update id-tag-map
                id_to_tag[current_id] = tag
                # Get next id
                current_id -= 1

        # Set of generated IDs
        ids = set(range(-1, current_id, -1))
        return point_to_id, ids, id_to_tag

    def get_visible_actor_ids(self) -> Set[int]:
        # Return set of actor IDs that can be used in Carla
        # Ignore internal IDs
        return {s for s in self.ids if s > 0}

    def update(self) -> None:
        # Preprocess data and simulate radar detections

        # Only process points with tags that are in the list of objects to simulate
        tags = self.raycast_sensor.tags.intersection(config.RADAR_PARAMS.keys())
        idx = np.isin(self.raycast_sensor.point_to_tag, list(tags))

        # Flatten the point cloud to 2D and reindex the data
        point_cloud = self.raycast_sensor.point_cloud[idx, :2]
        velocities = self.raycast_sensor.velocities[idx]
        point_to_tag = self.raycast_sensor.point_to_tag[idx]
        point_to_actor = self.raycast_sensor.point_to_id[idx]

        # Downsample the flattened point cloud
        point_cloud, idx_filtered = uniform_filter(
            point_cloud, self.raycast_sensor.downsample_grid_size, point_to_tag
        )
        # Reindex the data
        velocities = velocities[idx_filtered]
        point_to_tag = point_to_tag[idx_filtered]
        point_to_actor = point_to_actor[idx_filtered]

        # The datatype of actor ids from the sensor is uints, cast it before assigning identities
        point_to_actor = point_to_actor.astype(np.int32)

        # Assign IDs to all points that don't have actor IDs
        point_to_id, ids, id_to_tag = self.cluster_actorless(
            point_cloud, point_to_actor, point_to_tag, tags
        )

        # Add Carla actor IDs
        id_to_tag.update(self.raycast_sensor.id_to_tag)
        ids.update(self.raycast_sensor.ids)

        # Simulate detections
        self.sim_objects(point_cloud, velocities, ids, id_to_tag, point_to_id)

    def detected(self, snr: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        # Samples and returns detected points based on SNR and Pd
        pd = self.radar_model.compute_pd(snr)
        return rng.random(pd.shape[0]) < pd
