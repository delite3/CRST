import carla
import numpy.typing as npt
import numpy as np


class CarlaRadar:
    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Vehicle,
        sensor_transform: carla.Transform,
        vertical_fov: float,
        horizontal_fov: float,
        range_: float,
    ) -> None:
        # Carla radar handler, structured in a similar way as the simulated radar to simplify plotting etc
        # It only supports 2D point clouds for now

        # This flag indicates to the main loop that the callback has finished
        self.working = False

        self.world = world
        self.ego_vehicle = ego_vehicle
        self.range_ = range_
        self.horizontal_fov = horizontal_fov

        self.point_cloud: npt.NDArray[np.float64]
        self.velocities: npt.NDArray[np.float64]

        # Dummy values
        self.point_to_id = None
        self.point_to_tag = None
        self.ids = None
        self.tags = None
        self.snr = None
        self.rcs = None

        self.sensor_rot: npt.NDArray[np.float64]
        self.sensor_pos: npt.NDArray[np.float64]

        self.init_sensor(sensor_transform, vertical_fov, horizontal_fov, range_)
        self.sim_radar = True

    def init_sensor(
        self,
        sensor_transform: carla.Transform,
        vertical_fov: float,
        horizontal_fov: float,
        range_: float,
    ) -> None:
        points_per_second = 500

        sensor_bp = self.world.get_blueprint_library().find("sensor.other.radar")

        sensor_bp.set_attribute("vertical_fov", str(vertical_fov))
        sensor_bp.set_attribute("horizontal_fov", str(horizontal_fov))
        sensor_bp.set_attribute("range", str(range_))
        sensor_bp.set_attribute("points_per_second", str(points_per_second))

        self.sensor: carla.Sensor = self.world.spawn_actor(
            sensor_bp, sensor_transform, attach_to=self.ego_vehicle
        )
        self.actor_id = self.sensor.id
        self.sensor.listen(lambda data: self.callback(data))

    def callback(self, data: carla.RadarDetection) -> None:
        self.working = True
        # Decode the data
        radar_data = np.frombuffer(
            data.raw_data,
            dtype=np.dtype(
                [
                    ("velocity", np.float32),
                    ("azimuth", np.float32),
                    ("altitude", np.float32),
                    ("range", np.float32),
                ]
            ),
        )

        self.velocities = np.array(radar_data["velocity"])

        # Compute cartesian point cloud from range-azimuth data
        self.point_cloud = np.array(
            [
                radar_data["range"] * np.cos(radar_data["azimuth"]),
                radar_data["range"] * np.sin(radar_data["azimuth"]),
            ]
        ).T

        self.working = False
