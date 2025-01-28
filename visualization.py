import enum
import random
from typing import Callable, Dict, List, Optional, Set, Union
import carla
import matplotlib
import numpy as np
import open3d as o3d
from carla_radar import CarlaRadar
import glfw_map
import numpy.typing as npt
import time

from radar import RadarDataHandler, SemanticRaycastHandler

VIRIDIS = np.array(matplotlib.colormaps["plasma"].colors)
VEHICLE_CM = np.array(matplotlib.colormaps["Set3"].colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
MAX_VELOCITY = 8
MAX_SNR = 100
MAX_RCS = 20


class ColorScheme(enum.IntEnum):
    VELOCITY = 0
    ID = enum.auto()
    TAG = enum.auto()
    SNR = enum.auto()
    RCS = enum.auto()


# From https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/image/CityScapesPalette.h
# Normalized to [0,1] for use in open3d
CITYSCAPES_PALETTE_MAP = (
    np.array(
        [
            [0, 0, 0],  # unlabeled     =   0
            # cityscape
            [128, 64, 128],  # road          =   1
            [244, 35, 232],  # sidewalk      =   2
            [70, 70, 70],  # building      =   3
            [102, 102, 156],  # wall          =   4
            [190, 153, 153],  # fence         =   5
            [153, 153, 153],  # pole          =   6
            [250, 170, 30],  # traffic light =   7
            [220, 220, 0],  # traffic sign  =   8
            [107, 142, 35],  # vegetation    =   9
            [152, 251, 152],  # terrain       =  10
            [70, 130, 180],  # sky           =  11
            [220, 20, 60],  # pedestrian    =  12
            [255, 0, 0],  # rider         =  13
            [0, 0, 142],  # Car           =  14
            [0, 0, 70],  # truck         =  15
            [0, 60, 100],  # bus           =  16
            [0, 80, 100],  # train         =  17
            [0, 0, 230],  # motorcycle    =  18
            [119, 11, 32],  # bicycle       =  19
            # custom
            [110, 190, 160],  # static        =  20
            [170, 120, 50],  # dynamic       =  21
            [55, 90, 80],  # other         =  22
            [45, 60, 150],  # water         =  23
            [157, 234, 50],  # road line     =  24
            [81, 0, 81],  # ground        =  25
            [150, 100, 100],  # bridge        =  26
            [230, 150, 140],  # rail track    =  27
            [180, 165, 180],  # guard rail    =  28
        ]
    )
    / 255.0
)


class PointCloudHolder:
    def __init__(
        self,
        sensors: List[Union[RadarDataHandler, SemanticRaycastHandler, CarlaRadar]],
        init_pts: npt.NDArray[np.float64],
    ) -> None:
        self.index = 0

        self.n_point_clouds = len(sensors)
        self.sensor_names = [sensor.__class__.__name__ for sensor in sensors]
        self.point_clouds = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(init_pts))
            for _ in range(self.n_point_clouds)
        ]
        self.bboxes = [o3d.geometry.LineSet() for _ in range(self.n_point_clouds)]

        self.bbox_color = np.array([0, 0, 0], dtype=np.float64)
        self.id_color_dict: Dict[int, npt.NDArray[np.float64]] = {}

    def get_active_point_cloud(self) -> o3d.geometry.PointCloud:
        return self.point_clouds[self.index]

    def get_active_bbox(self) -> o3d.geometry.LineSet:
        return self.bboxes[self.index]

    def toggle_point_cloud(self) -> None:
        self.index = (self.index + 1) % self.n_point_clouds
        print(f"Rendering point cloud {self.sensor_names[self.index]}")

    def update_point_cloud(
        self,
        point_cloud_np: npt.NDArray[np.float64],
        point_cloud_o3d: o3d.geometry.PointCloud,
        velocities: npt.NDArray[np.float64],
        ids: Optional[Set[int]],
        point_to_id: Optional[npt.NDArray[np.int_]],
        point_to_tag: Optional[npt.NDArray[np.int_]],
        snr: Optional[npt.NDArray[np.float64]],
        rcs: Optional[npt.NDArray[np.float64]],
        color_scheme: ColorScheme,
    ) -> bool:
        is_2d = point_cloud_np.shape[1] == 2
        if point_cloud_np.shape[0] > 0:
            if is_2d:
                # Add third row of zeros for plotting
                point_cloud_flipped = np.zeros(
                    (point_cloud_np.shape[0], point_cloud_np.shape[1] + 1)
                )
                point_cloud_flipped[:, :-1] = point_cloud_np.copy()
            else:
                point_cloud_flipped = point_cloud_np.copy()
            point_cloud_flipped[:, 1] *= -1

            point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_flipped)
            point_cloud_o3d.colors = o3d.utility.Vector3dVector(
                self.color_point_cloud(
                    velocities,
                    ids,
                    point_to_id,
                    point_to_tag,
                    snr,
                    rcs,
                    point_cloud_np.shape[0],
                    color_scheme,
                )
            )
            return is_2d
        else:
            point_cloud_o3d.points = o3d.utility.Vector3dVector()
            point_cloud_o3d.colors = o3d.utility.Vector3dVector()
            return is_2d

    def update_bboxes(
        self,
        bbox_vertices: npt.NDArray[np.float64],
        bbox_edges: npt.NDArray[np.float64],
        bbox: o3d.geometry.LineSet,
        is_2d: bool,
    ) -> None:
        if bbox_vertices.shape[0] > 0:
            if is_2d:
                bbox_vertices = bbox_vertices[::2].copy()
                bbox_vertices[:, 2] = 0
            else:
                bbox_vertices = bbox_vertices.copy()

            bbox_vertices[:, 1] *= -1
            if is_2d:
                # Pick 4 rows, skip 8 rows etc etc
                bbox_edges = (
                    bbox_edges.copy()
                    .reshape(-1, 12, bbox_edges.shape[1])[:, :4, :]
                    .reshape(-1, bbox_edges.shape[1])
                )

                # Correct vertix indices after reducing dimension
                bbox_edges = (
                    bbox_edges.reshape(-1, 4, bbox_edges.shape[1])
                    - (np.arange(bbox_edges.shape[0] / 4) * 4)[
                        :, np.newaxis, np.newaxis
                    ]
                ).reshape(-1, bbox_edges.shape[1])
            else:
                bbox_edges = bbox_edges.copy()
            bbox.points = o3d.utility.Vector3dVector(bbox_vertices)
            bbox.lines = o3d.utility.Vector2iVector(bbox_edges)
            bbox.paint_uniform_color(np.full(3, 0))
        else:
            bbox.points = o3d.utility.Vector3dVector()
            bbox.lines = o3d.utility.Vector2iVector()

    def color_point_cloud(
        self,
        velocities: npt.NDArray[np.float64],
        ids: Optional[Set[int]],
        point_to_id: Optional[npt.NDArray[np.int_]],
        point_to_tag: Optional[npt.NDArray[np.int_]],
        snr: Optional[npt.NDArray[np.float64]],
        rcs: Optional[npt.NDArray[np.float64]],
        n_pts: int,
        color_scheme: ColorScheme,
    ) -> npt.NDArray[np.float64]:
        if color_scheme == ColorScheme.VELOCITY and velocities is not None:
            vels_norm = velocities + MAX_VELOCITY
            vels_norm /= MAX_VELOCITY * 2

            colors = np.c_[
                np.interp(vels_norm, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(vels_norm, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(vels_norm, VID_RANGE, VIRIDIS[:, 2]),
            ]

        elif color_scheme == ColorScheme.ID and ids is not None:
            colors = np.full((n_pts, 3), np.full(3, 0.5))

            for id_ in ids:
                if id_ in self.id_color_dict.keys():
                    color = self.id_color_dict[id_]
                else:
                    color = random.choice(VEHICLE_CM)
                    self.id_color_dict[id_] = color
                colors[point_to_id == id_] = color
        elif color_scheme == ColorScheme.TAG and point_to_tag is not None:
            colors = CITYSCAPES_PALETTE_MAP[point_to_tag]
        elif color_scheme == ColorScheme.SNR and snr is not None:
            snr_norm = snr / MAX_SNR

            colors = np.c_[
                np.interp(snr_norm, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(snr_norm, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(snr_norm, VID_RANGE, VIRIDIS[:, 2]),
            ]
        elif color_scheme == ColorScheme.RCS and rcs is not None:
            rcs_norm = rcs / MAX_RCS

            colors = np.c_[
                np.interp(rcs_norm, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(rcs_norm, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(rcs_norm, VID_RANGE, VIRIDIS[:, 2]),
            ]
        else:
            colors = np.full((n_pts, 3), np.full(3, 0.5))

        return colors

    def update(
        self,
        sensors: List[Union[RadarDataHandler, SemanticRaycastHandler, CarlaRadar]],
        color_scheme: ColorScheme,
        bbox_vertices: npt.NDArray[np.float64],
        bbox_edges: npt.NDArray[np.float64],
    ) -> None:
        for i, sensor in enumerate(sensors):
            is_2d = self.update_point_cloud(
                sensor.point_cloud,
                self.point_clouds[i],
                sensor.velocities,
                sensor.ids,
                sensor.point_to_id,
                sensor.point_to_tag,
                sensor.snr,
                sensor.rcs,
                color_scheme,
            )
            self.update_bboxes(bbox_vertices, bbox_edges, self.bboxes[i], is_2d)


class PointCloudPlotter:
    def __init__(
        # self, n_point_clouds: int, set_sim_radar: Callable[[bool], None]
        self,
        sensors: List[Union[RadarDataHandler, SemanticRaycastHandler, CarlaRadar]],
        set_sim_radar: Callable[[bool], None],
    ) -> None:
        self.init_pts = np.array([[0, 100, 0], [0, -100, 0]])
        self.color_scheme = ColorScheme.VELOCITY
        self.show_bboxes = True
        self.sim_radar = True
        self.set_sim_radar = set_sim_radar
        self.point_cloud_holder = PointCloudHolder(sensors, self.init_pts)
        self.quit = False
        self.init_vis()
        self.add_circles()
        self.add_sensor()

    def update(
        self,
        sensors: List[Union[RadarDataHandler, SemanticRaycastHandler, CarlaRadar]],
        bbox_vertices: npt.NDArray[np.float64] = np.array([]).reshape(0, 2),
        bbox_edges: npt.NDArray[np.float64] = np.array([]).reshape(0, 3),
    ) -> None:
        self.point_cloud_holder.update(
            sensors,
            self.color_scheme,
            bbox_vertices,
            bbox_edges,
        )

        self.vis.update_geometry(self.point_cloud_holder.get_active_point_cloud())
        if self.show_bboxes:
            self.vis.update_geometry(self.point_cloud_holder.get_active_bbox())

        self.vis.poll_events()
        self.vis.update_renderer()

    def init_vis(self) -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="Radar", width=960, height=540, left=480, top=270
        )
        self.vis.get_render_option().background_color = [1, 1, 1]
        self.vis.get_render_option().point_size = 2

        self.register_callback("C", self.toggle_color_scheme)
        self.register_callback("T", self.toggle_point_cloud)
        self.register_callback("B", self.toggle_bboxes)
        self.register_callback("S", self.toggle_sim)
        self.register_callback("Q", self.exit_simulator)
        self.register_callback("ESCAPE", self.exit_simulator)

        self.vis.add_geometry(self.point_cloud_holder.get_active_point_cloud())
        self.vis.add_geometry(
            self.point_cloud_holder.get_active_bbox(), reset_bounding_box=False
        )

    def register_callback(
        self, key: str, func: Callable[[o3d.visualization.Visualizer], None]
    ) -> None:
        print(f"Callback {func.__name__} registered to {key}")
        self.vis.register_key_callback(glfw_map.KEY[key], func)

    def exit_simulator(self, vis: o3d.visualization.Visualizer) -> None:
        self.quit = True

    def toggle_color_scheme(self, vis: o3d.visualization.Visualizer) -> None:
        color_scheme = ColorScheme((self.color_scheme.value + 1) % len(ColorScheme))
        print(
            f"Changing color scheme from {self.color_scheme.name} to {color_scheme.name}"
        )
        self.color_scheme = color_scheme

    def toggle_point_cloud(self, vis: o3d.visualization.Visualizer) -> None:
        vis.remove_geometry(
            self.point_cloud_holder.get_active_point_cloud(), reset_bounding_box=False
        )
        if self.show_bboxes:
            vis.remove_geometry(
                self.point_cloud_holder.get_active_bbox(), reset_bounding_box=False
            )
        self.point_cloud_holder.toggle_point_cloud()
        vis.add_geometry(
            self.point_cloud_holder.get_active_point_cloud(),
            reset_bounding_box=False,
        )
        if self.show_bboxes:
            vis.add_geometry(
                self.point_cloud_holder.get_active_bbox(), reset_bounding_box=False
            )

    def toggle_bboxes(self, vis: o3d.visualization.Visualizer) -> None:
        self.show_bboxes = not self.show_bboxes
        if self.show_bboxes:
            vis.add_geometry(
                self.point_cloud_holder.get_active_bbox(), reset_bounding_box=False
            )
        else:
            vis.remove_geometry(
                self.point_cloud_holder.get_active_bbox(), reset_bounding_box=False
            )
        print(f"Render bounding boxes = {self.show_bboxes}")

    def toggle_sim(self, vis: o3d.visualization.Visualizer) -> None:
        self.sim_radar = not self.sim_radar
        self.set_sim_radar(self.sim_radar)
        print(f"Radar simulation = {self.sim_radar}")

    def add_circles(self) -> None:
        n_circles = 10
        outer_r = 100
        r_per_circle = outer_r / n_circles
        circles = []
        colors = [np.array([0.95, 0.95, 0.95]), np.array([1, 1, 1])]
        for i in range(n_circles):
            circle = o3d.geometry.TriangleMesh.create_cylinder(
                radius=outer_r - i * r_per_circle,
                height=0.001 * (i + 1),
                resolution=20,
                split=1,
                create_uv_map=False,
            )
            circle.paint_uniform_color(colors[i % 2])
            circle.translate(np.array([0, 0, -0.1]))
            circles.append(circle)
            self.vis.add_geometry(circle, reset_bounding_box=False)

    def add_sensor(self) -> None:
        sensor_render = o3d.geometry.TriangleMesh.create_box()
        self.vis.add_geometry(sensor_render, reset_bounding_box=False)

    def close(self) -> None:
        self.vis.close()


class CameraVisualizer:
    def __init__(self, width: int, height: int) -> None:
        self.image = o3d.geometry.Image()
        self.working = False
        self.width = width
        self.height = height
        self.init_vis()

    def init_vis(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Camera",
            width=self.width,
            height=self.height,
        )

    def to_rgb_array(self, image: carla.Image) -> npt.NDArray[np.uint8]:
        """Convert a CARLA raw image to a RGB numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return np.ascontiguousarray(array)

    def callback(self, carla_image: carla.Image) -> None:
        # Originally, self.vis was updated here in the callback as well, which worked fine on Ubuntu.
        # Those parts had to be moved to update() and called a bit later, as it caused crazy issues on Windows.
        # Best guess: open3d-stuff must be updated from the main thread on Windows but not on Linux.
        self.working = True
        image_np = self.to_rgb_array(carla_image)
        self.image = o3d.geometry.Image(image_np)
        time.sleep(1)
        self.working = False

    def update(self) -> None:
        self.vis.clear_geometries()
        self.vis.add_geometry(self.image)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        self.vis.close()
