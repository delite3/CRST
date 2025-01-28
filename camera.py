from datetime import datetime
from typing import Callable, Optional
import carla
from pathlib import Path


class CameraHandler:
    def __init__(
        self,
        width: int,
        height: int,
        fov: int,
        transform: carla.Transform,
        vehicle: carla.Vehicle,
        world: carla.World,
        visualizer_callback: Callable[[carla.Image], None],
        log_path: Optional[Path] = None,
        record = bool,
    ) -> None:
        # Handles rendering and saving image data from the Carla camera sensor
        self.log_path = log_path
        self.visualizer_callback = visualizer_callback

        blueprint = self.generate_bp(
            world.get_blueprint_library(),
            width,
            height,
            fov,
        )
        self.sensor: carla.Sensor = world.spawn_actor(
            blueprint, transform, attach_to=vehicle
        )
        self.sensor.listen(lambda image: self.callback(image, record))

    def generate_bp(
        self,
        blueprint_library: carla.BlueprintLibrary,
        width: int,
        height: int,
        fov: int,
    ) -> carla.ActorBlueprint:
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        cam_bp.set_attribute("fov", str(fov))
        return cam_bp

    def callback(self, img: carla.Image, record) -> None:
        # Show the image captured by the camera, and save it if a log path was provided
        if self.log_path and record == True:
            img_path = self.log_path / Path(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            ).with_suffix(".jpg")
            img.save_to_disk(str(img_path))
        self.visualizer_callback(img)
