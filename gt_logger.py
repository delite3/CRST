from datetime import datetime
from pathlib import Path
from typing import List
import carla
from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class State:
    x: float
    y: float
    vx: float
    vy: float
    yaw: float
    t: float


@dataclass(frozen=True)
class Measurement:
    state: State
    id: int
    ego: bool
    t: float


class GtLogger:
    def __init__(
        self,
        log_path: Path,
        ego_vehicle_id: int,
        vehicle_ids: List[int],
        world: carla.World,
    ) -> None:
        # Logs ground truth data on all vehicles given by vehicle_ids to a jsonl-file

        self.log_file = log_path.open(mode="w")
        self.ego_vehicle_id = ego_vehicle_id
        self.vehicle_ids = vehicle_ids
        self.world = world

    def state_from_actor(self, vehicle: carla.Vehicle, t: datetime) -> State:
        # Fill a State with the required data from the actor
        velocity = vehicle.get_velocity()
        transform = vehicle.get_transform()

        return State(
            transform.location.x,
            transform.location.y,
            velocity.x,
            velocity.y,
            transform.rotation.yaw,
            t.timestamp(),
        )

    def log(self, t: datetime) -> None:
        # Log data to the file for time t

        # Add the ego vehicle
        ego_measurement = Measurement(
            state=self.state_from_actor(self.world.get_actor(self.ego_vehicle_id), t),
            id=self.ego_vehicle_id,
            ego=True,
            t=t.timestamp(),
        )
        measurements = [ego_measurement]

        # Add the rest of the vehicles
        measurements += [
            Measurement(
                state=self.state_from_actor(self.world.get_actor(vehicle_id), t),
                id=vehicle_id,
                ego=False,
                t=t.timestamp(),
            )
            for vehicle_id in self.vehicle_ids
        ]

        # Dump the measurements as a json-string
        measurements_json = json.dumps(
            [asdict(measurement) for measurement in measurements]
        )

        # Write the string on a new line in the jsonl-file
        self.log_file.write(measurements_json + "\n")

    def close(self) -> None:
        self.log_file.close()
