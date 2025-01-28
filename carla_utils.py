from typing import Dict, List
import carla
import carla.libcarla
import numpy as np
import numpy.typing as npt
import math

# To through all the available Carla labels, and create dicts to simplify
# tag handling throughout the code
tag_to_string: Dict[int, str] = {
    k: {v: k for k, v in carla.CityObjectLabel.names.items()}[v]
    for k, v in carla.CityObjectLabel.values.items()
}
string_to_tag: Dict[str, int] = {v: k for k, v in tag_to_string.items()}

# First in list = highest priority
# Not in list = don't care
priority = ["Car", "Motorcycle", "Truck", "Bus"]
# Map tag to priority. Higher value = higher priority
tag_to_priority = {
    string_to_tag[st]: len(priority) - i for i, st in enumerate(priority)
}


def get_prioritized(tags: List[int]) -> int:
    # Returns the tag with the highest priority from a list of tags
    # Useful for actors that contain multiple different tags
    # If no tags from the input list are in the priority list, the first
    # tag is returned.
    prio_tag = tags[0]
    prio = -1
    for tag in tags:
        if tag in tag_to_priority and tag_to_priority[tag] > prio:
            prio = tag_to_priority[tag]
            prio_tag = tag
    return prio_tag


def location_to_array(location: carla.Location) -> npt.NDArray[np.float64]:
    return np.array([location.x, location.y, location.z])


def array_to_location(array: npt.NDArray) -> carla.Location:
    return carla.Location(array[0], array[1], array[2])


def force_green_lights(world: carla.World) -> None:
    # Force all traffic lights in the map to stay green most of the time
    for actor in world.get_actors():
        if isinstance(actor, carla.TrafficLight):
            actor.set_state(carla.TrafficLightState.Green)
            actor.set_green_time(1000.0)
            actor.set_red_time(1.0)
            actor.set_yellow_time(1.0)


def get_yaw_rate(actor: carla.Actor) -> float:
    return actor.get_angular_velocity().z


def get_velocity(actor: carla.Actor) -> float:
    return actor.get_velocity().length()


def vehicle_at_junction(world_map, vehicle):
    """
    Check if the given vehicle is at a junction.

    Args:
        vehicle: carla.Vehicle object.

    Returns:
        bool: True if the vehicle is at a junction, False otherwise.
    """
    
    # Get the current location of the vehicle
    vehicle_location = vehicle.get_location()
    
    # Get the waypoint corresponding to the current location
    waypoint = world_map.get_waypoint(vehicle_location)
    
    # Check if the waypoint is within a junction
    return waypoint.is_junction

def actor_near_location(actor, location):

    actor_location = actor.get_location()
    if ((actor_location.x - location.x) < actor.bounding_box.extent.x) and ((actor_location.y - location.y) < actor.bounding_box.extent.y):
        return True
    else:
        return False
    

def draw_fov_lines(radar_location, radar_rotation, distance, fov_angle, world):
        
    #radar_location = carla.Location(x=wsp02[rsp]['x'], y = wsp02[rsp]['y'], z = wsp02[rsp]['z'])  # Example location, set according to your setup
    #radar_rotation = carla.Rotation(pitch=0, yaw=wsp02[rsp]['yaw'], roll=0)

    end_point_left = carla.Location(
        x=radar_location.x + distance * math.cos(math.radians(radar_rotation.yaw - fov_angle)),
        y=radar_location.y + distance * math.sin(math.radians(radar_rotation.yaw - fov_angle)),
        z=radar_location.z
    )
    end_point_right = carla.Location(
        x=radar_location.x + distance * math.cos(math.radians(radar_rotation.yaw + fov_angle)),
        y=radar_location.y + distance * math.sin(math.radians(radar_rotation.yaw + fov_angle)),
        z=radar_location.z
    )

    # Draw the lines
    world.debug.draw_line(radar_location, end_point_left, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.0, persistent_lines=True)
    world.debug.draw_line(radar_location, end_point_right, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.0, persistent_lines=True)

import carla
import math

def calculate_distance(location1, location2):
    """
    Calculate Euclidean distance.
    """
    return math.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)

def junction_jam(world, junction_vehicles, distance_threshold=20.0):
    """
    Determine if three or more stopped vehicles are within close proximity of each other.

    Args:
    vehicles: list of carla.Vehicle objects.
    distance_threshold: float, the maximum distance to consider vehicles at the same junction.

    Returns:
    list of tuples, each tuple contains vehicles that are close together.
    """
    traffic_jam = False
    close_vehicles = []
    vehicle_ids = list(junction_vehicles.keys())
    for i in range(len(vehicle_ids)):
        for j in range(i + 1, len(vehicle_ids)):
            if calculate_distance(junction_vehicles[vehicle_ids[i]]['location'], junction_vehicles[vehicle_ids[j]]['location']) < distance_threshold:
                close_vehicles.append(vehicle_ids[j])
        if len(close_vehicles) > 1:
            close_vehicles.append(vehicle_ids[i])
            break
    counter = 0
    if len(close_vehicles) > 2:
        for id in close_vehicles:
            stand_still_time = junction_vehicles[id]['times'][1]
            if stand_still_time> 5:
                counter += 1
    if counter == 3:
        traffic_jam = True
    
    return traffic_jam
