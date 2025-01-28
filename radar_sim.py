# Mainfil som kör allt
import os
import signal
import argparse
from pathlib import Path
import time
from datetime import datetime, timedelta
import random
from typing import List, Union
import numpy as np
import carla
from gt_logger import GtLogger
from gt_logger_radar import GtLoggerRadar
from radar import RadarDataHandler, SemanticRaycastHandler
from carla_utils import force_green_lights, get_yaw_rate, get_velocity, vehicle_at_junction, draw_fov_lines, junction_jam
from bboxes import BboxHandler, get_actor_bbox_corners
import visualization
from can_logger import CanLogger
from carla_radar import CarlaRadar
from camera import CameraHandler
import subprocess

def start_carla():
    # Path to your CARLA executable, e.g., CarlaUE4.sh or CarlaUE4.exe
    return subprocess.Popen(["C:/Users/David/GIT/CARLA15/CarlaUE4.exe"], stdout=subprocess.PIPE)

def stop_carla(carla_process):
    try:
        carla_process.terminate()
        carla_process.wait(timeout=5)  # Wait for 5 seconds
    except subprocess.TimeoutExpired:
        print("Process did not terminate, forcing it to stop...")
        subprocess.run(["taskkill", "/F", "/carla_process.pid", str(carla_process.pid)])

def main(args: argparse.Namespace, rsp) -> None:
    vehicle_ids = []
    sensor_ids = []

    # Sensors that should be plotted by Visualizer
    plottable_sensors: List[
        Union[RadarDataHandler, SemanticRaycastHandler, CarlaRadar]
    ] = []
    t = datetime.now()
    t0 = time.time()
    t_delta = timedelta(milliseconds=50)  # Time between simulation steps

    # Carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world: carla.World = client.load_world(args.map)
    world_map = world.get_map()

    # Settings
    settings = world.get_settings()
    settings.fixed_delta_seconds = t_delta.total_seconds()
    settings.synchronous_mode = True
    settings.no_rendering_mode = args.no_rendering
    world.apply_settings(settings)

    force_green_lights(world)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    # Vehicles
    spawn_points = world.get_map().get_spawn_points()
    #spawn_points.pop(23) # Invisible collison at SP23 on Town02 with CyberTruck
    if args.spawn_shuffle:
        random.shuffle(spawn_points)

    if args.vehicles > len(spawn_points):
        raise Exception("Too many vehicles")

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.audi.a2")

    #all_vehicle_blueprints = blueprint_library.filter("vehicle.*")

    # List of vehicle blueprints to exclude
    exclude_list = ["vehicle.tesla.cybertruck", "vehicle.bh.crossbike", "vehicle.gazelle.omafiets",
                    "vehicle.diamondback.century", "vehicle.mitsubishi.fusorosa", "vehicle.carlamotors.firetruck", 
                    "vehicle.carlamotors.european_hgv", "vehicle.yamaha.yzf", "vehicle.harley-davidson.low_rider", 
                    "vehicle.vespa.zx125", "vehicle.kawasaki.ninja", "vehicle.carlamotors.carlacola", 
                    "vehicle.carlamotors.firetruck", "vehicle.carlamotors.european_hgv", "vehicle.mitsubishi.fusorosa", 
                    "vehicle.mercedes.sprinter", "vehicle.mercedes.sprinter", "vehicle.ford.ambulance",
                    "vehicle.volkswagen.t2", "vehicle.volkswagen.t2_2021"]
    mc_group = ['low_rider', 'ninja', 'yzf']

    # Filter out the vehicles you don't want
    #vehicle_blueprints = [bp for bp in all_vehicle_blueprints if bp.id not in exclude_list]
    
    vehicle_blueprints = blueprint_library.filter("vehicle.audi.a2")

    blueprints = random.choices(vehicle_blueprints, k=args.vehicles)
    for blueprint in blueprints:
        transform = spawn_points.pop(0)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(
                blueprint.get_attribute("driver_id").recommended_values
            )
            blueprint.set_attribute("driver_id", driver_id)
        else:
            blueprint.set_attribute("role_name", "autopilot")

        actor: carla.Vehicle = world.spawn_actor(blueprint, transform)
        actor.set_autopilot(True)
        #vehicle.set_autopilot(True, behavior_type=carla.VehicleAutopilotBehavior.Aggressive) #incorporate agressive behaviour
        vehicle_ids.append(actor.id)

        if blueprint.tags[2] in mc_group: 
            traffic_manager.distance_to_leading_vehicle(actor,np.random.randint(2))
            traffic_manager.vehicle_percentage_speed_difference(actor, -20)
        else:
            traffic_manager.distance_to_leading_vehicle(actor,np.random.randint(4,8))
            traffic_manager.vehicle_percentage_speed_difference(actor, np.random.randint(-30,10))
    
    # radar spawn points
    wsp01 = [{'x': 0, 'y': 0, "z": 1, "yaw": 0}]
             
    wsp02 = [{'x': -10, 'y': 108, "z": 1, "yaw": 0},    # 0
            {'x': -5, 'y': 103, "z": 1, "yaw": 90},   
           {'x': -11, 'y': 189.5, "z": 1, "yaw": 0},
           {'x': -8.1, 'y': 305, "z": 1, "yaw": 0},
           {'x': -5, 'y': 308, "z": 1, "yaw": -90},
           {'x': 44, 'y': 310, "z": 1, "yaw": -90},     # 5
           {'x': 37, 'y': 238.5, "z": 1, "yaw": 0},     
           {'x': 37, 'y': 265, "z": 1, "yaw": -55},
           {'x': 37, 'y': 210, "z": 1, "yaw": 55},
           {'x': 37, 'y': 220, "z": 1, "yaw": -55},     
           {'x': 70, 'y': 183, "z": 1, "yaw": 145},     # 10
           {'x': 110, 'y': 245, "z": 1, "yaw": -20},     
           {'x': 110, 'y': 245, "z": 1, "yaw": -90}
           ]
    
    rsp = 12 # radar spawn point
    radar_transform = carla.Transform(carla.Location(x = wsp02[rsp]['x'], y = wsp02[rsp]['y'], z = wsp02[rsp]['z']),
                                       carla.Rotation(yaw = wsp02[rsp]['yaw'], pitch = 0))

    radar_blueprints = blueprint_library.filter("vehicle.gazelle.omafiets")
    ego_radar_bp = radar_blueprints[0]
    ego_radar: carla.Vehicle = world.spawn_actor(ego_radar_bp, radar_transform)
    ego_id = ego_radar.id

    # Logging
    if args.log:
        log_path = Path("RSP" + str(rsp) + "_" + t.strftime("%Y-%m-%d_%H%M%S"))
        log_path.mkdir()
        """can_logger = CanLogger(
            dbc_path=Path("dbc/carla/combined.dbc"),
            log_path=log_path / log_path.with_suffix(".blf"),
            time_between_msgs=timedelta(microseconds=10),
        )"""
        gt_logger = GtLogger(
            log_path / log_path.with_suffix(".jsonl"), ego_id, vehicle_ids, world
        )
        gt_logger_radar = GtLoggerRadar(
            log_path / log_path.with_suffix(".jsonl"),
            ego_id,
            vehicle_ids,
            world
        )
    else:
        log_path = None 

    # Sensors---
    sensor_transform = carla.Transform(carla.Location(args.x, args.y, args.z))
    radar_data_handler = RadarDataHandler(
        world,
        #ego_vehicle,
        ego_radar,
        sensor_transform,
        args.upper_fov,
        args.lower_fov,
        args.horizontal_fov,
        args.range,
    )
    plottable_sensors.append(radar_data_handler)
    vertical_fov = abs(args.upper_fov) + abs(args.lower_fov)
    carla_radar = CarlaRadar(
        world,
        #ego_vehicle,
        ego_radar,
        sensor_transform,
        vertical_fov,
        args.horizontal_fov,
        args.range,
    )
    plottable_sensors.append(carla_radar)

    if args.camera:
        vis_cam = visualization.CameraVisualizer(args.cam_width, args.cam_height)
        camera = CameraHandler(
            args.cam_width,
            args.cam_height,
            args.cam_fov,
            sensor_transform,
            #ego_vehicle,
            ego_radar,
            world,
            vis_cam.callback,
            log_path,
            args.record
        )
        sensor_ids.append(camera.sensor.id)

    for sensor in plottable_sensors:
        sensor_ids.append(sensor.actor_id)

    # Bounding boxes
    bbox_handler = BboxHandler(world)
    radar_data_handler.set_vehicle_bbox_corners(
        get_actor_bbox_corners(vehicle_ids, world)
    )

    if args.vis_pc:
        vis_pc = visualization.PointCloudPlotter(plottable_sensors, radar_data_handler.set_sim_radar)
    junction_list = {}
    
    #Vizualise FOV 
    #radar_location = carla.Location(x=wsp02[rsp]['x'], y = wsp02[rsp]['y'], z = wsp02[rsp]['z'])  
    #radar_rotation = carla.Rotation(pitch=0, yaw=wsp02[rsp]['yaw'])
    #draw_fov_lines(radar_location, radar_rotation, distance=args.range, fov_angle=args.horizontal_fov/2 , world=world)
    
    try:
        while True:
            ego_radar.set_target_velocity(carla.Vector3D(0, 0, 0))
            ego_radar.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            # Tick and increment time stamp
            world.tick()
            
            # Wait until callbacks are done
            while radar_data_handler.working or (args.camera and vis_cam.working):
                time.sleep(0.001)

            # Update visualizers
            if args.camera:
                vis_cam.update()
            
            if args.vis_pc:
                if vis_pc.show_bboxes:
                    vertices, edges = bbox_handler.get_transformed_vertices_edges(
                        radar_data_handler.get_visible_actor_ids(),
                        radar_data_handler.sensor_pos,
                        radar_data_handler.sensor_rot,
                    )
                    vis_pc.update(plottable_sensors, vertices, edges)
                else:
                    vis_pc.update(plottable_sensors)
            
            #Get bboxes from the vehicles
            EOT=bbox_handler.get_connected_dynamic_vertices(vehicle_ids)
            ground_ids=[]
            unique_ground_ids=[]

            #Find visible bboxes
            ego_radar_transform_matrix = np.array(ego_radar.get_transform().get_inverse_matrix())[0:3, 0:3]
            activebboxes={}
            radar_visible_ids = np.unique(radar_data_handler.point_to_id[radar_data_handler.point_to_id > -1])
            for id in radar_visible_ids:
                if world.get_actor(int(id)).attributes != {}: # ignore poles
                    ground_ids.append(int(id),)
                    #pos = np.array([EOT.get(id)[0][:,0] - ego_radar.get_location().x,
                                    #EOT.get(id)[0][:,1] - ego_radar.get_location().y,
                                    #EOT.get(id)[0][:,2] - ego_radar.get_location().z])
                    #pos_trans = ego_radar_transform_matrix.dot(pos)
                    #[EOT.get(id)[0][:,0], EOT.get(id)[0][:,1], EOT.get(id)[0][:,2]] = [pos_trans[0]- 3, pos_trans[1], pos_trans[2]]
                    # Vizualise cars in FOV
                    #print(world.get_actor(int(id)).type_id)
                    #print("########################")
            
            unique_ground_ids=list(set(ground_ids))
            for id in unique_ground_ids:
                vehicle = world.get_actor(id)

                vehicle_gt = []
                pos_vehicle = [vehicle.get_location().x - ego_radar.get_location().x, 
                               vehicle.get_location().y - ego_radar.get_location().y, 
                               vehicle.get_location().z - ego_radar.get_location().z]
                pos_vehicle = ego_radar_transform_matrix.dot(pos_vehicle)[:-1]
                vehicle_gt.append(pos_vehicle)
                id_velocity = [vehicle.get_velocity().x, vehicle.get_velocity().y, vehicle.get_velocity().z]
                id_velocity = ego_radar_transform_matrix.dot(id_velocity)[:-1]
                vehicle_gt.append(id_velocity)
                vehicle_gt.append([2*vehicle.bounding_box.extent.x, 2*vehicle.bounding_box.extent.y])
                vehicle_gt = np.array(vehicle_gt)

                activebboxes[id] = vehicle_gt.tolist()

                #vehicle_gt = np.array([])
                #vehicle_gt = np.append(vehicle_gt, [vehicle.get_location().x - ego_radar.get_location().x, vehicle.get_location().y - ego_radar.get_location().y])
                #vehicle_gt = np.append(vehicle_gt, [vehicle.get_velocity().x, vehicle.get_velocity().y])
                #vehicle_gt = np.append(vehicle_gt, [2*vehicle.boundng_box.extent.x, 2*vehicle.bounding_box.extent.y])
                
                #id_velocity = [vehicle.get_velocity().x, vehicle.get_velocity().y, vehicle.get_velocity().z]
                #id_velocity = ego_radar_transform_matrix.dot(id_velocity)[:-1]

                #a = np.append(EOT.get(id)[0][::2,:-1], id_velocity).reshape(5,2)
                #activebboxes[id] = a.tolist()
            
    
            for vehicle in world.get_actors(vehicle_ids):
                vehicle_velocity = np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2 + vehicle.get_velocity().z**2)
                if vehicle_at_junction(world_map, vehicle) and vehicle_velocity < 0.05:
                    if vehicle.id not in junction_list.keys(): # Adds new car to junctionlist
                        junction_list[vehicle.id] = {'times': [time.time(), 0], 'location': [vehicle.get_location().x, vehicle.get_location().y]}
                    else: #Adds to timestamp to existing car at junction 
                        cum_stand_time = time.time() - junction_list[vehicle.id]['times'][0]
                        junction_list[vehicle.id]['times'][1] = cum_stand_time  # Append time car been standing at junc
                        junction_list[vehicle.id]['location'] = [vehicle.get_location().x, vehicle.get_location().y]  # Append time car been standing at junc
                        
                else:
                    junction_list.pop(vehicle.id, None)
                
            if len(junction_list.keys()) > 2:
                junction_vehicle_ids = [int(key) for key in junction_list.keys()]
                junction_vehicles = world.get_actors([int(key) for key in junction_list.keys()])
                if junction_jam(world, junction_list):
                    raise KeyboardInterrupt

            # Log
            if args.log:
                gt_logger.log(t)
                vx=[]
                vy=[]
                
                
                for id in radar_data_handler.point_to_id:
                    try:
                        vels = [world.get_actor(int(id)).get_velocity().x, 
                                world.get_actor(int(id)).get_velocity().y, 
                                world.get_actor(int(id)).get_velocity().z]
                        vels = ego_radar_transform_matrix.dot(np.array(vels))
                        vx.append(vels[0])
                        vy.append(vels[1])
                        #vz.append(vels[2])
                        #vx.append(world.get_actor(int(id)).get_velocity().x)
                        #vy.append(world.get_actor(int(id)).get_velocity().y)                        
                    except:
                        vx.append(0.0)
                        vy.append(0.0)

                gt_logger_radar.log(
                    t,
                    radar_data_handler.point_cloud,
                    radar_data_handler.point_to_tag,
                    radar_data_handler.point_to_id,
                    #get_velocity(ego_vehicle),
                    #get_yaw_rate(ego_vehicle),
                    get_velocity(ego_radar),
                    get_yaw_rate(ego_radar),
                    radar_data_handler.r,
                    vx,
                    vy,
                    radar_data_handler.velocities,
                    radar_data_handler.phi,
                    radar_data_handler.snr,
                    radar_data_handler.rcs,
                    unique_ground_ids,
                    activebboxes
                    )

            if args.vis_pc:
                if vis_pc.quit:
                    raise KeyboardInterrupt
            
            if time.time()-t0 >500:
                if args.log:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        # Clean up and return
        if args.log:
            #can_logger.close()
            gt_logger.close()
            gt_logger_radar.close()
        if args.camera:
            vis_cam.close()
        if args.vis_pc:
            vis_pc.close()
        [actor.destroy() for actor in world.get_actors(vehicle_ids)]
        [actor.destroy() for actor in world.get_actors(sensor_ids)]
        world.get_actor(ego_id).destroy()
        print(f"Time Elapsed: {(time.time()-t0):.2f}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="localhost",
        help="IP of the host CARLA Simulator",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port of CARLA Simulator",
    )
    argparser.add_argument(
        "--no-rendering",
        action="store_true",
        help="use the no-rendering mode which will provide some extra"
        " performance but you will lose the articulated objects in the"
        " lidar, such as pedestrians",
    )
    argparser.add_argument(
        "--autopilot",
        type=bool,
        default=True,
        help="Use the autopilot, set for False to make the ego vehicle stand still",
    )
    argparser.add_argument(
        "--upper-fov",
        metavar="FOV",
        default=10.0,
        type=float,
        help="Upper field of view [deg]",
    )
    argparser.add_argument(
        "--lower-fov",
        metavar="FOV",
        default=0,
        type=float,
        help="Lower field of view [deg]",
    )
    argparser.add_argument(
        "--horizontal-fov",
        metavar="FOV",
        default=70,
        type=float,
        help="Horizontal field of view [deg]",
    )
    argparser.add_argument(
        "--range", default=100.0, type=float, help="Sensor range [m]"
    )
    argparser.add_argument("-x", default=3, type=float, help="Sensor position X [m]")
    argparser.add_argument("-y", default=0.0, type=float, help="Sensor position Y [m]")
    argparser.add_argument("-z", default=0.5, type=float, help="Sensor position Z [m]")
    argparser.add_argument("--map", default="Town02", help="Map name")
    argparser.add_argument(
        "--camera",
        action="store_true",
        help="add a camera at the same position as the radar and render the output",
    )
    argparser.add_argument(
        "--vis_pc",
        default = True,
        help = "visualize bboxes on 2D plot"
    )
    argparser.add_argument(
        "--log",
        action="store_true",
        help="Log to file",
    )
    argparser.add_argument(
        "--record",
        metavar="REC",
        default=False,
        type=bool,
        help="Records and saves image of camera POV every timestep",
    )
    argparser.add_argument(
        "--cam-width",
        metavar="W",
        default=1280,
        type=int,
        help="Number of pixels horizontally generated by the camera",
    )
    argparser.add_argument(
        "--cam-height",
        metavar="H",
        default=720,
        type=int,
        help="Number of pixels vertically generated by the camera",
    )
    argparser.add_argument(
        "--cam-fov",
        metavar="FOV",
        default=105,
        type=int,
        help="Field of view captured by the camera",
    )
    argparser.add_argument(
        "--vehicles",
        metavar="N",
        default=10,
        type=int,
        help="Number of vehicles to spawn",
    )
    argparser.add_argument(
        "--spawn-shuffle", action="store_true", help="Randomize vehicle spawn points"
    )
    args = argparser.parse_args()
    main(args, 12)
    for _ in range(40):
            radar_spawn_point = random.randint(0,11)
            for radar_spawn_point in range(0,11):
                main(args, radar_spawn_point)
            time.sleep(10)
    
    """ 
    carla_process = start_carla()
    time.sleep(20)
    stop_carla(carla_process)
    main(args)
       try:
        for _ in range(20):
            main(args)
            time.sleep(10)
            stop_carla(carla_process)
            time.sleep(5)  # Give it a moment before restarting
            carla_process = start_carla()
    finally:
        stop_carla(carla_process) """


