"""
Plot stuff to help tune in radar simulation params
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time
import carla
from carla_utils import location_to_array, array_to_location
from bboxes import BBOX_EDGE_PAIRS
from config import SCATTERERS

from radar_model import RadarModel


def plot_n_pts() -> None:
    ##### n_pts in sim_radar_pts ######

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Plot n pts vs angle and range

    max_pts = 4
    n_samples = 1000
    range_ = 100
    fov = np.deg2rad(90)
    model = RadarModel()

    x = np.linspace(-fov / 2, fov / 2, n_samples)
    y = np.linspace(0, range_, n_samples)
    x, y = np.meshgrid(x, y)
    z = (max_pts * ((range_ - y) / range_) ** 2 * model.gaussian_gain(x)).astype(
        int
    ) + 1

    surf = ax.plot_surface(
        x, y, z, cmap=cm.coolwarm, linewidth=0.1, edgecolors="k", antialiased=True
    )
    ax.set_zlabel("n_pts")
    ax.set_xlabel("phi")
    ax.set_ylabel("r")

    # Plot max pts
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    n_pts = 5000
    x = np.linspace(0, n_pts, n_pts)
    y = (x ** (1 / 4)).astype(int)

    ax2.plot(x, y)
    ax2.set_xlabel("n pts in")
    ax2.set_ylabel("max pts")

    plt.show()


###################################


def plot_bosch_vs_model() -> None:
    ##### Plot bosch Pd vs radar model ######

    fig, axes = plt.subplots(1, 3, figsize=(10, 10), subplot_kw=dict(projection="3d"))

    # Plot n pts vs angle and range

    model = RadarModel()
    rcs_table = [10.0, 1.0, 0.1]
    deg_bosch_table = np.array([10, 21, 25, 35, 42.6])
    typ_table_bosch = {
        10.0: np.array([95, 78, 73, 46, 29]),
        1.0: np.array([52, 42, 40, 25, 15]),
        0.1: np.array([28, 23, 22, 13, 8]),
    }
    min_table_bosch = {
        10.0: np.array([50, 39, 36, 16, 12]),
        1.0: np.array([27, 21, 19, 8, 6]),
        0.1: np.array([15, 11, 10, 4, 3]),
    }
    for i, rcs in enumerate(rcs_table):
        ax = axes[i]
        print(rcs, i)
        deg_bosch = np.hstack([-deg_bosch_table[::-1], deg_bosch_table])
        typ_bosch = np.hstack([typ_table_bosch[rcs][::-1], typ_table_bosch[rcs]])
        min_bosch = np.hstack([min_table_bosch[rcs][::-1], min_table_bosch[rcs]])
        pd_typ_bosch = model.compute_pd(
            model.compute_snr(typ_bosch, np.deg2rad(deg_bosch), rcs)
        )
        pd_min_bosch = model.compute_pd(
            model.compute_snr(min_bosch, np.deg2rad(deg_bosch), rcs)
        )
        n_samples = 1000
        range_ = 100
        fov = np.deg2rad(90)

        phi = np.linspace(-fov / 2, fov / 2, n_samples)
        r = np.linspace(0.1, range_, n_samples)
        phi, r = np.meshgrid(phi, r)
        z = model.compute_pd(model.compute_snr(r, phi, rcs))
        limit_typ = 0.5
        limit_min = 0.01
        # limit = model.pf
        colors = np.full(phi.shape, "r")
        colors[z > limit_min] = "y"
        colors[z > limit_typ] = "g"
        # exit()
        surf = ax.plot_surface(
            np.rad2deg(phi),
            r,
            z,
            cmap=cm.get_cmap("jet"),
            linewidth=0,
            rcount=50,
            ccount=50,
            antialiased=True,
            alpha=0.8,
            zorder=1,
        )
        (typ_plt,) = ax.plot(
            deg_bosch,
            typ_bosch,
            pd_typ_bosch,
            "w.",
            markersize=10,
            markeredgewidth=0.5,
            markeredgecolor="k",
            zorder=4,
            label="typ",
        )
        (min_plt,) = ax.plot(
            deg_bosch,
            min_bosch,
            pd_min_bosch,
            "k.",
            markersize=10,
            markeredgewidth=0.5,
            markeredgecolor="w",
            zorder=4,
            label="min",
        )
        ax.set_zlabel("Pd")
        ax.set_xlabel("phi")
        ax.set_ylabel("range")
        ax.set_zlim(0, 1)
        ax.view_init(elev=90, azim=-90, roll=0)
        ax.set_title(f"RCS {rcs}")

    fig.legend(handles=[typ_plt, min_plt])
    cbar = fig.colorbar(surf, ax=axes.ravel().tolist(), orientation="horizontal")
    cbar.set_label("Pd")

    plt.show()


def plot_rcs_sim() -> None:
    model = RadarModel()
    r = np.linspace(0, 100, 1000)
    rcs_sim = model.sim_rcs(r, 10)
    plt.gca().invert_xaxis()
    plt.plot(r, rcs_sim, ".")
    plt.show()


def plot_compare_bbox_methods() -> None:
    bbox_clr = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:grey",
    ]

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world: carla.World = client.get_world()
    settings = world.get_settings()
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.filter("vehicle.tesla.cybertruck")[0]
    transform = world.get_map().get_spawn_points()[0]
    transform = carla.Transform(transform.location, carla.Rotation(0, 25, 0))
    act: carla.Vehicle = world.spawn_actor(blueprint, transform)
    world.wait_for_tick()

    act_transform = act.get_transform()
    bbox_vehicle = act.bounding_box
    act.destroy()

    n_pts = 10
    n_actors = 20
    world_to_ego = np.array(act_transform.get_matrix())
    world_to_ego_rot = world_to_ego[0:3, 0:3]
    ego_to_world_rot = world_to_ego_rot.T
    ego_pos = world_to_ego[0:3, 3]

    bboxes = []
    for scatterer in SCATTERERS:
        extent = carla.Vector3D(
            scatterer.extent_ratio.x * bbox_vehicle.extent.x,
            scatterer.extent_ratio.y * bbox_vehicle.extent.y,
            scatterer.extent_ratio.z * bbox_vehicle.extent.z,
        )
        center = carla.Location(
            bbox_vehicle.location.x + extent.x * scatterer.center_ratio.x,
            bbox_vehicle.location.y + extent.y * scatterer.center_ratio.y,
            bbox_vehicle.location.z + extent.z * scatterer.center_ratio.z,
        )
        bboxes.append(carla.BoundingBox(center, extent))

    pts = (np.random.rand(n_pts, 3) - np.array([0.5, 0.5, 0])) * np.array([6, 2, 2])
    pts = pts @ ego_to_world_rot
    pts += ego_pos

    t0 = time.time()
    for _ in range(n_actors):
        in_box = np.full(n_pts, False, dtype=np.bool_)
        for bbox in bboxes:
            for i, pt in enumerate(pts):
                loc = array_to_location(pt)
                in_box[i] |= bbox.contains(loc, act_transform)
    t1 = time.time()
    print("carla naive: ", t1 - t0)

    t0 = time.time()
    for _ in range(n_actors):
        pts_to_check = np.full(n_pts, fill_value=True, dtype=np.bool_)
        for bbox in bboxes:
            to_check = np.nonzero(pts_to_check)[0]
            for pt in to_check:
                loc = array_to_location(pts[pt])
                pts_to_check[pt] = not bbox.contains(loc, act_transform)
        np.invert(pts_to_check, out=pts_to_check)
    t1 = time.time()
    print("carla no redundancy: ", t1 - t0)

    # These are precomputed
    vertices = np.array(
        [[location_to_array(loc) for loc in b.get_local_vertices()] for b in bboxes]
    )
    vertices_max = np.max(vertices, axis=1, keepdims=True)
    vertices_min = np.min(vertices, axis=1, keepdims=True)

    t0 = time.time()
    for _ in range(n_actors):
        pts_tr = pts - ego_pos
        pts_tr = pts_tr @ world_to_ego_rot
        # assignment = np.nonzero(((vertices_min <= pts_tr) & (pts_tr <= vertices_max)).all(2))
        any_bbox = ((vertices_min <= pts_tr) & (pts_tr <= vertices_max)).all(2).any(0)
    t1 = time.time()
    print("numpy: ", t1 - t0)

    print(pts_tr.shape)
    pts_2d = pts_tr[:, :2]
    print(pts_2d[0])
    print(pts_2d.shape)
    print(vertices_min.shape)
    v_min_2d = vertices_min[:, :, :2]
    print(v_min_2d[0])
    print(v_min_2d.shape)
    v_max_2d = vertices_max[:, :, :2]
    print(v_max_2d[0])
    print(v_max_2d.shape)
    any_bbox_2d = ((v_min_2d <= pts_2d) & (pts_2d <= v_max_2d)).all(2).any(0)
    print(any_bbox_2d)
    print(any_bbox)

    print(
        "Same result for all methods:",
        np.logical_and(
            (any_bbox == pts_to_check).all(), (pts_to_check == in_box).all()
        ),
    )

    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(projection="3d")

    pts_to_check = np.full(pts.shape[0], fill_value=True, dtype=np.bool_)
    pts_clr = np.full(pts.shape[0], fill_value="r", dtype=object)
    assignment_carla = np.zeros(pts.shape[0], dtype=np.int_)
    for i, bbox_vehicle in enumerate(bboxes):
        vertices = np.array(
            [
                location_to_array(loc)
                for loc in bbox_vehicle.get_world_vertices(act_transform)
            ]
        )
        to_check = np.nonzero(pts_to_check)[0]
        tr = act_transform
        for pt in to_check:
            loc = array_to_location(pts[pt, :])
            if bbox_vehicle.contains(loc, tr):
                pts_clr[pt] = bbox_clr[i]
                pts_to_check[pt] = False
                assignment_carla[pt] = i + 1
        for edge in BBOX_EDGE_PAIRS:
            ax.plot(
                vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], c=bbox_clr[i]
            )
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts_clr)
    ax.set_aspect("equal")

    plt.show()


# plot_bosch_vs_model()
# plot_n_pts()
# plot_rcs_sim()
plot_compare_bbox_methods()
