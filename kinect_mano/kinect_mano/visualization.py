"""Local Matplotlib visualization; headless export does not require ROS/X11."""
from pathlib import Path

import numpy as np


def render_frame(vertices, faces, joints, output, *, views=(), title="MANO reconstruction", truth_joints=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(9, 8))
    ax = figure.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices[faces], alpha=.3, facecolor="lightskyblue", edgecolor="none")
    ax.add_collection3d(mesh)
    for index, view in enumerate(views):
        points = view.points_m[::max(1, len(view.points_m) // 2000)]
        ax.scatter(*points.T, s=2, alpha=.35, color=("tab:orange", "tab:green")[index % 2], label=view.camera_id)
    ax.scatter(*joints.T, s=15, color="navy", label="fitted joints")
    for start in (1, 5, 9, 13, 17):
        chain = joints[[0, *range(start, start + 4)]]
        ax.plot(*chain.T, color="navy")
    if truth_joints is not None:
        ax.scatter(*truth_joints.T, marker="x", color="red", s=18, label="synthetic truth")
    center = joints.mean(0)
    radius = max(float(np.ptp(vertices, axis=0).max()) * .65, .05)
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius), xlabel="world x [m]", ylabel="world y [m]", zlabel="world z [m]", title=title)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=-70, azim=-90)
    ax.legend(loc="upper right")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def render_reconstruction(path, output, frame=0):
    from .surface import SurfaceView
    with np.load(path, allow_pickle=False) as archive:
        if not 0 <= frame < len(archive["joints_m"]):
            raise ValueError("frame out of range")
        views = ()
        if "cloud_m" in archive:
            points = archive["cloud_m"][frame]
            points = points[np.isfinite(points).all(1)]
            views = (SurfaceView(points, np.zeros(3), "observations"),)
        return render_frame(archive["vertices_m"][frame], archive["faces"], archive["joints_m"][frame], output, views=views, title=f"Frame {frame}, valid={archive['valid'][frame]}")
