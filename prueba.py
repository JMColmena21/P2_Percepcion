import open3d as o3d
import numpy as np
import copy

pcd = o3d.io.read_point_cloud("clouds/scenes/snap_0point.pcd")
# Mostrar nube
o3d.visualization.draw_geometries([pcd])