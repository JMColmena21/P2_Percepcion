import open3d as o3d
import numpy as np
import copy

pcd = o3d.io.read_point_cloud("clouds/scenes/snap_0point.pcd")
# Mostrar nube

plane_model, inliers = pcd.segment_plane(distance_threshold = 0.05, ransac_n  = 3, num_iterations = 1000)
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)

plane_model, inliers2 = outlier_cloud.segment_plane(distance_threshold = 0.05, ransac_n  = 3, num_iterations = 1000)
inlier_cloud2 = outlier_cloud.select_by_index(inliers2)
inlier_cloud2.paint_uniform_color([1.0, 0, 0])
outlier_cloud2 = outlier_cloud.select_by_index(inliers2, invert=True)

plane_model, inliers3 = outlier_cloud2.segment_plane(distance_threshold = 0.005, ransac_n  = 3, num_iterations = 1000)
inlier_cloud3 = outlier_cloud2.select_by_index(inliers3)
inlier_cloud3.paint_uniform_color([1.0, 0, 0])
outlier_cloud3 = outlier_cloud2.select_by_index(inliers3, invert=True)

outlier_cloud3_sub = outlier_cloud3.voxel_down_sample(0.005) # Tamaño de la hoja de 0.1

o3d.visualization.draw_geometries([outlier_cloud3_sub])

pcd = o3d.io.read_point_cloud("clouds/objects/s0_piggybank_corr.pcd")

pcd_sub = pcd.voxel_down_sample(0.005) # Tamaño de la hoja de 0.1

o3d.visualization.draw_geometries([pcd_sub])