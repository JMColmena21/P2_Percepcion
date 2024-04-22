import open3d as o3d
import numpy as np
import copy

def keypoints_to_spheres(keypoints, radius= 0.001):
    spheres = o3d.geometry.TriangleMesh()

    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(keypoint)
        spheres += sphere

    spheres.paint_uniform_color([1, 0, 0.75])

    return spheres

def extraer_keypoints():

    pcd = o3d.io.read_point_cloud("clouds/objects/s0_piggybank_corr.pcd")
    pcd_sub = pcd.voxel_down_sample(0.005) # Tamaño de la hoja de 0.1

    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd_sub, salient_radius= 0.015, non_max_radius = 0.01, gamma_21= 0.99, gamma_32= 0.99)

    spheres = keypoints_to_spheres(keypoints)

    o3d.visualization.draw_geometries([pcd_sub, spheres])


def main():
    pcd = o3d.io.read_point_cloud("clouds/scenes/snap_0point.pcd")
    # Mostrar nube
    #o3d.visualization.draw_geometries([pcd])

    plane_model, inliers = pcd.segment_plane(distance_threshold = 0.05, ransac_n  = 3, num_iterations = 1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold = 0.05, ransac_n  = 3, num_iterations = 1000)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

    plane_model, inliers3 = outlier_cloud.segment_plane(distance_threshold = 0.005, ransac_n  = 3, num_iterations = 1000)
    outlier_cloud = outlier_cloud.select_by_index(inliers3, invert=True)

    outlier_cloud_sub = outlier_cloud.voxel_down_sample(0.005) # Tamaño de la hoja de 0.1

    #o3d.visualization.draw_geometries([outlier_cloud_sub])

    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(outlier_cloud_sub, salient_radius= 0.015, non_max_radius = 0.01, gamma_21= 0.99, gamma_32= 0.99)

    spheres = keypoints_to_spheres(keypoints, radius=0.002)

    o3d.visualization.draw_geometries([outlier_cloud_sub, spheres])

    pcd = o3d.io.read_point_cloud("clouds/objects/s0_piggybank_corr.pcd")


if __name__ == "__main__":
    main()