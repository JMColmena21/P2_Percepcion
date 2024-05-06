import open3d as o3d
import numpy as np
import copy
import time

CELL_SIZE = 0.0025
THRESHOLD = 0.0025

def keypoints_to_spheres(keypoints, radius= 0.001):
    spheres = o3d.geometry.TriangleMesh()

    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(keypoint)
        spheres += sphere

    spheres.paint_uniform_color([1, 0, 0.75])

    return spheres

def fpfh(pcd):

    pcd_sub = pcd.voxel_down_sample(CELL_SIZE) 

    radius_normal = CELL_SIZE * 4
    pcd_sub.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn = 30))

    radius_feature = CELL_SIZE * 10
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_sub, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn = 100))

    return pcd_sub, pcd_fpfh


def extraer_keypoints(pcd):

    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius= 0.0075, non_max_radius = 0.005, gamma_21= 0.975, gamma_32= 0.975)

    spheres = keypoints_to_spheres(keypoints)

    o3d.visualization.draw_geometries([pcd, spheres])

    return keypoints

def correspondencias(fpfh_escena, fpfh_objeto, mutua = True):

    escena_tree = o3d.geometry.KDTreeFlann(fpfh_escena)
    objeto_tree = o3d.geometry.KDTreeFlann(fpfh_objeto)

    corr = []

    for i in range(len(fpfh_objeto[0,:])):

        [k, idx_escena, _] = escena_tree.search_knn_vector_xd(fpfh_objeto[:,i], 1)

        if not mutua:

            corr.append([i,idx_escena[0]])

        else:
            
            [k, idx_objeto, _] = objeto_tree.search_knn_vector_xd(fpfh_escena[:,idx_escena[0]], 1)

            if idx_objeto[0] == i:
                corr.append([i,idx_escena[0]])

    return o3d.cpu.pybind.utility.Vector2iVector(corr)





def main():

    pcd_objeto = o3d.io.read_point_cloud("more_clouds/charmander_obj/pcd_9.pcd")
    pcd_escena = o3d.io.read_point_cloud("more_clouds/clutter_scene/pcd_26.pcd")

    plane_model, inliers = pcd_escena.segment_plane(distance_threshold = 0.05, ransac_n  = 3, num_iterations = 100)
    outlier_cloud = pcd_escena.select_by_index(inliers, invert=True)

    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold = 0.12, ransac_n  = 3, num_iterations = 100)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

    plane_model, inliers3 = outlier_cloud.segment_plane(distance_threshold = 0.01, ransac_n  = 3, num_iterations = 100)
    outlier_cloud = outlier_cloud.select_by_index(inliers3, invert=True)

    pcd_sub_escena, pcd_fpfh_escena = fpfh(outlier_cloud)
    pcd_sub_objeto, pcd_fpfh_objeto = fpfh(pcd_objeto)

    keypoints_escena = extraer_keypoints(pcd_sub_escena)
    keypoints_objeto = extraer_keypoints(pcd_sub_objeto)

    pcd_sub_escena_tree = o3d.geometry.KDTreeFlann(pcd_sub_escena)
    index_array_escena = []

    for point in keypoints_escena.points:

        [k, idx, _] = pcd_sub_escena_tree.search_knn_vector_3d(point, 1)
        index_array_escena.append(idx[0])

    pcd_sub_objeto_tree = o3d.geometry.KDTreeFlann(pcd_sub_objeto)
    index_array_objeto = []

    for point in keypoints_objeto.points:

        [k, idx, _] = pcd_sub_objeto_tree.search_knn_vector_3d(point, 1)
        index_array_objeto.append(idx[0])
    

    fpfh_escena = pcd_fpfh_escena.data
    fpfh_objeto = pcd_fpfh_objeto.data


    fpfh_escena = fpfh_escena[:, index_array_escena]
    fpfh_objeto = fpfh_objeto[:, index_array_objeto]


    corr = correspondencias(fpfh_escena, fpfh_objeto)

    distance_threshold = CELL_SIZE * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        keypoints_objeto, keypoints_escena, corr, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    evaluation = o3d.pipelines.registration.evaluate_registration(pcd_sub_objeto, pcd_sub_escena, THRESHOLD, result.transformation)

    print(f"Solo RANSAC: fitness-{evaluation.fitness} rmse-{evaluation.inlier_rmse}")

    result_icp = o3d.pipelines.registration.registration_icp(pcd_sub_objeto, pcd_sub_escena, THRESHOLD, result.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))

    print(f"ICP: fitness-{result_icp.fitness} rmse-{result_icp.inlier_rmse}")

    pcd_objeto.transform(result_icp.transformation)
    pcd_objeto.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd_objeto, pcd_escena])

if __name__ == "__main__":
    main()