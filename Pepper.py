import open3d as o3d
import numpy as np
import copy
import time

CELL_SIZE = 0.01
THRESHOLD = 0.005

def keypoints_to_spheres(keypoints, radius= 0.01):
    spheres = o3d.geometry.TriangleMesh()

    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(keypoint)
        spheres += sphere

    spheres.paint_uniform_color([1, 0, 0.75])

    return spheres

def fpfh(pcd):

    pcd_sub = pcd.voxel_down_sample(CELL_SIZE) 

    radius_normal = 0.02
    pcd_sub.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn = 30))

    radius_feature = 0.03
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_sub, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn = 300))

    return pcd_sub, pcd_fpfh


def extraer_keypoints(pcd):

    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius= 0.04, non_max_radius = 0.04, gamma_21= 0.8, gamma_32= 0.9)

    spheres = keypoints_to_spheres(keypoints)

    o3d.visualization.draw_geometries([pcd, spheres])

    return keypoints

def correspondencias(fpfh_escena, fpfh_objeto, mutua = True):

    escena_tree = o3d.geometry.KDTreeFlann(fpfh_escena)
    objeto_tree = o3d.geometry.KDTreeFlann(fpfh_objeto)

    corr = []

    for i in range(len(fpfh_objeto[0,:])):

        [k, idx_escena, distance_kdtree] = escena_tree.search_knn_vector_xd(fpfh_objeto[:,i], 1)

        if distance_kdtree[0] < 2000:
            if not mutua:

                corr.append([i,idx_escena[0]])

            else:
                
                [k, idx_objeto, _] = objeto_tree.search_knn_vector_xd(fpfh_escena[:,idx_escena[0]], 1)

                if idx_objeto[0] == i:
                    corr.append([i,idx_escena[0]])
    return o3d.cpu.pybind.utility.Vector2iVector(corr)





def main():

    pcd_objeto = o3d.io.read_point_cloud("more_clouds/pepper_obj/pcd_33.pcd")
    pcd_escena = o3d.io.read_point_cloud("more_clouds/pepper_scene/pcd_21.pcd")

    pcd_objeto = pcd_objeto.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-0.3,-1,1.6],dtype=np.float64), np.array([0.3,0.4,2.2],dtype=np.float64)))

    inicio_keypoints = time.time()

    pcd_sub_escena, pcd_fpfh_escena = fpfh(pcd_escena)
    pcd_sub_objeto, pcd_fpfh_objeto = fpfh(pcd_objeto)

    keypoints_escena = extraer_keypoints(pcd_sub_escena)
    keypoints_objeto = extraer_keypoints(pcd_sub_objeto)

    
    index_array_escena = []

    for point in keypoints_escena.points:

        index_array_escena.append(np.where(np.all(np.asarray(pcd_sub_escena.points)==point, axis=1))[0][0])

    index_array_objeto = []

    for point in keypoints_objeto.points:

        index_array_objeto.append(np.where(np.all(np.asarray(pcd_sub_objeto.points)==point, axis=1))[0][0])
    

    fpfh_escena = pcd_fpfh_escena.data
    fpfh_objeto = pcd_fpfh_objeto.data

    fpfh_escena = fpfh_escena[:, index_array_escena]
    fpfh_objeto = fpfh_objeto[:, index_array_objeto]
    fin_keypoints = time.time()


    inicio_correspondecias = time.time()
    corr = correspondencias(fpfh_escena, fpfh_objeto)
    fin_correspondencias = time.time()
    
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(keypoints_objeto, keypoints_escena, corr)
    line_set.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd_sub_objeto, pcd_sub_escena, line_set])

    inicio_RANSAC = time.time()
    distance_threshold = CELL_SIZE * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        keypoints_objeto, keypoints_escena, corr, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.99))
    fin_RANSAC = time.time()


    evaluation = o3d.pipelines.registration.evaluate_registration(pcd_sub_objeto, pcd_sub_escena, THRESHOLD, result.transformation)

    inicio_ICP = time.time()
    result_icp = o3d.pipelines.registration.registration_icp(pcd_sub_objeto, pcd_sub_escena, THRESHOLD, result.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
    fin_ICP = time.time()

    print(f"(Keypoints - {round(fin_keypoints-inicio_keypoints,4)})")
    print(f"(Correspondencias - {round(fin_correspondencias-inicio_correspondecias,4)})")
    print(f"(RANSAC - {round(fin_RANSAC-inicio_RANSAC,4)}) Fitness:{evaluation.fitness} RMSE:{evaluation.inlier_rmse}")
    print(f"(ICP - {round(fin_ICP-inicio_ICP,4)}) Fitness:{result_icp.fitness} RMSE:{result_icp.inlier_rmse}")

    pcd_objeto.transform(result.transformation)
    pcd_objeto.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd_objeto, pcd_escena])

if __name__ == "__main__":
    main()