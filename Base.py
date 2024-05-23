import open3d as o3d
import numpy as np
import copy
import time

# NECESARIO QUE LAS NUBES DE PUNTOS 
# SE ENCUENTREN EN LA MISMA LOCALIZACION 
# QUE GITHUB:
#  
# https://github.com/JMColmena21/P2_Percepcion


# Variables globales
CELL_SIZE = 0.0025
THRESHOLD = 0.0025

# Función que genera esferas en los puntos que recibe
def keypoints_to_spheres(keypoints, radius= 0.002):
    spheres = o3d.geometry.TriangleMesh()   # Generacion de un Mesh formado por triangulos

    for keypoint in keypoints.points:   
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius) # Crear una esfera del radio determinado
        sphere.translate(keypoint)  # Trasladarlo al punto 
        spheres += sphere

    spheres.paint_uniform_color([1, 0, 0.75])   # Pintar las esferas de color rosa

    return spheres


def fpfh(pcd_sub):

    radius_normal = 0.01
    # Estimacion de las normales
    pcd_sub.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn = 30))

    radius_feature = 0.02
    # Calculo de los descriptores
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_sub, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn = 300))

    return pcd_fpfh


def extraer_keypoints(pcd):

    #keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius= 0.005, non_max_radius = 0.015, gamma_21= 0.9, gamma_32= 0.9)
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius= 0.01, non_max_radius = 0.01, gamma_21= 0.9, gamma_32= 0.9)
    
    # Obtener esferas para visualizar mejor los keypoints
    spheres = keypoints_to_spheres(keypoints)

    # Visualizar resultados
    o3d.visualization.draw_geometries([pcd, spheres])

    return keypoints


def correspondencias(fpfh_escena, fpfh_objeto, mutua = True):

    # Generacion de KDTree para obtencion rapida de vecinos cercanos
    escena_tree = o3d.geometry.KDTreeFlann(fpfh_escena)
    objeto_tree = o3d.geometry.KDTreeFlann(fpfh_objeto)

    # Matriz donde se guardaran los indices de las correspondencias
    corr = []

    # Para cada descriptor del objeto
    for i in range(len(fpfh_objeto[0,:])):

        # Buscar el vecino mas cercano en la escena
        [k, idx_escena, distance_kdtree] = escena_tree.search_knn_vector_xd(fpfh_objeto[:,i], 1)

        # Si el parecido es mayor que un umbral
        if distance_kdtree[0] < 2000:

            # Correspondencia en un sentido
            if not mutua:

                corr.append([i,idx_escena[0]])
            
            # Correspondencia mutua
            else:
                # Buscar el vecino mas cercano en el objeto
                [k, idx_objeto, _] = objeto_tree.search_knn_vector_xd(fpfh_escena[:,idx_escena[0]], 1)

                # Si se trata del mismo descriptor se acepta
                if idx_objeto[0] == i:
                    corr.append([i,idx_escena[0]])

    # Tipo de dato requerido en RANSAC
    return o3d.cpu.pybind.utility.Vector2iVector(corr)



def registration(pcd_escena, pcd_objeto):

    #print(len(pcd_escena.points))

    inicio = time.time()
    
    # Segmentacion de planos principales para eliminar paredes y la mesa
    plane_model, inliers = pcd_escena.segment_plane(distance_threshold = 0.05, ransac_n  = 3, num_iterations = 200)
    outlier_cloud = pcd_escena.select_by_index(inliers, invert=True)

    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold = 0.05, ransac_n  = 3, num_iterations = 200)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold = 0.02, ransac_n  = 3, num_iterations = 200)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

    # Simplificacion de las nubes de puntos
    pcd_sub_escena = outlier_cloud.voxel_down_sample(CELL_SIZE) 
    pcd_sub_objeto = pcd_objeto.voxel_down_sample(CELL_SIZE) 
    
    #print(1-(len(pcd_sub_escena.points)/len(outlier_cloud.points)))

    inicio_keypoints = time.time()

    # Extraccion de keypoints
    keypoints_escena = extraer_keypoints(pcd_sub_escena)
    keypoints_objeto = extraer_keypoints(pcd_sub_objeto)

    # Obtencion de descriptores de toda la nube de puntos
    pcd_fpfh_escena = fpfh(pcd_sub_escena)
    pcd_fpfh_objeto = fpfh(pcd_sub_objeto)

    #Filtrado de los descriptores de los keypoints, encontrar indices
    index_array_escena = []

    for point in keypoints_escena.points:

        index_array_escena.append(np.where(np.all(np.asarray(pcd_sub_escena.points)==point, axis=1))[0][0])

    index_array_objeto = []

    for point in keypoints_objeto.points:

        index_array_objeto.append(np.where(np.all(np.asarray(pcd_sub_objeto.points)==point, axis=1))[0][0])

    # Obtener los descriptores 
    fpfh_escena = pcd_fpfh_escena.data
    fpfh_objeto = pcd_fpfh_objeto.data

    # Filtrado de los descriptores
    fpfh_escena = fpfh_escena[:, index_array_escena]
    fpfh_objeto = fpfh_objeto[:, index_array_objeto]

    fin_keypoints = time.time()
    
    #print(f"Keypoints Objeto: {len(keypoints_objeto.points)}")
    #print(f"Keypoints Escena: {len(keypoints_escena.points)}")

    inicio_correspondecias = time.time()
    
    # Calculo de las correspondencias
    corr = correspondencias(fpfh_escena, fpfh_objeto)
    fin_correspondencias = time.time()

    # Visualizacion de las correspondencias, lineas de color rojo
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(keypoints_objeto, keypoints_escena, corr)
    line_set.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd_sub_objeto, pcd_sub_escena, line_set])

    inicio_RANSAC = time.time()
    distance_threshold = CELL_SIZE * 1.5

    # RANSAC que necesita keypoints y descriptores
    '''fpfh_objeto = o3d.pipelines.registration.Feature()
    fpfh_objeto.data = pcd_fpfh_objeto.data[:, index_array_objeto]

    fpfh_escena = o3d.pipelines.registration.Feature()
    fpfh_escena.data = pcd_fpfh_escena.data[:, index_array_escena]

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        keypoints_objeto, keypoints_escena, fpfh_objeto, fpfh_escena, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))'''

    # RANSAC que necesita las correspondencias y los keypoints
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        keypoints_objeto, keypoints_escena, corr, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(5000))
    fin_RANSAC = time.time()

    # Evaluar la transformacion obtenida con RANSAC
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd_sub_objeto, pcd_sub_escena, THRESHOLD, result.transformation)

    inicio_ICP = time.time()
    # Ejecutar ICP para refinar
    result_icp = o3d.pipelines.registration.registration_icp(pcd_sub_objeto, pcd_sub_escena, THRESHOLD, result.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    fin_ICP = time.time()

    print(f"(Keypoints - {round(fin_keypoints-inicio_keypoints,4)})")
    print(f"(Correspondencias - {round(fin_correspondencias-inicio_correspondecias,4)})")
    print(f"(RANSAC - {round(fin_RANSAC-inicio_RANSAC,4)}) Fitness:{evaluation.fitness} RMSE:{evaluation.inlier_rmse}")
    print(f"(ICP - {round(fin_ICP-inicio_ICP,4)}) Fitness:{result_icp.fitness} RMSE:{result_icp.inlier_rmse}")
    print(f"Tiempo Total: {round(fin_ICP-inicio,4)}")

    # Ejecutar el desplazamiento y rotacion requerido para superponer ambas nubes
    pcd_objeto.transform(result_icp.transformation)
    
    # Visualizar el resultado
    pcd_objeto.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd_objeto, pcd_escena])

if __name__ == "__main__":

    # Lectura de los archivos que contienen las nubes de puntos
    pcd_piggy = o3d.io.read_point_cloud("clouds/objects/s0_piggybank_corr.pcd")
    pcd_plant = o3d.io.read_point_cloud("clouds/objects/s0_plant_corr.pcd")
    pcd_mug = o3d.io.read_point_cloud("clouds/objects/s0_mug_corr.pcd")
    pcd_plc = o3d.io.read_point_cloud("clouds/objects/s0_plc_corr.pcd")
    pcd_escena = o3d.io.read_point_cloud("clouds/scenes/snap_0point.pcd")

    # Ejecucion del pipeline para cada uno de los objetos
    registration(pcd_escena, pcd_piggy)
    registration(pcd_escena, pcd_plant)
    registration(pcd_escena, pcd_mug)
    registration(pcd_escena, pcd_plc)