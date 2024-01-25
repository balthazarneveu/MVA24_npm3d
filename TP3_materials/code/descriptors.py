#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from typing import Tuple
# Import time package
import time
from pathlib import Path
here = Path(__file__).parent
out_dir = here / 'out'
out_dir.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def PCA(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eigenvalues = None
    eigenvectors = None
    barycenter = np.mean(points, axis=0)
    diff = points - barycenter
    cov_mat = np.dot(diff.T, diff)/points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    # Take the first component to get the normal
    return eigenvalues, eigenvectors


def compute_local_PCA(query_points: np.ndarray, cloud_points: np.ndarray, radius: float = None, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    assert (k is not None) or (radius is not None), "You must set k or radius"

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    tree = KDTree(cloud_points, leaf_size=40, metric='minkowski')
    if radius is not None:
        neighbors = tree.query_radius(query_points, radius)
    else:
        neighbors = tree.query(query_points, k=k, return_distance=False)
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    for i in range(len(neighbors)):
        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[neighbors[i]])
    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):

    verticality = None
    linearity = None
    planarity = None
    sphericity = None

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':
    cloud_path = here.parent/'data'/'Lille_street_small.ply'
    # PCA verification
    # ****************
    if False:

        # Load cloud as a [N x 3] matrix

        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    if True:
        # Load cloud as a [N x 3] matrix
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # # Compute PCA on the whole cloud
        for radius in [0.1, 0.2, 0.5]:
            all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, radius=radius)
            normals = all_eigenvectors[:, :, 0]

            # Save cloud with normals
            write_ply(str(out_dir/f"Lille_street_small_normals_r={radius:.2f}.ply"),
                      (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])

        # Compute PCA on the whole cloud - k=30
        for k in [30, 10, 6]:
            all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, k=k)
            normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply(str(out_dir/f"Lille_street_small_normals_k={k}.ply"),
                  (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
