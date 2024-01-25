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

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def PCA(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print(points.shape)
    eigenvalues = None
    eigenvectors = None
    barycenter = np.mean(points, axis=0)
    diff = points - barycenter
    cov_mat = np.dot(diff.T, diff)/points.shape[0]
    print(cov_mat.shape)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    # Take the first component to get the normal
    return eigenvalues, eigenvectors


def compute_local_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))

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
    if True:

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
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
