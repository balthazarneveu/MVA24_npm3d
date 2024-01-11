#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time
from pathlib import Path
from typing import List
here = Path(__file__).parent


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(
        queries: np.ndarray,  # (N, 3)
        supports: np.ndarray,  # (M, 3)
        radius: np.ndarray) -> List[np.ndarray]:
    # [1, N, 3] - [M, 1, 3]
    queries = np.expand_dims(queries, axis=1)
    supports_exp = np.expand_dims(supports, axis=0)
    distances = np.linalg.norm(supports_exp-queries, axis=-1)
    neighborhood_indices = np.argwhere(distances <= radius)
    return neighborhood_indices


def brute_force_KNN(
    queries: np.ndarray,
    supports: np.ndarray,
    k: int = 100
) -> np.ndarray:
    queries = np.expand_dims(queries, axis=1)
    supports_exp = np.expand_dims(supports, axis=0)
    distances = np.linalg.norm(supports_exp-queries, axis=-1)
    neighborhoods = np.argsort(distances)[:, :k]
    return neighborhoods


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#
if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = here.parent/'data'/'indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(
            points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        total_iterations = 1
        t0 = time.time()
        for _ in range(total_iterations):
            neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()
        # Search KNN
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(
            num_queries, (t1 - t0)/total_iterations))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * \
            (t1 - t0)/total_iterations / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(
            total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(
            total_KNN_time / 3600))

    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 1000

        # YOUR CODE
