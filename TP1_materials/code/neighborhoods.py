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
from ply import read_ply

# Import time package
import time
from pathlib import Path
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt

here = Path(__file__).parent


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    # assert len(queries.shape) == 2 and queries.shape[1] == 3
    # assert len(supports.shape) == 2 and supports.shape[1] == 3
    
    supports = supports[:, None] # shape [N, 1 3]
    queries = queries[None] # shape [1, M, 3]
    
    dist = np.linalg.norm(supports - queries, axis=-1)
    
    valid_ids = np.argwhere(dist <= radius) # shape [N, M]
    
    return valid_ids.transpose()


def brute_force_KNN(queries, supports, k):
    # assert len(queries.shape) == 2 and queries.shape[1] == 3
    # assert len(supports.shape) == 2 and supports.shape[1] == 3
    
    supports = supports[:, None] # shape [N, 1 3]
    queries = queries[None] # shape [1, M, 3]
    
    dist = np.linalg.norm(supports - queries, axis=-1) # shape [N, M]
    
    sorted_sample_id = np.argsort(dist, axis=0) # for each query, order samples by growing distance
    closest_sample_id = sorted_sample_id[:k] # keep the k closest sample for each query point
    
    return closest_sample_id.transpose()





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
    if False:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(
            points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:
        radius = 0.2
        # Define the search parameters
        num_queries = 1000

        leaf_sizes = []
        timings_creation = []
        timings_query = []
        for leaf_size in tqdm(range(0, 10)):
            if leaf_size == 0:
                leaf_size = 1
            leaf_sizes.append(leaf_size)

            # create the kd tree
            t0 = time.time()
            tree = KDTree(points,
                          leaf_size=leaf_size,
                          metric='minkowski')
            t1 = time.time()
            # print('leaf_size {}, kd-tree computed in {:.3f} seconds'.format(leaf_size, t1 - t0))
            timings_creation.append(t1 - t0)

            random_indices = np.random.choice(
                points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            t0 = time.time()
            ind = tree.query_radius(queries, r=radius)
            t1 = time.time()
            timings_query.append(t1 - t0)
            # print('leaf_size {}, {:d} tree query computed in {:.3f} seconds'.format(leaf_size, num_queries, t1 - t0))
            total_time = points.shape[0] * (t1 - t0) / num_queries
            # print('leaf_size {}, Computing kd-tree on whole cloud : {:.2f} hours'.format(leaf_size, total_time / 3600))
            # print(10*"_")

        plt.figure()
        plt.plot(leaf_sizes, timings_creation)
        plt.title("Creation")
        plt.grid()
        plt.ylabel("Duration (sec)")
        plt.xlabel("Leaf size")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(leaf_sizes, timings_query)
        plt.title("10 queries for radius = 0.2")
        plt.grid()
        plt.xlabel("Leaf size")
        plt.ylabel("Duration (sec)")
        plt.legend()
        plt.show()

    # If statement to skip this part if wanted
    if True:
        radius = 0.2
        # Define the search parameters
        num_queries = 1000

        leaf_size = 50
        timings_query = []
        radia = []
        for radius in tqdm(np.linspace(0.1, 0.5, 20)):
            radia.append(radius)

            # create the kd tree
            t0 = time.time()
            tree = KDTree(points,
                          leaf_size=leaf_size,
                          metric='minkowski')
            t1 = time.time()
            # print('leaf_size {}, kd-tree computed in {:.3f} seconds'.format(leaf_size, t1 - t0))

            random_indices = np.random.choice(
                points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            t0 = time.time()
            ind = tree.query_radius(queries, r=radius)
            t1 = time.time()
            timings_query.append(t1 - t0)
            # print('leaf_size {}, {:d} tree query computed in {:.3f} seconds'.format(leaf_size, num_queries, t1 - t0))
            total_time = points.shape[0] * (t1 - t0) / num_queries
            print(
                'radius {}, Computing kd-tree on whole cloud : {:.2f} minutes'.format(radius, total_time / 60))
            print(10*"_")

        plt.figure()
        plt.plot(radia, timings_query)
        plt.title(f" {num_queries} queries for radius = 0.2")
        plt.grid()
        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Radius")
        plt.ylabel("Duration (sec)")
        plt.legend()
        plt.show()
