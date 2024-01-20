#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
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
from visu import show_ICP
from pathlib import Path
from typing import List, Tuple
from sklearn.neighbors import KDTree
from tqdm import tqdm
import time
import sys
here = Path(__file__).parent

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(dat: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
         dat = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''
    assert dat.shape == ref.shape, f"{dat.shape} =! {ref.shape}"
    dat_m = np.mean(dat, axis=1, keepdims=True)
    ref_m = np.mean(ref, axis=1, keepdims=True)
    dat_c = dat - dat_m
    ref_c = ref - ref_m
    dat_m.shape
    h_mat = dat_c.dot(ref_c.T)
    u, _, vt = np.linalg.svd(h_mat)
    rot = (vt.T).dot(u.T)
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = rot = (vt.T).dot(u.T)
    return rot, ref_m - rot.dot(dat_m)


def icp_point_to_point(dat, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        dat = (d x N_dat) matrix where "N_dat" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration

    '''

    # Variable for aligned data
    data_aligned = np.copy(dat)
    leaf_size = 2
    tree = KDTree(ref.T, leaf_size=leaf_size, metric='minkowski')

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []
    d = ref.shape[-2]
    rot_prev = np.eye(d)
    trans_prev = np.zeros((d, 1))
    # YOUR CODE
    rms = np.inf
    for it in range(max_iter):
        if rms < RMS_threshold:
            break
        ref_nearest_index = tree.query(data_aligned.T, k=1, return_distance=False)[:, 0]
        ref_nearest = ref[:, ref_nearest_index]
        neighbors_list.append(ref_nearest_index.copy())
        rot, trans = best_rigid_transform(data_aligned, ref_nearest)
        data_aligned = np.dot(rot, data_aligned) + trans
        trans = np.dot(rot, trans_prev) + trans
        rot = np.dot(rot, rot_prev)
        rot_prev = rot
        trans_prev = trans
        R_list.append(rot.copy())
        T_list.append(trans.copy())
        rms = np.sqrt(np.linalg.norm(data_aligned - ref_nearest, axis=0).mean())
        RMS_list.append(rms)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_point_to_point_fast(dat, ref, max_iter, RMS_threshold, tree=None, sampling_limit=100, true_rmse=True):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        dat = (d x N_dat) matrix where "N_dat" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        Tree = pre-buit KDTree (leaf size=150 is a good value)
        true_rmse = Full RMSE computation
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
        total = total time spent in the function (for benchmarking)

    '''

    # Variable for aligned data
    data_aligned = np.copy(dat)
    if tree is None:
        leaf_size = 150
        tree = KDTree(ref.T, leaf_size=leaf_size, metric='minkowski')
        print("Done KDTree")
    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []
    d = ref.shape[-2]
    rot_prev = np.eye(d)
    trans_prev = np.zeros((d, 1))
    # YOUR CODE
    rms = np.inf
    total = 0.
    for it in tqdm(range(max_iter)):
        if rms < RMS_threshold:
            break
        t0 = time.time()
        selection_indexes = np.random.choice(data_aligned.shape[-1], size=sampling_limit, replace=False)
        data_selection = data_aligned[:, selection_indexes]
        ref_nearest_index = tree.query(data_selection.T, k=1, return_distance=False)[:, 0]
        ref_nearest = ref[:, ref_nearest_index]
        neighbors_list.append(ref_nearest_index.copy())
        rot, trans = best_rigid_transform(data_selection, ref_nearest)
        data_aligned = np.dot(rot, data_aligned) + trans
        trans = np.dot(rot, trans_prev) + trans
        rot = np.dot(rot, rot_prev)
        rot_prev = rot
        trans_prev = trans
        R_list.append(rot.copy())
        T_list.append(trans.copy())
        t1 = time.time()
        total = t1-t0
        if not true_rmse:
            rms = np.sqrt(np.linalg.norm(data_aligned[:, selection_indexes] - ref_nearest, axis=0).mean())
        else:
            # Compute RMSE on the whole point cloud - not taken into account in the timing
            ref_nearest_index = tree.query(data_aligned.T, k=1, return_distance=False)[:, 0]
            ref_nearest = ref[:, ref_nearest_index]
            rms = np.sqrt(np.linalg.norm(data_aligned - ref_nearest, axis=0).mean())
        RMS_list.append(rms)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list, total


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Transformation estimation
    # *************************
    #
    out_dir = here/"_out"
    out_dir.mkdir(exist_ok=True, parents=True)

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = here/'../data/bunny_original.ply'
        bunny_r_path = here/'../data/bunny_returned.ply'

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud

        write_ply(out_dir/'bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = here/'../data/ref2D.ply'
        data2D_path = here/'../data/data2D.ply'

        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)

        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)

        # Plot RMS
        plt.close()
        plt.figure()
        plt.plot(RMS_list)
        plt.ylim(0, None)
        plt.grid()
        plt.title("RMS evolution")
        plt.show()

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = here/'../data/bunny_original.ply'
        bunny_p_path = here/'../data/bunny_perturbed.ply'

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)

        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

        # Plot RMS
        plt.plot(RMS_list)
        plt.grid()
        plt.ylim(0, None)
        plt.title("RMS evolution - ICP 3D")
        plt.show()

        # If statement to skip this part if wanted
    if True:

        # Cloud paths
        cloud_o_path = here/'../data/Notre_Dame_Des_Champs_1.ply'
        cloud_p_path = here/'../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        cloud_o_ply = read_ply(cloud_o_path)
        cloud_p_ply = read_ply(cloud_p_path)
        cloud_o = np.vstack((cloud_o_ply['x'], cloud_o_ply['y'], cloud_o_ply['z']))
        cloud_p = np.vstack((cloud_p_ply['x'], cloud_p_ply['y'], cloud_p_ply['z']))

        # Apply ICP

        # for leaf_size in [200, 100, 20]:
        # for leaf_size in [50, 100, 150]:
        for leaf_size in [150]:
            t_build_tree = time.time()
            print("Build tree")
            tree = KDTree(cloud_o.T, leaf_size=leaf_size, metric='minkowski')
            t_build_tree = time.time() - t_build_tree
            for sampling_limit in [1000, 10000]:
                cloud_p_opt, R_list, T_list, neighbors_list, RMS_list, total_iter_time = icp_point_to_point_fast(
                    cloud_p, cloud_o, 20, 1e-4, tree=tree, sampling_limit=sampling_limit, true_rmse=True)
                # Plot RMS
                plt.subplot(211)
                plt.plot(RMS_list, "-o", label=f"sampling_limit ={sampling_limit} - KD{leaf_size}")
                plt.subplot(212)
                plt.plot(np.linspace(0, total_iter_time, len(RMS_list)), RMS_list,
                         "-o", label=f"sampling_limit ={sampling_limit} - KD{leaf_size}" +
                         f" - Total Time {t_build_tree+total_iter_time:.2f}s (={total_iter_time:.2f} + KD{leaf_size} {t_build_tree:.2f}s)")
        plt.subplot(211)
        plt.xlabel("iteration")
        plt.ylabel("RMS")
        plt.grid()
        plt.legend()
        plt.ylim(0.5, None)
        plt.subplot(212)
        plt.xlabel("elapsed time")
        plt.ylabel("RMS")
        plt.grid()
        plt.legend()
        plt.ylim(0.5, None)
        plt.suptitle("RMS evolution - ICP 3D")
        plt.show()
