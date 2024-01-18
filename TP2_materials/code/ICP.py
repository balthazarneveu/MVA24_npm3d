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

    for it in range(max_iter):
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
        # data_aligned 
        rms = np.sqrt(np.linalg.norm(data_aligned - ref_nearest, axis=0).mean())
        RMS_list.append(rms)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


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
    if True:

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
        plt.plot(RMS_list)
        plt.show()

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'

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
        plt.show()
