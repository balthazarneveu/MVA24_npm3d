#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    ### compute the plance defined by the three first points given
    
    ## compute the normal as cross product
    point_plane = points[0, None]
    
    # Extract the three points from the matrix
    p0 = points[0, :]
    p1 = points[1, :]
    p2 = points[2, :]
    
    # Compute vectors p0p1 and p0p2
    v0 = p1 - p0
    v1 = p2 - p0
    
    # Compute the normal vector using the cross product
    normal_plane = np.cross(v0, v1)[:, None]
    
    # Normalize the normal vector
    normal_plane /= np.linalg.norm(normal_plane)
    return point_plane, normal_plane



def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):
    N_points, _ = points.shape
    indexes = np.zeros(len(points), dtype=bool)
    
    # TODO:
    dif = points - pt_plane
    dot = np.sum(dif * normal_plane.T, axis=-1)
    distance = np.abs(dot)
    
    indexes = (distance <= threshold_in).astype(int)
        
    return indexes



def RANSAC(points, nb_draws=100, threshold_in=0.1):
    
    best_vote = -1
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    n_points = points.shape[0]
    
    # TODO:
    for draw_id in tqdm(range(nb_draws)):
        ## Fetch 3 points
        random_indices = np.random.choice(n_points, size=3, replace=False)
        
        ## Fecth random plane
        point_plane, normal_plane = compute_plane(points[random_indices, :])
        
        ## Test plane
        n_votes = np.sum(in_plane(points, point_plane, normal_plane, threshold_in))
        
        if n_votes > best_vote:
            best_vote = n_votes
            best_pt_plane = point_plane
            best_normal_plane = normal_plane
                
    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):
    nb_points = len(points)
    
    remaining_inds = np.arange(nb_points)
    
    remaining_points = points
	
    plane_inds = []
    plane_labels = []
    # TODO:
    for plane_id in range(nb_planes):
        print(f"Fitting plane {plane_id}")
        ## Fit with ransac
        pt_plane, normal_plane, best_vote = RANSAC(remaining_points)
        
        ## Find explainable points
        is_explainable = in_plane(remaining_points, pt_plane, normal_plane, threshold_in)

        plane_inds.append(remaining_inds[is_explainable.astype(bool)])
        plane_labels.append(plane_id * np.ones(np.sum(is_explainable)))
        
        ## Mask them
        remaining_points = remaining_points[(1 - is_explainable).astype(bool)]
        remaining_inds = remaining_inds[(1 - is_explainable).astype(bool)]
    
    plane_inds = np.concatenate(plane_inds)
    plane_labels = np.concatenate(plane_labels)
    return plane_inds, remaining_inds, plane_labels



#------------------------------------------------------------------------------------------
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
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 100
    threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    threshold_in = 0.10
    nb_planes = 2
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    
    
    print('Done')
    