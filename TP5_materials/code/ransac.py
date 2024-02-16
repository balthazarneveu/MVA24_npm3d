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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree# Import functions from scikit-learn

from tqdm import tqdm

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

def plane_aligned(normal_planes, normals, threshold_angle=15):
    normals = normals.unsqueeze(0)
    dot_product = torch.sum(normal_planes * normals, dim=-1)
    norm1 = torch.norm(normal_planes, dim=-1)
    norm2 = torch.norm(normal_planes, dim=-1)
    
    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = cos_theta * (1 - 1e-5) # To avoid nan
    angle = torch.acos(cos_theta) * 180 / torch.pi # angle in degrees
    
    mask = torch.min(angle, torch.pi - angle) < threshold_angle
    return mask
    
    

def RANSAC(
    points: torch.Tensor,
    nb_draws: int = 100,
    threshold_in: int = 0.1,
    normals: torch.Tensor=None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Performs the RANSAC algorithm to estimate the best plane model from a set of 3D points.

    Args:
        points (torch.Tensor): [N, 3] input 3D points.
        nb_draws (int): RANSAC parameter - number of random draws to perform.
        threshold_in (int): RANSAC parameter - inlier threshold distance.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: A tuple containing 
        - estimated point on the plane,
        - estimated normal of the plane
        - number of inliers.
    """
    if normals is None:
        selection_index = torch.randint(0, len(points), (nb_draws, 3), device=device)
        selection = points[selection_index]
        point_planes, normal_planes = compute_plane(selection)
    else:
        selection_index = torch.randint(0, len(points), (nb_draws, 1), device=device)
        point_planes, normal_planes = points[selection_index], normals[selection_index]
    in_planes = in_plane(points, point_planes, normal_planes, threshold_in)
    
    if normals is not None:
        in_planes = in_planes * plane_aligned(normal_planes, normals)
    
    total_votes = in_planes.squeeze(-1).sum(dim=-1)
        
    best_index = np.argmax(total_votes.cpu().numpy())
    best_vote = int(total_votes[best_index])
    point_plane = point_planes[best_index]
    normal_plane = normal_planes[best_index]
    return point_plane, normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2, normals=None):
    nb_points = len(points)
    device = points.device
    
    remaining_inds = torch.arange(nb_points, device=device)
    remaining_points = points
    remaining_normals = normals
	
    plane_inds = []
    plane_labels = []

    for plane_id in range(nb_planes):
        print(f"Fitting plane {plane_id}")
        ## Fit with ransac
        pt_plane, normal_plane, best_vote = RANSAC(remaining_points,
                                                   nb_draws,
                                                   threshold_in,
                                                   normals=remaining_normals)
        
        ## Find explainable points
        is_explainable = in_plane(remaining_points, pt_plane, normal_plane, threshold_in).squeeze(-1)
        if normals is not None:
            is_explainable = is_explainable * plane_aligned(normal_plane, remaining_normals).squeeze(0)

        plane_inds.append(remaining_inds[is_explainable.bool()])
        plane_labels.append(plane_id * torch.ones(is_explainable.sum().item(), device=device))
        
        ## Mask them
        remaining_points = remaining_points[~is_explainable]
        remaining_inds = remaining_inds[~is_explainable]
        if remaining_normals is not None:
            remaining_normals = remaining_normals[~is_explainable]
    
    plane_inds = torch.cat(plane_inds)
    plane_labels = torch.cat(plane_labels)
    return plane_inds.cpu().numpy(), remaining_inds.cpu().numpy(), plane_labels.cpu().numpy()



    return plane_inds, remaining_inds

def PCA(points):
    assert points.dim() == 3
    # [batch, neighbours, xyz]
    barycenter = torch.mean(points, dim=1, keepdim=True)
    diff = points - barycenter
    cov_mat = torch.matmul(diff.transpose(1, 2), diff) / points.size(1)
    eigenvalues, eigenvectors = torch.linalg.eig(cov_mat)
    # Take the first component to get the normal
    # Torch eigh seems broken atm, so we get complex of imaginary part 0
    # with eig instead
    return torch.real(eigenvalues), torch.real(eigenvectors)


def compute_normals(query_points, cloud_points, k: int = 20):
    query_points_np = query_points.cpu().numpy()
    cloud_points_np = cloud_points.cpu().numpy()
    device = cloud_points.device

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    tree = KDTree(cloud_points_np, leaf_size=40, metric='minkowski')
    neighbors = tree.query(query_points_np, k=k, return_distance=False)
    neighbors = torch.from_numpy(neighbors).to(device)
        
    eigenvalues, eigenvectors = PCA(cloud_points[neighbors])
    normals = eigenvectors[:, :, 0]
    return normals


def run_plane_passing_through_3_points(
    points: torch.Tensor,
    threshold_in: float = 0.10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the plane passing through 3 random points, extract close points and remaining points
    # >>>> QUESTION 1 and 2
    Args:
        points (torch.Tensor): [N, 3]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: plane_inds, remaining_inds (close to the plane, too far from the plane)
    """
    # Define parameter

    nb_points = len(points)
    # [3038661, d=3]
    # Take randomly three points
    selection = np.random.randint(0, nb_points, size=3)
    pts = points[selection]
    # [n=3, d=3]

    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))

    # Find points in the plane and others
    t0 = time.time()
    in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds, remaining_inds = select_points_in_plane(points, pt_plane, normal_plane, nb_points, threshold_in)
    for sel in selection:
        assert sel in plane_inds
    return plane_inds, remaining_inds


def run_ransac(points: torch.Tensor, nb_draws: int = 100, threshold_in: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    # >>> QUESTION 3

    Args:
        points (torch.Tensor): [N, 3]
        nb_draws (int, optional): Ransac parameters, number of random sampled planes to test. Defaults to 100.
        threshold_in (float, optional): Ransac parameters, distance to plane. Defaults to 0.1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    nb_points = len(points)
    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

    # Find points in the plane and others
    plane_inds, remaining_inds = select_points_in_plane(
        points, best_pt_plane, best_normal_plane, nb_points, threshold_in)
    return plane_inds, remaining_inds

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
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    question_list = [4, 5]
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
    if 4 in question_list:
        print('\n--- 4) ---\n')

        # Define parameters of recursive_RANSAC
        nb_draws = 100
        threshold_in = 0.10
        nb_planes = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
        t1 = time.time()
        print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply(output_path/'Q4_best_planes.ply', [points_np[plane_inds], colors[plane_inds], labels[plane_inds],
                                                  plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
        write_ply(output_path/'Q4_remaining_points_best_planes.ply',
                  [points_np[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        
    if 5 in question_list:
        print('\n--- 5) ---\n')

        # Define parameters of recursive_RANSAC
        nb_draws = 100
        threshold_in = 0.10
        nb_planes = 5
        
        # Compute normals
        normals = compute_normals(points, points, k=50)

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes, normals)
        # plane_inds = plane_inds.cpu().numpy()
        # remaining_inds = remaining_inds.cpu().numpy()
        # plane_labels = plane_labels.cpu().numpy()
        
        t1 = time.time()
        print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply(output_path/'Q5_best_planes.ply', [points_np[plane_inds], colors[plane_inds], labels[plane_inds],
                                                  plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
        write_ply(output_path/'Q5_remaining_points_best_planes.ply',
                  [points_np[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print('Done')
    