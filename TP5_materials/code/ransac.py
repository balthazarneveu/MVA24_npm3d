#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
# ------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

from pathlib import Path
from math import ceil

# Import numpy package and name it "np"
import numpy as np
import torch

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import functions from scikit-learn
from sklearn.neighbors import KDTree  # Import functions from scikit-learn

from tqdm import tqdm

from typing import Tuple

# Import time package
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import time package
HERE = Path(__file__).parent


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the plane passing through 3 points

    Args:
        points (torch.Tensor): A tensor of shape (B, n=3, d=3) representing the n=3 points.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - the point on the plane.
        - the normal vector to the plane.
    """
    with torch.no_grad():
        p0p1 = points[..., 1, :] - points[..., 0, :]
        p0p2 = points[..., 2, :] - points[..., 0, :]
        normal_plane = torch.cross(p0p1, p0p2).unsqueeze(-2)
        normal_plane_norm = torch.norm(normal_plane, dim=-1, keepdim=True)
        assert (normal_plane_norm != 0.).all()
        normal_plane = normal_plane / normal_plane_norm
        point_plane = points[..., 0, :].unsqueeze(-2)
        # normal_plane = normal_plane.transpose(-1, -2)
        # point_plane = point_plane.transpose(-1, -2)
        # assert normal_plane.shape == (3, 1), normal_plane.shape
        # assert point_plane.shape == (3, 1), point_plane.shape
    return point_plane, normal_plane


def in_plane(
    points: torch.Tensor,
    pt_plane: torch.Tensor,
    normal_plane: torch.Tensor,
    threshold_in: float = 0.1
) -> torch.Tensor:
    """
    Compute where the distance between the points and the plane under a certain threshold
    """
    with torch.no_grad():
        if len(pt_plane.shape) == 2:
            diff = (points - pt_plane)
            dist_to_plane = torch.abs(torch.matmul(diff, normal_plane.transpose(-1, -2)))
        else:
            # Do this computation in 2 steps to limit memory footprint
            # When naively computing the differences between points and pt_plane
            # The matrix mulitplication with the normal vector gets too big
            # n^T.(x-x0) = n^T*x - n^T*x0... as simple as that.
            # x0 -> [B, 3], x -> [N, 3]
            # (x-x0) -> [B, N, 3] - this is a large matrix for multiplication...
            # Better broadcast afteward
            n_t = normal_plane.transpose(-1, -2)
            x_n = torch.matmul(points, n_t).squeeze(-1)
            x0_n = torch.matmul(pt_plane, n_t).squeeze(-1)
            dist_to_plane = torch.abs(x_n-x0_n)
        indexes = dist_to_plane < threshold_in
    return indexes


def plane_parallel(normal_planes, normals, threshold_angle=15):
    normals = normals.unsqueeze(0)
    dot_product = torch.sum(normal_planes * normals, dim=-1)
    norm1 = torch.norm(normal_planes, dim=-1)
    norm2 = torch.norm(normal_planes, dim=-1)

    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = cos_theta * (1 - 1e-5)  # To avoid nan
    angle = torch.acos(cos_theta) * 180 / torch.pi  # angle in degrees

    mask = torch.min(angle, 180 - angle) < threshold_angle
    return mask


def RANSAC(
    points: torch.Tensor,
    nb_draws: int = 100,
    threshold_in: int = 0.1,
    normals: torch.Tensor = None,
    threshold_angle=15,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
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
    with torch.no_grad():
        if normals is None:
            selection_index = torch.randint(0, len(points), (nb_draws, 3), device=device)
            selection = points[selection_index]
            # Compute in parallel.
            point_planes, normal_planes = compute_plane(selection)
        else:
            selection_index = torch.randint(0, len(points), (nb_draws, 1), device=device)
            point_planes, normal_planes = points[selection_index], normals[selection_index]

        in_planes = in_plane(points, point_planes, normal_planes, threshold_in)
        # in_planes is a mask! [B]
        # Count the number of inliers for each plane and vote

        if normals is not None:
            planes_parallel = plane_parallel(normal_planes, normals, threshold_angle)
            in_planes = in_planes * planes_parallel

        total_votes = in_planes.squeeze(-1).sum(dim=-1)
    best_index = np.argmax(total_votes.cpu().numpy())
    best_vote = int(total_votes[best_index])
    point_plane = point_planes[best_index]
    normal_plane = normal_planes[best_index]
    return point_plane, normal_plane, best_vote, in_planes[best_index]


def faster_RANSAC(
        points: torch.Tensor,
        beta=0.01,
        threshold_in: int = 0.1,
        normals: torch.Tensor = None,
        threshold_angle=15,
        B=20,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
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
    N_iter = 1_000_000  # init
    best_best_vote = -1
    with torch.no_grad():
        for n in range(N_iter):

            point_plane, normal_plane, best_vote, inliers = RANSAC(
                points,
                nb_draws=B,
                threshold_in=threshold_in,
                normals=normals,
                threshold_angle=threshold_angle,
            )

            if best_vote > best_best_vote:
                # update N_iter
                N_iter = ceil(
                    np.log(beta) / (1e-6 + np.log(1 - best_vote/points.size(0)))
                )
                best_best_vote = best_vote
                best_point_plane = point_plane
                best_normal_plane = normal_plane

            if n >= N_iter:
                break

        # Evaluate the best model on the whole dataset
        in_planes = in_plane(points, best_point_plane, best_normal_plane, threshold_in)
        # in_planes is a mask! [B]
        # Count the number of inliers for each plane and vote

        if normals is not None:
            planes_parallel = plane_parallel(best_normal_plane, normals, threshold_angle).T
            in_planes = in_planes * planes_parallel

    return best_point_plane, best_normal_plane, best_best_vote, in_planes.squeeze(1)


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2, normals=None, threshold_angle=15,
                     faster_ransac_variant=False, beta=0.01, B=5):

    nb_points = len(points)
    device = points.device

    remaining_inds = torch.arange(nb_points, device=device)
    remaining_points = points
    remaining_normals = normals

    plane_inds = []
    plane_labels = []

    for plane_id in tqdm(range(nb_planes), desc="Processing planes", unit="plane"):

        # Fit with ransac
        if faster_ransac_variant:
            pt_plane, normal_plane, best_vote, is_explainable = faster_RANSAC(
                remaining_points,
                beta,
                threshold_in,
                normals=remaining_normals,
                threshold_angle=threshold_angle,
                B=B)
        else:
            pt_plane, normal_plane, best_vote, is_explainable = RANSAC(remaining_points,
                                                                       nb_draws,
                                                                       threshold_in,
                                                                       normals=remaining_normals,
                                                                       threshold_angle=threshold_angle)

        plane_inds.append(remaining_inds[is_explainable.bool()])
        plane_labels.append(plane_id * torch.ones(is_explainable.sum().item(), device=device))

        # Mask them
        remaining_points = remaining_points[~is_explainable]
        remaining_inds = remaining_inds[~is_explainable]
        if remaining_normals is not None:
            remaining_normals = remaining_normals[~is_explainable]

    plane_inds = torch.cat(plane_inds)
    plane_labels = torch.cat(plane_labels)
    return plane_inds.cpu().numpy(), remaining_inds.cpu().numpy(), plane_labels.cpu().numpy()


def PCA(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plane fitting by performing PCA on the covariance matrix of a neighborhood.

    Args:
        points (torch.Tensor): Neighborhoods of points. [batch, neighbours, xyz]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: eigenvalues [batch, 1], eigenvectors [batch, xyz]

    WARNING:
    ========
    Uses numpy for eigenvectors computation
    For some reasons torch.eig and torch.eigh did not work.

    """
    assert points.dim() == 3
    # [batch, neighbours, xyz]
    barycenter = torch.mean(points, dim=1, keepdim=True)
    diff = points - barycenter
    cov_mat = torch.matmul(diff.transpose(1, 2), diff) / points.size(1)

    # torch eig and eigh looked broken. None of them worked in my install.
    # They do a stupid cpu sync anyways.
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat.cpu().numpy())

    eigenvalues = torch.from_numpy(eigenvalues).to(device)
    eigenvectors = torch.from_numpy(eigenvectors).to(device)

    return eigenvalues, eigenvectors


def compute_normals(
    query_points: torch.Tensor,
    cloud_points: torch.Tensor,
    k: int = 20
) -> torch.Tensor:
    """Compute normals of a point cloud using PCA on the k neighborhoods of the points.

    Args:
        query_points (torch.Tensor): [N, 3]
        cloud_points (torch.Tensor): [total_num_points, 3]
        k (int, optional): number of points in the neighborhood. Defaults to 20.

    Returns:
        torch.Tensor: normals [N, 3]
    """
    query_points_np = query_points.cpu().numpy()
    cloud_points_np = cloud_points.cpu().numpy()
    device = cloud_points.device

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    tree = KDTree(cloud_points_np, leaf_size=40, metric='minkowski')
    neighbors = tree.query(query_points_np, k=k, return_distance=False)
    neighbors = torch.from_numpy(neighbors).to(device)

    _eigenvalues, eigenvectors = PCA(cloud_points[neighbors])
    normals = eigenvectors[:, :, 0]

    return normals


def select_points_in_plane(points, pt_plane, normal_plane, nb_points, threshold_in=0.1):
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    plane_inds = torch.where(points_in_plane)[0]
    remaining_inds = torch.where(~points_in_plane)[0]
    plane_inds = plane_inds.cpu()
    remaining_inds = remaining_inds.cpu()
    print(
        f"plane_inds: {plane_inds.shape[0]} {plane_inds.shape[0]/nb_points:.1%}\n" +
        f"remaining_inds: {remaining_inds.shape[0]} {remaining_inds.shape[0]/nb_points:.1%}")

    return plane_inds, remaining_inds


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


def run_ransac(
    points: torch.Tensor,
    nb_draws: int = 100,
    threshold_in: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    best_pt_plane, best_normal_plane, best_vote, _ = RANSAC(points, nb_draws, threshold_in)
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


def main():
    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = HERE.parent/'data'/'indoor_scan.ply'
    output_path = HERE.parent/'__output'
    output_path.mkdir(exist_ok=True, parents=True)

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points_np = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    points = torch.from_numpy(points_np).to(device)

    question_list = [6]
    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #

    if 1 in question_list or 2 in question_list:
        print('\n--- 1) and 2) ---\n')
        plane_inds, remaining_inds = run_plane_passing_through_3_points(points)
        # Save extracted plane and remaining points
        write_ply(
            (output_path/'Q1_plane.ply').as_posix(),  # Windows friendly
            [points_np[plane_inds], colors[plane_inds], labels[plane_inds]],
            ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply(
            (output_path/'Q1_remaining_points_plane.ply').as_posix(),
            [points_np[remaining_inds], colors[remaining_inds],
             labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    if 3 in question_list:
        print('\n--- 3) ---\n')
        plane_inds, remaining_inds = run_ransac(points)
        # Save the best extracted plane and remaining points
        write_ply(
            (output_path/'Q3_best_plane.ply').as_posix(),
            [points_np[plane_inds], colors[plane_inds], labels[plane_inds]],
            ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply(
            (output_path/'Q3_remaining_points_best_plane.ply').as_posix(),
            [points_np[remaining_inds], colors[remaining_inds], labels[remaining_inds]],
            ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    if 4 in question_list:
        print('\n--- 4) ---\n')

        # Define parameters of recursive_RANSAC
        nb_draws = 200
        threshold_in = 0.1
        nb_planes = 2

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
        t1 = time.time()
        print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply((output_path/'Q4_best_planes.ply').as_posix(),
                  [points_np[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
        write_ply((output_path/'Q4_remaining_points_best_planes.ply').as_posix(),
                  [points_np[remaining_inds], colors[remaining_inds], labels[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    if 5 in question_list:
        print('\n--- 5) ---\n')

        # Define parameters of recursive_RANSAC
        nb_draws = 500
        threshold_in = 0.20
        threshold_angle = 10  # degrees
        nb_planes = 5

        # Compute normals
        normals = compute_normals(points, points, k=20)

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC(
            points, nb_draws, threshold_in, nb_planes, normals, threshold_angle)

        t1 = time.time()
        print('recursive RANSAC with normals done in {:.3f} seconds'.format(t1 - t0))

        normals = normals.cpu().numpy()
        # write_ply((output_path/'Q5_test_normals.ply').as_posix(),
        #           [points_np, colors, labels, normals],
        #           ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'nx', 'ny', 'nz'])

        # Save the best planes and remaining points
        write_ply((output_path/'Q5_best_planes.ply').as_posix(),
                  [points_np[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32),
                   normals[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label', 'nx', 'ny', 'nz'])
        write_ply((output_path/'Q5_remaining_points_best_planes.ply').as_posix(),
                  [points_np[remaining_inds], colors[remaining_inds], labels[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print('Done')

    if 6 in question_list:
        print('\n--- 6) ---\n')

        # Define parameters of recursive_RANSAC
        nb_draws = 500
        threshold_in = 0.20
        threshold_angle = 10  # degrees
        nb_planes = 5

        # Compute normals
        normals = compute_normals(points, points, k=20)

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC(
            points, nb_draws, threshold_in, nb_planes, normals, threshold_angle, faster_ransac_variant=True)

        t1 = time.time()
        print('Faster recursive RANSAC with normals done in {:.3f} seconds'.format(t1 - t0))

        normals = normals.cpu().numpy()
        # write_ply((output_path/'Q5_test_normals.ply').as_posix(),
        #           [points_np, colors, labels, normals],
        #           ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'nx', 'ny', 'nz'])

        # Save the best planes and remaining points
        write_ply((output_path/'Q6_best_planes.ply').as_posix(),
                  [points_np[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32),
                   normals[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label', 'nx', 'ny', 'nz'])
        write_ply((output_path/'Q6_remaining_points_best_planes.ply').as_posix(),
                  [points_np[remaining_inds], colors[remaining_inds], labels[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print('Done')


if __name__ == '__main__':
    main()
