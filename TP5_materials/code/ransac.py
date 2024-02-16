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


# Import numpy package and name it "np"
from pathlib import Path
import time
from ply import write_ply, read_ply
import numpy as np
import torch
from typing import Tuple
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Import functions to read and write ply files

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


def in_plane(points: torch.Tensor, pt_plane: torch.Tensor, normal_plane: torch.Tensor, threshold_in: float = 0.1) -> torch.Tensor:
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


def RANSAC(
    points: torch.Tensor,
    nb_draws: int = 100,
    threshold_in: int = 0.1
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
        selection_index = torch.randint(0, len(points), (nb_draws, 3), device=device)
        selection = points[selection_index]
        # Compute in parallel.
        point_planes, normal_planes = compute_plane(selection)
        in_planes = in_plane(points, point_planes, normal_planes, threshold_in)
        # in_planes is a mask! [B]
        # Count the number of inliers for each plane and vote
        total_votes = in_planes.squeeze(-1).sum(dim=-1)
    best_index = np.argmax(total_votes.cpu().numpy())
    best_vote = int(total_votes[best_index])
    point_plane = point_planes[best_index]
    normal_plane = normal_planes[best_index]
    return point_plane, normal_plane, best_vote, in_planes[best_index]


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):

    nb_points = len(points)
    plane_inds = np.arange(0, 0)
    plane_labels = np.arange(0, 0)
    remaining_inds = np.arange(0, nb_points)

    # TODO:

    return plane_inds, remaining_inds, plane_labels


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
    threshold_in: float = 0.1
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

    question_list = [1, 3]
    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #

    if 1 in question_list or 2 in question_list:
        print('\n--- 1) and 2) ---\n')
        plane_inds, remaining_inds = run_plane_passing_through_3_points(points)
        # Save extracted plane and remaining points
        write_ply(output_path/'plane.ply', [points_np[plane_inds], colors[plane_inds], labels[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply(output_path/'remaining_points_plane.ply', [points_np[remaining_inds], colors[remaining_inds],
                                                             labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    if 3 in question_list:
        print('\n--- 3) ---\n')
        plane_inds, remaining_inds = run_ransac(points)
        # Save the best extracted plane and remaining points
        write_ply(output_path/'best_plane.ply', [points_np[plane_inds], colors[plane_inds],
                                                 labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply(output_path/'remaining_points_best_plane.ply', [points_np[remaining_inds], colors[remaining_inds],
                                                                  labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    if 4 in question_list:
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
        write_ply(output_path/'best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds],
                                                  plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
        write_ply(output_path/'remaining_points_best_planes.ply',
                  [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print('Done')


if __name__ == '__main__':
    main()
