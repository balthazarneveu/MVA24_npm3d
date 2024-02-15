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
    point_plane = np.zeros((3, 1))
    normal_plane = np.zeros((3, 1))
    p0p1 = points[..., 1, :] - points[..., 0, :]
    p0p2 = points[..., 2, :] - points[..., 0, :]
    normal_plane = torch.cross(p0p1, p0p2).unsqueeze(-2)
    normal_plane = normal_plane / torch.norm(normal_plane, dim=-1, keepdim=True)
    point_plane = points[..., 0, :].unsqueeze(-2)
    # normal_plane = normal_plane.transpose(-1, -2)
    # point_plane = point_plane.transpose(-1, -2)
    # assert normal_plane.shape == (3, 1), normal_plane.shape
    # assert point_plane.shape == (3, 1), point_plane.shape
    return point_plane, normal_plane


def in_plane(points: torch.Tensor, pt_plane: torch.Tensor, normal_plane: torch.Tensor, threshold_in: float = 0.1) -> torch.Tensor:
    # (N, 3), (1, 3)

    diff = (points - pt_plane)
    dist_to_plane = torch.abs(torch.matmul(diff, normal_plane.transpose(-1, -2)))
    indexes = dist_to_plane < threshold_in

    return indexes


def RANSAC(points, nb_draws=100, threshold_in=0.1):

    best_vote = 3
    best_pt_plane = np.zeros((3, 1))
    best_normal_plane = np.zeros((3, 1))

    # TODO:

    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):

    nb_points = len(points)
    plane_inds = np.arange(0, 0)
    plane_labels = np.arange(0, 0)
    remaining_inds = np.arange(0, nb_points)

    # TODO:

    return plane_inds, remaining_inds, plane_labels


def compute_plane_passing_through_3_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Define parameter
    threshold_in = 0.10
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
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    # plane_inds = points_in_plane.nonzero()[0]
    # remaining_inds = (1-points_in_plane).nonzero()[0]
    plane_inds = torch.where(points_in_plane)[0]
    remaining_inds = torch.where(~points_in_plane)[0]
    plane_inds = plane_inds.cpu()
    remaining_inds = remaining_inds.cpu()
    for sel in selection:
        assert sel in plane_inds
    print(
        f"plane_inds: {plane_inds.shape[0]} {plane_inds.shape[0]/nb_points:.1%}\n" +
        f"remaining_inds: {remaining_inds.shape[0]} {remaining_inds.shape[0]/nb_points:.1%}")
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

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #

    print('\n--- 1) and 2) ---\n')
    plane_inds, remaining_inds = compute_plane_passing_through_3_points(points)
    # Save extracted plane and remaining points
    write_ply(output_path/'plane.ply', [points_np[plane_inds], colors[plane_inds], labels[plane_inds]],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply(output_path/'remaining_points_plane.ply', [points_np[remaining_inds], colors[remaining_inds],
              labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    return

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
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds],
              labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds],
              labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

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
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds],
              plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply',
              [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print('Done')


if __name__ == '__main__':
    main()
