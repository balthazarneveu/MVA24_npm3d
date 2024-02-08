#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh
import torch
from tqdm import tqdm

# Hoppe surface reconstruction


def compute_hoppe(points,
                  normals,
                  scalar_field,
                  grid_resolution,
                  min_grid,
                  size_voxel):

    tree = KDTree(points,
                  leaf_size=40)

    # Create a voxel grid
    x = np.arange(grid_resolution) * size_voxel[0] + min_grid[0]
    y = np.arange(grid_resolution) * size_voxel[1] + min_grid[1]
    z = np.arange(grid_resolution) * size_voxel[2] + min_grid[2]

    x = x[:, None, None]
    x = np.repeat(x, repeats=grid_resolution, axis=1)
    x = np.repeat(x, repeats=grid_resolution, axis=2)

    y = y[None, :, None]
    y = np.repeat(y, repeats=grid_resolution, axis=0)
    y = np.repeat(y, repeats=grid_resolution, axis=2)

    z = z[None, None, :]
    z = np.repeat(z, repeats=grid_resolution, axis=0)
    z = np.repeat(z, repeats=grid_resolution, axis=1)

    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    cloud_id = tree.query(xyz, k=1, return_distance=False)[:, 0]
    dist = xyz - points[cloud_id, :]
    n = normals[cloud_id, :]

    f = np.sum(n*dist, axis=-1)  # batched dot product

    out = f.reshape(grid_resolution, grid_resolution, grid_resolution)
    return out

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMLS surface reconstruction
def compute_imls(points, normals, scalar_field, grid_resolution, min_grid, size_voxel, knn=30, h=0.01, device=DEVICE):
    inv_h_square = 1/(h**2)
    tree = KDTree(points,
                  leaf_size=40)
    print("done kdtree")
    # Create a voxel grid
    x = np.arange(grid_resolution) * size_voxel[0] + min_grid[0]
    y = np.arange(grid_resolution) * size_voxel[1] + min_grid[1]
    z = np.arange(grid_resolution) * size_voxel[2] + min_grid[2]

    x = x[:, None, None]
    x = np.repeat(x, repeats=grid_resolution, axis=1)
    x = np.repeat(x, repeats=grid_resolution, axis=2)

    y = y[None, :, None]
    y = np.repeat(y, repeats=grid_resolution, axis=0)
    y = np.repeat(y, repeats=grid_resolution, axis=2)

    z = z[None, None, :]
    z = np.repeat(z, repeats=grid_resolution, axis=0)
    z = np.repeat(z, repeats=grid_resolution, axis=1)

    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    cloud_id = tree.query(xyz, k=knn, return_distance=False)
    print("done kdtree")
    knn_points = points[cloud_id, :]
    
    knn_points = torch.from_numpy(knn_points).to(device)
    knn_normals = torch.from_numpy(normals[cloud_id, :]).to(device)
    # knn_normals = knn_normals.unsqueeze(-1)
    print(knn_normals.shape)
    xyz = torch.from_numpy(xyz).to(device).unsqueeze(1)
    dist = xyz - knn_points
    kernel = (-dist.norm(dim=-1).square()*inv_h_square).exp()
    dot_product = (knn_normals*dist).sum(dim=-1)
    sdf = (dot_product*kernel).sum(dim=-1)/(kernel.sum(dim=-1))
    # print(knn_points.shape, xyz.shape, kernel.shape)
    # print(dist.shape)
    sdf = sdf.reshape(grid_resolution, grid_resolution, grid_resolution)
    sdf = sdf.cpu().numpy()
    return sdf


if __name__ == '__main__':
    from pathlib import Path
    here = Path(__file__).parent
    outdir = here/'..'/'__out'
    outdir.mkdir(exist_ok=True)
    t0 = time.time()

    # Path of the file
    file_path = here/'..'/'data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

    # Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)

    # Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)


    # grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1), (max_grid[1]-min_grid[1]) /
                     (grid_resolution-1), (max_grid[2]-min_grid[2])/(grid_resolution-1)])

    size_voxel = (size_voxel, ) * 3

    # Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)

    # Compute the scalar field in the grid
    method = "hoppe"
    # scalar_field = compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel)
    method = "imls"
    scalar_field = compute_imls(points, normals, scalar_field, grid_resolution, min_grid, size_voxel, 30)

    # Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri =\
        measure.marching_cubes(scalar_field,
                               level=0.0,
                               spacing=(size_voxel[0],
                                        size_voxel[1],
                                        size_voxel[2])
                               )
    verts += min_grid

    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(file_obj=outdir/f'bunny_mesh_{method}_{grid_resolution}.ply', file_type='ply')

    print("Total time for surface reconstruction : ", time.time()-t0)
