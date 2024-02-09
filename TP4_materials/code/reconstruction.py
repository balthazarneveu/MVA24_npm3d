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
from time import perf_counter

from skimage import measure

import trimesh
import torch
from tqdm import tqdm
import math



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hoppe surface reconstruction
def compute_hoppe(points,
                  normals,
                  grid_resolution,
                  min_grid,
                  size_voxel,
                  device=DEVICE):

    # Build kDtree
    t0 = perf_counter()
    tree = KDTree(points,
                  leaf_size=40)
    print(f"Built kdtree in {perf_counter()-t0} sec")

    ## Move everything on gpu
    points = torch.from_numpy(points).to(device)
    normals = torch.from_numpy(normals).to(device)

    t1 = perf_counter()
    # Create a voxel grid (meshgrids acts weird in 3D)
    x = torch.arange(grid_resolution, device=device) * size_voxel[0] + min_grid[0]
    y = torch.arange(grid_resolution, device=device) * size_voxel[1] + min_grid[1]
    z = torch.arange(grid_resolution, device=device) * size_voxel[2] + min_grid[2]

    x = x[:   , None, None].expand((-1, grid_resolution, grid_resolution))
    y = y[None, :   , None].expand((grid_resolution, -1, grid_resolution))
    z = z[None, None, :   ].expand((grid_resolution, grid_resolution, -1))

    xyz = torch.stack((x, y, z), axis=-1).view(-1, 3)
    xyz_np = xyz.cpu().numpy() # Induces copy

    ## batched Tree query
    t0 = perf_counter()
    cloud_id = tree.query(xyz_np, k=1,
                          return_distance=False)[:, 0]
    _kd_tree_query = perf_counter()-t0
    print(f"Query kdtree in {_kd_tree_query} sec")
    cloud_id = torch.from_numpy(cloud_id).to(device)
    
    dist = xyz - points[cloud_id, :]
    n = normals[cloud_id, :]

    f = torch.sum(n*dist, axis=-1)  # batched dot product

    out = f.view(grid_resolution, grid_resolution, grid_resolution)
    _total = perf_counter() - t1
    print(f"Total time without kdTree : {_total - _kd_tree_query} sec")
    
    return out.cpu().numpy()


# IMLS surface reconstruction
def compute_imls(
    points, normals, grid_resolution, min_grid, size_voxel,
    knn: int=30,
    h: float=0.01,
    device=DEVICE,
    batch_size: int = 4096*4
    ) -> np.array:
    
    inv_h_square = 1/(h*h)
    
    # Build kDtree
    t0 = perf_counter()
    tree = KDTree(points,
                  leaf_size=40)
    _kd_tree_init = perf_counter()-t0
    print(f"Built kdtree in {_kd_tree_init} sec")
    
    ## Move everything on gpu
    points = torch.from_numpy(points).to(device)
    normals = torch.from_numpy(normals).to(device)
    
    t1 = perf_counter()
    # Create a voxel grid (meshgrids acts weird in 3D)
    x = torch.arange(grid_resolution, device=device) * size_voxel[0] + min_grid[0]
    y = torch.arange(grid_resolution, device=device) * size_voxel[1] + min_grid[1]
    z = torch.arange(grid_resolution, device=device) * size_voxel[2] + min_grid[2]

    x = x[:   , None, None].expand((-1, grid_resolution, grid_resolution))
    y = y[None, :   , None].expand((grid_resolution, -1, grid_resolution))
    z = z[None, None, :   ].expand((grid_resolution, grid_resolution, -1))


    xyz = torch.stack((x, y, z), axis=-1).view(-1, 3)
    xyz_np = xyz.cpu().numpy() # Induces copy

    n_batches = math.ceil(xyz.shape[0]/batch_size)
    sdf_list = [None] * n_batches
    # for idx in tqdm(range(0, xyz.shape[0]-batch_size, batch_size)):
    for idx in tqdm(range(n_batches)):
        beg, end = idx*batch_size, min((idx+1)*batch_size, xyz.shape[0])
        xyz_batch = xyz[beg:end]
        xyz_batchs_np = xyz_np[beg:end]

        ## batched Tree query
        t0 = perf_counter()
        cloud_id = tree.query(xyz_batchs_np, k=knn,
                              return_distance=False)
        _kd_tree_query = perf_counter()-t0
        print(f"\Query kdtree in {_kd_tree_query} sec")
        
        ## Move ids to gpu
        cloud_id = torch.from_numpy(cloud_id).to(device)
        
        knn_points = points[cloud_id, :]
        knn_normals = normals[cloud_id, :]

        xyz_batch = xyz_batch.unsqueeze(1)
        dist = xyz_batch - knn_points
        kernel = (-dist.norm(dim=-1).square()*inv_h_square).exp()
        dot_product = (knn_normals*dist).sum(dim=-1)
        sdf = (dot_product*kernel).sum(dim=-1)/(kernel.sum(dim=-1))
        sdf_list[idx] = sdf
        
    sdf = torch.cat(sdf_list, dim=0)
    sdf = sdf.view(grid_resolution, grid_resolution, grid_resolution)

    _total = perf_counter() - t1
    print(f"Total time without kdTree : {_total - _kd_tree_query*n_batches} sec")
    
    sdf = sdf.cpu().numpy()
    sdf = np.nan_to_num(sdf, nan=float("inf"))
    return sdf


if __name__ == '__main__':
    from pathlib import Path
    here = Path(__file__).parent
    outdir = here/'..'/'__out'
    outdir.mkdir(exist_ok=True)
    t0 = perf_counter()

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
    size_voxel = max([
        (max_grid[0]-min_grid[0])/(grid_resolution-1),
        (max_grid[1]-min_grid[1])/(grid_resolution-1),
        (max_grid[2]-min_grid[2])/(grid_resolution-1)
        ])

    size_voxel = (size_voxel, ) * 3

    # Create a volume grid to compute the scalar field for surface reconstruction

    # Compute the scalar field in the grid
    # method = "hoppe"
    # scalar_field = compute_hoppe(points, normals, grid_resolution, min_grid, size_voxel)
    method = "imls"
    scalar_field = compute_imls(points, normals, grid_resolution, min_grid, size_voxel,
                                30, batch_size=1_000 * 4_096)
    # scalar_field = compute_imls(points, normals, grid_resolution, min_grid, size_voxel,
    #                             30, batch_size=1_000 * 4_096)

    # Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri =\
        measure.marching_cubes(scalar_field,
                               level=0.0,
                               spacing=size_voxel
                               )
    verts += min_grid

    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(file_obj=outdir/f'bunny_mesh_{method}_{grid_resolution}.ply', file_type='ply')

    print("Total time for surface reconstruction : ", perf_counter() - t0)


    #%% Test different h
    for h in [1e-3, 1e-2, 1e-1]:
        scalar_field = compute_imls(points, normals, grid_resolution, min_grid, size_voxel,
                                    30, h=h, batch_size=1_000 * 4_096)
        verts, faces, normals_tri, values_tri =\
            measure.marching_cubes(scalar_field,
                                   level=0.0,
                                   spacing=size_voxel
                                   )
        verts += min_grid
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(file_obj=outdir/f'bunny_mesh_{method}_{grid_resolution}_h={h}.ply', file_type='ply')
        print(f"done h={h}")
        