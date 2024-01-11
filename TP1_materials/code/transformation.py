#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
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

# Import functions to read and write ply files
from ply import write_ply, read_ply
from pathlib import Path
here = Path(__file__).parent


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#

def rescale_point_cloud(pc: np.ndarray, scale: float = 1., offset=np.zeros(3)) -> np.ndarray:
    center = pc.mean(axis=0)
    return (pc - center) * scale - offset

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

    # Path of the file
    file_path = here.parent/'data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data['red'], data['green'], data['blue'])).T

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #

    # Replace this line by your code
    transformed_points = rescale_point_cloud(
        points, scale=1/2, offset=np.array([0., -10.E-2, 0.]))

    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply('../little_bunny.ply',
              [transformed_points, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    print('Done')
