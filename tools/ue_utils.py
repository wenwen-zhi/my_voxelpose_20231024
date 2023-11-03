import numpy as np
from scipy.spatial.transform import Rotation as R
def transform_unreal_to_stadium(_pts):
    r_unreal_stadium = R.from_euler('z', -90, degrees=True).as_matrix()
    # placed a cube in the unreal stadium and measured its position
    # origin_coords = [-3428, -4070, 265] # before scale up
    origin_coords = [-3608, -4250, 280]  # with scale up
    # in unreal, 1 unit is 1cm, we have 1 unit = 1m
    scale = 100

    orig = np.array(origin_coords.copy())
    orig[1] *= -1
    #     transform unreal to stadium!!

    pts = _pts.copy()
    pts -= orig
    pts = pts.dot(r_unreal_stadium) / scale
    return pts
