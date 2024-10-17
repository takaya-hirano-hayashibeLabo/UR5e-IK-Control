import numpy as np
from scipy.spatial.transform import Rotation as R

import mink

def transform_matrix(quaternion, position):
    """
    クォータニオンと位置座標からTmatrixを計算
    :param quaternion: [x,y,z,w]
    :param position: [x,y,z]
    :return: 4x4 transformation matrix
    """
    # Create a rotation object from the quaternion
    rotation = R.from_quat(quaternion)

    # Convert the rotation object to a 3x3 rotation matrix
    rotation_matrix = rotation.as_matrix()

    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix


def np2SE3_target(quaternion:np.ndarray,position:np.ndarray):
    """
    numpyのクォータニオンと位置をminkのtargetに変換する
    :param quaternion: [x,y,z,w]
    :param position: [x,y,z]
    :return: 4x4 transformation matrix
    """
    t_matrix = transform_matrix(np.array(quaternion), np.array(position))
    target=mink.SE3.from_matrix(t_matrix)
    
    return target

