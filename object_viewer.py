import os

import open3d as o3
import numpy as np
from datetime import datetime as dt

from open3d_chain import Open3D_Chain
from utils import DataManagement, poses_masks_from_npz


def pcd_from_masks(file_path):
    _, masks = poses_masks_from_npz(file_path)
    print(len(masks))

    mask_ids = list(masks.keys())
    mask_pcs = []
    for id in mask_ids:
        np_arr = masks[id]
        print(np_arr.shape)
        mask_pcs.append(np_arr)

    pcd = o3.PointCloud()
    pcd.points = o3.Vector3dVector(np.concatenate(mask_pcs, axis=0))

    return pcd


if __name__=='__main__':
    dm = DataManagement()
    static_path = os.path.join('./static_data')

    after = dt(2018, 9, 9, 13, 2, 0)
    before = dt(2018, 9, 9, 13, 9, 0)
    datetimes = dm.get_datetimes_in(after, before)

    datetime = datetimes[0]
    print(datetime)

    data_path = dm.get_save_directory(datetime)

    files = os.listdir(data_path)
    filename = files[446]

    file_path = os.path.join(data_path, filename)

    pc = pcd_from_masks(file_path)

    room_ply = os.path.join(static_path, 'room_A.ply')
    room = o3.read_point_cloud(room_ply)
    o3.draw_geometries([room, pc])
