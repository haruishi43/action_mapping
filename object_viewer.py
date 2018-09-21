import os

import open3d as o3
import numpy as np
from datetime import datetime as dt

from open3d_chain import Open3D_Chain
from utils import DataManagement, poses_masks_from_npz


def 




if __name__=='__main__':
    dm = DataManagement()
    static_path = os.path.join('./static_data')

    after = dt(2018, 9, 9, 0, 0, 0)
    before = dt(2018, 9, 10, 0, 0, 0)
    datetimes = dm.get_datetimes_in(after, before)

    datetime = datetimes[2]
    print(datetime)

    data_path = dm.get_save_directory(datetime)

    files = os.listdir(data_path)
    filename = files[30]

    file_path = os.path.join(data_path, filename)

    _, masks = poses_masks_from_npz(file_path)
    print(len(masks))

    mask_ids = list(masks.keys())
    mask_pcs = []
    for id in mask_ids:
        np_arr = masks[id]
        mask_pcs.append(np_arr)

    pc = o3.PointCloud()
    pc.points = o3.Vector3dVector(np.concatenate(mask_pcs, axis=0))

    room_ply = os.path.join(static_path, 'room_A.ply')
    room = o3.read_point_cloud(room_ply)
    o3.draw_geometries([room, pc])
