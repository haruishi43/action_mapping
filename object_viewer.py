import os
import argparse
import open3d as o3
import numpy as np
from datetime import datetime as dt

from open3d_chain import Open3D_Chain
from getter_models import coco_label_colors
import utils
from utils import DataManagement, poses_masks_from_npz



def pcd_from_masks(file_path):
    _, masks = poses_masks_from_npz(file_path)
    print(len(masks))

    mask_pcs = []
    if not (masks is None):
        mask_ids = list(masks.keys())
        for id in mask_ids:
            pcd = o3.PointCloud()
            np_arr = masks[id]
            print(np_arr.shape)
            pcd.points = o3.Vector3dVector(np_arr)
            
            
            # pc = o3.voxel_down_sample(pc, voxel_size=10) 
            i = int(id.split('_')[0])
            color = coco_label_colors[i]
            pcd.paint_uniform_color(color)
            mask_pcs.append(pcd)

    return mask_pcs


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Object Pose Getter')
    parser.add_argument('--data', default= '/mnt/extHDD/raw_data',help='relative data path from where you use this program')
    parser.add_argument('--save', default= './data_mask',help='relative data path from where you use this program')
    parser.add_argument('--frame', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    dm = DataManagement(args.data, args.save)
    static_path = os.path.join('./static_data')

    after = dt(2018, 9, 9, 13, 7, 0)
    before = dt(2018, 9, 9, 13, 9, 0)
    datetimes = dm.get_datetimes_in(after, before)

    datetime = datetimes[0]
    print(datetime)

    data_path = dm.get_save_directory(datetime)

    files = os.listdir(data_path)
    filename = files[args.frame]

    file_path = os.path.join(data_path, filename)
    pc = pcd_from_masks(file_path)

    room_ply = os.path.join(static_path, 'room_A.ply')
    room = o3.read_point_cloud(room_ply)

    pc.append(room)
    o3.draw_geometries(pc)
