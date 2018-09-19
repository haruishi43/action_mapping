import os

import open3d as o3
import numpy as np
from datetime import datetime as dt

from open3d_chain import Open3D_Chain
from utils import DataManagement



def main():
    dm = DataManagement()
    after = dt(2018, 7, 23, 14, 0, 0)
    before = dt(2018, 7, 23, 14, 1, 0)
    datetimes = dm.get_datetimes_in(after, before)

    print(datetimes)

    assert len(datetimes) == 1

    datetime = datetimes[0]

    data_path = dm.get_save_directory(datetime)
    data_path = os.path.join(data_path, 'objects')

    print(os.listdir(data_path))

    ob_name = 'keyboard'
    ob_path = os.path.join(data_path, ob_name)

    # just get one
    files = os.listdir(ob_path)
    filename = files[0]
    csv_path = os.path.join(ob_path, filename)
    
    np_pc = np.loadtxt(csv_path, delimiter=',')
    pc = o3.PointCloud()
    pc.points = o3.Vector3dVector(np_pc)


    o3.draw_geometries([pc])


    pass


if __name__=='__main__':
    main()
