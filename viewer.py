import chainer
import open3d as o3
import numpy as np

import argparse
import sys
import os
import time

from getter_models import coco_label_colors
from utils import poses_masks_from_npz


dir_path = os.path.dirname(os.path.realpath(__file__))
abs_op_lib = os.path.join(dir_path, 'openpose')
from openpose import params, JointType


class Joint:

        def __init__(self, index, coord):
            '''
            Init with
            index: joint index
            coord: joint's camera coordinate
            '''
            self.point = coord
            self.index = index
            self.name = JointType(index).name


class Joints:

    def __init__(self, raw_jnts):
        assert len(raw_jnts) == 18, "Not enough points to make Joints."
        self.joints = {}
        nan = np.nan
        for i, jointType in enumerate(JointType):
            if (raw_jnts[i] == [nan, nan, nan]).all():
                # create a zero vector for place holder
                self.joints[jointType.name] = np.zeros(3)
            else:
                joint = Joint(i, raw_jnts[i])
                self.joints[jointType.name] = joint.point

    def get_array(self):
        arr = []
        for k, v in self.joints.items():
            if np.all(v != 0):
                arr.append(v)
        return np.asarray(arr)

    def get_array_of_joint(self, joint):
        if np.all(self.joints[joint] != 0):
            return np.asarray(self.joints[joint])

    def to_pointcloud(self):
        pc = o3.PointCloud()
        pc.points = o3.Vector3dVector(np.array(self.get_array()))
        return pc

    def normalize(self, v):
        '''
        Normalize a vector
        '''
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def get_rotation(self, a, b):
        '''
        Calculate rotation matrix from two 3D unit vectors
        - maps vector 'a' onto vector 'b'

        FIXME: when the two vectors are parallel
        '''
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        
        V = np.cross(a, b)
        s = np.linalg.norm(V)
        c = np.dot(a, b)
        I = np.array([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]
                     ])
        Vx = np.array([[0, -V[2], V[1]], 
                      [V[2], 0, -V[0]], 
                      [-V[1], V[0], 0]
                      ])
        R = I + Vx + np.matmul(Vx, Vx) * (1 / (1 + c))
        return R
    
    def create_skeleton_geometry(self):
        '''
        Create human skeleton geometry
        '''
        geometries = []
        
        joint_colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
            [255, 0, 255], [255, 0, 170], [255, 0, 85]
        ]

        limb_colors = [
            [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
            [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0.],
            [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
            [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
        ]

        for i, (jointType, color) in enumerate(zip(JointType, joint_colors)):
            if np.all(self.joints[jointType.name] != 0):
                sphere = o3.create_mesh_sphere(radius = 10.0)
                pos = np.concatenate([np.asarray(self.joints[jointType.name]), [1]])
                # create translation matrix
                Tm = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], pos]).T
                # move sphere
                sphere.transform(Tm)
                # paint sphere
                sphere.paint_uniform_color([v / 255 for v in color])

                geometries.append(sphere)

        for i, (limb, color) in enumerate(zip(params['limbs_point'], limb_colors)):
            if i != 9 and i != 13:  # don't show ear-shoulder connection
                l1 = limb[0].name
                l2 = limb[1].name
                pl1 = self.joints[l1]
                pl2 = self.joints[l2]

                if np.any(pl1) and np.any(pl2):
                    dist = np.linalg.norm(pl1 - pl2)
                    midpoint = np.concatenate([(pl1 + pl2) / 2, [1]])

                    # orientation of cylindar (z axis)
                    vec_cylindar = np.array([0, 0, 1])
                    # normalized vector of the two points connected
                    norm = self.normalize(pl2 - pl1)
                    
                    # get rotation matrix
                    R = self.get_rotation(vec_cylindar, norm).T
                    
                    # create translation matrix
                    tm1 = np.concatenate([R[0], [0]])
                    tm2 = np.concatenate([R[1], [0]])
                    tm3 = np.concatenate([R[2], [0]])
                    Tm = np.array([tm1, tm2, tm3, midpoint]).T

                    # create the cylinder
                    cylinder = o3.create_mesh_cylinder(radius = 5.0, height = dist)
                    # move the cylinder
                    cylinder.transform(Tm)
                    # paint the cylinder
                    cylinder.paint_uniform_color([v / 255 for v in color])
                    
                    geometries.append(cylinder)
        
        return geometries


class CustomVisualizer:

    def __init__(self, base):
        self.base = base
        
    def intialize_visualizer(self):
        '''
        Function to add geometry (cannot destroy)
        '''
        
        h, w = self._get_window_size()
        self.vis = o3.Visualizer()
        self.vis.create_window('pose', width=int(w), height=int(h))
        self.vis.add_geometry(self.base)

        self.render_option = self.vis.get_render_option().load_from_json(
            "static_data/renderoption.json")
        
        self.trajectory = o3.read_pinhole_camera_trajectory("static_data/pinholeCameraTrajectory.json")
        self.custom_view()
        self.vis.update_renderer()
        self.vis.run()

    def _get_window_size(self):
        intrinsics = o3.read_pinhole_camera_intrinsic("static_data/pinholeCameraIntrinsic.json")
        h = intrinsics.height
        w = intrinsics.width
        return h, w

    def update_geometry(self, pcd):
        
        for p in pcd:
            self.vis.add_geometry(p)
        
        self.vis.update_geometry()
        
        self.vis.reset_view_point(False)
        self.custom_view()
        self.vis.poll_events()
        self.vis.update_renderer()

        for p in pcd:
            p.clear()
    
    def custom_view(self):
        ctr = self.vis.get_view_control()
        intrinsic = self.trajectory.intrinsic
        extrinsic = self.trajectory.extrinsic
        ctr.convert_from_pinhole_camera_parameters(intrinsic, np.asarray(extrinsic)[0])


# ROOT: /mnt/extHDD/save_data/

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Getter')
    parser.add_argument('--data', default= '/mnt/extHDD/save_data/20180913_1908/',help='relative data path from where you use this program')
    parser.add_argument('--static', default='static_data', help='static data location')
    args = parser.parse_args()

    # get directory of data (rgb, depth)
    data_path = os.path.join(dir_path, args.data)
    static_path = os.path.join(args.static)
    assert os.path.exists(data_path), "Could not find data directory in the path: {}".format(data_path)
    assert os.path.exists(static_path), "Could not find static data directory in the path: {}".format(static_path)
    print('Getting data from: {}'.format(data_path))

    # Load room
    room_ply = os.path.join(static_path, 'room_A.ply')
    pc_room = o3.read_point_cloud(room_ply)

    # initialize visualizer
    vis = CustomVisualizer(pc_room)
    vis.intialize_visualizer()

    files = os.listdir(data_path)
    filenames = sorted(files, key=lambda f: int(''.join(filter(str.isdigit, f))))

    for fn in filenames:  #FIXME: didn't sort by number, but name
        print(fn)
        file_path = os.path.join(data_path, fn)
        poses, masks = poses_masks_from_npz(file_path)

        if not (poses is None):
            pose_ids = list(poses.keys())
            print("number of poses: ", len(pose_ids))

            joints = Joints(poses[pose_ids[0]])
            # get skeleton geometries
            
            pc_joints = joints.create_skeleton_geometry()

            pcds = []
            if not (masks is None):
                mask_ids = list(masks.keys())
                for id in mask_ids:
                    pcd = o3.PointCloud()
                    np_arr = masks[id]
                    pcd.points = o3.Vector3dVector(np_arr)
                    # pcd.points = o3.Vector3dVector(np.concatenate(mask_pcs, axis=0))

                    i = int(id.split('_')[0])
                    color = coco_label_colors[i]
                    pcd.paint_uniform_color(color)
                    pcds.append(pcd)


            pcds += pc_joints

            vis.update_geometry(pcds)
            
            time.sleep(0.025)
