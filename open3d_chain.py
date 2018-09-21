import open3d as o3
import numpy as np
import os


#FIXME: change name to Open3DChai
class Open3D_Chain:
    '''
    Open3D Chain for easy rendering
    '''
    def __init__(self, pname='T.csv'):
        self.camera_intrinsic = o3.read_pinhole_camera_intrinsic("static_data/realsense_intrinsic.json")
        self.K = np.asarray(self.camera_intrinsic.intrinsic_matrix)
        P_matrix_filename = os.path.join('static_data', pname)
        self.P = np.loadtxt(P_matrix_filename, delimiter=',')

    def change_image(self, rgb_path, depths_path):
        assert os.path.exists(rgb_path), 'Could not find corresponding rgb image in: {}'.format(rgb_path)
        assert os.path.exists(depths_path), 'Could not find corresponding depth image in: {}'.format(depths_path)
        self.rgb =  self.read_image(rgb_path)
        self.depths =  self.read_image(depths_path)
    
    def create_rgbd(self):
        self.rgbd = o3.create_rgbd_image_from_color_and_depth(self.rgb, self.depths)

    def read_image(self, path):
        return o3.read_image(path)
    
    def get_rgb(self, rgbd=False):
        if rgbd:
            # this returns grayscale
            #FIXME: separate to grayscale
            return self.rgbd.color
        else:
            return np.asarray(self.rgb)

    def get_grayscale(self):
        if self.rgbd:
            return np.asarray(self.rgbd.color)
        else:
            print("No RGB-D created!")

    def get_depths(self, rgbd=False):
        if rgbd:
            return self.rgbd.depth
        else:
            return np.asarray(self.depths)
    
    def get_pcd(self):
        pcd = o3.create_point_cloud_from_rgbd_image(self.rgbd, self.camera_intrinsic)
        # create_point_cloud_from_rgbd_image makes it to be 1cm instead of 1mm and is flipped in the x axis
        pcd.transform(
            [[1000, 0, 0, 0], 
             [0, -1000, 0, 0], 
             [0, 0, -1000, 0], 
             [0, 0, 0, 1]])
        return pcd

    def get_K(self):
        return self.K

    def get_P(self):
        return self.P

    def save_pcd(self):
        pcd = self.get_pcd()
        o3.write_point_cloud('static_data/camera_pc.ply', pcd)
    
    def calc_xy(self, x, y, z=None):
        '''
        K: intrinsic matrix
        x: pixel value x
        y: pixel value y
        z: mm value of z
        '''
        if not z:
            z = self.get_depths()[y][x]

        fx = self.K[0][0]
        fy = self.K[1][1]
        u0 = self.K[0][2]
        v0 = self.K[1][2]

        _x = (x - u0) * z / fx
        _y = (y - v0) * z / fy
        return _x, _y

    def convert2world(self, coord):
        '''
        Convert from Camera coordniate to World coordinate (according to P)
        '''
        _coord = np.concatenate([np.asarray(coord), [1.000]])
        # For visualization, flip it in x-axis
        rotate = np.array([[1,0,0,0], [0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        n = rotate.dot(_coord)
        return self.P.dot(n)[:3]

    def compare_with_room(self):
        pc_room = o3.read_point_cloud('static_data/room_A.ply')
        pcd = self.get_pcd()

        P = np.loadtxt('static_data/T.csv', delimiter=',')
        pcd.transform(P)

        o3.draw_geometries([pc_room, pcd])

    def downsample_nparray(self, arr, ratio=1):  # arr = [[x, y, z], [...] ]
        pcd = o3.PointCloud()
        pcd.points = o3.Vector3dVector(arr)
        downpcd = o3.voxel_down_sample(pcd, voxel_size = (10*ratio))  # 5 millimeters
        return np.asarray(downpcd.points)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    chain = Open3D_Chain()
    chain.change_image("static_data/rgb.png", "static_data/depth.png")

    chain.create_rgbd()
    plt.subplot(1, 2, 1)
    plt.title('Grayscale image')
    plt.imshow(chain.get_rgb(True))
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(chain.get_depths(True))
    plt.show()
    chain.save_pcd()
    chain.compare_with_room()
