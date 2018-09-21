import argparse
import sys
import os
from datetime import datetime as datetime


import matplotlib.pyplot as plot
from datetime import datetime as dt

from utils import DataManagement
from open3d_chain import Open3D_Chain



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Getter')
    parser.add_argument('--data', default='/mnt/extHDD/raw_data',help='relative data path from where you use this program')
    parser.add_argument('--save', default='objects',help='relative saving directory from where you use this program')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--modelfile', default= os.path.join(abs_maskrcnn, 'modelfiles/e2e_mask_rcnn_R-50-C4_1x_d2c.npz'))
    args = parser.parse_args()

    print('Getting data from: {}'.format(args.data))
    dm = DataManagement(args.data)
    after = dt(2018, 9, 9, 0, 0, 0)
    before = dt(2018, 9, 10, 0, 0, 0)
    datetimes = dm.get_datetimes_in(after, before)

    # setup chainer mask-rcnn
    roi_size = 14
    roi_align = True

    model = MaskRCNNResNet(n_fg_class = 80,
                        roi_size = roi_size,
                        pretrained_model = args.modelfile,
                        n_layers = 50,  # resnet 50 layers (not 101 layers)
                        roi_align = roi_align,
                        class_ids = test_class_ids)
    chainer.serializers.load_npz(args.modelfile, model)

    assert args.gpu >= 0, "You must use a GPU"
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
    bn_to_affine(model)

    # open3d chain
    o3_chain = Open3D_Chain()
    w = 1280

    for datetime in datetimes:
        print(datetime)
        # get directory of data (rgb, depth)
        save_path = dm.get_save_directory(datetime)
        save_path = os.path.join(save_path, args.save)
        if not dm.check_path_exists(save_path):
            print('Making a save directory in: {}'.format(save_path))
            os.makedirs(save_path)
        else:
            continue

        rgb_path = dm.get_rgb_path(datetime)
        depth_path = dm.get_depth_path(datetime)

        # sort rgb files before looping
        # order matters!
        filenames = dm.get_sorted_rgb_images(datetime)
        
        # Loop:
        for fn in filenames:
            if fn.endswith(".png"): 
                print(fn)
                # find the corresponding depth image
                rgb_img = os.path.join(rgb_path, fn)
                depth_img = os.path.join(depth_path, fn)
                if not os.path.exists(depth_img):
                    print('Could not find corresponding depth image in: {}'.format(depth_img))
                    continue

                # read image
                o3_chain.change_image(rgb_img, depth_img)
                rgb_frame = o3_chain.get_rgb().swapaxes(2, 1).swapaxes(1, 0)  # you need to swap the numpy array for inference
                depth_frame = o3_chain.get_depths()

                # inference
                _, _labels, _scores, _masks = model.predict([rgb_frame])
                labels, scores, masks = np.asarray(_labels[0], dtype=np.int32), _scores[0], _masks[0]

                for i, label in enumerate(labels):
                    object_save_path = os.path.join(save_path, coco_label_names[label])
                    
                    # create a directory for the object
                    if not os.path.exists(object_save_path):
                        os.makedirs(object_save_path)
                    
                    mask = masks[i]
                    score = scores[i]

                    if score < 0.70:
                        # don't get objects that's not really accurate
                        continue
                    
                    # multiply mask with depth frame
                    mask_with_depth = np.multiply(depth_frame, mask)

                    mask_flattened = mask_with_depth.flatten()
                    # Get all indicies that has depth points
                    non_zero_indicies = np.nonzero(mask_flattened)[0]

                    points = np.zeros((len(non_zero_indicies), 3))
                    for i, index in enumerate(non_zero_indicies):
                        Z = mask_flattened[index]
                        x, y = index % w, index // w

                        # get X and Y converted from pixel (x, y) using Z and intrinsic
                        X, Y = o3_chain.calc_xy(x, y, Z)
                        # print('x: {}, y: {}, depth: {}'.format(X, Y, Z))

                        # append to points
                        points[i] = np.asarray([X, Y, Z])

                    
                    csv_name = fn.split('.')[0] + '.csv'
                    csv_path = os.path.join(object_save_path, csv_name)
                    n = 0

                    while csv_name in os.listdir(object_save_path):
                    # if csv_name in os.listdir(object_save_path):
                        #FIXME: Currently, just append if there are multiple instances of objects in frame
                        # this is bad since when this code is rerun, it will keep updating the csv...
                        # print("before: ", points.shape)
                        # points = np.concatenate([points, np.loadtxt(open(csv_path, "rb"), delimiter=",")], axis=0)
                        # print("after: ", points.shape)

                        csv_name = fn.split('.')[0] + '_' + str(n) + '.csv'
                        csv_path = os.path.join(object_save_path, csv_name)
                        n += 1
                                        
                    np.savetxt(csv_path, points, delimiter=",")
