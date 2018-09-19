import os
import json
import pyrealsense2 as rs
import numpy as np

import cv2

class PyRS:

    def __init__(self, w=640, h=480, depths=True, frame_rate=30):
        '''
        Initializing the Python RealSense Control Flow:
        w: Int (default = 640)
        h: Int (default = 480)
        depth: Bool (default = True)
        frame_rate: Int (default = 30)

        RGB and Depths formats are: bgr8, z16

        Note: In this class, variables should not be directly changed.
        '''
        self.depths_on = depths
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, frame_rate)
        self.intrinsic = None
        if depths:
            self._preset = 1
            # Presets:
            # 0: Custom
            # 1: Default
            # 2: Hand 
            # 3: High Accuracy
            # 4: High Density
            # 5: Medium Density
            self._config.enable_stream(rs.stream.depth, w, h, rs.format.z16, frame_rate)
        
        print("Initialized RealSense Camera\nw: {}, h: {}, depths: {}, frame_rate: {}".format(w, h, depths, frame_rate))

    def __del__(self):
        if not self._pipeline:
            self._pipeline.stop()

    ## Using `with PyRS(...) as pyrs:`:
    # https://stackoverflow.com/questions/1984325/explaining-pythons-enter-and-exit

    def __enter__(self):
        self.start_pipeline()
        print("Started pipeline")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._pipeline:
            self._pipeline.stop()
        print("Closed pipeline")
        
    ## Supporting functions:

    def __initialize_depths_sensor(self):
        '''Don\'t Call'''
        device = self._context.get_device()
        #FIXME: `device.first_depths_sensor` does not work well when multiple RS devices are connected
        self._depths_sensor = device.first_depth_sensor()
        
    def start_pipeline(self):
        '''Always call this function to start the capturing pipeline'''
        self._context = self._pipeline.start(self._config)
        if self.depths_on:
            self.__initialize_depths_sensor()
            self.set_depths_preset(self._preset)
            
    ## Depths sensor settings:
    
    def get_depths_preset(self):
        '''Return depths sensor\'s preset index'''
        return self._preset
    
    def get_depths_preset_name(self, index):
        '''Return depths sensor\'s preset name form index'''
        return self._depths_sensor.get_option_value_description(rs.option.visual_preset, index)
    
    def get_depths_visual_preset_max_range(self):
        '''Returns the depths sensor's visual preset range in Int'''
        assert self.depths_on, 'Error: Depths Sensor was not enabled at initialization (turn `depths` to `True`)'
        return self._depths_sensor.get_option_range(rs.option.visual_preset).max

    def set_depths_preset(self, index):
        '''Sets the depths sensor preset'''
        # http://intelrealsense.github.io/librealsense/doxygen/rs__option_8h.html#a8b9c011f705cfab20c7eaaa7a26040e2
        assert self._preset <= self.get_depths_visual_preset_max_range(), "Error: Desired preset exceeds range"
        self._depths_sensor.set_option(rs.option.visual_preset, index)

    ## Frames:

    def update_frames(self):
        '''Updates frames to pipeline (same as calling `_pipeline.wait_for_frames()`)'''
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        self._color_image = np.asanyarray(color_frame.get_data())
        if self.depths_on:
            depths_frame = frames.get_depth_frame()
            self._depths_image = np.asanyarray(depths_frame.get_data())

    def get_color_image(self):
        '''Returns color image as Numpy array'''
        return self._color_image
    
    def get_depths_frame(self):
        '''Returns depth image as Numpy array'''
        return self._depths_image

    def get_intrinsic(self, as_json=False, path=None):
        '''Return Camera Intrinsic as json'''
        assert self._context, "Has not started pipeline yet"
        if self.intrinsic is None:
            print("Getting Intrinsics for Camera...")
            profile = self._context.get_stream(rs.stream.depth)
            self.intrinsic = profile.as_video_stream_profile().get_intrinsics()
        intrinsic = self._intrinsic2dict(self.intrinsic)  # dict
        if as_json:
            assert path, "You must add path for saving intrinsics"
            intrinsic_as_json = json.dumps(intrinsic)
            with open(os.path.join(path, 'realsense_intrinsic.json'), 'w') as f:
                    intrinsic_as_json = json.dump(intrinsic, f, sort_keys=False,
                                                                indent=4,
                                                                ensure_ascii=False)
            return intrinsic_as_json
        return intrinsic

    def _intrinsic2dict(self, i):
        mat = [i.fx, 0, 0, 0, i.fy, 0, i.ppx, i.ppy, 1]
        return {'width': i.width, 'height': i.height, 'intrinsic_matrix': mat}
        

if __name__ == '__main__':
    with PyRS(h=720, w=1280) as pyrs:
        print('Modes:')
        print('\tSave RGB and Depths:\tp')
        print('\tChange preset:\tc')
        print('\tSave Intrinsic:\ti')
        print('\tExit:\tq')

        preset = pyrs.get_depths_preset()
        preset_name = pyrs.get_depths_preset_name(preset)
        print('Preset: ', pyrs.get_depths_preset_name(preset))
        print("Intrinsics: ", pyrs.get_intrinsic())

        while True:
            # Wait for a coherent pair of frames: depth and color
            pyrs.update_frames()

            # Get images as numpy arrays
            color_image = pyrs.get_color_image()
            depths_image = pyrs.get_depths_frame()

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depths_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depths_image, None, 0.5, 0), cv2.COLORMAP_RAINBOW)

            # Stack both images horizontally
            images = np.hstack((color_image, depths_colormap))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(images, preset_name, (60,80), font, 4,(255,255,255),2, cv2.LINE_AA)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)

            if key == ord('q'):
                # end OpenCV loop
                break
            elif key == ord('p'):
                # save rgb and depths
                cv2.imwrite("static_data/rgb.png", color_image)
                cv2.imwrite("static_data/depth.png", depths_image)
            elif key == ord('c'):
                # change preset
                preset = preset + 1
                max_ = pyrs.get_depths_visual_preset_max_range()
                preset = preset % max_
                pyrs.set_depths_preset(preset)
                preset_name = pyrs.get_depths_preset_name(preset)
            elif key == ord('i'):
                # save intrinsics
                intrinsic_as_json = pyrs.get_intrinsic(True, "./static_data/")
                


