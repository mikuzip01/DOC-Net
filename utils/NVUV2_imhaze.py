import numpy as np
import h5py
import os
import cv2 as cv
from tqdm import tqdm

class NVUV2_imhaze(object):
    def __init__(self, rgbim_path:str, depthim_path:str):
        self.rgbim_path = rgbim_path
        self.depthim_path = depthim_path
        self.rgbim_name = os.listdir(rgbim_path)
        self.rgbim_name.sort(key= lambda x:int(x[:-4]))
        self.depthim_name = os.listdir(depthim_path)
        self.depthim_name.sort(key= lambda x:int(x[:-4]))

    def haze(self, im: np.array, depth: np.array, A: int = 255, beta = 1.5):
        t = np.exp(-beta*depth/255.0)
        res = im * t + A * (1 - t)
        return res

    def output_haze_image(self, output_dir:str, test = False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, (rgb_im_name, depth_im_name) in tqdm(enumerate(zip(self.rgbim_name, self.depthim_name))):
            if rgb_im_name is not depth_im_name:
                RuntimeError('NE!')
            rgb_im_path = os.path.join(self.rgbim_path, rgb_im_name)
            depth_im_path = os.path.join(self.depthim_path, depth_im_name)

            rgbim = cv.imread(rgb_im_path)
            depthim = cv.imread(depth_im_path)
            save_name = os.path.join(output_dir, '%d.jpg'%(i))
            cv.imwrite(save_name, self.haze(rgbim, depthim))
            if test:
                break


if __name__  == '__main__':
    nv = NVUV2_imhaze("K:\Dehaze/nyu_depth_v1/rgb",
                      "K:\Dehaze/nyu_depth_v1/depth")
    nv.output_haze_image("K:\Dehaze/nyu_depth_v1\haze")