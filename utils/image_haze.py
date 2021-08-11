import os
import cv2 as cv
import numpy as np

def image_haze(clear_image_dir:str, save_haze_dir:str, A:int=255, t:float=0.5):
    clears_image = os.listdir(clear_image_dir)
    if not os.path.exists(save_haze_dir):
        os.makedirs(save_haze_dir)
    for clear in clears_image:
        clear_image_full_path = os.path.join(clear_image_dir, clear)
        image = cv.imread(clear_image_full_path)
        haze_image = image * t + A*(1-t)
        cv.imwrite(os.path.join(save_haze_dir, clear), haze_image)

if __name__ == "__main__":
    image_haze("D:\Dataset\Geographic image\clear", "D:\Dataset\Geographic image\haze")
    pass