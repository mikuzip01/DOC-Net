import os
import random


def nvu_split(im_dir:str, save_path:str,train_per=0.9):
    f = open(os.path.join(save_path, 'split.csv'), mode='w')
    im_list = os.listdir(im_dir)
    im_list_len = len(im_list)
    random.shuffle(im_list)
    train_num = int(im_list_len * train_per)
    for i in range(train_num):
        s = "%s,%d\n"%(im_list[i], 0)
        f.write(s)
    for i in range(train_num, im_list_len):
        s = "%s,%d\n" % (im_list[i], 1)
        f.write(s)
    f.close()

if __name__ == '__main__':
    nvu_split("K:/Dehaze/nyu_depth_v2/rgb", "K:\Dehaze/nyu_depth_v2")