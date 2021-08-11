import os
import shutil
import random
from tqdm import tqdm


class its_ots_selector(object):
    def __init__(self, gt_path:str, haze_path:str, gt_save_path:str, haze_save_path:str, start_num:int, per:float=0.5):
        self.gt_path = gt_path
        self.haze_path = haze_path
        self.gt_save_path = gt_save_path
        self.haze_save_path = haze_save_path
        self.start_num = start_num
        self.per = per

    def run(self):
        gt_list = os.listdir(self.gt_path)
        haze_list = os.listdir(self.haze_path)

        gt_split = []
        haze_split = {}
        print('generate list')
        for file in gt_list:
            file_name, extentin = os.path.splitext(file)
            gt_split.append([file_name, extentin])

        for file in haze_list:
            file_name, extentin = os.path.splitext(file)
            file_split = file_name.split('_')
            if file_split[0] not in haze_split:
                haze_split[file_split[0]] = []
            haze_split[file_split[0]].append([file_split[1], file_split[2], extentin])

        # copy file
        ls = [i for i in range(len(gt_split))]
        select_list = random.sample(ls, int(len(gt_split)*self.per))
        print('start copy file')
        for slct in tqdm(select_list):
            gt_s_file_path = os.path.join(self.gt_path, '%s%s'%(gt_split[slct][0], gt_split[slct][1]))
            gt_d_file_path = os.path.join(self.gt_save_path, '%s%s'%(int(gt_split[slct][0])+self.start_num, gt_split[slct][1]))
            ch_haze = random.choice(haze_split[gt_split[slct][0]])
            haze_s_file_path = os.path.join(self.haze_path, '%s_%s_%s%s' % (gt_split[slct][0], ch_haze[0], ch_haze[1], ch_haze[2]))
            haze_d_file_path = os.path.join(self.haze_save_path, '%s%s' % (int(gt_split[slct][0])+self.start_num, ch_haze[2]))
            shutil.copy(gt_s_file_path, gt_d_file_path)
            shutil.copy(haze_s_file_path, haze_d_file_path)

    def run_mk2(self):
        gt_list = os.listdir(self.gt_path)
        haze_list = os.listdir(self.haze_path)

        gt_split = []
        haze_split = {}
        print('generate list')
        for file in gt_list:
            file_name, extentin = os.path.splitext(file)
            gt_split.append([file_name, extentin])

        for file in haze_list:
            file_name, extentin = os.path.splitext(file)
            file_split = file_name.split('_')
            if file_split[0] not in haze_split:
                haze_split[file_split[0]] = []
            haze_split[file_split[0]].append([file_split[1], extentin])

        # copy file
        print('start copy file')
        for gt in tqdm(gt_split):
            gt_s_file_path = os.path.join(self.gt_path, '%s%s' % (gt[0], gt[1]))
            gt_d_file_path = os.path.join(self.gt_save_path, '%s%s' % (int(gt[0]) + self.start_num, gt[1]))
            ch_haze = random.choice(haze_split[gt[0]])
            haze_s_file_path = os.path.join(self.haze_path,
                                            '%s_%s%s' % (gt[0], ch_haze[0], ch_haze[1]))
            haze_d_file_path = os.path.join(self.haze_save_path, '%s%s' % (int(gt[0]) + self.start_num, ch_haze[1]))
            shutil.copy(gt_s_file_path, gt_d_file_path)
            shutil.copy(haze_s_file_path, haze_d_file_path)



if __name__ == '__main__':
    select = its_ots_selector(gt_path='K:\Dehaze\OTS_ALPHA\clear\clear_images', haze_path='K:\Dehaze\OTS_ALPHA\haze\OTS',
                              gt_save_path='K:\Dehaze\OTS_ITS\gt', haze_save_path='K:\Dehaze\OTS_ITS\haze', start_num=10000, per=0.5)
    select.run()
    pass