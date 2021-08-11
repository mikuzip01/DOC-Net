import os
import time

class Loger(object):
    def __init__(self, log_file_save_path: str, log_file_name: str, title: list):
        self._init_csv_file_name(log_file_name, log_file_save_path)
        self._init_title(title)

    def _init_csv_file_name(self, log_file_name, log_file_save_path):
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        log_file_name = log_file_name + '_' + cur_time + '.csv'
        self.f = open(os.path.join(log_file_save_path, log_file_name), mode='w')

    def _init_title(self, title):
        t = None
        for i in title:
            if t is None:
                t = i
            else:
                t = t + ',' + i
        t = t + '\n'
        self.f.write(t)
        self.f.flush()


    def log(self, name, ssim, psnr):
        input_str = "%s,%f,%f\n"%(name, ssim, psnr)
        self.f.write(input_str)
        self.f.flush()

    def close(self):
        self.f.close()


if __name__ == "__main__":
    loger = Loger('', 'test')
    loger.log(1)
    loger.close()