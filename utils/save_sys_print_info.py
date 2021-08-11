import os
import sys
import errno
import os.path as osp
import datetime

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, logdir=None):
        fileName = datetime.datetime.now().strftime('log-' + '%Y_%m_%d_%H_%M_%S'+'.log')
        self.console = sys.stdout
        self.file = open(os.path.join(logdir, fileName), 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


if __name__ == '__main__':
    sys.stdout = Logger('')

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print("1234124")
    print("--")
    print(":;;;")
    print("")
    print("阿斯顿发11111111111111111")
    print("zzzzz")