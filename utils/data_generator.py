import os
import csv
import random


def data_generator(clear_dir:str, haze_dir:str, save_path:str, file_name:str="data", train_sample_rate:float=0.5):
    train_f = open(os.path.join(save_path, "%s_train.csv"%(file_name)), "w", newline='')
    train_csv = csv.writer(train_f)
    test_f = open(os.path.join(save_path, "%s_test.csv"%(file_name)), "w", newline='')
    test_csv = csv.writer(test_f)
    image_name = os.listdir(clear_dir)

    random.shuffle(image_name)

    train_len = int(len(image_name) * train_sample_rate)
    test_len = len(image_name) - train_len

    for i in range(train_len):
        train_csv.writerow([os.path.join(haze_dir, image_name[i]), os.path.join(clear_dir, image_name[i])])
    for i in range(train_len, train_len+test_len):
        test_csv.writerow([os.path.join(haze_dir, image_name[i]), os.path.join(clear_dir, image_name[i])])

    pass


if __name__ == "__main__":
    data_generator("D:\Dataset\Geographic image\clear", "D:\Dataset\Geographic image\haze", "D:\Dataset\Geographic image")
    pass