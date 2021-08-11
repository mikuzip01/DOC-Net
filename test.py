import os
import torch
import torchvision
from torchvision import transforms
from dataloaders.Tester import Tester
from utils import SSIM, PSNR, tool_box, loger
from utils.remove_module import remove_module
from model.GeIDN_mk3_1 import GeIDN

def test():
    test_name = "User_test"
    G_haze2clear = GeIDN(img_ch=3,exp=3)
    ckpt = torch.load("checkpoint/best_model.ckpt")
    G_haze2clear.load_state_dict(remove_module(ckpt['G_haze2clear']))

    test_transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    save_path = os.path.join("res", test_name)
    tool_box.mkdir(save_path)
    tester = Tester(model=G_haze2clear, transform=test_transform, device="cuda:0", save_res_path=save_path, get_full_name=True)
    tester.test_user_dir("Geographic image")


if __name__ == "__main__":
    test()