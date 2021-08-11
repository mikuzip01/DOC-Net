import os
import torch
import torchvision
from torchvision import transforms
from dataloaders.Geo_dataloader import GeoDataset
from utils import SSIM, PSNR, tool_box, loger

path = {
    "GeoDataset": "D:\Dataset\Geographic image\data_test.csv"
}


class Tester(object):
    def __init__(self, model, transform, device, save_res_path: str, use_Normalize:bool = True, skip_exist_res_fold:bool=True, model_eval:bool=True,
                 get_full_name:bool=False, one_by_one:bool=True):
        self.skip_exist_res_fold = skip_exist_res_fold
        self.use_N = use_Normalize
        self.model_eval = model_eval
        self.get_full_name = get_full_name
        self.one_by_one = one_by_one
        self.transform = transform
        self.__init_transform()
        self.device = device
        self.__init_device()
        self.model = model
        self.__init_model()
        self.save_res_path = save_res_path
        self.test_dataset = {
            "GeoDataset": GeoDataset(data_csvpath=path["GeoDataset"], transform=self.transform, random_crop=False, random_flip=False)
        }
        self.test_dataloader = {}
        self.__init_test_dataloader()

    def __init_model(self):
        self.model = self.model.to(self.device)
        if self.model_eval:
            self.model.eval()

    def __init_transform(self):
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            ])

    def __init_device(self):
        if self.device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init_test_dataloader(self):
        for name, dataset in self.test_dataset.items():
            self.test_dataloader[name] = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    def run_test(self):
        tool_box.mkdir(self.save_res_path)
        if self.skip_exist_res_fold:
            print("skip exist res fold!!")
        self.model.eval()
        print("Start test!")
        with torch.no_grad():
            for name, dataloader in self.test_dataloader.items():
                ssim, psnr = 0.0, 0.0
                if os.path.exists(os.path.join(self.save_res_path, name)):
                    print("skip dataset %s"%(name))
                    continue
                tool_box.mkdir(os.path.join(self.save_res_path, name))
                ssim_psnr_loger = loger.Loger(log_file_save_path=os.path.join(self.save_res_path, name),
                                              log_file_name=name,
                                              title=["name", "ssim", "psnr"])
                for i, (test_data) in enumerate(dataloader):
                    if len(test_data)==3:
                        haze_img, clear_img, file_name = test_data[0], test_data[1], test_data[2][0]
                    else:
                        haze_img, clear_img = test_data[0], test_data[1]

                    haze_img = haze_img.to(self.device)
                    clear_img = clear_img.to(self.device)

                    # generate test image
                    clear_img_gen = self.model(haze_img)
                    if self.use_N:
                        # SSIM
                        temp_ssim = SSIM.ssim((clear_img_gen + 1) / 2.0, (clear_img + 1) / 2.0)
                        ssim += temp_ssim

                        # PSNR
                        temp_psnr = PSNR.psnr((clear_img_gen.cpu().detach().numpy() + 1) / 2.0,
                                               (clear_img.cpu().detach().numpy() + 1) / 2.0, PIXEL_MAX=1.0)
                        psnr += temp_psnr
                    else:
                        # SSIM
                        temp_ssim = SSIM.ssim(clear_img_gen, clear_img)
                        ssim += temp_ssim

                        # PSNR
                        temp_psnr = PSNR.psnr(clear_img_gen.cpu().detach().numpy(),
                                              clear_img.cpu().detach().numpy(), PIXEL_MAX=1.0)
                        psnr += temp_psnr

                    ssim_psnr_loger.log("%s_(%.5d).jpg"%(name, i), float(temp_ssim), float(temp_psnr))

                    # save res
                    if self.use_N:
                        ps_im = (torch.cat([haze_img, clear_img_gen, clear_img], dim=0) + 1) / 2.0
                    else:
                        ps_im = torch.cat([haze_img, clear_img_gen, clear_img], dim=0)

                    if len(test_data)==3:
                        torchvision.utils.save_image(ps_im, '%s/%s/%s_%s_(%.5d).jpg' % (
                            self.save_res_path, name, name, file_name,i), nrow=4)
                        if self.one_by_one:
                            for im_i in range(ps_im.shape[0]):
                                tool_box.mkdir('%s/%s/%d' % (self.save_res_path, name, im_i))
                                torchvision.utils.save_image(ps_im[im_i:im_i+1], '%s/%s/%d/%s_%s_(%.5d).jpg' % (
                                    self.save_res_path, name, im_i, name, file_name, i))
                    else:
                        torchvision.utils.save_image(ps_im, '%s/%s/%s_(%.5d).jpg' % (
                            self.save_res_path, name, name, i), nrow=4)
                        if self.one_by_one:
                            for im_i in range(ps_im.shape[0]):
                                tool_box.mkdir('%s/%s/%d' % (self.save_res_path, name, im_i))
                                torchvision.utils.save_image(ps_im[im_i:im_i+1], '%s/%s/%d/_%s_(%.5d).jpg' % (
                                    self.save_res_path, name, im_i, name, i))
                ssim_psnr_loger.close()
                print("%s SSIM:%f, PSNR:%f"%(name, float(ssim/len(dataloader)), float(psnr/len(dataloader))))

    def test_user_dir(self, user_dir_path:str):
        print("test user dir")
        tool_box.mkdir(self.save_res_path)
        save_img_dir = os.path.join(self.save_res_path, "user_test")
        tool_box.mkdir(save_img_dir)

        from PIL import Image
        from torchvision import transforms
        import torch.utils.data as data
        class User_test(data.Dataset):
            def __init__(self, img_dir: str, transform=None, get_full_name: bool = False):
                self.img_dir = img_dir
                self.get_full_name = get_full_name
                self.haze_img_list = os.listdir(os.path.join(img_dir, 'data'))
                self.list = []
                self.transform = transform
                self.__init_transform()
                self.__init_list()

            def __getitem__(self, item):
                haze_img = Image.open(os.path.join(self.img_dir, 'data', self.list[item][0])).convert('RGB')
                haze_img = self.transform(haze_img)
                if self.get_full_name:
                    return haze_img, self.list[item][0]
                return haze_img

            def __len__(self):
                return len(self.list)

            def __init_transform(self):
                if self.transform is None:
                    self.transform = transforms.Compose([
                        # transforms.ColorJitter(contrast=[1.5, 2]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
                    ])

            def __init_list(self):
                for file in self.haze_img_list:
                    self.list.append([file])


        test_data = User_test(img_dir=user_dir_path, transform=self.transform, get_full_name=self.get_full_name)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for i, (data) in enumerate(test_dataloader):
                # set image to device
                if len(data)==2:
                    haze_img = data[0]
                    filename = data[1]
                else:
                    haze_img = data

                haze_img = haze_img.to(self.device)

                # generate test image
                clear_img_gen = self.model(haze_img)

                if len(data)==2:
                    filename = '%s_%.5d.jpg'%(filename, i)
                else:
                    filename = '%.5d.jpg' % (i)
                if self.use_N:
                    ps_im = (torch.cat([haze_img, clear_img_gen], dim=0) + 1) / 2.0
                else:
                    ps_im = torch.cat([haze_img, clear_img_gen], dim=0)

                torchvision.utils.save_image(ps_im, os.path.join(save_img_dir, filename))
                if self.one_by_one:
                    for im_i in range(ps_im.shape[0]):
                        tool_box.mkdir('%s/%d' % (save_img_dir, im_i))
                        torchvision.utils.save_image(ps_im[im_i:im_i + 1], '%s/%d/%s_(%.5d).jpg' % (
                            save_img_dir, im_i, filename, i))



if __name__ == "__main__":
    pass