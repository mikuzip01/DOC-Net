import torch
import torch.nn as nn
import torch.nn.functional as F
from model.octconv import OctaveConv


class Spatial_Attention(nn.Module):
    def __init__(self, ch, alpha=0.5):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(ch, int(ch/2), kernel_size=1, stride=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(int(ch/2), 1, kernel_size=1, stride=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        down_x_l = F.interpolate(x[1],x[0].shape[2:], mode="bilinear")
        att_x = torch.cat([down_x_l, x[0]], dim=1)
        att_x = self.conv1(att_x)
        att_x = self.act1(att_x)
        att_x = self.conv2(att_x)
        att_x = self.act2(att_x)
        return x[0]*att_x, x[1]*F.interpolate(att_x, x[1].shape[2:], mode="bilinear")


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out) * 0.1)
        out = torch.add(out, residual)
        return out


class FeatureExtract(torch.nn.Module):
    def __init__(self, channels: int, oct_num: int = 3):
        super(FeatureExtract, self).__init__()
        self.channels = channels
        f_e = []
        for i in range(oct_num):
            f_e.append(OctaveConv(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,res=True))
        self.f_e = nn.Sequential(*f_e)

    def forward(self, x):
        x = self.f_e(x)
        return x


class FeatureReconstruct(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int, residual_num: int = 3):
        super(FeatureReconstruct, self).__init__()
        self.output_channels = output_channels
        self.conv = ConvLayer(input_channels, output_channels, kernel_size=3, stride=1)
        f_e = []
        for i in range(residual_num):
            f_e.append(ResidualBlock(output_channels))
        self.f_e = nn.Sequential(*f_e)

    def forward(self, x):
        x = self.conv(x)
        x = self.f_e(x)
        return x


class DownExpansionChannels(nn.Module):
    def __init__(self, input_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, alpha=0.5):
        super(DownExpansionChannels, self).__init__()
        self.convh = nn.Conv2d(int(input_channels*alpha), out_channels - int(out_channels*alpha), kernel_size, stride, padding)
        self.convl = nn.Conv2d(int(input_channels*alpha), out_channels - int(out_channels*alpha), kernel_size, stride, padding)

    def forward(self, x):
        x_h = self.convh(x[0])
        x_l = self.convl(x[1])
        return x_h, x_l



class UpReconstructChannels(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1, alpha=0.5):
        super(UpReconstructChannels, self).__init__()
        self.conv2d1 = nn.ConvTranspose2d(int(input_channels*alpha), output_channels - int(output_channels*alpha), kernel_size=kernel_size, stride=stride,
                                         padding=padding)
        self.conv2d2 = nn.ConvTranspose2d(int(input_channels*alpha), output_channels - int(output_channels*alpha), kernel_size=kernel_size, stride=stride,
                                         padding=padding)

    def forward(self, x, match_x):
        x_h = self.conv2d1(x[0])
        x_h = F.interpolate(x_h, match_x[0].shape[2:], mode="bilinear")
        x_l = self.conv2d2(x[1])
        x_l = F.interpolate(x_l, match_x[1].shape[2:], mode="bilinear")
        return x_h, x_l


class GeIDN(nn.Module):
    def __init__(self, img_ch, exp:int=1):
        super(GeIDN, self).__init__()
        # input part
        self.input_conv = ConvLayer(in_channels=img_ch, out_channels=4, kernel_size=11, stride=1)
        self.feature_extract_init = OctaveConv(in_channels=8, out_channels=8*exp, kernel_size=3, padding=1)
        self.feature_extract_00 = FeatureExtract(8*exp)

        self.U_down_exp_ch_01 = DownExpansionChannels(8*exp, 32*exp)
        self.U_down_feature_extract_01 = FeatureExtract(32*exp)


        self.U_down_exp_ch_02 = DownExpansionChannels(32*exp, 64*exp)
        self.U_down_feature_extract_02 = FeatureExtract(64*exp)


        self.U_down_exp_ch_03 = DownExpansionChannels(64*exp, 128*exp)
        self.U_down_feature_extract_03 = FeatureExtract(128*exp)


        self.U_down_exp_ch_04 = DownExpansionChannels(128*exp, 256*exp)
        self.U_down_feature_extract_04 = FeatureExtract(256*exp, oct_num=6)


        self.U_up_rec_ch_04 = UpReconstructChannels(256*exp, 128*exp)
        self.U_up_sa_03 = Spatial_Attention(128*exp)
        self.U_up_feature_extract_03 = FeatureExtract(128*exp)


        self.U_up_rec_ch_03 = UpReconstructChannels(128*exp, 64*exp)
        self.U_up_sa_02 = Spatial_Attention(64*exp)
        self.U_up_feature_extract_02 = FeatureExtract(64*exp)


        self.U_up_rec_ch_02 = UpReconstructChannels(64*exp, 32*exp)
        self.U_up_sa_01 = Spatial_Attention(32*exp)
        self.U_up_feature_extract_01 = FeatureExtract(32*exp)


        self.U_up_rec_ch_01 = UpReconstructChannels(32*exp, 8*exp)
        self.U_up_sa_out = Spatial_Attention(8*exp)
        self.U_up_feature_extract_out = FeatureExtract(8*exp)

        # output part
        self.out_conv = ConvLayer(in_channels=8*exp, out_channels=img_ch, kernel_size=3, stride=1)

    def octadd(self, x, y, y_lam=0.5):
        x_h = x[0] + y_lam * y[0]
        x_l = x[1] + y_lam * y[1]
        return x_h, x_l

    def octsub(self, x, y, y_lam=0.5):
        x_h = x[0] - y_lam * y[0]
        x_l = x[1] - y_lam * y[1]
        return x_h, x_l


    def forward(self, x: torch.Tensor):
        input_image = x
        image_size = input_image.shape[2:]
        # input part
        x = self.input_conv(x)
        x = self.feature_extract_init(x)
        x = self.feature_extract_00(x)
        # U Hat net
        u_x = self.U_down_exp_ch_01(x)
        u_x_down_01 = self.U_down_feature_extract_01(u_x)

        u_x_down_02 = self.U_down_exp_ch_02(u_x_down_01)
        u_x_down_02 = self.U_down_feature_extract_02(u_x_down_02)

        u_x_down_03 = self.U_down_exp_ch_03(u_x_down_02)
        u_x_down_03 = self.U_down_feature_extract_03(u_x_down_03)

        u_x_down_04 = self.U_down_exp_ch_04(u_x_down_03)
        u_x_down_04 = self.U_down_feature_extract_04(u_x_down_04)

        u_x_up_03 = self.U_up_rec_ch_04(u_x_down_04, u_x_down_03)
        u_x_up_03 = self.U_up_sa_03(u_x_up_03)
        u_x_up_03 = self.octadd(self.U_up_feature_extract_03(self.octsub(u_x_up_03, u_x_down_03)), u_x_up_03)

        u_x_up_02 = self.U_up_rec_ch_03(u_x_up_03, u_x_down_02)
        u_x_up_02 = self.U_up_sa_02(u_x_up_02)
        u_x_up_02 = self.octadd(self.U_up_feature_extract_02(self.octsub(u_x_up_02, u_x_down_02)), u_x_up_02)

        u_x_up_01 = self.U_up_rec_ch_02(u_x_up_02, u_x_down_01)
        u_x_up_01 = self.U_up_sa_01(u_x_up_01)
        u_x_up_01 = self.octadd(self.U_up_feature_extract_01(self.octsub(u_x_up_01, u_x_down_01)), u_x_up_01)

        u_x_out = self.U_up_rec_ch_01(u_x_up_01, x)
        u_x_out = self.U_up_sa_out(u_x_out)
        u_x_out = self.octadd(self.U_up_feature_extract_out(self.octsub(u_x_out, x)), u_x_out)

        u_x_out = torch.cat([u_x_out[0],F.interpolate(u_x_out[1],u_x_out[0].shape[2:])], dim=1)

        out = self.out_conv(u_x_out)
        return out

if __name__ == "__main__":
    geidn = GeIDN(img_ch=3, exp=3)
    feature_h = torch.randn(4,3,256,256)
    feature_l = torch.randn(4, 3, 128, 128)
    # out_h, out_l = conv((feature_h,feature_l))
    out= geidn(feature_h)
    print(out.shape)