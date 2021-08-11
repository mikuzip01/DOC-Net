# DOC-Net: [Remote Sensing Image Defogging Networks Based on Dual Self-Attention Boost Residual Octave Convolution](https://www.mdpi.com/2072-4292/13/16/3104/htm)
Official implementation.
# Dependencies
- Python 3.6
- Pytorch 1.16.0
- NVIDIA GPU+CUDA
# Test
1. Download the Pretrained model from [Google Drive](https://drive.google.com/file/d/1ivzPoQyw8mykQFFuwgqLzWVst44vCmGY/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1bOEi9yeKIcc4qDPp6uyzdA) with code `vkuj`, and put the weight in the `checkpoint` folder.
2. To test the image under `Geographic image/data` folder, please execute:
    ```bash
    python test.py
    ```
    The result will be save in `res/User_test/user_test` folder.

# Sample

![](fig/('019.jpg',)_00000.jpg)

![](fig/('025.jpg',)_00001.jpg)
