# DOC-Net: [Remote Sensing Image Defogging Networks Based on Dual Self-Attention Boost Residual Octave Convolution](https://www.mdpi.com/2072-4292/13/16/3104/htm)
Official implementation.
# Abstruct
Remote sensing images have been widely used in military, national defense, disaster emergency response, ecological environment monitoring, among other applications. However, fog always causes definition of remote sensing images to decrease. The performance of traditional image defogging methods relies on the fog-related prior knowledge, but they cannot always accurately obtain the scene depth information used in the defogging process. Existing deep learning-based image defogging methods often perform well, but they mainly focus on defogging ordinary outdoor foggy images rather than remote sensing images. Due to the different imaging mechanisms used in ordinary outdoor images and remote sensing images, fog residue may exist in the defogged remote sensing images obtained by existing deep learning-based image defogging methods. Therefore, this paper proposes remote sensing image defogging networks based on dual self-attention boost residual octave convolution (DOC). Residual octave convolution (residual OctConv) is used to decompose a source image into high- and low-frequency components. During the extraction of feature maps, high- and low-frequency components are processed by convolution operations, respectively. The entire network structure is mainly composed of encoding and decoding stages. The feature maps of each network layer in the encoding stage are passed to the corresponding network layer in the decoding stage. The dual self-attention module is applied to the feature enhancement of the output feature maps of the encoding stage, thereby obtaining the refined feature maps. The strengthen-operate-subtract (SOS) boosted module is used to fuse the refined feature maps of each network layer with the upsampling feature maps from the corresponding decoding stage. Compared with existing image defogging methods, comparative experimental results confirm the proposed method improves both visual effects and objective indicators to varying degrees and effectively enhances the definition of foggy remote sensing images.
# Dependencies
- Python 3.6
- Pytorch 1.6.0
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

# Citation
If you use these models in your research, please cite:
```
@Article{rs13163104,
AUTHOR = {Zhu, Zhiqin and Luo, Yaqin and Qi, Guanqiu and Meng, Jun and Li, Yong and Mazur, Neal},
TITLE = {Remote Sensing Image Defogging Networks Based on Dual Self-Attention Boost Residual Octave Convolution},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {16},
ARTICLE-NUMBER = {3104},
URL = {https://www.mdpi.com/2072-4292/13/16/3104},
ISSN = {2072-4292},
DOI = {10.3390/rs13163104}
}
```
