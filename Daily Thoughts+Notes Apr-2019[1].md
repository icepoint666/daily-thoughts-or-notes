# Daily Thought (2019.4.1 - 2019.4.10)
**Do More Thinking!** ♈ 

**Ask More Questions!** ♑

**Nothing But the Intuition!** ♐
## 图像翻译（Image-to-image Translation）生成高质量图像策略
就是指那种不用调参就可以得到不错效果的实现

### 1. 来源于pix2pixHD的方法
**采用`multi-scale`的`Discriminator`以及`coarse-to-fine`的`Generator`

所谓multi-scale的Discriminator是指多个D，分别判别不同分辨率的真假图像。比如采用3个scale的判别器，分别判别256x256，128x128，64x64分辨率的图像。

至于如何获得不同分辨率的图像，不同分辨率的GT，直接经过pooling下采样即可。

Coarse2fine的Generator是指先训练一个低分辨率的网络，训好了再接一个高分辨率的网络，高分辨率网络融合低分辨率网络的特征得到更精细的生成结果。

具体介绍可以参考pix2pixHD

### 2. progressive growing的训练方式

先训小分辨率，再逐渐增加网络层数以增大分辨率，这个跟coarse2fine有点像。

### 3. 关于生成样本多样性

### 4. checkerboard现象
