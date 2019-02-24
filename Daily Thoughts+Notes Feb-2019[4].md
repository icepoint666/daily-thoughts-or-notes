# Daily Thought (2019.2.18 - 2019.2.25)
**Do More Thinking!** ♈ 

**Ask More Questions!** ♑

**Nothing But the Intuition!** ♐

## GAN专题

### 1. pix2pix (Image-to-Image Translation with Conditional Adversarial Networks) （2019.2.24）

2016年11月21日发表于arxiv 已投CVPR 2017

**根据cGAN提出可以用于Image-to-Image Translation中多个任务的通用框架**

通用框架主要的任务有：
- semantic labels <-> photo
- BW (gray) <-> color photos
- Edges <-> photos
- Sketch <-> photos
- Day <-> night

之前阅读的用于image inpainting的edge-connect，stage2是将想象的边缘图转化为彩色图从而达到修复效果，stage2的网络架构就是应用这篇里的网络结构。

**使用的是cGAN:**
- GAN: from random noise vector z to output image y, G: x → y
- cGAN: learn a mapping from observed image x and random noise vector z, to y, G : {x, z} → y

cGAN loss (D最大化，G最小化):

![](__pics/pix2pix.png)

cGAN 结构：

![](__pics/pix2pix_1.png)
 
可以拿前面的loss函数与下面的loss函数对比一下，前者在discriminartor也加了condition

![](__pics/pix2pix_2.png)

之前有的方法证明 conditioning the discriminator + generator 的效果要优于只condition generator的效果

### 2. L1/L2的pix2pix loss有什么问题呢？（pix2pix启发）

直接设置两张图片（预测与ground-truth）对应位置的pixel与pixel之间的euclidean distance 或者 曼哈顿距离为loss，会造成 **图像模糊**

因为Euclidean distance 是通过平均所有plausible outputs来最小化loss，肯定会造成blurring，极度low-level的信息

相对而言GAN损失就是一个极度high-level的语义信息，根据人为判断真假来决定的。

### 3. patchGAN （pix2pix启发）

思想最早提出是在 （ECCV2016）Precomputed real-time texture synthesis with markovian generative adversarial networks. 

叫作markovian generative adversarial networks

应用完善并且命名“patchGAN”是在（CVPR2017）Image-to-Image Translation with Conditional Adversarial Networks.

并且调查改变patch size的大小产生的影响

### 4. 单纯使用GAN损失为什么不行，L1 pix2pix loss 与 L2的pix2pix 有什么区别？（pix2pix启发）
