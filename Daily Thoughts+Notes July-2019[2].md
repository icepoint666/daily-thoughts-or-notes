# Daily Thought (2019.7.14 - 2019.7.24)
**Do More Thinking!** ♈ 

**Ask More Questions!** ♑

**Nothing But the Intuition!** ♐
### 1.训练GAN的一些技巧
**1. 更多的filter**：

filter数目的增加将极大增加网络的参数量，通常来说，filter的数目越多越好

当使用很少的filter时，尤其生成器包含很少的filter时，生成的图像会特别模糊。所以，更多的filter能够获取更多的信息，并最终保证生成的图像具有足够的清晰度。

**2. 不要early stopping**

刚刚接触GAN的训练时，一个常见的错误：当我们发现损失值不变时，或者生成的图像一直模糊时，通常会终止训练，调整模型。这个时候我们也要注意一下，GAN的训练通常非常耗时，所以有时候多等一等会有意想不到的“收获”。
值得注意的是，当判别器的损失值快速接近0时，通常生成器很难学到任何东西了，就需要及时终止训练，修改网络、重新训练。

https://zhuanlan.zhihu.com/p/74663048

### 2.SANet (ECCV2018, Crowd Counting)
**Scale Aggregation Network for Accurate and Efficient Crowd Counting**

论文关键点：

- **multi-scale representation** is of great value for crowd counting task
- the crowd density estimation based approaches aim to **incorporate the spatial information** of crowd images.

说出了以往的crowd counting工作，核心就是**multi-scale的特征表示**，**以及空间信息的整合**

**以往工作的缺陷：**

- Multi-column的网络缺陷是scale的diversity完全是由column的数目决定

- 大多数工作的loss就是只有pixel-wise的Eculidean loss，但是生成的density map肯定不是每个pixel都是独立的，所以必然会导致模糊的情况，在图像生成领域只用这样一个loss肯定不太科学

- 有一些解决办法，就是附加一个adversarial loss来防止模糊，但是这样带来的问题就是会引入high-level信息，而且也会增加很多discriminator的计算量

**SANet主要是受到inception modules的启发**

利用scale aggregation modules in encoder去提升the representation ability and scale diversity of features

loss函数：
- Euclidean loss
- local pattern consistency loss （to exploit the local correlation in density maps）

**SANet的insights**

- multi-scale feature representations
- high-resolution density maps

结构：

![](__pics/sanet.png)

使用了3个pooling层，降采样到原图的1/8大小

**提出的局部SSIM loss**

对生成的density map与ground truth都用11x11的gaussian kernel filter处理，得到的两个图像的均值，方差，参与计算SSIM，把这些局部的SSIM相加求平均，SSIM的值范围属于(-1，1)，local pattern consistency loss具体公式：

![](__pics/sanet_2.png)

![](__pics/sanet_3.png)

