# Daily Thought (2019.4.6 - 2019.4.8)
**Do More Thinking!** ♈ 

**Ask More Questions!** ♑

**Nothing But the Intuition!** ♐

## Crowd Counting 方向 state-of-the-art总结
### 论文：Learning from Synthetic Data for Crowd Counting in the Wild (CVPR2019)

**主要贡献** 创造了新的数据集，这个数据集是在GTA5上制作的，经过domain adaption将虚拟风格的图片转换成现实风格的图片

数据集概况简介：

![](__pics/crowd_counting_1.png)

GTA5 Crowd Counting(GCC) 数据集概况：

![](__pics/crowd_counting_2.png)

**解决方法** 使用一种`Spatial Fully Convolutional Networks`

![](__pics/crowd_counting_3.png)

**SFCN**

FCN就是用于关注 pixel-wise task (such as semantic segmentation,  saliency detection）

这里的`Spatial CNN`来自于 **Spatial As Deep: Spatial CNN for Traffic Scene Understanding** `AAAI2018`

For encoding the context information, 提出了 a spatial encoder via a sequence of convolution on the four directions (down, up,left-to-right and right-to-left).

Spatial CNN(CNN),它将传统的卷积层接层(layer-by-layer)的连接形式的转为feature map中片连片卷积(slice-by-slice)的形式，使得图中像素行和列之间能够传递信息。这特别适用于检测长距离连续形状的目标或大型目标，有着极强的空间关系但是外观线索较差的目标，例如交通线，电线杆和墙.

CNN将视觉理解推向了一个新的高度。但是这依然不能很好地处理外形线索不多的有强结构先验的目标，而人类可以推断它们的位置并填充遮挡的部分,为了解决这个问题，论文提出了SCNN，将深度卷积神经网络推广到丰富空间层次。

传统的CNN，任意层接收上层的数据作输入，再作卷积并加激活传给下一层，这个过程是顺序执行的。与之类似的是，SCNN将feature map的行或列也看成layer，也使用卷积加非线性激活，从而实现空间上的深度神经网络。这使得空间信息能够在同层的神经元上传播，增强空间信息进而对于识别结构化对象特别有效。

**Spatial CNN 与空间建模**
