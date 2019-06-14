# Daily Thought (2019.6.1 - 2019.6.15)
**Do More Thinking!** ♈ 

**Ask More Questions!** ♑

**Nothing But the Intuition!** ♐

### 1. 论文 Deep Exemplar-based Colorization 基于参考样例来上色

**对于上色问题往往有一个问题就是，一般情况设计L1 loss因为会尝试去取一个中和的值，所以出来的颜色比较灰，暗淡**

![](__pics/colorize_1.png)

**网络结构**

![](__pics/colorize_2.png)

这里的参考图片保证与目标语义具有相关性

这里是很困难的去衡量reference与target，尤其是鉴于reference是彩色图，target是灰度图，为了解决这个问题使用了`gray-VGG-19`一个训练在图像分类任务的网络，这里只使用`luminance channel`去extract它们的特征，计算它们的特征之间的差异性

**对于上色问题，一般不是使用RGB颜色空间作为输入，而是使用CIE Lab颜色空间**

因为它是 **perceptually linear**

其次可以被分成为 a luminance channel L and two chrominance channels a and b

**输入维度**

- 灰度target image：H × W × 1
- color reference image： H x W x 3
- 双向映射函数: The bidirectional mapping function is a `spatial warping function` defined with bidirectional correspondences. It returns the transformed pixel location given a source location ”p”. The two functions are respectively denoted as φT->R (mapping pixels from T to R) and φR->T (mapping pixels from R to T )

有两个子网络：

- **The Similarity sub-net** computes the semantic similarities between the reference and the target, and outputs bidirectional similarity maps `simT<->R`
- **The Colorization sub-net** takes `simT<->R`, `T_L` and `R_ab` as the input, and outputs the predicted ab channels of the target `P_ab` 维度H×W×2, which are then combined with TL to get the colorized result `P_Lab` (P_L = T_L)

**similarity子网络**

双向相似maps计算方法：

![](__pics/colorize_3.png)

前向相似性map `simT->R`反映了从T_L 到 R_L的匹配置信度

反向相似性map `simR->T`衡量了反向的匹配精确度

**colorization子网络**

![](__pics/colorize_4.png)
