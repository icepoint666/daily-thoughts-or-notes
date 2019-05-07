# Daily Thought (2019.5.5 - 2019.5.6)
**Do More Thinking!** ♈ 

**Ask More Questions!** ♑

**Nothing But the Intuition!** ♐

### 1. 知识蒸馏（Knowledge Distillation）

知识蒸馏是一种**模型压缩常见方法**，用于模型压缩指的是在teacher-student框架中，将复杂、学习能力强的网络学到的特征表示“知识”蒸馏出来，传递给参数量小、学习能力弱的网络。蒸馏可以提供student在one-shot label上学不到的soft label信息，这些里面包含了类别间信息，以及student小网络学不到而teacher网络可以学到的特征表示‘知识’，所以一般可以提高student网络的精度。
 
**Attention Transfer** 

传递teacher网络的attention信息给student网络。

首先，CNN的attention一般分为两种，`spatial-attention`, `channel-attention`。

本文利用的是`spatial-attention`.所谓spatial-attention即一种热力图，用来解码出输入图像空间区域对输出贡献大小。文章提出了两种可利用的spatial-attention,基于响应图的和基于梯度图的。

**spatial-attention ———— Activation-based**

基于响应图（特征图），取出CNN某层输出特征图张量A，尺寸：(C, H, W).定义一个映射F：

![](__pics/KD_1.png)

将3D张量flat成2D.这个映射的形式有三种供选择：

![](__pics/KD_2.png)

![](__pics/KD_3.png)

attention transfer的目的是将teacher网络某层的这种spatial attention map传递给student网络，让student网络相应层的spatial attention map可以模仿teacher，从而达到知识蒸馏目的。 teacher-student框架设计如下

![](__pics/KD_4.png)

**Mutual Learning**

Deep Mutual Learning VS Knowledge Distillation:
- Deep Mutual Learning(DML)与用于模型压缩的一般知识蒸馏不同的地方在于知识蒸馏是将预训练好的、不进行反向传播的“静态”teacher网络的知识单项传递给需要反向传播的"动态"student网络。
- DML是在训练过程中，一众需要反向传播的待训student网络协同学习，互相传递知识。
- 所以区别就在是否teacher、student网络都需要反向传播。

![](__pics/KD_5.png)

https://zhuanlan.zhihu.com/p/51563760

### 2. 
