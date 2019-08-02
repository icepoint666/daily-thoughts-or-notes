# Daily Thought
**Do More Thinking!** 🐉🐲🦕

**Ask More Questions!** 🏍🛵🚲

**Nothing But the Intuition!** 🌏🌍🌎



**有些特征，误以为DL可以学习到，但是其实网络没办法学习出来，所以如果加一些人为引导会提升效果。**

**例如：Deformable convolution, SEnet, Group convolution都是引入人为思考模式加以限制成功提升效果的案例。**

**但也有的特征，DL自身可以很好的学习出来，如果再加人为干预引导，就会画蛇添足，限制了参数优化的自由度。**

**注意 模块与模块 / 模块与loss监督 之间1+1>2的化学反应**

## Contents
**GAN**

主题 | 索引 | 备注   
-|-|-
GAN与VAE关系和区别 | 2019 - January[1] - 3 |  
Spectral Normalization | 2019 - February[2] - 17 |  
pix2pix (CVPR2017) | 2019 - February[4] - 1 |  
patchGAN | 2019 - February[4] - 3, 6 | 
GAN中的L1 loss | 2019 - February[4] - 2 | 
L1 loss 与 L2 loss区别 | 2019 - February[4] - 4 /  March[3] - 8 | 
cGAN中加z这样的随机性因子影响 | 2019 - February[4] - 5 | 
partial convolution | 2019 - March[1] - 3 |
gated convolution | 2019 - March[1] - 3 |
perceptual loss | 2019 - March[1] - 4 |
PGGAN | 2019 - March[1] - 5 ~ 7 |
Minibatch stddev 层 | 2019 - March[1] - 7 |
GAN Dissection (ICLR2019) | 2019 - March[3] - 2 |
StyleGAN | 2019 - March[4] - 1 ~ 6 |
GAN引入噪声 | 2019 - March[4] - 4， 5 |
Disentanglement理论 | 2019 - March[4] - 6 |
cGAN的mode collapse问题 (CVPR2019) | 2019 - March[6] - 1 |
GAN创造了新的信息吗 | 2019 - March[6] - 2 |
InfoGAN (NIPS 2016) | 2019 - April[5] - 1 | 互信息
Disentangled Person Image Generation (CVPR 2018）| 2019 - April[5] - 2 | 去纠缠合成
几种GAN loss简介 | 2019 - April[6] - 3 |
学习率对GAN的影响 | 2019 - April[6] - 1 |
上采样策略对GAN的影响 | 2019 - April[6] - 2 |
GAN的checkerboard效应 | 2019 - May[1] - 1 |
GAN的模型架构一览 | 2019 - May[1] - 3 |
GAN的loss实践经验 | 2019 - May[3] - 1 |
将crop小个体图片插入全景图片中 (CVPR2019) | 2019 -June[1] - 2 |

**Text-to-image**

主题 | 索引 | 备注   
-|-|-
ObjGAN (CVPR2019) | 2019 - March[3] - 1 | 

**Image-to-image**

主题 | 索引 | 备注   
-|-|-
pix2pix (CVPR2017) | 2019 - February[4] - 1 |  
SPADE (CVPR2019) | 2019 - March[7] - 1 ~ 7 |
Image-to-image Translation 生成高质量图像策略 | 2019 - April[1] - 1 |
Image-to-image translation for cross-domain disentanglement (NIPS 2018）| 2019 - April[5] - 3 |

**Image Inpainting**

主题 | 索引 | 备注   
-|-|-
FE-SNGAN | 2019 - March[1] - 1 ~ 4 | 

**网络架构**

主题 | 索引 | 备注   
-|-|-
Encoder-Decoder生成latent variable | 2019 - January[1] - 1 |  
bottleneck降维升维结构 | 2019 - January[1] - 2 |  
U-net | 2019 - January[1] - 4 | 
U-net用于医学影像 | 2019 - February[1] - 2 | 
skip-connection | 2019 - January[1] - 5 | 
Resnet残差网络理解 | 2019 - January[1] - 6 |
Resnet目的 | 2019 - January[1] - 7 |
小卷积核作用 | 2019 - January[1] - 8 |
空洞卷积 | 2019 - January[1] - 9 ~ 12 |
最佳downsampling方案 | 2019 - January[1] - 13 |
Deformable Conv v1 | 2019 - February[1] - 6 |
Deformable Conv v2 | 2019 - February[1] - 7 |
反卷积的两种理解方式 | 2019 - March[2] - 1 |
DSConv (Depthwise Separable Convolution) | 2019 - March[5] - 2 |
Xception / ResNeXt | 2019 - March[5] - 2 |
Spatial CNN | 2019 - April[1] - 2 |
Spatial FCN | 2019 - April[2] - 1 |
sub-pixel convolution | 2019 - May[1] - 2 | 用作上采样，超分
1 x 1 conv 理解 | 2019 - May[2] - 2 |

**目标检测**

主题 | 索引 | 备注   
-|-|-
Faster-RCNN | 2019 - February[1] - 3 |  
RPN网络 | 2019 - February[1] - 4 |
RoI pooling层 | 2019 - February[1] - 5 |
SPP (Spatial Pyramid Pooling)层 | 2019 - May[3] - 3 |
ROI pooling 与 ROI align | 2019 - March[5] - 1 |
guided anchoring （CVPR2019）| 2019 - March[5] - 3 |
CSP（Center and Scale Prediction）（CVPR2019）| 2019 - April[4] - 2 | point supervision

**Normalization**

主题 | 索引 | 备注   
-|-|-
Normalization相关理解 | 2019 - February[2] - 1 ~ 10, 15 |
Batch Normalization | 2019 - February[2] - 11 |
Layer Normalization | 2019 - February[2] - 12 |
Weight Normalization | 2019 - February[2] - 13 |
Cosine Normalization | 2019 - February[2] - 14 |
Batch Normalization 与 Instance Normalization对比 | 2019 - February[2] - 16 |
Local Response Normalization (LRN) | 2019 - March[2] - 2 |
Local Response Normalization (LRN) 变体 | 2019 - March[1] - 9 |
Normalization层总结与分类 | 2019 - March[7] - 2 |
sychronized batch normalization | 2019 - March[7] - 5 |

**Attention机制**

主题 | 索引 | 备注   
-|-|-
RA-CNN (CVPR2017) | 2019 - February[3] - 1 |
Multiple Granularity Descriptors for Fine-grained Categorization | 2019 - February[3] - 2 |

**Crowd Counting**

主题 | 索引 | 备注   
-|-|-
Learning from Synthetic Data for Crowd Counting (CVPR2019) | 2019 - April[3] - 1 |
ADCrowdNet (CVPR2019) | 2019 - April[3] - 2 |
Almost Unsupervised Learning for Dense Crowd Counting (AAAI2019) | 2019 - April[3] - 3 |
Point in, Box out: Beyond Counting Persons in Crowds (CVPR2019) | 2019 - April[3] - 4 / April[4] - 1| point supervision

**评价指标**

主题 | 索引 | 备注   
-|-|-
PSNR | 2019 - February[3] - 5 |
SSIM | 2019 - February[3] - 5 |
MS-SSIM | 2019 - February[3] - 5 |
Sliced Wasserstein distance （SWD）| 2019 - March[1] - 10 |
Inception Score | 2019 - March[2] - 4 |
Frechet inception distance(FID) | 2019 - March[2] - 4 |
mAP | 2019 - April[4] - 0 |

**其他**

主题 | 索引 | 备注   
-|-|-
调参过程中初始学习率的设置 | 2019 - February[1] - 1 |
FlowNet生成光流图 | 2019 - February[3] - 7 |
ColorNet数据增广新思路 | 2019 - February[3] - 4 |
权值初始化 | 2019 - March[1] - 8 |
Adam回顾 | 2019 - March[6] - 3 |
WTA(winner take all) | 2019 - April[3] - 3 |
focal loss | 2019 - April[4] - 3 | 平衡样本
工程上对细小样本loss加权从而增加效果 | 2019 - May[3] - 2 |
知识蒸馏（Knowledge Distillation）| 2019 - May[2] - 1 | 压缩模型
深度学习多个loss收敛的决定要素 | 2019 - May[2] - 3 |
参考上色问题 (siggraph 2018) | 2019 - June[1] - 1 |
