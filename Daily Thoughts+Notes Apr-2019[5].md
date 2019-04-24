# Daily Thought (2019.4.22 - 2019.4.24)
**Do More Thinking!** ♈ 

**Ask More Questions!** ♑

**Nothing But the Intuition!** ♐

## 关于 Disentangled image generation 论文汇总（将输入信息分解去纠缠的图像生成）
### 1. InfoGAN: Interpretable Representation Learning byInformation Maximizing Generative Adversarial Nets (NIPS 2016)

InfoGAN能够以一种无监督的方式去学习disentangled representations,主要是通过encoder结构以及互信息的loss

**motivation:**

在标准的GAN中，生成数据的来源一般是一段连续单一的噪声z，这样带来的一个问题是，Generator往往会将z高度耦合处理，我们无法通过控制z的某些维度来控制生成数据的语义特征，也就是说，z是不可解释的。比如，假设我们打算生成像MNIST那样的手写数字图像，每个手写数字可以分解成多个维度特征：代表的数字、倾斜度、粗细度等等，在标准GAN的框架下，我们无法在上述维度上具体指定Generator生成什么样的手写数字。
为了解决这一问题，文章对GAN的目标函数进行了一些小小的改进，成功让网络学习到了可解释的特征表示（即论文题目中的interpretable representation）。

**latent code**

既然原始的噪声是杂乱无章的，那就人为地加上一些限制，于是作者把原来的噪声输入分解成两部分：一是原来的z；二是由若干个latent variables拼接而成的latent code c，这些latent variables会有一个先验的概率分布，且可以是离散的或连续的，用于代表生成数据的不同特征维度，比如MNIST实验的latent variables就可以由一个取值范围为0-9的离散随机变量（用于表示数字）和两个连续的随机变量（分别用于表示倾斜度和粗细度）构成。
但仅有这个设定还不够，因为GAN中Generator的学习具有很高的自由度，它很容易找到一个解，使得 P_G(x|c) = P_G(x)

从而使c完全不起作用

latent code可以自己选择维度分布，例如：c_i ∼ Unif(−1, 1) with 1 ≤ i ≤ 5.

**mutual information**

![](__pics/infoGAN_1.png)

**Variational Mutual Information Maximization**

实际上`I(c; G(z, c))`很难直接最大化，因为需要posterior `P(c|x)`,幸运的是我们可以获取一个lower bound，定义一个辅助的distribution `Q(c|x)`去近似`P(c|x)`

![](__pics/infoGAN_2.png)

**loss function**

![](__pics/infoGAN_3.png)

前面一部分标准GAN的loss

![](__pics/infoGAN_4.png)

后面一部分根据引理推导

![](__pics/infoGAN_5.png)

（这里Q相当于encoder模型，D是discriminator，G是generator）

https://www.jianshu.com/p/1b84adec15e7

### 2. Disentangled Person Image Generation (CVPR 2018）

本身目的合成一个person image

将一个person image看成是由 `background features`, `foreground featrues`, `pose features`组合而成

**本文的目标是将person images中的外观因素(Appearance factors)与结构因素(structure factors)去纠缠**

类似于将群体个体与背景去纠缠的任务

**实现方法**

输入一张condition image，另一张目标pose, 然后生成目标图像

![](__pics/person_generator_1.png)

分为两个阶段
- stage1: multi-branched reconstruction architencture，将几部分去纠缠，采用一种分治策略，将去纠缠的部分encode到embedding vector中，然后concat到一起，之后把输入图像恢复，而且按照encode进去的feature恢复
- stage2: 将这些feature看作是real的，去对抗性的学习一个mapping functions，能够从 gaussian distribution 映射到 embedding feature distribution.

![](__pics/person_generator_2.png)

**Stage-I : Disentangled image reconstruction**

![](__pics/person_generator_3.png)

先通过之前文章的方法Pose guided person image generation，去生成 `pose heatmap` 以及与pose有关的 `mask`.

`foreground branch`

输入的不是图像，而是加了coarse pose mask的feature maps

为了从pose信息中更好的将fore ground去纠缠，encode pose invariant features with 7 Body Regions-Of-Interest instead of the whole image

对于每一个ROI，extract feature maps然后resize到48x48

之后放入一个weight sharing的encoder

最终得到7个body ROI embedding 然后 concatenate 成为一个 224维特征向量 （每个之前是32维）

`background branch`

应用 inverse pose mask 到原来的 feature map 去得到 background feature map

然后放进background encoder中，得到128维向量

最后将128维，224维向量concatenate在一起，然后tile成为 128x64x352维的 appearance feature map

`pose branch`

将pose keypoints的 map 经过卷积得到 18 channels 的 feature maps

然后与之前的 352 channels的 appearance feature maps concatenate在一起，通过一个"U-net"的结构（convolutional autoencoder with skip connections）（结构来自于 Pose guided person image generation. In NIPS 2017）然后生成一个重构的图片

这样的限制会迫使网络去学习怎样使用 pose structure 信息去选择每个像素有用的外观信息。

因为pose需要一个embedding用在第二阶段，所以pose branch还有一个encoder + decoder结构

![](__pics/person_generater_4.png)

这里使用fully-connected networks（全连接）去重建pose information，所以我们能够decode embedded pose feature to obtain the heatmaps

这里因为一些body regions处于被遮挡的情况，是看不到的，所以在引入一个visibility variable，表示每个pose keypoint的visibility state

The pose information can be represented by a 54-dim vector (36-dim keypoint coordinates γ and 18-dim keypoint visibility α).

然后经过全连接的encoder得到32-dim的向量，再经过全连接decoder得到重构的pose information

**Stage-II: Embedding feature mapping**

![](__pics/person_generater_5.png)

图像可以被represented成为一个low-dimensional，continuous的feature embedding space

这些低维manifold的feature embedding space更加容易学习

本文不是直接学习guassian noise vector 映射到 image

而是先学习一个mapping function能够将 `a Gaussian space Z` 映射到 `continuous feature embedding space E`

再使用 stage-I 预训练的decoder去将 `continuous feature embedding space E` 映射到 `real image space X`

对抗策略：

把 features mapped from Gaussian noise z 视作为fake的embedding features

对抗性的学习mapping function，这样就可以sample fake embedding features from noise，然后可以map成为图片，利用stage-I的方法

**理解**

对于这种image generation任务，把condition信息编码进入低维vector是非常有效的，然后再通过decoder生成图像。

想通过pix2pix解决类似问题是不现实的，因为降采样次数不够，pix2pix的降采样只可以应对low-level级别的image generation

而且pix2pix没有引入噪声，会使GAN很难模拟正确分布。

对于pix2pix的discriminator输入是condition + image，discriminator很难学习到两者的关联因为很浅层，所以效果不好。

**网络架构**

stage-I的架构：encoder, auto-decoder

![](__pics/person_generater_7.png)

![](__pics/person_generater_8.png)

(里面的feature之间都是经过的convolutional residual blocks)

stage-II的架构：mapping function

![](__pics/person_generater_6.png)

**loss function**

stage-I, we use both `L1` and `adversarial` loss to optimize the image (i.e. foreground and background) reconstruction network生成更加逼真的图片

stage-I的`generator`与`discriminator`的loss:

![](__pics/person_generater_9.png)

`pose reconstruction`的loss：

![](__pics/person_generater_10.png)

使用WGAN loss去优化第二阶段的全连接mapping的loss

![](__pics/person_generater_11.png)

**实现细节**

- 所有卷积kernel size都使用的是3x3
- 全连接层中间维度是512
- 激活函数ReLU
- convolutional residual blocks的数量依赖于input size的大小
- Each residual block consists of two convolution layers with stride=1, followed by one sub-sampling convolution layer with stride=2, except for the last block

对于Market1501数据集，, containing 32,668 images of 1,501 persons

All images are resized to 128×64 pixels. We use the same train/test split (12,936/19,732)

We train the foreground and background models with a mini-batch of `size 16 for ∼70k` iterations at `stage-I` 

and with a mini-batch of `size 32 for ∼30k` iterations at `stage-II`.

The pose models are trained with a mini-batch of `size 64 for ∼30k` iterations at `stage-I` 

and with a mini-batch of `size 32 for ∼60k` iterations at `stage-II`.

On both datasets, we use the Adam optimizer with weights β1 = 0.5 and β2 = 0.999. The initial learning rate is set to 2e-5. For adversarial training, we optimize the discriminator and generator alternatively.
