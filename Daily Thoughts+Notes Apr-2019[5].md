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

**任务描述**

输入一张condition image，另一张目标pose, 然后生成目标图像

![](__pics/person_generator_1.png)

分为两个阶段
- stage1: multi-branched reconstruction architencture，将几部分去纠缠，采用一种分治策略，将去纠缠的部分encode到embedding vector中，然后concat到一起，之后把输入图像恢复，而且按照encode进去的feature恢复
- stage2: 将这些feature看作是real的，去对抗性的学习一个mapping functions，能够从 gaussian distribution 映射到 embedding feature distribution.

![](__pics/person_generator_2.png)

**Stage-I : Disentangled image reconstruction**

![](__pics/person_generator_3.png)

先通过之前文章的方法Pose guided person image generation，去生成 `pose heatmap` 以及与pose有关的 `mask`.



