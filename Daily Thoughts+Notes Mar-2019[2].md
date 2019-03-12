# Daily Thought (2019.3.6 - 2019.3.13)
**Do More Thinking!** ♈ 

**Ask More Questions!** ♑

**Nothing But the Intuition!** ♐

### 1.反卷积的两种理解方式
**方式1**：

![](__pics/deconv_1.jpg)

完全可以看成为，先进行比例放缩，在卷积，替代实现如下：
```python
x = torch.nn.functional.interpolate(input, scale_factor=2)
x = conv2d(x)
```

**方式2**：
反卷积又名Fractionally Strided Convolution，也就是步长为分数的卷积。

我们知道步长大于1，一般可以达到降采样的效果，那么把反卷积说成步长为分数的卷积，也就可以达到上采样的效果了。

![](__pics/deconv_2.gif)
