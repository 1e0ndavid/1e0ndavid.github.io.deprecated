---
title: "深入浅出Pytorch"
style: post
categories: Pytorch
---

没有系统学习过pytorch，最近复习了tensorflow，趁下班时间把这些pytorch复习一下。

第一章：PyTorch深度学习基础知识

这一部分的内容都很基础，原先就已经掌握了，之后会多记录一下有必要记录的点

第二章：PyTorch基础知识

之前有tf1的经验，其中常量的创建用tf.constant，与pytorch不同，在pytorch中，使用如torch.tensor(1.0, dtype=torch.float)的方式来创建，注意到一般dtype的与实际给的data类型应该一样，如果不一样会有warning，但是如果某个入参data的类型未知，则可以用dtype来做强制转换。

可以通过把一个用g = np.array([[1, 2, 3],[4, 5, 6]])创建的ndarray直接用h = torch.tensor(g) 或 i = torch.from_numpy(g)的方式将g变成一个tensor，也可以通过j = h.numpy()的方式将tensor转成numpy类型。

以下是常见的构造Tensor的函数：

{% highlight python %}
k = torch.rand(2, 3)
l = torch.ones(2, 3)
m = torch.zeros(2, 3)
n = torch.arange(0, 10, 2)
{% endhighlight %}

以及查看tensor的维度信息用k.shape或k.size()，以及torch.add(k, l)等，索引方式，用view改变tensor形状。

tensor的广播机制，当维度不一样的时候运算时会对齐维度，注意。

拓展与压缩tensor维度，如r = o.unsqueeze(1)会在第将torch.Size([2, 3])的张量变成torch.Size([2, 1, 3])的。此时，假设我们使用r.squeeze(0)去压缩维度，其Size并不会变化因为该处维度并不是1，而用r.squeeze(1)则确实可以实现想要的效果。

接下去关于自动求导，基本都会的，Tensor数据结构是实现自动求导的基础，数学基础是多元函数求导的雅可比矩阵及复合函数的链式求导。

动态求导用动态计算图（DCG）具体实现，即张量和运算结合起来创建动态计算图，动静态图的区别主要在是否需要预先定义计算图的结构。

用x1.grad.data查看梯度，在未反向传播时值为None，通过如y.backward()，导数会累积，重复运算相同命令grad会增加，因此每次计算前要清除当前导数值避免累积，记得设置张量的requires_grad参数为True

接下去是并行计算，为什么？（能计算——显存占用，算得快——计算速度，效果好——大batch提升训练效果），cuda，并行的三种方法（网络结构分布到不同设备中Network Partitioning，同一层的任务发布到不同设备中Layer-wise Partitioning，不同数据发布到不同的设备中Data Parallelism）。cuDNN时用于深度神经网络的加速库，cuDNN基于cuda完成深度学习的加速。
