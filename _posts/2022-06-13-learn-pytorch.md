---
title: "深入浅出Pytorch"
style: post
categories: Pytorch
---

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

