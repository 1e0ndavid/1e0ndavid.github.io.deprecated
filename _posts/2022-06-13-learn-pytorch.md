---
title: "深入浅出Pytorch"
style: post
categories: Pytorch
---

没有系统学习过pytorch，最近复习了tensorflow，趁下班时间把这些pytorch复习一下。

[课程链接](https://datawhalechina.github.io/thorough-pytorch/index.html)

第一章：PyTorch深度学习基础知识

这一部分的内容都很基础，原先就已经掌握了，之后会多记录一下有必要记录的点

第二章：PyTorch基础知识

2.1 张量：

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

2.2 自动求导:

接下去关于自动求导，基本都会的，Tensor数据结构是实现自动求导的基础，数学基础是多元函数求导的雅可比矩阵及复合函数的链式求导。

动态求导用动态计算图（DCG）具体实现，即张量和运算结合起来创建动态计算图，动静态图的区别主要在是否需要预先定义计算图的结构。

用x1.grad.data查看梯度，在未反向传播时值为None，通过如y.backward()，导数会累积，重复运算相同命令grad会增加，因此每次计算前要清除当前导数值避免累积，记得设置张量的requires_grad参数为True

2.3 并行计算:

接下去是并行计算，为什么？（能计算——显存占用，算得快——计算速度，效果好——大batch提升训练效果），cuda，并行的三种方法（网络结构分布到不同设备中Network Partitioning，同一层的任务发布到不同设备中Layer-wise Partitioning，不同数据发布到不同的设备中Data Parallelism）。cuDNN时用于深度神经网络的加速库，cuDNN基于cuda完成深度学习的加速。

第三章：PyTorch的主要组成模块

3.1 思考：完成深度学习的必要部分：

和机器学习类似，在完成任务时，我们都先要进行数据预处理，其中重要的步骤包括如数据格式统一和必要的数据变换，同时划分训练集和测试集。接下来选择模型，并设定损失函数和优化函数，以及对应超参，最后用模型去拟合训练集数据，并在验证集/测试集上计算模型表现。

当然深度学习也有一些特殊性，如1. 样本量大，通常需要分批batch加载 2. 逐层、模块化搭建网络 3. 多样化的损失函数和优化器设计 4. GPU的和使用 5. 以上各模块的配合。

3.2 基本配置：

主要有三部分1. 导入必要的packages 2. 配置训练过程的超参数如batch size，learning rate，max_epochs和num_works 3. 配置训练用的硬件设备

3.3 数据读入：

数据读入是通过Dataset+DataLoader的方式完成，Dataset定义好数据的格式和数据变换形式，DataLoader用iterative的方式不断读入批次数据。Dataset主要包含三个函数：
{% highlight python %}
__init__  # 用于想类中传入外部参数，同时定义样本集
__getitem__  # 用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
__len__  # 用于返回数据集的样本数
{% endhighlight %}

在构建训练和测试数据完成后，需要定义DataLoader类以在训练和测试时加载数据，以下是一点样例代码，包括了Dataset和DataLoader的创建：
{% highlight python %}
class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        label_info = pd.read_csv(info_csv)
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_file[index].strip('\n')
        raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
        label = raw_label.iloc[:,0]
        image_name = os.path.join(self.data_dir, image_name)
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_file)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)
{% endhighlight %}

3.4 模型构建：

神经网络的构造，基于nn.Module，主要有初始化函数和forward函数（backward自动实现），通过层定义+层顺序的方式构建起来。

Parameter类是Tensor的子类，如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表，当然还可以使用ParameterList和ParameterDict分别定义参数的列表和字典。一个模型的可学习参数可以通过net.parameters()返回，如params = list(net.parameters())

一个神经网络的典型训练过程如下：

1. 定义包含一些可学习参数(或者叫权重）的神经网络
2. 在输入数据集上迭代
3. 通过网络处理输入
4. 计算loss(输出和正确答案的距离）
5. 将梯度反向传播给网络的参数
6. 更新网络的权重，一般使用一个简单的规则：weight = weight - learning_rate * gradient

torch.nn只支持小批量处理（mini-batch）不支持单个样本的输入，可以通过比如input.unsqueeze(0)来拓展维度。

3.5 模型初始化：

PyTorch在torch.nn.init中为我们提供了常用的初始化方法。初始化函数多种多样，一点案例如下，也可以用如initialize_weights()的函数将各种初始化方法放在一起。

{% highlight python %}
torch.nn.init.kaiming_normal_(conv.weight.data)
conv.weight.data
torch.nn.init.constant_(linear.weight.data,0.3)
linear.weight.data
{% endhighlight %}

3.6 损失函数：

常用操作backward()，主要有以下一些损失函数：
1. 二分类交叉熵损失函数torch.nn.BCELoss，一般来说input为sigmoid激活层的输出或者softmax的输出
2. 交叉熵损失函数torch.nn.CrossEntropyLoss
3. L1损失函数torch.nn.L1Loss
4. MSE损失函数torch.nn.MSELoss，差的平方
5. 平滑L1损失函数torch.nn.SmoothL1Loss，其功能是减轻离群点带来的影响
6. 目标泊松分布的负对数似然损失torch.nn.PoissonNLLLoss
7. KL散度torch.nn.KLDivLoss
8. MarginRankingLoss
9. 多标签边界损失函数torch.nn.MultiLabelMarginLoss，对于多标签分类问题计算损失函数
10. 二分类损失函数torch.nn.SoftMarginLoss，计算二分类的logistic损失
11. 多分类的折页损失torch.nn.MultiMarginLoss，计算多分类的折页损失
12. 三元组损失torch.nn.TripletMarginLoss
13. HingeEmbeddingLoss，对输出的embedding结果做Hinge损失计算
14. 余弦相似度torch.nn.CosineEmbeddingLoss，对两个向量做余弦相似度
15. CTC损失函数，torch.nn.CTCLoss，CTC损失函数

3.7 训练和评估：

模型状态设置：model.train(), model.eval()，模型训练流程包括读取、转换、梯度清零、输入、计算损失、反向传播、参数更新，验证流程包括读取、转换、输入、计算损失、计算指标。训练和评估的代码样例如下，注意区别。

{% highlight python %}
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(label, output)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
		print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


def val(epoch):       
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
            running_accu += torch.sum(preds == label.data)
    val_loss = val_loss/len(val_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, val_loss))
{% endhighlight %}

3.8 可视化：

略

3.9 PyTorch优化器：

常见的优化器如下，这些算法都继承于Optimizer，其有三个属性，defaults存储的是优化器的超参数，state参数的缓存，param_groups管理的参数组，Optimizer还有这些方法zero_grad(), step(), add_param_group(), load_state_dict(), state_dict()
* torch.optim.ASGD
* torch.optim.Adadelta
* torch.optim.Adagrad
* torch.optim.Adam
* torch.optim.AdamW
* torch.optim.Adamax
* torch.optim.LBFGS
* torch.optim.RMSprop
* torch.optim.Rprop
* torch.optim.SGD
* torch.optim.SparseAdam

第四章：FashionMNIST

比较简单，略
