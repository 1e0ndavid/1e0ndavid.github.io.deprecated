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

第四章：PyTorch基础实战

比较简单，我也不做CV，略

第五章：PyTorch模型定义

5.1 PyTorch模型定义的方式

Module类是torch.nn模块里提供的一个模型构造类(nn.Module)，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型。PyTorch模型定义应包括两个主要部分：各个部分的初始化`(__init__)`；数据流向定义(forward)，基于nn.Module，我们可以通过Sequential, ModuleList和ModuleDict三种方式定义PyTorch模型。

Sequential，对应模块为nn.Sequential()，当模型的前向计算为简单串联各个层的计算时，Sequential类可以通过更简单的方式定义模型。它可以接收一个子模块的有序字典(OrderedDict)或者一系列字模块作为参数来逐一添加Module的实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。使用Sequential定义模型的好处在于简单、易读，同时使用Sequential定义的模型不需要再写forward，因为顺序已经定义好了。但缺点是会使得模型定义失去灵活性，比如需要在模型中间加入一个外部输入时就不适合用Sequential的方式实现。结合Sequential和定义方式加以理解：

{% highlight python %}
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            input = module(input)
        return input
{% endhighlight %}

如下是两种用Sequential来定义模型的例子：

{% highlight python %}
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
print(net)

import collections
import torch.nn as nn
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
print(net2)
{% endhighlight %}

ModuleList，对应模块为nn.ModuleList()，其接收一个子模块（或层，需属于nn.Module类）的列表作为输入，也可以类似list那样进行append和extend操作。同时，子模块或层的权重也会自动添加到网络中来。

{% highlight python %}
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
{% endhighlight %}

要特别注意的是，nn.ModuleList并没有定义一个网络，它只是将不同的模块储存在一起。ModuleList中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序才算完成了模型的定义。具体实现时用for循环即可完成。

{% highlight python %}
class model(nn.Module):
  def __init__(self, ...):
    super().__init__()
    self.modulelist = ...
    ...
    
  def forward(self, x):
    for layer in self.modulelist:
      x = layer(x)
    return x
{% endhighlight %}

ModuleDict，对应模块为nn.ModuleDict()，其与ModuleList作用类似，只是ModuleDict能更方便地为神经网络的层添加名称。
{% highlight python %}
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
{% endhighlight %}

Sequential适用于快速验证结果，因为已经明确了要用哪些层，总结写一下就好了，不需要同时写`__init__`和forward。ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时非常方便实现，可以一行顶多行。当我们需要之前层的信息的时候，比如ResNets中的残差计算，当千层的结果和之前层中的结果进行融合，一般使用ModuleList/ModuleDict比较方便。

5.2 利用模型快速搭建复杂网络

一个U-Net的例子，比较常规，没什么要特别注意的点。

5.3 PyTorc修改模型

修改模型若干层，可以deepcopy某一层然后修改一下参数。添加额外输入，可以在加forward参数个数。添加额外输出，直接加forward的返回值。

5.4 PyTorch模型保存与读取

PyTorch存储模型主要采用pkl, pt, pth三种格式，就使用层面没有区别。

一个PyTorch模型有俩部分：模型结构与权重，其中模型是继承nn.Module的类，权重的数据结构是一个字典。对于单卡而言方式如下没有太大区别，除了大小差了一点：

{% highlight python %}
unet.state_dict()

torch.save(unet, "./unet_exp.pth")
loaded_unet = torch.load("./unet_exp.pth")
unet.state_dict()

torch.save(unet.state_dict(), "./unet_weight_exp.pth")
loaded_unet_weights = torch.load("./unet_weight_exp.pth")
unet.load_state_dict(loaded_unet_weights)
unet.state_dict()
{% endhighlight %}

对于多卡情况下就不一样了，PyTorch中将模型和数据放到GPU上有两种方式————.cuda()和.to(device)，暂时对前面一种方式讨论。如果要使用多卡训练的话，需要对模型使用torch.nn.DataParallel，代码样例如下：

{% highlight python %}
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 如果是多卡改成类似0,1,2
model = model.cuda()  # 单卡
model = torch.nn.DataParallel(model).cuda()  # 多卡
{% endhighlight %}

查看model对应layer名称，可以看到差别在于多卡并行的模型每层的名称前多了一个“module”，如单卡是layer: conv1.weight，多卡是layer: module.conv1.weight。这种模型表示的不同可能会导致模型保存和加载过程中因为单GPU和多GPU环境的不同带来模型不匹配等问题，需要处理一些矛盾点，以下做分类讨论：

* 单卡保存+单卡加载：这个比较简单，即使保存和读取时使用的GPU不同也无妨。
* 单卡保存+多卡加载：这种情况比较简单，读取单卡保存的模型后，用nn.DataParallel进行分布式训练即可。
* 多卡保存+单卡加载：此时核心问题是如何去掉权重字典键名中的“module”，以保证模型统一性，对于加载整个模型，直接提取模型的module属性即可。对于加载模型圈子，有多种思路：去除字典里的module麻烦，往model里添加module简单（推荐），这样即便是单卡也能开始训练（相当于分布到单卡上）；遍历字典去除module；使用replace操作去除module。

{% highlight python %}
import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'   #这里替换成希望使用的GPU编号

model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存+读取整个模型
torch.save(model, save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
loaded_model = torch.load(save_dir)
loaded_model = loaded_model.module
{% endhighlight %}

{% highlight python %}
import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'   #这里替换成希望使用的GPU编号

model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model = nn.DataParallel(loaded_model).cuda()
loaded_model.state_dict = loaded_dict
{% endhighlight %}

* 多卡保存+多卡加载：由于模型保存和加载都用多卡，因此不存在模型层名前缀不同的问题。但多卡状态下存在一个device（使用的GPU）匹配的问题，既保存整个模型时会同时保存所使用的GPU id等信息，读取时若这些信息和当前使用的GPU信息不符可能会报错或者程序不按预定状态运行，具体表现为：1. 读取整个模型再使用nn.DataParallel进行分布式训练设置————这种情况很可能会造成保存的整个模型中GPU id和读取环境下设置的GPU id不符，训练时数据所在device和模型所在device不一致而报错。2. 读取整个模型而不使用nn.DataParallel进行分布式训练设置————这种情况可能不会报错，测试中发现程序会自动使用设备的前n个GPU进行训练（n是保存的模型使用的GPU个数）。此时如果指定的GPU个数少于n，则会报错。在这种情况下，只有保存模型时环境的device id和读取模型时环境的device id一致，程序才会按照预期在指定的GPU上进行分布式训练。相比之下，读取模型权重，之后再使用nn.DataParallel进行分布式训练设置则没有问题。因此多卡模式下建议使用权重的方式存储和读取模型。如果只有保存的整个模型，也可以采用提取权重的方式构建新的模型。

{% highlight python %}
import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'   #这里替换成希望使用的GPU编号

model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存+读取模型权重，强烈建议！！
torch.save(model.state_dict(), save_dir)
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model = nn.DataParallel(loaded_model).cuda()
loaded_model.state_dict = loaded_dict
{% endhighlight %}

{% highlight python %}
# 读取整个模型
loaded_whole_model = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.state_dict = loaded_whole_model.state_dict
loaded_model = nn.DataParallel(loaded_model).cuda()
{% endhighlight %}

另外，上面所有对于loaded_model修改权重字典的形式都是通过赋值来实现的，在PyTorch中还可以通过"load_state_dict"函数来实现。
