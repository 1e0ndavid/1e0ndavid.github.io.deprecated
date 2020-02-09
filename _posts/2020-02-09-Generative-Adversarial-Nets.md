---
title:  "Generative Adversarial Nets"
mathjax: true
layout: post
categories: GAN
---
<center><font size="6">生成对抗网络</font></center>

作者：xxx

<center><font size="5">生成对抗网络</font></center>

我们提出了一个应用对抗过程来估算生成模型的新框架，在这个框架中我们同时训练了两个模型，分别是一个能捕捉数据分布的生成模型 *G* 和一个判别模型
*D*, *D* 可以估计一个来自训练数据而非模型 *G* 的样本的概率。