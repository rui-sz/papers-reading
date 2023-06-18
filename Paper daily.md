# papers-weekly

每周论文学习记录

# pending list

> paper1 = video 2

- [ ] Opening the black box of Deep Neural Networks via Information
- [ ] Learning phrase representations using rnn encoder-decoder for statistical machine translation2014年，提出了GRU
- [ ] 使用Xenon-Generation Finetune LLama文档阅读
- [ ] GNN 论文总结
- [ ] Deeper insights into graph convolutional networks for semi-supervised learning
- [ ] StyleDrop: Text-to-Image Generation in Any Style 通过文字改变发型等
- [ ] CLIP 得分，衡量文本、图像的对齐程度
- [ ] Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
- [ ] Big Transfer (BiT): General Visual Representation Learning
- [ ] DALLE
- [ ] DL三大特征抽取器（CNN,RNN,Transformer）总结
  TBD

## 2023.6.18

CLIP 系列文章整体串一下

《CLIPasso: Semantically-Aware Object Sketching》

本文核心要解决的问题：图像->简笔画

<img src="pic/CLIPasso1.png" width="400" height="200">

之前的研究：

    搜集 sketch 数据集，风格和类别受限，已有素描数据集比较有限。

    如何摆脱对有标签数据集的依赖，而CLIP对物体很敏感（不受风格印象），具有出色的zero shot能力，是否可以结合

<img src="pic/CLIPasso3.png" width="400" height="200">

本文核心内容：

1. 贝斯曲线，用了图形学已有的光栅化器；同时控制笔画多少，实现不同程度的抽象，兼顾几何和语义性
2. 进行更好的Saliency初始化（训练好的 VIT，最后一层多注意力看hot area，画贝斯曲线）
3. 设计合理的loss function：Ls语义接近 + Lg空间限制，其中Lg把模型前几层的特征拿出来算loss，有长宽概念，几何敏感
4. train阶段，进行了2000 iteration，1张V100 GPU，一天train
5. 效果优势：能达到任意程度抽象，控制贝斯曲线的点数；具备强大的Zero shot 能力
6. limitations，要求有背景的图，否则效果大打折扣；CLIPasso是同时生成，而不是序列生成，因此结果不够diverse；不同图片需要的抽象程度不一样，现在的模型需要显式指定

   本文综合利用了图形学知识，以及CLIP模型的强大多模态+zero shot能力。

《DepthCLIP: Can Language Understand Depth?》

    如何通过模型理解图片中的深度信息

<img src="pic/DepthCLIP.png" width="600" height="300">

《AudioCLIP: Extending CLIP to Image, Text and Audio》

<img src="pic/AudioCLIP.png" width="500" height="260">

把音频模态添加进来：

1. CLIP 框架完全适应，比较优雅简单
2. 在音频模态上，具备了zero shot 能力

《PointCLIP: Point Cloud understanding》

    将3D point cloud 投射到2D平面，变成2D深度图，再用经典的CLIP框架处理

《ActionCLIP: A New Paradigm for Video Action Recognition》

<img src="pic/ActionCLIP.png" width="400" height="350">

本文针对动作识别问题，一个图像领域更热的研究领域

本质是分类问题，传统方法：

    依赖大量标注数据集

    具有巨大局限性，动作非常难以标注，label 接近无穷

    类别太多，softmax不工作了

如何在动作识别里也做zero shot？

思路如上图：

    图像变成视频，如前文

    图像文本对，ground-truth 是一个矩阵

《CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval》

<img src="pic/CLIP4clip.png" width="500" height="400">

针对视频检索问题 Empirical 的 study

    CLIP 很适合做，图文相似性

关键点：

    视频的时序处理：视频->图片单帧->图像特征

    三种方式：mean pooling，（frame）transformer Encoder/LSTM，（text+frame）transformer Encoder

结果：

    怒刷了5个 Video retrieval 常用数据集，效果都很好

## 2023.6.14

Fine Tuning(by 李沐)
    用高质量预训练模型，初始化目标模型的权重，再在小数据集上微调
    微调技巧：
    1，更强的正则化：小学习率；用少量数据做FT
    2，如果pretrain数据集中有类似的数据，可以在FC层复用其权重

《NCI: A Neural Corpus Indexer for Document Retrieval》
    2022.10 NIPS 最佳论文之一
    核心idea：
    1，本文提出比较新颖的E2E做文本检索的方法，部分参考了同会DSI
    2，用transformer训练一个s2s的模型，输入query，输出docid
    3，train关键点：用层次化聚类的方式生成具有一定语义信息的docid，用docT2query为doc生成query；32w doc
    优势：
    a）区别于传统搜索基于倒排索引方案，本文E2E的检索很简洁
    b）模型记住了文档信息，对新query、长query的泛化能力比较好
    劣势：
    a）性能上还差的比较多，要消耗大量GPU
    b）数据集选择的特殊性，选择了比较复杂的QA数据集，短query存在隐患

    比较有意思的一篇文章，瑕不掩瑜

## 2023.6.13

《CLIP: Learning Transferable Visual Models From Natural Language Supervision》
    2021.2 的文章，核心idea：
    1，利用自然语言的监督信号，学习迁移性很好的视觉模型，在train时采用对比学习paradiam，推理时可以做zero shot 和 few shot
    3，模型架构，预训练阶段train2个编码器文本 + 图片，推理时构建文本prompt，用2个编码器编码，再计算cos相似度
    4，训练数据：作者构建的一个高质量数据集WIT/WebImageText，有4亿(image, text) pair，是CLIP如此强大的主要原因之一
    5，评估结果：在30个数据集上取得非常好的效果，zero shot 推理在Imagenet上超过ResNet50监督模型，同时表现出非常强的鲁棒性，很炸裂的结果；few shot 之后效果更好，说明模型学习到泛化性很好的特征
    6，模型在 MNIST数据集上效果不好，还是强依赖数据，DL的脆弱性所在；同时细分类任务、抽象任务、计数类任务等也表现不太好，原始数据中难以获得高质量信号
    7，整体方案比较scalable，不用针对特定任务去搜集监督数据集，同时打破了固定种类标签的范式，无限种类、不限使用场景

    总体上属于新意度、问题规模、有效性都非常高的一篇文章。
    后续继续看下CLIP衍生出来的一些工作

## 2023.6.10

GD, SGD, RMSProp, AdaGrad, Adam 之间关系及优化路径，还是比较搞混的，简单梳理下：
    GD是NN优化非常重要的部分，标准梯度下降需要计算全量样本，计算量太大，后续优化思路主要有2个（减少计算量、优化下降路径）
    SGD 通过随机选择样本减少计算量，可以解决这个问题，minibatch SGD可以加速收敛
    但是下降路径却不一定最优，学习率不应该是一个固定值，每一步应该随着梯度变化关系动态适应
    如何优化路径？
    牛顿法，通过对梯度公式进行泰勒展开保留到二阶导（梯度的梯度），可以优化这个问题，不过计算量也比较大
    动量法，把历史梯度数据考虑进来，对梯度更新的0次项做修正，指数滑动加权平均；Nestro方法则结合历史梯度和超前数据
    学习率应该逐渐变小，简单的办法随着迭代进行而逐渐减小，这种做法未必好。AdaGrad 提出学习率自适应的方法，在学习率下面除以一个数值
    这个数值跟历史梯度有关，修正在不同维度上的学习率。进一步优化也采用指数滑动加权平均，降低历史数据权重，即RMSprop

    动量法 + RMSprop = Adam(Adaptive moment estimation) optimizer，几个超参：
    α：初始学习率，0.001
        β1：0次项的加权系数，0.9
    β2：1次项的加权系数，0.999
    epsilon: 10^-8，这个一般不动
    通常固定β1,β2,epsilon，保持默认值，调参初始学习率

## 2023.5.22

《ResNet: Deep Residual Learning for Image Recognition》
    2016年文章，短短9页，16w引用，绝对是人狠话不多的典范
    核心idea：
    1，VGG 证明了深的网络更好，但是深的网络难以训练（收敛速度慢、梯度 vanish/explode等）
    2，本文提出的残差网络优化了这个问题，网络更深但是计算复杂度并没有提高很多
    3，对比VGG net，ResNet-152有8倍的深度，跟 VGG，GoogleNet 等网络的实验对比，效果非常的好，收敛速度也很快
    4，本文把网络做深过程中的几个核心部件：skip connection，bottleneck layer等

    Residual net 和 skip connection 都不是本文的首创，但是却很好的解决了图像分类等问题，非常赞
    不足之处是只讲了调参后工程上的做法，却没有很多理论解释（可能也比较难吧），无碍于神作

## 2023.5.20

《Very deep conversational networks for large-scale image recognition》
    2015 ICLR, 大名鼎鼎的VGG
    核心idea：
    1，2012年 AlexNet 已经展现了CNN处理图片分类的强大能力，之后不同工作有不同演进方向，例如感受野大小、网络深度等
    2，本文采用3*3 receptive fields，步长为1，堆叠8~16层卷积+3层FC，想要研究网络深度的影响

    部分研究结论：
    1，3个3*3卷积层叠加，感受野与1个7*7是一样的，参数量却小得多 3*9*C^2 vs 49*C^2
    2，网络深度的增加对于性能提升有显著作用，VGG19对比VGG16效果没有进一步提升，说明参数量饱和
    3，1*1卷积层的作用，是一种增加非线性的方式，实验结果显示能提升性能
    4，A-LRN 的效果并不比 A好，说明 LRN（AlexNet的创新）效果不大
    5，multi scale 技巧，无论用在train（数据增广）还是test（集成）都有效果，也容易理解

## 2023.5.15

《Semi-supervised classification with graph convolutional networks》
    2017 年的文章
    昨天 PPNP 的文章基于GCN做了一些创新，今天将GCN回顾一下，这是图卷积的开创性论文，非常经典，今天看来其思路或许比较平常，但是在17年的时候，其将CNN思想用于Graph是非常创新的
    核心idea：
    1，GCN 相对于GNN（下一篇会回顾）来说，是对信息聚合过程做了一定的简化和近似，之后就类似CNN的卷积了
    2，2层卷积网络 + CE loss，为什么只有两层卷积，卷积太浅，信息不能大范围传播，卷积太深，容易过平滑（PPNP解决的问题之一）
    3，GCN 要基于整个Graph的邻接矩阵去计算（参见5.14的公式），进行整体求解，因此其范式是 transductive 的，计算开销也大（GraphSAGE后续会改进）
    4，从评估结果来看，GCN是要显著优于DeepWalk, LP等传统方法的，从原理看也容易理解。不过遗憾的是没有与GNN做对比，我想其效果应该差于GNN？

## 2023.5.14

《Predict then propagate: Graph neural networks meet personalized pagerank》
    2019 ICLR 的一篇论文，核心思想：
    GCN 的公式：
    ![image](https://github.com/rui-sz/papers-weekly/assets/69101330/dc1506c0-c7c9-4fe5-aea7-1532f449a377)
    其原理直观理解，就是不断聚合邻居节点的信息，并学习聚合权重。原论文推荐使用2层GCN-layers
    那么就只能感知到两步以内的邻居节点信息，要进一步提高层深，会存在2个问题：
    1，过平滑问题，如果使用太多层节点表示会趋向一致
    2，增加要学习的参数量

    为了解决这2个问题，PPNP 提出了新的方案，思路：
    1，先用一个MLP基于提取特征做预测（特征变换），这里包含了所有要学习的参数
    2，传播策略，采用 personalized PR 的思想，将邻接矩阵替换为带有 self loop 的矩阵
    这样PPNP就包含两步，NN+传播，可以提高模型复杂度以及传播距离而不用担心GCN的2个问题

    PPNP 宏观的理解是：GNN=AXW 表达的是节点特征能够在图结构上进行传播，经过抽象解耦之后
    PPR 的引入能缓解过平滑问题，NN 结构也可以做深了

    由于 PPNP 信息传播过程中的矩阵求逆运算花销比较大，作者提出了一个近似的 APPNP 模型

## 2023.4.23

Anomaly Detection with Robust Deep Autoencoders
    2017年
    想法：
    1，提高autoencoder模型的抗噪声能力，启发于RPCA模型，降原始数据分成正常数据、噪声数据两部分，交替训练
    2，L1 正则，L21正则（对矩阵每一列求2范数再求和）
    3，先优化AE，再优化正则项

## 2023.4.22

Deep Visual-Semantic Alignments for Generating Image Descriptions
    2015年的文章，二作李飞飞

    本文的核心idea，训练模型来为图片生成文本描述，分为两步
    1，基于pretrained RCNN模型 和 BRNN 模型来学习 image 和 text 的emb，建立他们之间的对齐关系
    2，训练 generative model MRNN，接收RCNN的信息作为初始输入，并训练MRNN模型
    a) 初始化h0，输入START，得到第一个word
    b) 输入第一个word，得到输出的第二个word
    c) 以此类推，期待输入最后一个word之后输出END

    建立图文emb映射的思路比较有意思，不足之处
    1，输入图片分辨率是固定的
    2，RNN只通过bias来接收图像信息，可以做得更好
    3，不是end-to-end的

## 2023.4.13

RNN & LSTM basics
The Unreasonable Effectiveness of Recurrent Neural Networks
    link: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    核心思想：
    1，RNN 引入了时间和依赖信息，适用于序列建模场景，包含Recurrent网络，允许信息持久化。
    2，RNN 的优势: 能够处理长度不限的输入, 以及"理论上"可以利用整个输入序列的信息
    3，由于Gradient vanishing 和Gradient explosion的问题，RNN 实际上取得的成绩并不大
    4，LSTM 的出现，很好的解决了RNN存在的梯度消失和梯度爆炸问题，也因此成为DL领域引用率最高的论文之一

Long short-term memory
    1997年的文章
    核心思想：
    1，LSTM 通过三个门结构的引入解决RNN网络存在的梯度消失和梯度爆炸问题，极大提高RNN网络的能力
    2，遗忘门，决定ht-1的信息有多少能传递到ht
    3，更新门/输入门，决定记住多少xt和ht-1的信息
    4，输出门，决定ct的信息以多少比例输出到ht
    5，三个门结构比较好的解决了梯度消失问题，但是并不能够避免梯度爆炸。不过由于LSTM有众多门控结构，跟普通RNN相比，发生梯度爆炸概率低很多

    遗憾的是，LSTM 论文并没有solid的理论解释，也没有解释门控为什么这么设计。再次支撑了深度学习是炼丹的说法~ lol
    LSTM 相比RNN的改进是比较好理解的，后来基于各种门控又有一些新的变体，例如GRU（Gated Recurrent Unit）等

## 2023.4.12

Learning convolutional neural networks for graphs
    2016年的文章
    本文提出了 PATCHY-SAN model，核心idea：
    1，CNN 对于CV/NLP等Euclidean数据场景，取得了SOTA效果，但是对于social network等Non-Euclidean数据，还不能很好的处理
    2，因此本文提出了一种方法，将图数据巧妙的转换成CNN可以处理的 Euclidean 数据格式
    3，核心过程：确定w个中心节点，并为它们创建邻域；然后将图表示映射为向量表示，使得相似结构的节点处于向量中相似位置

    具体逻辑比较复杂，就不详述了。我个人的看法是：这篇文章设计了巧妙（且复杂）的方法，将图数据映射为CNN能够处理的欧式数据，以便可以使用CNN这一利器，好是好，不过以当今的视角来看，肯定是信息有损的了。后续关注一下其它方法与之的对比，尤其是GNN的对比。

## 2023.4.9

Imagenet classification with Deep CNN
    2017年的文章，用一个改进的CNN网络解决基于ImageNet的1k类分类问题
    核心idea：
    1，当时最大的一个CNN网络，基于2块GPU
    2，2D卷积的高性能GPU实现
    3，采用了一些技巧（Relu instead of tanh/multiple GPUs），提高模型性能和训练效率
    4，采用了一些方法避免过拟合：数据增强、dropout等

## 2023.4.8

Graph Convolutional Networks for Text Classification
    本文提出 TextGCN model，核心idea：
    1，利用 word-doc TF-IDF, word-word 共现来构建 heterogeneous graph
    2，在此graph之上用GCN模型，半监督方式train

    对比 LR/CNN/LSTM 等模型，表现出了更优的效果，即便CNN和LSTM使用了pre train emb，TextGCN只使用graph和部分标签信息。

    不足之处：因为用了GCN，整个方案是transductive的，不过要改为inductive应该也不困难，参考GraphSAGE即可
    (原文的代码基于TF1，现在跑不了了，git上 找了一个TF2的版本)
