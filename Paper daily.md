# papers-weekly

每周论文学习记录

# pending list

> paper1 = video 2

领域主线：

1. 大模型（NLP, CV, 多模态等)
2. GNN系列
3. Anomaly Detection 系列
4. 搜广推（+智能运营)
5. ABtest

具体文章：

- [ ] Opening the black box of Deep Neural Networks via Information
- [ ] Learning phrase representations using rnn encoder-decoder for statistical machine translation2014年，提出了GRU
- [ ] 使用Xenon-Generation Finetune LLama文档阅读
- [ ] 写综述：GNN 论文总结
- [ ] Deeper insights into graph convolutional networks for semi-supervised learning
- [ ] 搜广推：MMoE，DeepFM等
- [ ] AlphaFold, AlphaFold2
- [ ] 不同的位置编码方式，1D(Bert的), 2D etc
- [ ] 通用内容

  - [ ] GAP global average pooling
  - [ ] Grad-cam: Visual explanations from deep networks via gradient-based localization  视觉DNN的可解释性
  - [ ] DL三大特征抽取器（CNN,RNN,Transformer）总结TBD
  - [ ] 自注意力原理，MHA 详解，CV和NLP的自注意力机制
  - [ ] viking, milvus等向量检索相关技术，检索和矢量搜索工具（如 LangChain、LlamaIndex 和 Pinecone）
- [ ] 大模型近年来的主要文章（目标：7月底写一篇综述）

  - [ ] Big Transfer (BiT): General Visual Representation Learning （CNN中做的比较大的)
  - [ ] StyleDrop: Text-to-Image Generation in Any Style 通过文字改变发型等
  - [ ] 详细看下ViLT跟Bert、ViT之间的相似之处
  - [ ] CLIP 得分，衡量文本、图像的对齐程度
  - [ ] Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
  - [ ] image GPT 的工作
  - [ ] 多模态预训练工作（by李沐)
  - [ ] MAE loss，BiT MAE重建图像
  - [ ] 图像生成系列合集：AE, DAE，DALLE，BiT，BLIP，imagen等

## 2023.7.3

《BLIP》

《DALLE》

《AE》

《DAE》

《VAE》

## 2023.7.2

《**Highly accurate protein structure prediction with AlphaFold**》

大名鼎鼎的 AlphaFold2，同时被科学和自然评为 2021年 AI 届在科学界最大的突破，为什么这是一个改变了整个生物学的跨时代工作？1972 年诺奖得主畅想过，未来应该根据氨基酸序列预测蛋白质3D结构。本文解决了一个50年的难题

问题定义：

1. 蛋白质的氨基酸序列结构 -> 预测蛋白质3D形状（蛋白质折叠问题），形状决定功能，并且结构和形状是一一对应的
2. 已知结构的只有10万，但现存有10亿种不同的氨基酸序列

模型架构：

<img src="pic/AlphaFold2_1.png" width=600" height="300">

总体分成3个部分：

1. 抽特征，分别抽取序列结构信息MSA，以及氨基酸互相之间的空间关系
2. encoder，输入前述2个3D张量，经过类Tfm的很复杂的运算，输出2类不同编码信息
3. decoder，将编码信息转换为3D空间结构，里面用到很多空间几何的运算。有点像LSTM的结构
4. 训练：主损失函数FAPE，128TPU v3，train 1周，因为整体参数量很大，内存比较吃紧，采用了一些优化技巧
5. 预测：256序列，单卡V100 跑 4.8分钟，2500序列，单卡V100 跑 18h~
6. 结果：从结果来看，相比同期的方法，取得了大幅度的提升，将误差降低到原子级别；消融实验证明了很多机制都有用

本文用了50页的补充材料来介绍算法细节，包含伪代码。模型是很复杂的，借鉴了很多前人的工作，像Tfm，MSA/空间信息的拆分等等。面对如此复杂的研究课题，我觉得要完成这样一个工作，需要极强的科研和工程能力，否则很容易迷失在其中。牛逼，致敬划时代的工作。


## 2023.6.30

《Hierarchical Text-Conditional Image Generation with CLIP Latents》

DALLE2，2022.4 OpenAI的工作，也是大热的一篇

一些有趣的应用场景：

1. 根据文本描述，生成原创性的图片，可以任意组合概念、属性、风格
2. 根据文本对图片进行编辑，因为文本图片共享语义空间
3. 根据输入图片生成相似风格的图片，如图1
4. 2个图像内插，如图2
5. 图像文本内插，如图3

图1：

<img src="pic/DALLE2_1.png" width=500" height="350">

图2：

<img src="pic/DALLE2_2.png" width=500" height="350">

图3：

<img src="pic/DALLE2_3.png" width=500" height="350">

模型主要内容：

1. 本质上是 CLIP+GLIDE 的模型，
2. Text -> text emb -> image emb -> image，与CLIP的过程是相反的，所以又叫UNCLIP
3. 模型结构如图4，上半部分是CLIP（在本文一直锁住)，下半部分采用两阶段建模：prior + decoder
4. prior：text emb->img emb，尝试了AR和Diffusion 两种，Tfm decoder-only 用classifier free方式的效果好一些
5. decoder：采用GLIDE 的方案，基于UNET，CNN base的模型，最后逐层上采样，级联生成高清大图
6. 效果评估，COCO FID分数高于GAN，DALLE1等；生成的图片质量很高
7. 若干局限性：物体属性结合的不好；文字生成效果不好；有自己自己的鸟语~

图4：

<img src="pic/DALLE2_4.png" width=500" height="300">

总体而言，是非常吸引眼球的工作，复刻艺术风格、基于文字生成图片等，相比人工内容创作，拓展了很大的自由空间，具有想象力。不过看得出来训练和推理成本都是很高的，短期内应该难以落地实用，科研的路费钱又不能短时间内看到收益，比较考验投入定力。

## 2023.6.29

《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》

2021.3 上传arxiv，ICCV最佳论文

ViT 初步证明了Tfm在CV领域的巨大潜力，它能够让模型架构更加优雅，同时显著降低计算复杂度。但是相比叱咤CV领域多年的CNN骨干网络，还没有普遍性的证明它作为骨干网络的完备性。NLP Tfm -> CV Tfm，面临的困难：

1. 多尺度问题，同物体大小多变，CNN 则应对的很好，ViT 对多尺度特征的把握能力，会弱一些，而对于检测、分割这类任务，多尺度特征很重要
2. Resolution 困扰，通过patch 初步得到解决

本文的出发点，就是想要证明 Swin Tfm 可以作为通用骨干网络，既然有CNN这么一个CV领域成熟的模型架构做参照，本文作者设计了一个精巧的披着Tfm皮的类CNN架构。并基本把视觉领域所有任务刷了一遍，并且效果很炸裂，基本实现了目标。

模型结构

<img src="pic/Swin1.png" width="700" height="300">

核心组件：

1. 局部自注意力，全局计算自注意力对于图像类任务来说，可能有点浪费了，图像的特性还是应该有局部性。本文主要贡献，提出局部自注意力，在7*7序列内计算，计算复杂度跟图片大小成正比，相比较而言ViT是平方
2. Patch merging，类似于卷积神经网络的卷积操作。基于窗口和移动窗口的注意力计算，算出多尺度特征输入给FPN，去做检测，也可以扔给 unet 去做分割
3. 掩码自注意力，在基于patch的移动窗口机制下，局部自注意力之上，进行跨窗口注意力计算不是很容易，作者进行了精巧的设计。

<img src="pic/Swin2.png" width="300" height="300">

<img src="pic/Swin3.png" width="300" height="300">

实验：

* 作者把NLP领域分类、目标检测、语义分割代表性数据集都刷了一遍，考虑到其初衷这也是需要做的。效果都相当不错
* 消融实验证明，shifted window 和 相对位置编码都比较有用。并且在 dense predict 场景（COCO, ADE20k) 的提升幅度大一些，分类提升稍小，结合模型改进点可以理解。

作者实现了其原始目标，Swin Tfm 由于其里程碑式的优秀表现，之后会成为视觉领域一个重要的baseline。本文也体现了作者对CNN，Transformer，MLP 几种架构的研究深度和醇熟运用，随意魔改~

个人看法，Shift window的自注意力计算机制有点太复杂了，太fancy不一定能长久，有生命力的还是简洁优雅的方案

《ALBEF: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation》

Transformer用于多模态领域的一篇文章，2021.7

核心内容：

1. 本文propose了一个新的 VL 表示学习框架，相比ViLT资源需求少，更容易复现。对比当前 VLP methods：LXMERT, UNITER, OSCAR 等，本文提出的方法具有更好的模型性能，以及更快的推理速度
2. 模型：三个 encoder, visual/text/multimodel，分别基于 ViT-B/16, Bert-base, Bert-base; 损失函数ITC对比学习，语言模型 MLM 和 图文匹配 ITM三者之和。引入了动量模型（滑动平均）的pseudo-labels对抗数据噪声
3. 训练数据：同ViLT/UNITER 采用4M数据集，COCO/VG/GCC/SBU，在A100 8卡机器上train 了30Epoch
4. 实验：从实验结果来看，在V-L多个任务上都取得很好的效果，如Retrieval/VQA/VE/NLVR/VG等

模型框架：

<img src="pic/ALBEF1.png" width="600" height="350">

准备复现一下论文结果，毕竟相比ViLT，资源需求少了很多

## 2023.6.27

《Whisper: Robust Speech Recognition via Large-Scale Weak Supervision》

2022年，OpenAI的文章，语音识别领域一个比较有潜力的工作

核心内容：

1. 已有的方法，主要是无标数据预训练 + 有标数据FT，局限性：没有高质量解码器，需要搜集数据FT；FT的时候容易过拟合
2. 本文模型，一个标准的Transformer encoder-decoder架构，emb层加了2层卷积，降低输入长度；all in one 的模型
3. 训练，68h 网上抓下来的多语言数据，做了一定预处理后分成30s/区间，train 了2~3个Epoch
4. 效果，在英语等语种上都不错，中韩阿拉伯语效果差一些；泛化性也比较好；对比几个商业系统也不错，可能会造成冲击

## 2023.6.26

《MAE: **Masked Autoencoders Are Scalable ****Vision**** Learners**》

    2021.11的一篇文章，一作是 ResNet 的作者 Kaiming He

本文思路比较straightforward，相当于Bert在CV上面的实现，a masked CV model

"***Simple*** algorithms that ***scale well*** are the core of deep learning."

核心内容：

1. 只用图片本身做自监督学习，属于Bert 在计算机视觉上的拓展
2. 把Bert从NLP用到CV的时候，会有什么问题？
   1. 卷积上面不好做掩码
   2. NLP里面完形填空并不简单，CV的完形填空则相对简单一些~图像中的很多像素是冗余的
   3. 解码器方面，NLP任务简单FC就行，CV则不行
3. 模型：非对称性的encoder-decoder架构，encoder follow ViT，decoder 标准 Transformer，损失函数 MSE in pixel space
4. 实验：下游分类、目标检测、语义分割等数据集上，都有比较好的表现；展示出来的例子也比较震撼，75%遮盖下能够很好的还原

<img src="pic/MAE1.png" width="500" height="350">

<img src="pic/MAE2.png" width="600" height="350">

<img src="pic/MAE3.png" width="600" height="400">

## 2023.6.25

《AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE》

2021 ICLR，大名鼎鼎的ViT，本文的重要意义：

1. 证明了在图像任务上CNN不是必须，可以用纯Transformer，在此之前ResNet 还是SOTA的方法
2. 同时注重可扩展性，基于Transformer尽量少做修改，架构简洁
3. 挑战了CNN在视觉领域的绝对统治地位，打破了CV和NLP的壁垒，打通CV、NLP的鸿沟，开启了CV新时代
4. 同时探索自监督预训练方式在CV的可行性，本文证明了也OK，但是相比有监督有不少差距

主要内容：

* 模型：尽量follow original Tfm，每个patch16X*16作为一个元素，完成token化（2020年类似的工作，2*X2patch)，MHA+Res+MLP。如图1
* 数据集：imagenet 1k，imagenet 21k，JFT 等
* 效果：对比ResNet 方法，ViT-H在多个数据集上都取得了最好的效果，性能强大。如图2
* 训练：在同等计算复杂度下，ViT的效果比CNN要好，不过从论文评估来看，CNN和ViT的效果随着计算复杂度都没有进入瓶颈，还有提升空间.如图3
* 其他分析：图4可以看出，在<14M数据集上，ResNet的效果还是更好一些，ViT 在更大的数据集上效果更好；自监督工作有潜力（关注MAE)；1D pos emb有学习到2D的信息

图1：

<img src="pic/ViT1.png" width="500" height="300">

图2：

<img src="pic/ViT2.png" width="600" height="300">

图3：

<img src="pic/ViT3.png" width="600" height="300">

图4：

<img src="pic/ViT4.png" width="600" height="300">

## 2023.6.22

《ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision》

多模态领域里程碑著作，把目标检测任务去除了，计算性能有上千倍提升，同时效果整体并没有降低很多。（有时候工作follow的慢一些，可能反而会更容易上手了~)

先上一个牛逼的对比图：

<img src="pic/ViLT1.png" width="400" height="400">

多模态领域 VLP 现状：

* 传统方法计算太贵，在2021年之前，VLP基本都依赖目标检测，识别bounding box，region->word，目标检测作为多模态的一部分（Visual Genome 预训练的目标检测器)。也是因为下游任务很多都是多模目标检测，这样更相关。
* 并且当前预训练模型的数据集不太大，因此泛化性不一定好。
* 2020年 Pixel BERT 只用用 ResNet 抽特征图，grid feature，7*7特征图拉直为线性序列，也还是有点贵，并且效果下降了很多，十几个点
* 传统方法非常依赖视觉特征部分的提升

本文核心idea：

1. 文本图像多模态，需要将图像 pixel 变成具有语义信息的序列，传统方法基于CNN+Region Supervision，比较贵
2. 怎么设计更简单的图像特征抽取方法，本文受启发于ViT，打patch获取 patch embedding。本质上属于ViT 思想在多模态领域的手快应用。因为是Linear emb 层，计算性能提升了很多，但效果上还是目标检测更好，迄今最简单的 v+l 模型，减少复杂度保证不掉点
3. 模型结构：2路特征concat为Transformer Encoder的输入，目标函数：image text matching + masked language loss + word patch alignment
4. 训练过程中小技巧：图像侧做了巧妙的图像数据增强，randAugment 这个工作，并尽量保证文本图片是匹配的；文本侧把文本word整个mask掉，以增强文本图像之间的联系
5. 训练成本：64个V100的GPU，训练三天，成本太高。相比而言ALBEF单机8卡训3天，以及BLIP等更容易跟进一些。

<img src="pic/ViLT4.png" width="500" height="200">

另外本文对多模态领域工作做了综述，非常扎实，值得学习！

<img src="pic/ViLT2.png" width="500" height="200">

## 2023.6.21

家里人来坡旅游，今天没时间看文章+summarize了，先占个位吧，明天补~

《ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision》

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
