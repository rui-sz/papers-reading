# papers-weekly  
每周论文学习记录  
  
  

# pending list

Opening the black box of Deep Neural Networks via Information  

Learning phrase representations using rnn encoder-decoder for statistical machine translation  
    2014年，提出了GRU  

使用Xenon-Generation Finetune LLama  
    文档阅读  

李沐论文精读一篇  
    https://www.zhihu.com/people/mli65/zvideos  


GCN 论文总结  


# 2023.5.14

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

# 2023.4.23

Anomaly Detection with Robust Deep Autoencoders  
    2017年  
    想法：  
    1，提高autoencoder模型的抗噪声能力，启发于RPCA模型，降原始数据分成正常数据、噪声数据两部分，交替训练  
    2，L1 正则，L21正则（对矩阵每一列求2范数再求和）  
    3，先优化AE，再优化正则项  

# 2023.4.22 
  
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

# 2023.4.13  
  
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

# 2023.4.12  
  
Learning convolutional neural networks for graphs  
    2016年的文章  
    本文提出了 PATCHY-SAN model，核心idea：  
    1，CNN 对于CV/NLP等Euclidean数据场景，取得了SOTA效果，但是对于social network等Non-Euclidean数据，还不能很好的处理  
    2，因此本文提出了一种方法，将图数据巧妙的转换成CNN可以处理的 Euclidean 数据格式  
    3，核心过程：确定w个中心节点，并为它们创建邻域；然后将图表示映射为向量表示，使得相似结构的节点处于向量中相似位置  
  
    具体逻辑比较复杂，就不详述了。我个人的看法是：这篇文章设计了巧妙（且复杂）的方法，将图数据映射为CNN能够处理的欧式数据，以便可以使用CNN这一利器，好是好，不过以当今的视角来看，肯定是信息有损的了。后续关注一下其它方法与之的对比，尤其是GNN的对比。  
  
  
# 2023.4.9  
  
Imagenet classification with Deep CNN  
    2017年的文章，用一个改进的CNN网络解决基于ImageNet的1k类分类问题  
    核心idea：  
    1，当时最大的一个CNN网络，基于2块GPU  
    2，2D卷积的高性能GPU实现  
    3，采用了一些技巧（Relu instead of tanh/multiple GPUs），提高模型性能和训练效率  
    4，采用了一些方法避免过拟合：数据增强、dropout等  
  
  
# 2023.4.8  
  
Graph Convolutional Networks for Text Classification  
    本文提出 TextGCN model，核心idea：  
    1，利用 word-doc TF-IDF, word-word 共现来构建 heterogeneous graph  
    2，在此graph之上用GCN模型，半监督方式train  
      
    对比 LR/CNN/LSTM 等模型，表现出了更优的效果，即便CNN和LSTM使用了pre train emb，TextGCN只使用graph和部分标签信息。  
  
    不足之处：因为用了GCN，整个方案是transductive的，不过要改为inductive应该也不困难，参考GraphSAGE即可  
    (原文的代码基于TF1，现在跑不了了，git上 找了一个TF2的版本)  
  
