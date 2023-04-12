# papers-weekly
每周论文学习记录

# 2023.4.8

Graph Convolutional Networks for Text Classification
    本文提出 TextGCN model，核心idea：
    1，利用 word-doc TF-IDF, word-word 共现来构建 heterogeneous graph
    2，在此graph之上用GCN模型，半监督方式train
    
    对比 LR/CNN/LSTM 等模型，表现出了更优的效果，即便CNN和LSTM使用了pre train emb，TextGCN只使用graph和部分标签信息。

    不足之处：因为用了GCN，整个方案是transductive的，不过要改为inductive应该也不困难，参考GraphSAGE即可
    (原文的代码基于TF1，现在跑不了了，git上 找了一个TF2的版本)

# 2023.4.9

Imagenet classification with Deep CNN


# 2023.4.12

Learning convolutional neural networks for graphs
    这是2016年的一篇文章
    本文提出了 PATCHY-SAN model，核心idea：
    1，CNN 对于CV/NLP等Euclidean数据场景，取得了SOTA效果，但是对于social network等Non-Euclidean数据，还不能很好的处理
    2，因此本文提出了一种方法，将图数据巧妙的转换成CNN可以处理的 Euclidean 数据格式
    3，核心过程：确定w个中心节点，并为它们创建邻域；然后将图表示映射为向量表示，使得相似结构的节点处于向量中相似位置

    具体逻辑比较复杂，就不详述了。我个人的看法是：这篇文章设计了巧妙（且复杂）的方法，将图数据映射为CNN能够处理的欧式数据，以便可以使用CNN这一利器，好是好，不过以当今的视角来看，肯定是信息有损的了。后续关注一下其它方法与之的对比，尤其是GNN的对比。

