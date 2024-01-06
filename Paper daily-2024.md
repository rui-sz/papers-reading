# papers daily-2024

2024年每日论文阅读记录

领域主线：

1. 大模型（NLP, CV, 多模态等)
2. GNN系列
3. Anomaly Detection 系列
4. 搜广推（+智能运营)

# pending list

> paper1 = video 2

具体文章：

- [ ] Opening the black box of Deep Neural Networks via Information
- [ ] Learning phrase representations using rnn encoder-decoder for statistical machine translation2014年，提出了GRU
- [ ] 使用Xenon-Generation Finetune LLama文档阅读
- [ ] 写综述：GNN 论文总结
- [ ] Deeper insights into graph convolutional networks for semi-supervised learning
- [ ] 不同的位置编码方式，1D(Bert的), 2D etc
- [ ] 通用内容

  - [ ] GAP global average pooling
  - [ ] Grad-cam: Visual explanations from deep networks via gradient-based localization  视觉DNN的可解释性
  - [ ] DL三大特征抽取器（CNN,RNN,Transformer）总结TBD
  - [ ] viking, milvus等向量检索相关技术，检索和矢量搜索工具（如 LangChain、LlamaIndex 和 Pinecone）
- [ ] 大模型近年来的主要文章（目标：7月底写一篇综述）

  - [ ] Big Transfer (BiT): General Visual Representation Learning （CNN中做的比较大的)
  - [ ] Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
  - [ ] 多模态预训练工作（by李沐)
  - [ ] MAE loss，BiT MAE重建图像
  - [ ] DALLE/DALLE2 效果体验

  ALIGN

  Imagen

  LLM 系列：

  Zero-Shot Text-to-Image Generation

# 2024.1

## 2024.1.6

TBD

## 2024.1.5

《Multimodal Foundation Models: From Specialists to General-Purpose Assistants》

2023.9 Microsoft出的多模态领域综述文章，review视角全面，细致深入，质量很高，要点：

1. Multimodal foundation models pre-trained for special purposes
   1. Topics:
      1. methods of learning vision backbones for **visual understanding**      CLIP，BEiT，SAM 等
      2. text-to-image **visual generation**      包括 SD 等
2. Multimodal foundation models aim to be **general-purpose assistants**
   1. General-purpose assistants 的三个要点：
      1. an unified network architecture
      2. an unified input-output data format
      3. a general interface that facilitates easy interaction with humans.
   2. Topics/三种方案:
      1. **unified vision models** inspired by large language models (LLMs)
      2. end-to-end training of **multimodal ****LLMs**
      3. Chaining multimodal tools with LLMs:**multimodal agents**

要说缺憾，就是主要覆盖的是 image/image-language，没有覆盖音频、3D、视频等更多领域。
