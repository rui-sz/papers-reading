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

  LLM 系列：

  Zero-Shot Text-to-Image Generation

# 2024.1

## 2024.1.17

《Minigpt-4: Enhancing vision-language understanding with advanced large language models》

2023.4 沙特阿布扎比大学的一篇多模态研究工作，要点：

1. 我们相信GPT4的加强的多模态生成能力源自于对复杂LLM的使用，将 visual features 对齐到一个先进的LLM，可以获得很多GPT4展示出来的高级multimodal能力
2. 模型，BLIP2 的Vit&QFormer + Vicuna，做一阶段预训练，加一轮finetune
3. 实验结果，对比先前使用 less powerful language models的模型（比如BLIP2），MiniGPT4 展示出了更强的基于图片的理解和生成能力；align visual features with LLM，基于detailed image desc pairs效果更好

总体上，本文用一个很简单的方法，将预训练图像encoder与比较强的LLM级联，然后做一定量的pretrain+finetune，极大的增强了模型的图像理解和文本生成能力，涌现出类似GPT4的很多能力。


《Minigpt-5: Interleaved vision-and-language generation via generative vokens》

2024.10 加州大学的一篇研究多模态理解和生成的工作，要点：

1. 本文出发点是增加LLM同时输出images+texts的能力，做法上基于 MiniGPT4 引入multimodal generation能力，核心是一种generative vokens的中间变量
2. 模型：范式 interleaved multimodal input -> interleaved multimodal output, Multimodal LLM + Stable diffusion；采用两阶段 train，一阶段 unimodal alignment stage，text as input, generate vokens，二阶段 vokens as input, generate vision and language实验结果，性能优于SD2、Divter等之前表现比较好的model

总体上，在设计上比较有创意，属于集成型的model，基于一些不错的model（MiniGPT4和SD2），使用一些训练技巧达到比较好的效果


## 2024.1.15

《LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding》

2023.6 Adobe Research 和 Stanford 合作的一篇文章，要点：

1. LLaVAR: LLaVA (Large Language and Vision Assistant) that can **R**ead. 顾名思义，本文目标是提升 instruction-tuned model 对于 text-rich image 的理解能力，设计了一套方法构建高质量的 instruction tuning 数据集
2. 数据，除了来自LLaVA的数据之外，设计了一个流程，用OCR提取text数据，并基于OCR和captain的结果 prompt GPT4，生成 high-quality instruction tuning 数据集
3. 模型，模型复用了LLaVA的模型架构，两阶段训练，pretrain + finetune，实验结果主要验证了对于图片中text的理解能力

总体上，本文通过构建一个精巧的数据集，加强了instruction-tuned model对于text-rich image中text的理解能力，是一篇方法比较巧妙的文章，成本也不高。

## 2024.1.13

《VISTA-LLAMA: Reliable Video Narrator via Equal Distance to Visual Tokens》

2023.12 字节和浙江大学合作的工作，关于 Video understanding 能力的，要点：

1. 本文目标是加强 visual understanding and facilitate temporal modeling within the language model
2. 模型，包括几个部分 visual encoder + visual projector + text encoder + language model，其中projector提升了时序建模能力，并且能把long video压缩到更少的tokens；attention 机制也做了改进，保持equal distance between visual tokens and any language tokens，同时ratain relative distance between two language tokens，也就是EDVT attention
3. 实验结果，在Video理解任务上表现不错，zero-shot VQA 多个数据集上达到SOTA结果，video-based text generation 表现也不错

总体上，属于基于 trained with LLM 的多模态路径上一个新颖的尝试，解决时序问题，提升video理解能力，计算量和人力投入应该都不大。

## 2024.1.11

《If LLM Is the Wizard, Then Code Is the Wand》

2024.1 UIUC的一篇文章，对于Code在LLM领域的作用做了系统性梳理，要点：

本文系统性梳理了 Code pretrain 在 LLMs 能力当中的作用，包括code 能力，reasoning能力，capture  structural knowledge能力等；增强LLM与digital tool, physical tool connect的能力等，对于增强 IAs 也非常有益。

没有读的很仔细，粗略扫了一遍，后面有需要再看吧

## 2024.1.10

《ALIGN: Scaling up visual and vision-language representation learning with noisy text supervision》

2021年Google Research的一篇视觉预训练文章，跟CLIP类似，要点：

1. 本文的主要目标是 scale up visual and vision-language representation learning，因此命名  **ALIGN** : **A L**arge-scale **I**ma**G**e and **N**oisy-text embedding，是一个非常好的V和V-L预训练模型
2. 数据，标题里有scaling，数据集的大小也绝不含糊，首先是一个1B的image alt-text pairs；后来增加多语种，扩充到1.8B大小数据集，cover 100+种语言
3. 模型，相对比较简单，image encoder + text encoder + contrastive loss（normalized softmax）
4. 实验结果，因为数据集非常大，预训练效果很好，在zero-shot/SFT的classification任务上，都取得非常好的效果，zero-shot超过CLIP；在T2I，I2T等retrieval任务上，也取得很好的效果

《REDCODER: Retrieval Augmented Code Generation and Summarization》

2021.9 的一篇文章，研究RAG增强Code场景，要点：

1. 想法是针对程序员的高频场景，利用existing high-quality code 和 desc，通过IR召回后引入generation process
2. 模型，retriever 参考了DPR的 two different parallel encoders；generator 用的PLBART
3. 实验结果，对比之前单独基于 retriever 和 generation 的方法，效果提升比较明显

## 2024.1.8

《ShareGPT4V: Improving Large Multi-Modal Models with Better Captions》

2023.11 中科大和上海AILab的一篇文章，关于构建高质量image-text数据集的，要点：

1. 多模态领域缺少足够的高质量数据集，传统数据集中的 Vision 在info和细粒度semantics上都很rich，但是brief的captions只focus在突出的objects上，造成巨大信息丢失；本文推出了一个ShareGPT4V数据集，以及相应的ShareGPT4V-7B 多模态model
2. 数据，先基于GPT4-Vision生成100k高质量captions数据，然后build a strong caption model，expand 数据集到1.2M，avg length 是942 characters
3. 模型，ShareGPT4V-7B model follow LLaVA-1.5 的设计，包含3个integral组件

本文主要是研究高质量captions数据的有效性，能实现LMMs更好的alignment

## 2024.1.7

《Imagen: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding》

2022.5 Google 的一篇text-to-image文章，要点：

1. 本文提出 text-to-image model Imagen，底层都是扩散模型diffusion model，整体框架比 DALLE2 简单很多
2. 模型，Combine transformer based LMs with high-fidelity diffusion models; T5 + 3个diffusion models
3. 实验结果，在COCO数据集，以及本文推出的DrawBench上，比DALLE2, GLIDE, VQGAN+CLIP的结果都要好，是新的SOTA；消融实验发现Increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model

Imagen 的关键发现是，在text-only corpora上预训练的LMs，其text embeddings对于text-to-image synthesis非常有效，效果优于基于CLIP text latents的DALLE2

《MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices》

2023.12 美团和浙大合作的一篇研究Moble VLM的工作，要点：

1. 本文推出 MobleVLM targeted to run on mobile devices，对开元VLM的arch和training做了一定总结
2. 模型，模型结构比较简单，类似 LLaVA 的架构，主干LLM模型采用了 MobileLLaMA，projection 采用了一个downscale的设计LDP结构，总体比较常规；训练数据用了1.3T tokens，是比较大的
3. 实验结果，相对做了比较全的对比，包括与SOTA模型在VLM benchmarks上的性能对比，推理速度，以及vision encoder scales 和 training paradigms 消融实验的对比等。不过这一块的结论不是那么清晰

总体属于对 MobileVLM 一篇还比较务实的研究工作，结论方面感觉不是很深刻

## 2024.1.6

《GLIP: Grounded Language-Image Pre-training》

2022.6 MSRA和UCLA的一篇文章，要点：

1. 本文统一了 object detection（找出图像中所有感兴趣目标，确定它们的类别和位置） 和 visual grounding（输入图片和物体文字描述，找到物体的bounding box） 2个任务，推出GLIP model, learn object-level, language-aware and semantic-rich visual representations，显示出很强的zero-shot和few-shot迁移性，到各种object-level识别任务上。reformulate objective detection as a phrase grounding problem
2. 模型，与CLIP model仅在最后一层fusion vision和language不同，GLIP model 采用deep cross-modality fusion，加强vision和language信号的交互；pre-trained on 27M grounding data, including 3M human annotated fine-grained data and 24M web-crawled image text pairs
3. 实验结果，GLIP show promising results on zero-shot and fine-tuning settings on several benchmarks。由于vision&text信号的深度融合，prompt tuning 结果接近full tuned model效果。

总体上，本文尝试统一目标检测和visual grounding任务，推出一个简洁且强大的预训练model GLIP，取得很好的性能。

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
