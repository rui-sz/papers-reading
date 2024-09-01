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

# 2024.9
## 2024.9.1
《**VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback**》
2016年的一篇文章，是研究 visual signal 引入RS的一篇早期工作，结果具有比较好的参考价值。内容要点：
1，背景：本文研究基于隐式 feedback 的个性化ranking问题，在BPR-MF基础上引入 visual signal
2，数据：用到了2个包含隐式反馈的数据集，amazon.com 和 tradesy.com，其中 cold-start 的占比比较高，可能是后续实验结果不错的一个原因
3，方法：在BPR-MF的基础上，引入 AlexNet 提取的CNN信号，用于捕捉视觉信息对用户决策的影响，BPR-MF 是一个pair-wise的框架（当时的SOTA框架）
4，实验结果，VBPR对比其他 pair-wise 方法有明显的AUC提升；当然对比point-wise方法就更好了。理论上 latent factors 可以uncover任意relevant的信号，但是cold/cool 的items会导致这种方法不是特别有效，在此基础上引入 explicit 的signal，有助于模型学习。实验结果也佐证了这一点。
未来的研究方向，包括 temporal dynamics，以及引入 visual signal 的 efficiency 问题


# 2024.8

## 2024.8.11

《**Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application**》
2024.5 快手的一篇将LLM引入RS的文章，要点：
1，背景：传统推荐引擎的 ID embeddings 忽视了 semantic info of items，导致rec system泛化能力不强，无法泛化到unseen data上，特别是在冷启动和长尾 user recommendation 问题上；LLM 通过预训练拥有 general knowledge，因此有理解和推理能力。而RS的ID emb则 fall within the collaborative knowledge domain，因此 bridge the gap between general and recommendation-specific domains is crucial，可以提高推荐系统的有效性和泛化性

2，想法：本文 explore the learning of user and item embeddings for recommender systems from LLM-based content representations. 提出了 a twin-tower structure supervised by the recommendation task and tailored for practical industrial application. Llm-driven knowlEdge Adaptive RecommeNdation (LEARN) framework

3，数据：快手线上数据，31 million items interacted with by 12 million users over a 10-month period

4，模型：将预训练的 LLM（Baichuan2-7B）用作 item encoders，冻结LLM参数避免遗忘问题，生成item emb，然后用 transformer model 训练得到user emb；模型整体是一个 twin tower 架构，item tower 有三个变种

5，实验结果：在用于快手App场景时，取得了不错的business benefit

总的来说，本文属于将RS和LLM结合的一篇有效实践，synergizes open-world knowledge with collaborative knowledge

## 2024.8.8

《**Retrieval-Augmented Generation for Large Language Models: A Survey**》
2024.3 的一篇文章，本文要点：
1，对 RAG 的发展做了归纳总结，三个阶段：Naive RAG, Advanced RAG, Modular RAG

2，RAG 优化的一些方法：Retrieval Source, Indexing Optimization, Query Optimization, Embedding, Adapter
3，PE，RAG，FT 几种方法的对比，做了比较好的总结归纳

总体来说，没有很多新意，但是从系统性梳理和抽象的角度，是一篇不错的综述文章


# 2024.6

## 2024.6.1

《**Chameleon: Mixed-Modal Early-Fusion Foundation Models**》

FAIR 在2024.5的一篇文章，属于 CM3Leon 工作的后续，本文要点：

1，本文介绍了一个 token based multimodal 模型，在精心设计的多模态数据集上预训练和SFT，最终在多模态理解生成和纯文本任务上都取得良好的效果。

2，数据，text-only data 2.9T，text-image data 1.5T，interleaved data 400B

3，模型，2个版本 7B/34B，采用了 early-fusion 的设计方式，与Gemini结构类似，在 pretrain 模型时为了保证stable有比较多技巧，例如dropout, QK-Norm 的使用

4，实验结果，在多模态任务上，例如VQA，取得了非常好的效果，outperform Flamingo等模型，同时在text任务上，保持了competitive 的结果，可以对抗LLaMA2 34B

# 2024.4

## 2024.4.29

《Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone》

2024.4 微软出的一篇文章，延续基于filtered高质量数据train小模型的思路，要点：

1. 本文工作属于 phi-1和phi-2的延续，基于过滤的高质量数据，以及模型生成数据。相当于phi-2的scale-up versino
2. 模型，在3.5T数据上训练了三个版本的model，训练分为两个阶段：phase-1 teach model general knowledge 和 language understanding；phase-2 使用 more heavily filtered webdata (a subset used in Phase-1)以及synthetic data，教会 model  logical reasoning and various niche skills
3. 实验结果，尽管只有3.8B参数，phi-3 mini 在achedemic datasets上的效果，与Mixtral 8*7B以及GPT-3.5相当

## 2024.4.3

停更了一段时间，近期有点忙于其他事情~论文阅读会继续catch up起来，同时增加代码阅读和实践。

《SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis》

2023.7 Stability 的一篇文章，关于improve SD的，要点：

1. 本文在SD的基础上进行优化，以提高image synthesis的质量。模型代码和参数开源
2. 模型，包括 text-to-image(base)和image-to-image(refine)的两个阶段；相比SD，base部分包含一个3倍大小的UNet；通过cross attention引入了conditioning tech；improved autoencoder；multi-aspect training 等
3. 实验结果，在human preference上，SDXL相比SD基线有很大提升，但是FID和CLIP Score未能反映这一点，体现了对应指标的局限性。

# 2024.3

## 2023.3.22

《(LDM)Diffusion Models: High-Resolution Image Synthesis with Latent Diffusion Models》

2022年一篇对diffusion model进行改进的文章，要点：

1. 本文是潜在扩散模型，LDMs work on compressed latent space of lower dimensionality，Stable diffusion 是 LDM 的一个开源预训练模型，由于它的开源迅速火爆起来
2. 模型，引入一个autoencoder 先对原始对象进行压缩编码，编码后的向量再应用到扩散模型，引入batch-based adversarial objective相比基于像素的L2/L1 loss更能保证图像质量。潜空间的尺寸远远小于像素空间（f是一个参数，本文试验4和8效果比较好），极大了降低了硬件资源的需求，同时也能加速整个过程。增加一个自编码器并没有改变 DDPM 的算法过程，所以并不需要对 DDPM 算法代码做任何改动。另外设计的conditional mechanism，通过在 UNET 中加入 Attention 机制，处理条件变量
3. 实验结果，Achieve new SOTA for image inpainting and class-conditional image synthesis，以及在 unconditional image gen，text-to-image synthesis，超分等任务上的 competitive perf，同时对比 pixel-based DMs [5177]显著降低计算开销

总体上，Pixel-based representations of images 包含 barely perceptible, high-freq details，与pixel-based diffusion approaches相比，本文提出在低维隐空间进行的LDM model，显著降低了计算开销，同时能生成百万像素级更高清的图像。

## 2024.3.16

《InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Models 》

2024.1 上海AI-Lab与香港中文大学、商汤的一篇关于多模态模型的文章，要点：

1. 模型，本文提出了 XComposer2 model，InternLM2 + CLIP ViT-L，引入partial LoRA（PLoRA）的设计，在保留LLM knowledge 的情况下，引入visual特征，本model不能生成图片，但是可以在生成的text中插入图片
2. 实验结果，在 vision-language understanding and free-form text-image composition 上展现了很强的能力，outperform 了一众 open-source models，并且在与closed-source model的对比中也展示出了竞争力

总体上，在 multimodal 理解，以及long-text multi-modal content composition 上能力很强

## 2024.3.11

《ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation》

2023.12 的一篇文章，研究T2I model评估指标和improve问题，要点：

1. 本文提出了一个基于 human preference feedback 评估和提高 text-to-image models 的方法
2. 模型，基于 BLIP backbone 训练了一个分类model，评估结果显示在与 Human preference 对齐上比 CLIP score 和 FID 明显要好，单张图片也更有区分度。

总体上，属于研究T2I模型评估的一篇很好的工作，ImageReward 指标比CLIP score和FID更有优势。

《CLIPScore: A Reference-free Evaluation Metric for Image Captioning》

2021年底2022年初，研究image caption任务的评估指标，要点：

1. 此前 image captioning 依赖 reference-based automatic evaluations，本文提出了基于CLIP model不依赖 reference 的model
2. 模型，用CLIP输出的text/image emb，计算cos距离，作为相似度衡量；ref 版本在此基础上引入了跟人工标注样本集的距离进行增强
3. 实验结果，跟人工判断的一致性比之前评估指标（BLEU, ROUGE-L, METEOR, CIDEr）都要高一些

总体上，属于利用CLIP强大的cross-modal emb对齐能力，方法比较简单，在早期应该效果不错。

## 2024.3.5

《COMET**: A Neural Framework for MT Evaluation》

2020.1 Unbabel AI 的一篇机器翻译评估指标文章，要点：

1. BLEU指标，是基于 MT-generated hypothesis and a human-generated reference translation in the target language. 存在一些弱点
2. 本文推出COMET model，一个多语言翻译模型，验证了三种目标：DA,HTER,MQM，基础模型就是 XLM-RoBERTa 的model，在上面加了结构做回归和分类（详细了解TODO）
3. 实验结果，相比BLEU效果要好很多，是更优秀的MT评估指标

## 2024.3.3

《Generative AI for Translation in 2024》

2024.2 的一篇文章，系统性的研究了主流LLMs在翻译性能上与NMT models的对比关系

要点

1. LLM 基本上在性能上可以match top MT model，PaLM2 的翻译性能表现最好，Gemini pro 的翻译性能还不错；在OpenAI model中，GPT4 -Turbo 的表现最好，GPT 3.5Turbo并不弱，差的不多；LLaMA2 对比闭源模型还有比较大差距
2. 1M长度文本的翻译成本方面GPT系列与传统MT服务差不多，Google 的大模型服务要明显低一些，低20倍以上
3. LLM 的翻译速度相比NMT要慢100倍以上

# 2024.2

## 2024.2.28

《ADM: Diffusion Models Beat GANs on Image Synthesis》

2021.5 OpenAI的一篇工作，研究用diffusion model生成图像，要点：

1. 本文在DDPM基础上做了一系列改进，以达到更好的 image sample quality to the current state-of-the-art generative models，也就是GANs
2. 模型，a method for using gradients from a classifier to guide a diffusion model during sampling，controlled generative models with a pre-trained classifier
3. 实验结果，SOTA on unconditional image synthesis and with classifier guidance on conditional image synthesis，match BigGAN-deep even with as few as 25 forward passes per sample

总体上，本文在diffusion模型结构上做了有效改进，同时通过引入guidance trade off diversity for fidelity

## 2024.2.25

《Improved denoising diffusion probabilistic models》

2021年OpenAI的一篇文章，基于DDPM做了一些改进，要点：

1. 之前尚未证明DDPM相比其它模型（例如autoregressive model）可以获得 competitive log-likelihood 性能，而log-likelihood 是广泛用于生成模型的指标，一般认为优化LL可以促使model capture所有数据分布形式。本文目标对DDPM模型进行优化，让其具有更好的 log-likelihood 性能，同时保持 sample quality 较高
2. 模型，几个改进项：正向diffusion从linear改为cosine；优化损失函数；通过消融实验可以看到这2个都有正向效果。同时 sampling 的速度也有提升，相比较而言DDPM需要上百steps sample得到好样本
3. 实验结果，FID性能与模型计算量呈log线性关系，NLL弱线性，说明diffusion模型是 scalable 的。在likelihood建模方面，DDPM相比CNN-based方法更好，但是比transformer-based方法要弱。

不过本文方法的计算量更大一些，1000 steps->4000 steps

## 2024.2.20

《VideoGPT: Video Generation using VQ-VAE and **Transformers**》

2021年UC Berkeley的一篇video生成模型的文章，要点：

1. 本文是用 transformer 生成 video 的一篇工作，如标题用到了VQ-VAE和transformer
2. 模型，VQ-VAE + Transformer，VQVAE encoder + autoregresive model + VQVAE decoder

总体上，是用 transformer 生成video一个不错的尝试

## 2024.2.19

《DDPM: Denoising diffusion probabilistic models》

2020年diffusion model用于图像生成的奠基作，要点：

1. 本文被认为是 diffusion model 用于图像生成领域的奠基作，解决了DPM的一些不足，使得图像生成质量大幅提高，使得扩散模型在图像生成领域大放异彩。对前向、后向的过程进行了比较严谨的分析和证明，对reverse process采用U-Net结构建模
2. 模型，reverse的预测过程是比较简单的，predict 添加的噪声（预测噪声比预测原始的x0要容易的多），一个细节是，最后生成的图像怎么来，不是 x0 hat 的采样值，而是其期望
3. 实验结果，能够生成高质量图像，并且可以在diffusion steps中间对不同图片做插值，来做图像融合

无论是 DDPM 还是 DDIM，这些扩散模型在生成图片时，都是输入一个随机高斯噪声数据， 然后逐步的产出一张有意的真实图片。这个过程中每一步都是一个随机过程，所以每次执行产出的图片都不一样， 生成的图像多样性非常好。但是不能对生成结果做控制。

## 2024.2.16

《Diffusion Transformer: Scalable Diffusion Models with Transformers》

2022.12 Meta 实习生的一篇工作，要点：

1. 本文研究基于 transformer architecture 扩展 diffusion models，VAE + diffusion transformer
2. 模型，transformer-based DDPMs 基于 image + condition 的input，输出 latent emb，VAE decoder 再输出最终图片，本文探索的 DiT design space 包括：patch size, transformer block arch，model size
3. 实验结果，adaLN-Zero 的DiT设计更有效，model size，patch size，Gflops对模型性能（FID，定性评估等）都有比较显著影响

总体上，本文属于 transformer based diffusion model 一篇很不错的文章，并且做了很好的消融实验。不足之处是，似乎本文主要还是为了解决图像生成问题，并不能作为大一统 foundation model

## 2024.2.15

《Prompting Large Language Model for Machine Translation: A Case Study》

2023.1 Edinburgh 大学基于ChatGLM研究LLM MT的一篇文章，重点关注prompt技巧，要点：

1. 本文重点研究 prompt LLM do MT，关注如何让 prompting works for MT，研究了 prompt template 和 few-shot examples selection
2. 方法，基于ChatGLM-130B model（这个模型本身性能应该一般，所以结论不一定具有普适性），研究了 template 和 template language 的影响，language 方面英语表现最好，尽管pretrain语料中英语和中文token相当；
3. 实验结果，*在选择few shot examples 时，high translation quality, high * *semantic similarity**, high **LLM** likelihood, long sequence length and high similarity to test inputs are all preferable strategies. 但是相关性都比较弱，迁移性一般；单语言examples以及random pairs表现都比较一般，pseudo parallel data表现不错，表明了在demo中保持正确mapping的作用*

## 2024.2.14

《One-peace: EXPLORING ONE GENERAL REPRESENTATION MODEL TOWARD UNLIMITED MODALITIES》

2023.5 达摩院和华科合作的一篇文章，要点：

1. 本文提出了一个 scalable way for building a general representation model across different modalities，其思想中有很强的 ImageBind 和 BEiT v3 的影子
2. 模型，modality Adaptors + Fusion Encoder(cross attention + FFNs) 的模型结构，loss 包括 cross-modal aligning 和 intra modal denoising 两种，分别用于跨模态对齐以及单模态特征学习；在训练时，两阶段使用 image-text 和 audio-text 数据； 模型尺寸4B
3. 实验结果，在 vision/audio/visio-language等任务上对比之前SOTA都展示出了很强的能力，同时展示出了很强的跨模态zero shot能力，尽管训练数据中并未出现相应的模态组合

总体上，在数据使用上的思想 Like Imagebind，只不过本文是将image/audio与text对齐，其训练思想有点类似 BEiT v3，分阶段训练网络的不同部分。从结果来看是一篇很不错的文章，只不过创新性稍有不足。

## 2024.2.12

VQVAE实际上只是一种类似于AE的编码解码器，并没有生成功能，它需要联合如PixelCNN等模型才能进行生成。

用 VQVAE 生成图像的步骤

1. 训练VQVAE
   1. Image1 -> Encoder（emb1） + cookbook（emb2） + decoder -> image2
2. 用VQVAE重建数据集里的随机数据
   1. 重建，就是模拟训练的过程，随机取一些图片，先编码后解码，看解码出来的图片和原图片是否一致。
   2. 重建效果还是很重要的，它决定了该方法做图像生成的质量上限
   3. 重建过程中的中间数据 image1->emb2 作为 PixelCNN 的训练数据
3. 训练PixelCNN
   1. 基于上述 image1->emb2 训练生成压缩图片（离散编码图片）
   2. 这样 PixelCNN 就可以用空白图片作为输入生成压缩图片了，再输入VQVAE做解码
4. 用PixelCNN+VQVAE随机生成图片
   1. PixelCNN 生成压缩图片，然后VQVAE解码

## 2024.2.11

《PixelCNN: Pixel recurrent neural networks》

2016年 DeepMind 的一篇自回归图像生成模型的文章，要点：

1. 本文属于运用 autoregressive modeling 方法建模图像生成的文章，这与VAE的方法显著不同？
2. 模型，PixelRNN 和 PixelCNN，主要区别在于第一层 conv Mask A 提取特征之后的若干层网络结构上，前者是LSTM，后者是CNN，Recurrent Neural Networks (RNN) are powerful models that offer a compact, shared parametrization of a series of conditional distributions，CNN-based model属于前者的变种，同时作者为了有效训练更深的网络，引入了 residual connection 结构

## 2024.2.7

《VQVAE: *Neural Discrete Representation Learning*》

2017年DeepMind的一篇工作，跟 PixelCNN强关联，要点如下：

1. AutoEncoder的重构思想就是 **用低纬度的latent code分布来表达高纬度的数据分布** ，但是潜空间的编码点相互之间关系不大，基本上只能用来抽特征做分类，不能用来做图像生成。VAE和VQVAE的重构思想是通过 **设计latent code的分布形式** ，进而控制图片生成的过程。
2. VAE是增大自由度，VQVAE 相比VAE控制了自由度。VQVAE通过Encoder学习出中间编码，然后通过最邻近搜索将中间编码映射为codebook中K个向量之一，然后通过Decoder对latent code进行重建。

VQ-VAE 与 VAE 的主要2点区别：

1. encoder 部分输出的是 discrete codes，而不是 continuous latent variables
2. learnt rather than static

VQVAE相比于VAE最大的不同是 **直接找每个属性的离散值，通过类似于查表的方式** ，计算codebook和中间编码的最近邻作为latent code。由于维护了一个codebook，编码范围更加可控，VQVAE相对于VAE， **可以生成更大更高清的图片** (这也为后续DALLE和VQGAN的出现做了铺垫)。

## 2024.2.6

《VAE: Auto-encoding variational bayes》

2013年Kingma的一篇文章，在AE基础上将其改造成能做数据生成，要点：

AE：

    x->z->x'，最小化损失函数 L(x,x')

    目的是提取抽象特征z，或者数据降维

VAE：

    总体是一个生成式模型，在AE思想的基础上引入变分的思想，使其能够进行数据生成。

    中间学习出一个分布（正态分布），再采样得到特征，而不是直接学特征

    VAE可以理解为通过网络学习出每个属性正态分布的mean和std编码，然后通过mean和std和N ( 0,1 )正态分布恢复每个属性的正态分布，最后随机采样得到每个属性的离散值。

    鲁棒性会更好

    总体上，VAE是AEVB算法的一个具体例子，证明了重参数化的变分下界的，使其可以用标准随机梯度法直接优化，同时VAE生成模型的缺陷就是生成的图片相对模糊（why？），VAE提出后，很多学者针对这个问题进行了优化。如结合GAN等。

    理论性比较强的一篇文章。

## 2024.2.4

《Sparse transformer: Generating Long Sequences with Sparse Transformers》

2019.4，OpenAI Ilya 领衔的一篇研究Transformer 平方计算问题的文章，要点：

1. 本文的出发点为了解决随着 sequence 增长，transformer attention 部分占用内存和计算开销呈平方复杂度的问题，Scaling up autoregressive generative models，让 Transformer Arch 更适合 modeling long sequences
2. 本文提出了一种对 attention matrix 进行sparse factorization 的方法，将attention复杂度降到 n*sqrt(n)，以及其他几种优化：一种网络变种能train更深的网络；在backward pass中对attention matrices进行recompution以节省存储；Fast attention kernels for training

总体上，本文是解决Transformer attention平方问题的早期论文，在图像生成问题上揭示了attention的原罪，就是其实不需要那么密集的注意力，启发了后续的一系列工作

《Emu: Generative Pretraining in Multimodality》

2023.7 BAAI 联合清华、北大的一篇多模态文章，要点：

1. 本文推出Emu，一个Transformer-based 多模态基础模型，采用了统一的 autoregressive objective of predicting the next element, including both visual and textual tokens，这种建模方式，可以被用作 generalist interface
2. 数据，除了传统的 image-text 数据集之外，本文基于 video 提取了 interleaved image-text 数据
3. 模型，visual encoder + causal transformer + LLM + diffusion decoder，14B 模型参数量；训练有三个阶段，pretraining、diffusion model FT、instruction finetuning
4. 实验结果，在 zero shot/few shot evaluation 上都表现出比较好的效果，模型有较好的ICL能力

## 2024.2.2

《Large Language Models Are State-of-the-Art Evaluators of Translation Quality》

2023.5 微软的一篇文章，研究GPT模型评估翻译质量，要点：

1. 本文的总体思路就是用GPT模型来对翻译结果进行评价，构建4种不同的prompt（2 scoring + 2 classification），以及2种不同的mode（human-ref, not）让GPT给出判断结果，再与人工打分对比给出accuracy
2. 实验结果，GEMBA-GPT4的评估准确率要显著高于model-based COMET22，以及大幅高于传统 BLEU 分

本文用的Zero shot prompt 的方式来询问GPT，后续few-shot，以及SFT都是继续可以尝试的方向，同时可能也会启发后续 document-level 的MT工作，构建更好用的metrics

《DALLE3: Improving Image Generation with Better Captions》

2023.10 CloseAI的一篇T2I工作，要点：

1. 本文展示了T2I model的prompt following abilities 可以被 highly descriptive generated image captions 显著增强，当前模型在这方面表现不好，假设是由于训练数据中 noisy and inaccurate image captions 导致的；同时T2I model 的一个严峻挑战是 controllability 不足。对于这些问题，本文通过caption improvement来解决
2. 模型，总体上采用2个阶段，re-captioning + diffusion model，通过训练一个 descriptive 的 re-captioning model，生成 Synthetic data，基于这个更丰富的 caption 数据来训练 diffusion model
3. 实验结果，对比 DALLE2 和 其他 SD model 都有显著效果提升，采用了 automatic/human evaluation等不同方法；评估工作总体讲的比较少

总体上，本文通过增强 caption 数据来优化了 T2I model的 prompt following 能力，取得很好的效果，遗憾的是对于模型细节披露太少，只讲了数据处理这部分工作。

# 2024.1

## 2024.1.30

《How Good Are GPT Models at Machine Translation? A Comprehensive Evaluation》

2023.2 Microsoft 的一篇关于 GPT model MT 能力的系统评估文章，要点：

1. 本文主要研究GPT model的MT性能，做了一个全面的评估，三种不同版本GPT model，对比一些基于WMT数据集的最优模型
2. 实验结果，在high resources 语言上，GPT models 获得了非常有竞争力的结果，但是在low resources语言上，能力相对有限。
3. 本文进一步全面分析理解了GPT translations的特征，对比NMT model，对于获取关于GPT models for translations的潜力和限制有启发。

## 2024.1.29

《MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models》

2023.12 腾讯优图实验室和厦门大学的一篇多模态模型评估工作，要点：

1. 本文提出了一个 comprehensive dataset MME, to evaluate MLLMs的性能
2. 数据，包括14个不同的数据集，涵盖 perception and cognition task，preception包含 coarse-grained 和 fine-grained，以及OCR任务；指令和回答经过特殊设计，非常简洁，以保证对所有模型公平，并且指标易于计算
3. 模型，评估了30个 up-to-date 的MLLMs，给出了 preception 和 cogintion 的leaderboards

## 2024.1.26

《Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data》

2023.12 MSRA 的一篇工作，围绕如何生成高质量数据FT 弱model，要点：

1. 本文提出了一个使用ChatGPT生成高质量 multi-turn chat corpus 数据的pipeline，接着使用 param-efficient LoRA方法来FT LLaMA model，提出了SDF方法（相当于蒸馏ChatGPT），得到Baize
2. 模型，基于LLaMA，用LoRA进行finetuning，在过程中用到了 self-distillation finetuning 技术
3. 实验结果，从定量定性的评估来看，本文方法都还比较有效，同样参数规模能compete Vicuna等

## 2024.1.23

《Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors》

2022.3 Meta 的一篇 text-to-image 优化工作，要点：

1. Text-to-image 面临的几个重要问题，Controllability，用文字控制 style/color 比较容易，但是structure, form, or arrangement就很难描述；bounding box 控制；Human preception，对齐人的图像偏好；Quality and resolution
2. 本文 provides a new type of control complementary to text，通过增加scene input的方式，有比较多好处，比如更好的控制性，在生成系列图片时更好的一致性（story teller）
3. 模型，骨干部分是一个GPT3的 decoder-only transformer，text/scene/image 分别通过相应的Encoder来编码成token，scene 在推理时是可选的；为了增强VQ编码的效果，使用了一些技巧，例如face-aware/object-aware等
4. 实验结果，从定性和定量评估上都取得很不错的效果，FID的对比超过DALLE和GLIDE等，消融实验表明 face-aware, CF, scene input 等优化机制都有不错的效果

总体上，本文对于 text-to-image 问题增加控制性和提高生成质量，是一个比较新颖的尝试，也取得不错效果。

## 2024.1.21

《DALLE: Zero-Shot Text-to-Image Generation》

2021.2 OpenAI的一篇text-to-image model文章，要点：

1. 传统的 text-to-image generation聚焦于寻找更好的模型假设，例如复杂 arch，辅助loss，或者伴随信息object part labels, seg masks等，直接用pixel来做token的话，造成大量的内存消耗，本文描述一个简单的方法，把text,image tokens当做data seq，
2. 模型，autoregressive model 是基于sparse transformer，image tokenizer 使用了VQ-VAE2，text 和 image 的token vocab size分别是16384和8192；模型大小12B，mask attention使用 conv 类型的
3. 实验结果，相比之前GAN的方法，在真实性和image-captain匹配上都在90+%评估中更好；FID分数，在MS-COCO上表现跟GAN相当，但是在CUB上差的比较多

总体上，DALLE使用统一的decoder-only transformer来训练text-to-image model，方法上比较新颖简洁，效果还可以。不过DALLE 论文给出的公式信息并不多，会影响部分理解。

《CM3: A causal masked multimodal model of the internet》

2022.1 FAIR 的一篇工作，研究decoder-only的多模态模型，要点：

1. 本文推出一个CM3（**C**ausally-**M**asked **M**ultimodal **M**odeling） model，基于 hyper-text 结构化数据（full document structure including images and hypertext links）训练，用到 causally masked objective，因此具备 bidirectional context 理解能力
2. 模型，decoder-only 的架构，引入了VQVAE-GAN来做Visual tokenization，重点训练了2个尺寸 2.7 billion（CM3-Medium） and 13 billion（CM3-Large）
3. 实验结果，在prompt的时候，结合html类型数据格式，可以适应于多种类型下游任务，例如conditional/unconditional image生成，text-image/image-text等

总体上本文工作属于[5157] HTML 的延伸，使用了VQVAE-GAN的visual tokens，以及 causal masked objective，并scaling up 了一个数量级，适用多种任务类型，同时与T5对比，masked language model 性能相当，未损失性能。

## 2024.1.20

《(CM3leon)Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning》

2023.9 FAIR的一篇工作，关于 token based 多模态模型，要点：

1. 本文展示了 text-only 的 autoregressive models 也可以很好的用于 text 和 image generation，推出了 CM3leon，RAG 增强训练，token-based，decoder-only multimodal model，可以生成和填充text, image
2. 模型，同CM3 decoder-only transformer architecture，去掉了 bias terms, dropout, and learnable parameters for layer norms and use a sequence length of 4096 instead of 2048，在3B的tokens数据集上训练；RAG 增强的方法，希望在pretrain阶段引入相关的、多样化数据。
3. 实验结果，pretrain + SFT，在图像编辑/生成，以及文本生成方面都展示出了比较好的能力。

总体上，本文用 decoder-only model来解决图像生成问题，extend the scope of autoregressive model，展示了compete with and exceed diffusion model的能力，在效率和质量方面。

## 2024.1.19

《Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond》

2023.10 alibaba 的一个多模态model，要点：

1. 本文推出Qwen-VL大概10B参数量，面向的任务类型：image captioning, question answering, text-oriented question answering/OCR, and visual grounding
2. 数据，三阶段训练分别采用了不同数据集，一阶段是weakly labelled image-text pairs，二阶段是基于高质量数据的multitask pretraining，三阶段是supervised finetuning
3. 模型，架构上采用了openclip ViT+Position-aware Vision-Language Adapter + QwenLM，总参数量大概10B
4. 实验结果，在多个种类的图文tasks上，比如captaining，VQA，OCR，grounding等任务上，都表现出很强的性能，可以说比BLIP、LLaVA、Flamingo等模型都要好，也表现出了很好的few-shot learning能力

总体上，本文基于两个预训练基座组装了一个V-L model，在训练上采用了一些技巧，取得很好的评估结果，未来发展方向，包括拓展更多的模态，继续scaling up the model size，增加多模态生成能力

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

《InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning》

2023.6 Salesforce 研究V-L模型指令微调的工作，要点：

1. 本文的出发点，是受language model启发要研究 V-L model 的 instruction tuning 问题，InstructBLIP 可以理解为是 BLIP2 + 指令微调
2. 数据，本文 uses a much wider range of vision-language instruction data, covering both template-based converted data and LLM-generated data. 26个dataset
3. 模型，在BLIP-2的模型架构上做了一些改进，引入instruction-aware visual feature extraction，Balancing training dataset 等
4. 实验结果，在多个数据集上outperform了BLIP-2和Flamingo；同时通过消融实验证明，instruction template 有多样性的 instruction tuning 对于模型泛化到 unseen tasks 上的性能contribute比较大，而单纯的 multitask learning 则作用有限；InstructBLIP在下游任务上FT的效果也好于BLIP-2，说明其是一个更好的 for specific-task FT的基线model

总体上，是 V-L model 做指令微调的一篇很好的文章，工作扎实，效果好，同时消融实验的insight比较有价值。

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
