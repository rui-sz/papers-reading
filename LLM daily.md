# LLM_follow
记录学习材料，daily


## TBD
《The growing-up story of Language models》  
ChatGPT账号


## 2023.6.9

《李沐：单卡、多卡 Bert/GPT-2 性能对比》  
    模型参数量：  
        Bert: 1亿、3亿  
        GPT-2: 1亿、3亿、7.6亿、large 1.5B  
    测试卡种：N厂30x系列显卡  
    模型实现：HuggingFace, Megatron  
    指标：samples/sec, per GPU TFLOPS, per GPU memory  
    单卡测试：  
        Batch 大小调整，尽可能大  
        内存占用组成：model parameters, layer output(activations), backend libraries  
        grad_accum，梯度累加也是增大batch的一种方式，减少计算量。不过pretrain可以尽量增加，FT时可能影响收敛  
        三个关键点：Batch 尽量大，浮点类型fp16/8，Megatron 库  
    多卡并行测试  
        数据并行（Megatron）
            NV link 在GPU通讯方面帮助很大，5~10倍提升  
        Tensor并行  
            把网络切分，分开计算，能够训练很大的模型  
            缺点：不再能通过梯度累加来降低开销  
        Zero  
            把模型、Adam状态等切分，每个GPU维持一部分  

## 2023.6.8

《HELM: Holistic Evaluation of Language Models》  
    2022.11, Percy Liang@Stanford 的一篇大作，系统性评估了当前主要的一些LLM  
    非常基础性的工作，其研究方法和结论目前时段来看都比较有价值  
    核心内容：  
    1，选择了现在常见的30个语言模型（参数量310M->530B），设计了7个评估指标，用到16个数据集  
    2，涉及到的NLP领域：QA,IR,Summarization,Sentiment Analysis,Toxicity detection,Classification  
    3，评估结果中 InstructGPT, Anthropic-LM, GLM 这几个模型可以重点说下，InstructGPT综合各方面是最好的，Anthropic 这个模型也比较牛，只有52B参数，各方面性能不虚 InstructGPT 很多，GLM 也很不错，国产之光！  
    4，摘要任务各个模型表现都不太好，作者分析了一下可能是当前摘要评估数据集标注都不太准  

    160+页的大制作，系统看完的话应该能对当前LLM领域有全面的认识，但是很难看得完啊，就过了一遍李沐的视频讲解。paper中的研究方法，对一些问题的处理思路还是挺新颖的，比如如何设计 prompt 来adapt各个模型，以评估我们需要的指标。作者们做了大量扎实的工作，respect  

## 2023.6.7

《Codex: Evaluating Large Language Models Trained on Code》  
    2020.8 的一篇文章，这会是若干年后程序员大规模失业的缘起吗，让我们师母已呆~  
    核心idea：  
    1，Codex 是基于GPT的模型，使用github代码做训练（微调真香），解决实际问题的一篇文章  
    2，评估方式：正确性判断（NLP BLEU score不合适） pass@k，用了HumanEval 评估集，164个编程问题，签名、文档、单元测试。生成多个答案，top K  
    3，数据集，github 2020年基于54M project，179G python 文件，经过过滤后得到159G  
    4，模型训练，用GPT3的参数，只是收敛的更快，精度上并没有变好，重新训练效果也OK。在推理时nucleus sampling 算法引入多样性，同时保证效果；精度 28.8%  
    5，supervised fine tune，HumanEval 评估跟github数据不对齐，另外搜集了一批更对齐的数据集（竞赛、CI数据等），Codex-S 模型 提升比较多  
    6，Docstring，函数实现->文档，改变训练方法，签名、实现、docstring  
    7，局限性：样本有效性不够；docstring写的越长，代码效果越差，不能理解长指令；数学（精确问题）做的不好；训练数据 bias 问题 等； 抄袭等法务风险  

    总体来说，模型思路与GPT/GPT3保持一致，作者花了大量时间准备数据，想评估方法，比较理想的一种范式  

paper 原文待读  

## 2023.6.6

《Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding》  
    对NLP领域贡献非常大的一篇文章，后续的很多任务可以用这个简单的统一架构来做  
    核心idea：
    1，基于双向 Transformer，可以同时利用左右两侧上下文信息，因此对语言理解类任务很擅长，融合了EMLo（双向RNN架构）和GPT（单向）的思想。名字也来自芝麻街，like EMLo~  
    2，在预训练阶段，使用了Masked LM 掩码语言模型（train 阶段随机mask，预测masked的词），Wordpiece 分词后15%的会做；FT 时不做Mask  
    3，两类预训练任务，Feature based（ELMo、word emb）, fine-tuning（GPT）  
    4，Bert的模型调参，调L/H/A 两个版本，base 1亿参数，large 3亿参数，当时已经是非常大的模型了  
    5，训练输入：Transformer 模型输入是一个seq pair，Bert 输入一个seq，过了嵌入层之后，输入的emb是几个emb的相加，token emb+sentence emb+pos emb，其中pos emb是学习得来的  
    6，Bert 把句子pair都放进去，self attention 可以看到所有的内容，这跟只能看到一部分的 Transformer不太一样，带来的代价是 Bert 就不能做机器翻译了，同时文本摘要等生成任务也不好做了  
    7，微调模型使用TPU跑1小时，在11个任务上取得SOTA，GLUE 多分类，SQuAD QA 数据集 ，SWAG 判断2个句子关系等  

    总体感觉模型的架构设计非常简洁，思路也很清楚，不愧为NLP里程碑著作。遗憾的是模型道路选择可能也间接注定了后续与GPT系列的效果对比  

paper 原文待读  

## 2023.6.4

《InstructGPT：Training language models to follow instructions
with human feedback》  
    GPT系列承上启下的一篇文章，引入了RLHF这个关键组件，也称GPT3.5  
    核心idea：  
    1，SFT（13k）, 人工标记prompt+标注答案，基于GPT3模型做了 16 轮迭代fine tune  
    2，RM（33k），基于6B GPT3模型，每个prompt生成9个答案标注排序，pair-wise ranking损失。RM用于模拟人的反馈  
    3，RL（31k样本），PPO 优化算法，也是OpenAI之前的工作，加散度项控制模型不要跑太远，同时加了原始语言模型目标项  
    4，model alignment 的训练代价是比较低的，只有万级别样本。不过到底align到哪里？  
    5，局限性，请了40个合同工，数据可能有bias；本文只显式优化了帮助性这一点，对真实性、无害等没有证明太多；另外RL步骤有可能并不需要做，增大SFT数据集可能也可以  

paper 原文待读  

## 2023.6.2

《GPT3: Language Models are Few-Shot Learners》  
    2020.5, GPT3  
    核心idea：  
    1，回到 few shot paradigm（预测时不做梯度更新），模型可学参数量拉到175B  
    2，FT，需要与任务相关的带标数据集，以及FT过程  
    3，FS/1S/0S：FS 不参与模型训练，不更新梯度，希望在attention+FFN等环节抽取出样例中的有用信息，学习在上下文中开展，因此叫做 in-context learning  
    4，训练数据：Common Crawl+WebText+Books+Wiki，约45TB，3000亿token  
    5，Prompt：自然语言prompt对性能提升有帮助，这也是现在prompt eng大行其道的原因吧  
    6，结果总体当然惊艳，也展示了在一些场景，few-hot GPT已经超过了当前 FT SOTA 模型，未来可能弃用FT  
    7，模型局限性：文本生成还比较弱，只能往前看，文本的重要性，多模态的缺失，样本有效性不够，成本，解释性等等  


## 2023.5.31

《GPT2: Language Models are Unsupervised Multitask Learners》
    2019年，GPT2  
    核心idea：  
    1，1.5B参数，no FT，zero-shot 的能力  
    2，复用了MQAN文章思想，把multitask以无标文本序列来表达，这种训练是可行的，但是比监督要慢。这就是unsupervised multitask learning  
    3，训练数据 WebText：Common Crawl数据选优 + Reddit，共40GB  
    4，测试结果显示当前模型 underfit WebText 数据集，还有进一步提升的空间  


## 2023.5.29

《GPT1: Improving Language Understanding by Generative Pre-Training》  
    2018年文章，GPT模型系列第一篇 GPT1  
    核心idea：  
    1，generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on specific task  
    2，无监督预训练，难点1目标函数怎么设计（语言模型），难点2怎么迁移学习到的表示到目标任务上面（精调），本文用了一个12层的transformer decoder 架构  
    3，fine tune 监督模型，对几类任务（分类、蕴含、相似度、多选）的输入输出做了设计  
    4，训练数据集：7k本书；4个月后的Bert-base数据集相当，Bert-large 用了4倍数据集3倍参数量  
    5，实验，9/12个任务outperform了当时SOTA结果，已经非常不错，同时作者验证了layers和updates对性能的影响，layers 达到12之后性能提升幅度不大了；消融实验也证明了预训练和FT时辅助语言目标的作用  
    6，作者对于继续提升 unsupervised learning paradigm 也充满期待  

    GPT 系列选择了一条更难的路，预测 instead of 完形填空，注定了其发展需要经历一段时间的蛰伏~  

## 2023.5.28

大模型部署  
    以 langchain-ChatGLM-6B 为例在公司开发机上部署了一下大模型  
    花了差不多3h，最终把 model launch 起来还挺激动的，不过这个模型效果初步试用很一般  
    意外惊喜是发现公司开发机竟然有很多空闲 GPU 资源可以用，太赞了！  
    下周把模型 train 和 fine-tune 跑一遍，打通这个路径之后，就可以各种耍起了，嘿嘿  

《PaLM2 technical report》  
    基本框架跟GPT4的类似，篇幅明显长很多  
    比较有意思的点，论证了训练token数量和模型参数量的关系，要同步提升效果才好，echo了Huffman 2022的论文  
    另外如果总成本（train+inference）受限，那么在小模型上用更多token训练，效率更高一些  

## 2023.5.27

《GPT-4 Technical Report》  
    2023.3 GPT4发布，Tech report之前只看过片段，今天完整看了下  
    核心内容：  
    1，讲了如何基于一个更小的模型去预测大模型性能，对于动辄百万刀的高昂训练成本这非常重要，主要参考loss、HumanEval dataset 等评估指标，比较好的符合 power law 分布（感觉AI现在的发展很实验科学了）  
    2，之前 Inverse Scaling Prize 中的几个问题，随着模型变大效果会变差，本文评估发现随着模型进一步变大，性能又回升了，也就是 U 形  
    3，模型能力评估（秀肌肉），在若干学术和专业测试上，GPT4比GPT3.5都取得了巨大的提升，并且GPT4(no vision) vs GPT4并没有出现明显性能下降，这说明模型泛化能力很好。另外在 Leetcode 问题上，模型提升很快，程序员距离失业是不是又进了一步 TT  
    4，在若干 NLP 的benchmarks 上，相比SOTA模型以及GPT3.5，继续取得较大提升，并且表现出强跨语言能力  
    5，在减少 hallucination 方面有比较大进步，但仍未彻底解决；另外在模型安全方面继续做了优化，主要由 RLHF 和 RBRMs 构成，从评估指标看，安全性有较大提升，同时也减少了GPT4-base的过度谨慎  

    在多模态能力方面讲的内容比较少，只给了一个例子~  

## 2023.5.25

今天没读paper，看了几篇网文，几个分享doc，说一些思考：  
    1，搜索引擎，是一个信息检索工具，它索引了海量的互联网已有信息，让用户通过query高效检索  
    2，ChatGPT，是一个生成式AI，在学习海量互联网数据后，它建立了一个超大的概率语言模型，能够针对问题生成回答，回答是基于概率计算的（因此目前它还没有Ground Truth的概念），因此它拥有知识，它也通过思考创造信息  
    3，人有哪些能力，人会学习，拥有知识，会思考，拥有情感，这些能力除了情感，ChatGPT似乎都有了，不过人是高度个性化的，人的学习能力、知识水平、思考方式都各不相同  
    4，搜索引擎和ChatGPT，对于使用者来说，都相当于一个信息库，它们最大的区别是搜索引擎只索引全网已有信息，而ChatGPT能创造，这是它们的差异。它们的共同点是，对于使用者都一无所知，因此给出的答案是不区分用户的，而实际上每个人的知识背景、学习能力、偏好不同，AGI 在与人交互时应该考虑到这一点，而要考虑到这一点，AGI 也必然需要充分了解每个用户，这就相当于我们在 AGI 的模型当中拥有一个数字分身。我想这是未来AGI要发展成个人助理的一个比较理想的状态，当然在此之前，在帮助人类高效获取信息、创造等方面也有巨大潜力。  
    5，另外我觉得一些核心价值是数据/信息的网站，未来被取代掉的可能性非常大，比如豆瓣这种，搜索引擎还会导流去你的网站，ChatGPT把你的数据都内化了，它是唯一跟用户交互的入口，同时它自己也可以理解内容、创造内容。  

    没时间看paper，就写点想法吧，好多东西要学习，时间不够~  

## 2023.5.24

《Attention is all you need》  
    2017 年的文章，Transformer 大作，作为近几年大伙的生成式NLP模型的基石，还是有必要仔细温故一遍的  
    Transformer 模型架构的核心要素有这些：  
    1，MHA，用 attention 机制替换 recurrent 机制，在性能和计算效率上都有大幅提升  
    2，Res connection，能加速模型学习，降低复杂度  
    3，Layer Norm，作者比较了基于样本的 Layer Norm 和基于Feature的Batch Norm，前者更好  
    4，Encoder 和 Decoder 各包含6层网络，输入分别是叠加了positional encoding的input emb和output emb  
    5，位于 Encoder/Decoder子网络内部的 Feed Forward 层，是简单的两层线性变换，相当于1*1卷积  
    
    测试结果非常好，对比之前的SOTA模型，BLEU值优化很多，同时计算cost有大幅降低。  

    另外，本文对比了几种不同Layer（Self attention、Recurrent、Conv）的几方面属性：
    1，计算复杂度  
    2，Sequential operations  
    3，Max Path Length
    这个对比差异也是构成三大特征抽取器的理论依据（这点后续再解释）  


## 2023.5.21

阅读《人工智能 LLM 革命前夜：一文读懂横扫自然语言处理的 Transformer 模型》  
    快速过了一遍麦克船长的这篇文章，算是一个回顾和复习吧，简要概括一下：  
    1，语言模型发展路径：ngram, MLP, CNN, RNN/LSTM, Transformer  
    2，RNN 没有体现注意力，会忘事儿，无法处理长序列，基于attention的E-D解决了该问题  
    3，Transformer 2017年横空出世，self-attention，MHA 多头注意力，ResNet，Short-Cut 等概念  
    4，GPT系列基于Transformer的Decoder架构发展，Bert基于Encoder架构演变，不同范式让他们适合不同任务类型  
    5，Transformer 的优势：并行性好、不健忘（ResNet）、处理变长序列 等  

另外，说下个人感受，感觉阿里系工程师还是挺乐于分享的，无论是各大顶会paper，或者知乎、微信等平台上的分享文章，数量都不少  
国内其他公司的分享数量相对就少一些，百度和腾讯好像也还可以。不太清楚是文化差异还是管理制度区别  
