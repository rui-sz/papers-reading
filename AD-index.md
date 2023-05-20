### Anomaly detection area tracker

I will use this project to track my progress in Anomaly Detection area.

## Anomaly detection

### Time-Series Anomaly Detection Service at Microsoft

这是一篇微软发表的关于检测产品数据异常的文章，是一类典型的Anomaly detection问题，本文关键点：  
  1，基于时间序列的异常检测，面临标签缺失、泛化性及检测效率多重挑战  
  2，采用 unsupervised manner，借鉴了视觉领域的显著性检验（Saliency detection）思路，模型：SR+CNN  
  3，作为特征引入 supervised learning model 中  

数据构建  
  本文引入了多个数据集，包括：  
  ![image](https://user-images.githubusercontent.com/69101330/184881474-7dbaad23-f3a5-4fba-b4e1-929c1879e27e.png)  
  并在数据集中引入了人工构造的基于策略的异常数据。  

模型  
  a，SR(Spectral Residual)  
  ![image](https://user-images.githubusercontent.com/69101330/184882076-b2691896-9570-4311-8990-20bf7c9c6dab.png)  
  传统SR模型，利用傅里叶变换和频谱残差，并结合简单的规则进行显著性检测  

  b，SR+CNN  
  ![image](https://user-images.githubusercontent.com/69101330/184881855-5f290f9a-3f1c-4e0c-b03a-1e03c821c006.png)  
  改进后的方法，基于SR生成的显著性图，用CNN提取图像特征进行识别  

评估  
  unsupervised learning  
  ![image](https://user-images.githubusercontent.com/69101330/184882719-b96ab358-63c7-4518-ad47-036fd9d50f0a.png)  

  supervised learning  
  ![image](https://user-images.githubusercontent.com/69101330/184882768-ee216268-a400-4174-9721-ceb1f08b4503.png)  
  
  可以看到，无论是无监督范式还是有监督范式，SR+CNN都取得了较好的效果，证明了方法的有效性。最后本文还尝试了SR+DNN，也取得了不错的效果  

## Fraud Detection

Fraud detection is a typical research branch of anomaly detection.

## AML

Anti money laundering is also an active area of anomaly detection.

## GNN

gnn models
