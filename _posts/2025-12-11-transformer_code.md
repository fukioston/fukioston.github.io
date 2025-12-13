---
title: Transformer相关代码
author: Fukioston
date: 2025-12-11 11:33:00 +0800
categories: [LLM,学习笔记]
tags: [LLM,学习笔记]
pin: true
math: true
mermaid: true
# image:
#   path: /../assets/img/te/img.png
#   lqip: false
#   alt: Responsive rendering of Chirpy theme on multiple devices.
---
# self-attention代码
```python
from math import sqrt

import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, dim_embedding, dim_qk, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_embedding = dim_embedding
        self.dim_qk = dim_qk
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_embedding, dim_qk, bias=False)
        self.linear_k = nn.Linear(dim_embedding, dim_qk, bias=False)
        self.linear_v = nn.Linear(dim_embedding, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_embedding)

    def forward(self,x):
        batch, n, dim_embedding = x.shape
        q=self.linear_q(x)
        k=self.linear_k(x)
        v=self.linear_v(v)
        # q乘k再除以根号dk
        dist=torch.bmm(q,k.transpose(1,2))*self._norm_fact
        # softmax一下
        dist=torch.softmax(dist,dim=-1)
        # 乘以v
        dist=torch.bmm(dist,v)
        return dist

```
nn.Linear(in_features, out_features) 的核心逻辑是对最后一维的做线性变换

torch.bmm就是batch的矩阵乘法

transpose更换维度下标