# 深度学习NLP概述

## 词向量
[友情链接](https://github.com/iwoods100/machine-learning-documents/blob/master/NLP/%E8%AF%8D%E5%90%91%E9%87%8F.md)

## RNN系列

NLP里最常用、最传统的深度学习模型就是循环神经网络 RNN（Recurrent Neural Network）。这个模型的命名已经说明了数据处理方法，是按顺序按步骤读取的。与人类理解文字的道理差不多，看书都是一个字一个字，一句话一句话去理解的。

*也有把CNN应用到NLP的例子，从归纳假设上来说，RNN表达的是在时间上共享参数，CNN表达的是在空间上共享参数。具体到某一应用实现，只要效果好，怎么用都可以。*

RNN的输入和输出可以是不定长且不等长的。

[见详细介绍](https://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)

对比普通的前馈网络（一层一层的计算变换），RNN更像一个黑盒子，内部有多份参数/变换，并加上了循环计算，所谓的GRU，LSTM，只是在基础RNN上，多增加了一些参数/变换罢了。

#### GRU（门控循环单元）

在基础RNN上引入了重置门（reset gate）和更新门（update gate）的概念，从而修改了循环神经网络中隐藏状态的计算方式。：

* 重置门有助于捕捉时间序列里短期的依赖关系
* 更新门有助于捕捉时间序列里长期的依赖关系

*门控单元的理解：每个gate可以理解为是权重系数，计算得到的值域为[0,1]*

这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

[见详细介绍](https://zh.gluon.ai/chapter_recurrent-neural-networks/gru.html)

### LSTM（长短时记忆）
LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），同时引入了记忆细胞的概念。

[见详细介绍](https://zh.gluon.ai/chapter_recurrent-neural-networks/lstm.html)

**除去以上形态的RNN外，还有[Deep-RNN 深度循环神经网络](https://zh.gluon.ai/chapter_recurrent-neural-networks/deep-rnn.html) 与 [BiRNN 双向循环神经网络](https://zh.gluon.ai/chapter_recurrent-neural-networks/bi-rnn.html)**

**观点：只要是RNN，就无法避免一点的是，权重始终是一层一层的积累下来的，而Attention机制可以说刚好是为了解决这个问题的，让每一步隐藏状态的权重分别进行点积。**

## Encoder-Decoder框架

Encoder-Decoder 不是一个具体的模型，是一种框架。

* Encoder：将不定长输入序列转成固定长度的向量
* Decoder：将固定长度的向量转成不定长输出序列
* Encoder与Decoder可以彼此独立使用，也可以一起使用（比如机器翻译，对话生成等场景）

## Attention

广义上，注意力机制的输入包括查询项以及一一对应的键项和值项，其中值项是需要加权平均的一组项。在加权平均中，值项的权重来自查询项以及与该值项对应的键项的计算。

Attention机制抽象出来其实就是三个部分：查询项Q、键项K、值项V

其中利用Q、K计算出权重向量W，再将W与V点乘，输出V在被加权平均之后的结果

*Q、K计算出权重向量W：这里计算方式有多种选择，如果两个输入向量长度相同，一个简单的选择是计算它们的内积。*

Attention里的K、V一般都是相等的，如果Q=K=V，则成为self-attention，transofrmer里就是用的self-attention。

*Attention提供了一种新的运算机制，可以被结合到不同的模型中使用，一般场景是，将原模型中V的部分 改为 V加权平均之后的值*

## seq2seq架构
可以认为是基于RNN的Encoder-Decoder实现，且也有结合attention的实现版本。

[普通的基于rnn的seq2seq](https://zh.gluon.ai/chapter_natural-language-processing/seq2seq.html)

[结合attention的seq2seq](https://zh.gluon.ai/chapter_natural-language-processing/attention.html)

## transformer架构
论文[attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)提出了transformer架构，transformer架构舍弃了rnn，完全基于attention机制进行实现，在很多任务上取得了更优秀的成绩。

[BERT](https://arxiv.org/pdf/1810.04805.pdf)是一个基于transformer的encoder部分，结合遮词+猜下一句任务的模型。

