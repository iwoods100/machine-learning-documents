# tensorflow开发笔记

## 特征存储

##### 数据序列化格式
||csv、tsv|tfrecord|parquet|
|---|---|---|---|
|文件可读性|人可直接阅读|需加载|需加载|
|保留数据结构|不支持|支持|支持|
|tensorflow内置读取|支持|支持|不支持|
|读取速度|一般|快|一般|
|文件存储size|小|大|大|
|训练时join多特征文件|是|否|是|

##### 在实际场景中，会根据特征的复杂度，数量级来决定将特征存储为什么样的格式
* 特征很简单，可直接用纯文本格式存储
* 特征复杂，可使用tfrecord或parquet，如果数据量在可接受范围内，建议用tfrecord
* 当特征复杂，且数据量很大时，可根据情况进行不同领域的特征分开存储，训练时进行join

以上的尺度标准由实际机器资源的上线来决定
训练时对特征进行join会影响速度，如果硬盘够大，建议存储为tfrecord格式

##### 特征处理的时机
* 线上特征：训练机器直接下载即可使用（需要衡量数据量以及机器宽带容量）
* 离线特征：训练机器下载原始特征，在本地机器上特征加工（可节省下载压力）
* 实时计算特征：比如依赖算法模型api得到的特征，可根据需求，在本地gpu机器上部署serving api，获取自己需要的那一部分特征，比如图片分类特征，基于BERT的文本特征等

由于线上能借助spark这类高效的数据并行处理框架，一些统计类特征处理起来非常快。


## tf内置特征类型

在训练阶段，会将我们计算好的特征与合适的特征列一一对应

[tensorflow feature_columns官方介绍](https://www.tensorflow.org/guide/feature_columns)

##### one-hot特征
* `bucketized_column` 将单个数值特征按范围进行分桶，切割为one-hot
* `categorical_column_***` 将特征值（字符串 or 整数）进行分类，转换为one-hot
* `crossed_column` 交叉组合特征，如地理位置经纬度的交叉
* `weighted_categorical_column` 支持对不同的category设置不同weight（如文本词汇特征里的词权重各不相同）

##### 稠密特征
* `numeric_column` 数值特征
* `indicator_column` 以categorical列作为输入，支持multi-hot的数据
* `embedding_column` 对categorical_column_*类型的列进行embedding


## embedding详解

```
tf提供的一般经验法则，在one-hot特征中：
embedding_dimensions =  number_of_categories**0.25
embedding维数应该是类别数量的 4 次方根。由于本示例中的词汇量为 81，建议维数为 3
```

不管embedding的实现方式如何，最终目的都是对稀疏特征进行降维

从不同方面对embedding进行理解

##### 按初始化方式
* `预训练模型`
* `随机初始化`: 结合模型训练进行参数更新

##### 按训练方式
* `non-trainable`: embedding vectors不会更新
* `trainable`: embedding vectors随训练一起更新

```
embedding table的Variable的trainable控制参数是否可训练，defaults=True
```

##### 按实现方式
* `tf.feature_column.embedding_column`：tf内置支持，只能针对one-hot
* `embedding_lookup`: non-trainable & trainable，无bias，不支持multi-hot，无激活函数，更快(An embedding layer performs select operation)
* `dense layer` : 只有trainable，有bias，支持multi-hot， 有激活函数(A dense layer performs dot-product operation, plus an optional activation)

|| embedding_column | embedding_lookup | dense layer | 自定义 |
|---|---|---|---|---|
|支持multi-hot|否|否|支持|支持|
|trainable|支持|支持|支持|支持|
|non-trainable|支持|支持|支持|支持|
|bias|无|无|支持|支持|
|activation|无|无|支持|支持|
|运算速度|快|快|一般|一般|

```python
# 自定义embedding variable
X = tf.get_variable("X", [vocab_size, embed_size])
b = tf.get_vairable("b", [vocab_size])
embeds = tf.mul(embeds, X) + b
```



## serving

