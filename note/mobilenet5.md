# 模型亮点
## v1

1. Depthwise Convolution减少计算量
2. Pointwise Convolution提取特征
3. 超参数alpha控制卷积核个数
4. 超参数beta控制分辨率

## v2

1. Inverted Residual倒残差结构，用了ReLU6激活函数，减少计算量
2. Linear Bottlenecks，倒残差结构最后一层没使用ReLU激活函数，使用线性激活函数Linear，ReLU激活函数对低维特征信息造成大量损失。

## v3

1. 更新block，bneck
2. 使用nas搜索参数

