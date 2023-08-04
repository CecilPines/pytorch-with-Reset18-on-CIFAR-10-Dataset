# Pytorch-with-Reset18-on-CIFAR-10-Dataset
本项目使用pytorch在CIFAR-10数据集上训练Resnet18模型并探究在有无残差连接的情况下在数据集上的表现。
## Breadcrumbspytorch-with-Reset18-on-CIFAR-10-Dataset.py
此文件为源代码。
## experiment
在此文件夹中包含本人的实验结果，使用ipynb的文件形式将实验结果与代码放置在一个文件中，可供参考。

所改参数由文件名显示，Resnet18.ipynb文件为保留残差链接的结果，Resnet18_none.ipynb文件为去除残差部分。

## 补充说明
1. 本实验藉由colab完成，使用相关库版本详情参见colab相关文档。
2. 可更改epoch变量来使代码运行训练或测试更多的epoch。
