# garbage_classification by pytorch
 Zsir_Henu人工智能导论实验作业，基于卷积神经网络ResNet50的垃圾分类

----

<h1>
    第一部分 模型
</h1>

<p>
    模型：ResNet 50 + 优化器 SGD
</p>

<p>
    数据集：Kaggle垃圾六分类数据集
</p>
<p>
    ImageNet预训练权重可自行网上获取
</p>

<p>
    若想使用由Zsir_Henu训练的权重，可以在百度网盘下载：
</p>

<p>
    链接：https://pan.baidu.com/s/1-yYLeN8gX4NPu-tjCwOhbw
</p>

<p>
    提取码：wzcl
</p>










| 训练集 | 验证集 | 测试集 |
| ------ | ------ | ------ |
| 1593   | 176    | 758    |



| 超参数 | Batch Size | Learning Rate | Epochs | Weight Decay | Momentum |
| :----- | ---------- | ------------- | ------ | ------------ | -------- |
| 参数   | 16         | 2e-4          | 30     | 5e-4         | 0.9      |

| 数据增广                   |
| -------------------------- |
| 随机裁剪                   |
| 随机水平翻转               |
| 随机改变亮、对比度、饱和度 |
| 图像标准化                 |

----



<h1>
    第二部分 结果
</h1>

<h3>
    训练
</h3>

| Epoch | logs                                                |
| ----- | --------------------------------------------------- |
| 26    | Epoch 26 loss 0.048, train acc 1.000, val acc 0.807 |
| 27    | Epoch 27 loss 0.276, train acc 0.889, val acc 0.859 |
| 28    | Epoch 28 loss 0.054, train acc 1.000, val acc 0.849 |
| 29    | Epoch 29 loss 0.045, train acc 1.000, val acc 0.865 |
| 30    | Epoch 30 loss 0.206, train acc 0.889, val acc 0.885 |

<h3>
    测试
</h3>

```
test acc 0.935
```

____

<h1>
    第三部分 How to train/test
</h1>

<h3>
    数据集格式：
</h3>
xxx/根目录

根目录内容如下：

> 根目录
>
> >
> >
> >cardboard
> >
> >> carboard1.jpg
> >>
> >> ...
> >>
> >> carboardn.jpg
> >
> >glass
> >
> >> glass1.jpg
> >>
> >> ...
> >>
> >> glassn.jpg
> >
> >metal
> >
> >> ...
> >
> >paper
> >
> >> ...
> >
> >plastic
> >
> >trash
<h3>
    运行项目之前
</h3>

在命令行中pip install -r requirements.txt

以配置项目所需环境

<h3>
    训练
</h3>

设置TRAIN_MODE 为True训练

设置TO_SAVE 保存模型训练参数

<H3>
    测试
</h3>

设置TRAIN_MODE为False为测试

设置CHECKPOINT_DIR加载用于测试的模型参数

配置完参数后只需运行main.py即可开始训练/测试
<h3>
    检测
</h3>

运行classify.py

设置IMAGE_DIR和CHECKPOINT_DIR可以将IMAGE_DIR下所有图像进行分类并用plt可视化



