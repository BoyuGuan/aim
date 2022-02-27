# My graduation project


## 0.概述
这是我的毕业设计项目，探究在AIMET框架下进行模型压缩与处理，并评估性能。

## 1.hanldeCifar.py 
将cifar数据集处理成AIMET要求的数据集的形式，也即分为val和train两个文件夹，每个文件夹中存放着10个文件夹的图片，这10个文件夹中每个文件夹的名字是该类的名称，里边是该类的图片。
## 2.pre_trian.py 
将模型进行预训练的文件，其中选取了torch.nn.Module里的ResNet18, ResNet50, mobileNetV2，三个模型，在并将该三个模型用cifar的训练集训练50个epoch，然后将这三个训练好的模型保存到./preTrainedModel中。
可以直接加载预训练好的模型，省却提前训练时间。因为原先的torch.nn.Module模型的pre-train是在ImageNet上作的。

##


## test
asdasdasd