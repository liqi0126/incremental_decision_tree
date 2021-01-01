#### MOA dataset generator

##### 要求

已安装Java，且配置了环境变量（可在终端中通过'java [some command]'访问）。

##### 使用方法

```bash
$ python ./generator.py [-n N] [-d D]
```

将分别生成四个数据集dataset1.csv ~ dataset4.csv

每个数据集的配置为：包含5个属性、每个属性包含5个离散的取值、将生成N条数据（默认值为5000）、dataset[i].csv中对应一个[i + 1]分类问题的数据集。
如果设置了-d属性，则会产生D次漂移。（默认D = 0）

##### 已生成的数据集

一组生成好的数据集已经上传到清华云盘（N = 3,000,000）：https://cloud.tsinghua.edu.cn/f/cb211fd2bf4f41369dc4/ 