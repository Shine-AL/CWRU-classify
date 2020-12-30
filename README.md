# 滚动轴承故障分类
### 数据集：CWRU（凯斯西储大学轴承数据中心）
官网提供数据下载：https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website


  Damage diameter  |  position  |  label  |  Training set  |  Test set
:----------|------------|--------|---------|---------
0          | no         | 0      | 270     | 30
0.007      | Ball       | 1      | 270     | 30
0.007      | Inner      | 2      | 270     | 30
0.007      | Outer      | 3      | 270     | 30
0.014      | Ball       | 4      | 270     | 30
0.014      | Inner      | 5      | 270     | 30
0.014      | Outer      | 6      | 270     | 30
0.021      | Ball       | 7      | 270     | 30
0.021      | Inner      | 8      | 270     | 30
0.021      | Outer      | 9      | 270     | 30

### 网络：LSTM（长短期记忆网络）
## 文件说明
dataset：原始数据

saveModel：保存训练好的模型。

data_init.py：数据初始化，将原始数据处理成需要的样子。

models.py：搭建网络结构

train.py：对模型进行训练并保存。

### cnn学习中 —— time：2020/12/22
CNN_train.py

迭代后10次准确率

index    |   accuracy  
:--------|------------------
1        |0.9466666579246521 
2        |0.9433333277702332
3        |0.9300000071525574 
4        |0.9366666674613953 
5        |0.9366666674613953 
6        |0.9399999976158142 
7        |0.95333331823349 
8        |0.9599999785423279 
9        |0.9300000071525574 
10       |0.9433333277702332 


