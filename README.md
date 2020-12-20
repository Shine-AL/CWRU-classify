#滚动轴承故障分类
###数据集：CWRU（凯斯西储大学轴承数据中心）
官网提供数据下载：https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website


class            | label
:----------------|------
0.007-Ball       | 1
0.007-InnerRace  | 2
0.007-OuterRace6 | 3
0.014-Ball       | 4
0.014-InnerRace  | 5
0.014-OuterRace6 | 6
0.021-Ball       | 7
0.021-InnerRace  | 8
0.021-OuterRace6 | 9
0.000-Normal     | 0

###网络：LSTM（长短期记忆网络）
##文件说明
dataTo3.py：将原始数据整理成2维数据。shape:(n_samples, timestamps, features)

dataTo3_H.py：将原始数据整理成2维数据,标签为独热编码。shape:(n_samples, timestamps, features)

LSTM.py：使用LSTM分类。

