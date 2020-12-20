import pandas as pd
# 加载mat文件
from scipy.io import loadmat


# # mat -> dict字典
# data_m1 = loadmat('./dataset/0.007-Ball.mat')
# DE_df = pd.DataFrame(data_m1['X118_DE_time'], columns=['X118_DE_time'])
# FE_df = pd.DataFrame(data_m1['X118_FE_time'], columns=['X118_FE_time'])
# df = pd.concat([DE_df,FE_df],axis=1)
def dataFormat(dir, ID):
    data = loadmat(dir)
    df = pd.DataFrame(data[ID + '_DE_time'])
    return df


Ball_7 = dataFormat('./dataset/0.007-Ball.mat', 'X118')[:120000]

result = []
item = []



