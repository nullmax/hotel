# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def nan_filter(data):
    nan_count = 4
    count = 0
    for d in data:
        if pd.isna(d):
            count = count + 1
            if count > nan_count:
                return False
    return True

def replaceLabels(data):
    
    if data[11] in labels:
        data[11] = labels.index(data[11]) 
    
raw_data = pd.read_csv("raw_data.csv")

#删除缺失值过多的行
x = raw_data.apply(nan_filter,axis=1)
temp=[i for i in range(len(x)) if x[i] != True]   
data = raw_data.drop(temp)

#删除不使用的列
data=data.drop(columns=['序号','酒店名','离中心地区距离'])

#将label替换为数值
labels=['评分','非常好', '好', '很棒', '优异的', '好极了']
for l in labels:
    data['综合评价'][data['综合评价'] == l] = labels.index(l)
    
data = data.sort_values(by = '综合评价')

#填补缺失值
items = list(data)
for item in items:
    data[item].fillna(np.mean(data[item]), inplace=True)

#保存为numpy数据        
data = np.array(data)
np.save("data.npy", data)