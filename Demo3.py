# -*- coding:utf-8 -*-
# K近邻(回归)    不需要模型训练，考虑不同的模型：算术平均、距离加权平均
# 1、读取数据
from sklearn.datasets import load_boston
boston=load_boston()

## 2、数据分割
from sklearn.model_selection import train_test_split
X=boston.data
y=boston.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)

#数据标准化处理
#preprocessing这个模块还提供了一个实用类StandarScaler，它可以在训练数据集上做了标准转换操作之后，
# 把相同的转换应用到测试训练集中。这是相当好的一个功能。可以对训练数据，测试数据应用相同的转换，
# 以后有新的数据进来也可以直接调用，不用再重新把数据放在一起再计算一次了。
# 调用fit方法，根据已有的训练数据创建一个标准化的转换器，另外，StandardScaler()中可以传入两个参数：
# with_mean,with_std.这两个都是布尔型的参数，默认情况下都是true,但也可以自定义成false.即不要均值中心化或者不要方差规模化为1.

from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1,1))
y_test=ss_y.transform(y_test.reshape(-1,1))

### 3、回归预测
# 从sklearn.neighbors导入KNeighborRegressor（K近邻回归器）。
from sklearn.neighbors import KNeighborsRegressor

# 初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归：weights='uniform'。
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# 初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归：weights='distance'。
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)


#### 4、性能评估    算术平均、加权平均
#使用R-squared、MSE、以及MAE三种指标对平均回归配置的K近邻模型在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print ('R-squared value of uniform-weighted KNeighorRegression:', uni_knr.score(X_test, y_test))
print ('The mean squared error of uniform-weighted KNeighorRegression:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print ('The mean absoluate error of uniform-weighted KNeighorRegression', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))

print('.'*50)
#使用R-squared、MSE、以及MAE三种指标对距离加权回归配置的K近邻模型在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print ('R-squared value of uniform-weighted KNeighorRegression:', dis_knr.score(X_test, y_test))
print ('The mean squared error of uniform-weighted KNeighorRegression:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print ('The mean absoluate error of uniform-weighted KNeighorRegression', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))

#事实证明，K近邻加权平均的回归策略可以获得更高的模型性能
