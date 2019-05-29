# -*- coding:utf-8 -*-
# 线性回归器  以最小二乘预测的损失，仍使用精确计算的解析方法和随即梯度下降法
# 三种评价机制 MAE(平均绝对误差)，MSE（均方误差），R-squared

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

### 3、回归预测  分别利用线性回归模型LinearRegression和SGDRegressor
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)
# 从sklearn.linear_model导入SGDRegressor。
from sklearn.linear_model import SGDRegressor

# 使用默认配置初始化线性回归器SGDRegressor。
sgdr = SGDRegressor()
# 使用训练数据进行参数估计。
sgdr.fit(X_train, y_train)
# 对测试数据进行回归预测。
sgdr_y_predict = sgdr.predict(X_test)# 使用LinearRegression模型自带的评估模块，并输出评估结果。

#### 4、性能评价
#利用LinearRegression自带的评估模块
print('LinearRegression测试的误差值是',lr.score(X_test,y_test))

#利用R2、MSE、MAE进行评估
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('线性回归模型的R—Squared误差值是',r2_score(y_test,lr_y_predict))

#inverse_transform是将标准化后的数据转换为原始数据
print('线性回归模型的MSE误差值是',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

print('线性回归模型的MAE误差值是',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

#使用SGDRegressor模型自带的评估模块，输出评估结果
print('The value of default measurement of SGDregressor is',sgdr.score(X_test,y_test))
print('The value of R-squared of SGDregressor is',r2_score(y_test,sgdr_y_predict))
print('The MSE of SGDregressor is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
print('The MAE of SGDregressor is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
