# -*- coding:utf-8 -*-
# 支持向量机 (回归)
# 三种核函数配置  线性核函数  多项式核函数  径向基核函数
# 1、读取数据
from sklearn.datasets import load_boston
boston=load_boston()

## 2、数据分割
from sklearn.model_selection import train_test_split
import numpy as np
X=boston.data
y=boston.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#数据标准化处理
#preprocessing这个模块还提供了一个实用类StandarScaler，它可以在训练数据集上做了标准转换操作之后，
# 把相同的转换应用到测试训练集中。这是相当好的一个功能。可以对训练数据，测试数据应用相同的转换，
# 以后有新的数据进来也可以直接调用，不用再重新把数据放在一起再计算一次了。
# 调用fit方法，根据已有的训练数据创建一个标准化的转换器，另外，StandardScaler()中可以传入两个参数：
# with_mean,with_std.这两个都是布尔型的参数，默认情况下都是true,但也可以自定义成false.即不要均值中心化或者不要方差规模化为1.

from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)    #reshape(1,-1)，表示将n*n的数据转换成1*n^2的数据
X_test=ss_X.transform(X_test)          #reshape(-1,1)，表示将n*n的数据转换成n^2*1的数据
y_train=ss_y.fit_transform(y_train.reshape(-1,1))
y_test=ss_y.transform(y_test.reshape(-1,1))

### 3、使用三种不同的核函数配置的支持向量机回归模型进行训练，分别对测试数据进行预测
from sklearn.svm import SVR

#使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr=SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict=linear_svr.predict(X_test)

#使用多形式核范数配置的支持向量机进行回归训练，并且对测试样本进行预测
poly_svr=SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict=poly_svr.predict(X_test)

#使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr=SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict=rbf_svr.predict(X_test)

#### 4、使用三种不同的核函数配置的支持向量机回归模型在相同的测试集上进行性能评估
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print('R-squared value of linear SVR is',linear_svr.score(X_test,y_test))
print('The mean squared error of linear SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print('The MAE of linear SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print('.'*40)
print('R-squared value of linear SVR is',poly_svr.score(X_test,y_test))
print('The mean squared error of linear SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
print('The MAE of linear SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
print('.'*40)
print('R-squared value of rbf SVR is',rbf_svr.score(X_test,y_test))
print('The mean squared error of rbf SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
print('The MAE of rbf SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
