# -*- coding:utf-8 -*- 
#集成模型(回归)    极端随机模型、随机森林模型、提升树模型
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
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict=rfr.predict(X_test)

etr=ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict=etr.predict(X_test)

gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_y_predict=gbr.predict(X_test)

#### 4、性能评估
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print('随机回归森林的R-squared值是：',rfr.score(X_test,y_test))
print('随机回归森林的MSE的值是：',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))
print('随机回归森林的MAE的值是：',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))

print('*'*50)

print('极端回归森林的R-squared值是：',etr.score(X_test,y_test))
print('极端回归森林的MSE的值是：',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))
print('极端回归森林的MAE的值是：',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))

print('*'*50)

print('梯度提升回归树的R-squared值是：',gbr.score(X_test,y_test))
print('梯度提升回归树的MSE的值是：',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))
print('梯度提升回归树的MAE的值是:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))

