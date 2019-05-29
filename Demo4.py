# -*- coding:utf-8 -*- 
# 回归树    与决策树不同，回归树叶结点的数据类型不是离散型的，而是连续型的，决策树的每个叶结点根据训练数据的表现的概率
#决定了最终的预测类别，回归树的叶结点是一个具体的值，而其返回的是一团训练数据的均值，不是具体的连续的预测值
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

### 3、回归预测  利用决策树回归模型
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
#使用默认配置的单一回归树模型对测试数据进行预测，并将预测值存储在变量dtr_y_predict中
dtr_y_predict=dtr.predict(X_test)

#### 4、性能评估
#使用R-squared、MSE、以及MAE三种指标对默认配置的回归树模型在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('回归树的R-squared值是：',dtr.score(X_test,y_test))
print('回归树的MSE值是：',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict)))
print('回归树的MAE值是：',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict)))


