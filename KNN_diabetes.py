import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#繪製學習曲線圖
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve

data = pd.read_csv(r'diabetes.csv') #糖尿病資料集
print(data.shape) #資料多寡
print(data.columns) #資料欄位
print(data.groupby('Outcome').size()) #資料結果groupby

X = data.iloc[:,0:8] #八個特徵欄位資料
Y = data.iloc[:,8] #結果欄資料

models = []
models.append(("KNN",KNeighborsClassifier(n_neighbors=5))) #一般
models.append(("KNN-distance",KNeighborsClassifier(n_neighbors=3, weights="distance"))) #距離越遠權重越低

#先將原始資料分成測試集和訓練集(20/80)
X_train , X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
result = []
for name, model in models:
    model.fit(X_train, Y_train) #將訓練資料帶入模型
    result.append((name, model.score(X_test, Y_test))) #將測試資料帶入訓練好之模型
print('Training Test')
for i in range(len(result)):
    print("name:{}, score:{}".format(result[i][0], result[i][1]))
print('')

results = []
for name, model in models:
    kfold = KFold(n_splits=10)  #K折交叉驗證器，將資料分成10份(9份訓練，1份測試)
    cv_result = cross_val_score(model, X, Y, cv=kfold)  #交叉驗證評估分數
    results.append((name, cv_result))
    cv_ShuffleSplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #劃出學習曲線
    plt_learn = plot_learning_curve(model, "Learn Curve for KNN Diabetes", X, Y, ylim=(0.,1.2), cv=cv_ShuffleSplit)

print("Cross Validation")
for i in range(len(results)):
    print("name:{}, score:{}".format(results[i][0], results[i][1].mean()))
print('')

#模型之後可以用下面的方法將資料帶入進行預測
predict = models[0][1].predict(X)
print("Predict",predict , predict.shape)

from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=2) #挑選兩個最佳特徵
X_new = selector.fit_transform(X, Y)
print(X_new[0:5]) #列出前五筆

#最佳特徵分別為Glucose和BMI指數
results = []
for name, model in models:
    kfold = KFold(n_splits=10)  #K折交叉驗證器，將資料分成10份(9份訓練，1份測試)
    cv_result = cross_val_score(model, X_new, Y, cv=kfold)  #交叉驗證評估分數
    results.append((name, cv_result))
for i in range(len(results)):
    print("name:{}, score:{}".format(results[i][0], results[i][1].mean()))

plt.figure(figsize=(10,6)) #畫布大小
plt.ylabel("BMI") #Y軸座標提
plt.xlabel("Glucose") #X軸作標題
plt.scatter(X_new[Y==0][:, 0], X_new[Y==0][:, 1], c='g', s=20, marker='o') #陰性
plt.scatter(X_new[Y==1][:, 0], X_new[Y==1][:, 1], c='r', s=20, marker='^') #陽性
plt.show()