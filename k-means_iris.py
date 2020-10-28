import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5) #設定隨機種子

iris = datasets.load_iris() #鳶尾花資料集
X = iris.data  #花的四種特徵
# print("IRIS.data", X, X.shape)
Y = iris.target   #花的種類
# print("IRIS.target", Y , Y.shape)

model = KMeans(n_clusters=3) #建立kmeans模型
model.fit(X)  #資料要為numpy格式，將資料帶入模型
labels = model.labels_  #模型自動產生的分類標籤
print("labels", labels, labels.shape)

predict = model.fit_predict(X) #模型之後可以用perdict方法進行資料預測
print("Predict" , predict,predict.shape)

i = 0
for labels_name , target_name in zip(labels, Y):
    #修正模型自動分群標籤labels_name, 1為0和0為1
    if labels_name ==1:
        labels_name = 0
    elif labels_name ==0:
        labels_name = 1
    if labels_name != target_name:
        print(labels_name, target_name) #列印出預測失誤之資訊
        i += 1
print("Accuracy:", str((len(X)-i)/len(X)), "%") #列印出準確率

fig = plt.figure('f0', figsize=(5,4))  #建立圖形
ax = Axes3D(fig, rect=[0,0,0.95,1], elev=48, azim=134)

#用三個特徵植數列，畫出3D模型(總共有四個特徵取其三)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), edgecolors='k')

#畫出模型重心
C = model.cluster_centers_
ax.scatter(C[:, 3], C[:, 0], C[:, 2], c='red', s=100, alpha=0.5)

ax.w_xaxis.set_ticklabels([]) #取消X軸刻度
ax.w_yaxis.set_ticklabels([]) #取消Y軸刻度
ax.w_zaxis.set_ticklabels([]) #取消Z軸刻度

ax.set_xlabel('花瓣寬度', fontproperties="SimSum") #宋體
ax.set_ylabel('花萼長度', fontproperties="SimSum")
ax.set_zlabel('花瓣長度', fontproperties="SimSum")
ax.set_title("K-means_iris_3D")
ax.dist = 12 #與3D圖距離
plt.show()
