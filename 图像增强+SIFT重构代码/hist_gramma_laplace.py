import copy
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from util import Get_label, encode
import cv2
from PIL import Image

label_Path = r"D:\pattern_recognition\facedection\facedection\facedect\face\face"
label_all = Get_label(label_Path)
# 提取性别标签

labe_dict = []
for dict in label_all :            #对于标签缺失的数据直接删除
	if dict['missing'] != 'true' :
		labe_dict.append(dict)
	else :
		continue
# 各类标签
sex_label,_ = encode(labe_dict,'sex')
age_label,_ = encode(labe_dict,'age')
face_label,_ = encode(labe_dict,'face')
race_label,_ = encode(labe_dict,'sex')
prop_label,_ = encode(labe_dict,'prop')

img_data = np.load("img.npy")  #读取图像文件，详见readme.txt
# data_projected = np.load("data_projected.npy")
pca_data = np.load("pca_img.npy")  #读取数据降维后的图像，详见readme.txt
# pca = PCA(n_components=0.9)
# pca_img = pca.fit_transform(img_data)

# bins 是一个长度为 257 的一维数组，表示 256 个像素值范围的边界，其中第一个元素为 0，最后一个元素为 256。可以用 bins 数组来绘制直方图。

# hist是所有图像的灰度出现情况
# hist,bins = np.histogram(img_data.flatten(),256,[0,256])
# # 计算累积分布图，
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img_data.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()




from matplotlib import pyplot as plt

#直方图均衡化,后面有保存，只需要执行一次，后面直接加载
#遍历每个图像，img_data.shape[0]
# for nnum1 in range(img_data.shape[0]):
# 	print('num=',nnum1)
# 	img_array = img_data[nnum1].reshape(128, 128)
# 	# img1 = Image.fromarray(img_array)
# 	equ = cv2.equalizeHist(img_array)
# 	# res = np.hstack((img_array, equ))  #对比拼接,拼接完的图像为res
# 	# cv2.imshow('img', res)
# 	# cv2.waitKey()
# 	# cv2.destroyAllWindows()
# 	img_temp=equ.flatten()
# 	if nnum1 ==0:
# 		img_hist=copy.deepcopy(img_temp)
# 		continue
# 	img_hist=np.vstack((img_hist,img_temp))

#存储img_hist数组
# np.save('img_hist.npy',img_hist)


#加载数据
# img_hist=np.load('img_hist.npy')

#对处理数据进行PCA
# method_pca=PCA()
# method_pca.fit(img_hist)
# #获取方差比例
# variance_ratio = method_pca.explained_variance_ratio_
# ratio1 = np.cumsum(variance_ratio)
# print(ratio1)
# #获取贡献率大于百分之九十五的数量
# components = np.argmax(ratio1 >= 0.95) + 1
# print(components)
#
# # 将n_components的数量作为主成分的个数
# pca_comp=PCA(n_components=components)
# img_hist_pca = pca_comp.fit_transform(img_hist)
# print('大小为',img_hist_pca.shape)
# np.save('img_hist_pca.npy',img_hist_pca)


img_hist_pca=np.load('img_hist_pca.npy')
img_gramma_pca=np.load('img_gramma_pca.npy')
img_laplace_pca=np.load('img_laplace_pca.npy')

#使用knn、贝叶斯、svm、决策树进行性别分类
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5) 
clf_tree = tree.DecisionTreeClassifier(criterion="entropy")
clf_svc = SVC(kernel='rbf', probability=True)
clf_gnb = GaussianNB()
models = [clf_knn, clf_gnb, clf_svc, clf_tree]
score = []

#采用hist+pca
for model in models :
	s = cross_val_score(model, img_hist_pca, prop_label,scoring='roc_auc', cv=5)
	score.append(s.mean())
print('采用hist+pca的score=',score)
print('--------------------------------------------------')
#采用gramma+pca
score2=[]
for model in models :
	s = cross_val_score(model, img_gramma_pca, prop_label,scoring='roc_auc', cv=5)
	score2.append(s.mean())
print('采用gramma+pca的score=',score2)
print('--------------------------------------------------')
#采用laplace+pca
score3=[]
for model in models :
	s = cross_val_score(model, img_laplace_pca, prop_label, scoring='roc_auc',cv=5)
	score3.append(s.mean())
print('采用laplace+pca的score=',score3)
print('--------------------------------------------------')
#采用pca
score1=[]
for model in models :
	s = cross_val_score(model, pca_data, prop_label,scoring='roc_auc' ,cv=5)
	score1.append(s.mean())
print('只采用pca的score=',score1)

#
# #使用bagging、随机森林、adaboost、stacking集成学习方法进行性别分类
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import StackingClassifier
#
# bagging = BaggingClassifier(clf_knn,n_estimators=20, max_samples=0.5, max_features=0.5)
# rf = RandomForestClassifier(n_estimators=20)
# adaboost = AdaBoostClassifier(base_estimator=clf_svc, n_estimators=10)
# stacking = StackingClassifier(estimators=[('KNN',clf_knn), ('SVC',clf_svc), ('NB',clf_gnb), ('DT',clf_tree)],  final_estimator=LogisticRegression())
# model_assume = [bagging,rf,adaboost,stacking] #adaboost跑得非常非常慢
# score_assume = []
# for model in model_assume :
# 	s = cross_val_score(model, pca_data, sex_label, cv=5)
# 	score_assume.append(s.mean())
# print(score_assume)
