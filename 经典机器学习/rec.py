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
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import StackingClassifier




import warnings
warnings.filterwarnings('ignore')

label_Path = "./人脸图像识别/face/"
label_all = Get_label(label_Path)
# 提取性别标签

labe_dict = []
for dict in label_all :            #对于标签缺失的数据直接删除
	if dict['missing'] != 'true' :
		labe_dict.append(dict)
	else :
		continue

sex_label,_ = encode(labe_dict, 'sex')
age_label,_ = encode(labe_dict, 'age')
race_label,_ = encode(labe_dict, 'race')
face_label,_ = encode(labe_dict, 'face')
prop_label,_ = encode(labe_dict, 'prop')



img_data = np.load("img.npy")  #读取图像文件，详见readme.txt
# data_projected = np.load("data_projected.npy")
pca_data = np.load("pca_img.npy")  #读取数据降维后的图像，详见readme.txt
print(pca_data.shape)
# pca = PCA(n_components=0.9)
# pca_img = pca.fit_transform(img_data)

# todo 加个特征选择
from sklearn.feature_selection import mutual_info_classif  #互信息法
X_mul = SelectKBest(mutual_info_classif, k=22).fit_transform(pca_data, prop_label)
print(X_mul.shape)
#使用knn、贝叶斯、svm、决策树进行性别分类
knn = neighbors.KNeighborsClassifier(n_neighbors=17) # knn
gnb = GaussianNB()                                  #bayes
#svm = SVC(kernel='rbf', probability=True)                     #svm
# tree = tree.DecisionTreeClassifier(criterion="entropy") #决策树
#clf_tree = tree.DecisionTreeClassifier(criterion="entropy")
models = [knn, gnb]
score = []
for model in models :
	print('1')
	s = cross_val_score(model, X_mul, prop_label, cv=5)
	score.append(s.mean())
	#print(score[0])
print(score)






# #使用bagging、随机森林、adaboost、stacking集成学习方法进行性别分类

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
