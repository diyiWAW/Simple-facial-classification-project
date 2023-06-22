import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from util import Get_label, encode, DeleteBadData

# 数据导入
label_Path = "./人脸图像识别/face/"
label_all = Get_label(label_Path)
img_data = np.load("img.npy")  # 读取图像文件，详见readme.txt
pca_data = np.load("pca_img.npy")

# 标签集提取
labe_dict = []
for dict in label_all:  # 对于标签缺失的数据直接删除
    if dict['missing'] != 'true':
        labe_dict.append(dict)  # labe_dict就是直接去除缺失数据后的label.txt的内容
    else:
        continue

sex_label, _ = encode(labe_dict, 'sex')

label_seleted = sex_label

# 归一化处理
scaler = StandardScaler()
pca_data_std = scaler.fit_transform(pca_data)
print('pca提取后的数据大小:', pca_data_std.shape)

# elasticnet特征选择
eCV_model = ElasticNetCV(cv=10)
eCV_model.fit(pca_data_std, label_seleted)

# 取特征选择后系数不为零的特征
selected_features = eCV_model.coef_ != 0
X_selected = pca_data_std[:, selected_features]
print("特征选择后数据大小：", X_selected.shape)

# # GBDT寻找最优参数
# clf_gbc = GradientBoostingClassifier(verbose=0)
# X_train, X_test, y_train, y_test = train_test_split(X_selected, label_seleted, test_size=0.3, random_state=42)
# param_find = {
#     'loss':['log_loss'],
# 	'learning_rate':[ 0.1,0.01,0.001],
# 	'n_estimators':[100,200,300,400],
# 	'max_depth':[3,4,5,6]
# }
# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(clf_gbc, param_grid=param_find, cv=5)
# grid_search.fit(X_train, y_train)
# # 显示最优参数
# print('Best parameters of GBDT: ',grid_search.best_params_)
#
# clf_best_gbc = grid_search.best_estimator_
# y_pred = clf_best_gbc.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print('Accuracy of easy gbc:', acc)

# xgboost找参数
import xgboost as xgb
# 根据标签编码不为零的
count = [0,0,0,0,0,0]
num_class = 0
for i in label_seleted:
    for j in range(5):
        if i ==j:
            count[i]+=1
print('该标签下各类别数据分布：',count)
for i in range(6):
    if count[i]!= 0:
        num_class+=1
print('该标签下是 %d 分类问题' % num_class)

X_train, X_test, y_train, y_test = train_test_split(X_selected, label_seleted, test_size=0.3, random_state=42)
param_find = {
    'booster': ['gbtree'], # 基于树的弱分类器模型
    'objective': ['multi:softmax'], # 多分类
    'num_class': [num_class],  # 类别数目
    'learning_rate': [0.1],
    'n_estimators': [160],
    'max_depth': [6,7,8], # 最大深度，合理设置能防止过拟合,key1
    'min_child_weight': [2],  # 叶子节点最小样本数,key2
    'subsample': [0.8,1], # 采样率
    'reg_alpha':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'reg_lambda':[0,0.1,0.5,1]
    }
# 将数据转换为 DMatrix 格式(XGBoost特有数据格式，可加速训练）
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)
clf_xgb = xgb.XGBClassifier()

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf_xgb, param_grid=param_find, cv=5)
grid_search.fit(X_train, y_train)
# 显示最优参数
print('Best parameters of XGboost: ',grid_search.best_params_)

clf_best_gbc = grid_search.best_estimator_
y_pred = clf_best_gbc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy of xgb:', acc)