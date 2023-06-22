
import numpy as np
from sklearn.decomposition import PCA


# #加载数据
# img=np.load('img.npy')
#
# #对处理数据进行PCA
# method_pca=PCA()
# method_pca.fit(img)
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
# img_pca = pca_comp.fit_transform(img)
# print('大小为',img_pca.shape)
#
#
# np.save('img_pca.npy',img_pca)
###########################################################################################################
#############################################################################################################

# #加载数据,hist
# img_hist=np.load('img_hist.npy')
#
# #对处理数据进行PCA
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
#
#
# np.save('img_hist_pca.npy',img_hist_pca)

########################################################################################################
#######################################################################################################

# gramma
# #加载数据,hist
# img_gramma=np.load('img_gramma.npy')
# print('1')
# #对处理数据进行PCA
# method_pca=PCA()
# method_pca.fit(img_gramma)
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
# img_gramma_pca = pca_comp.fit_transform(img_gramma)
# print('大小为',img_gramma_pca.shape)


# np.save('img_gramma_pca.npy',img_gramma_pca)


##########################################################################################################
###########################################################################################################
# laplace增强
#加载数据,hist
img_laplace=np.load('img_laplace.npy')
print('1')
#对处理数据进行PCA
method_pca=PCA()
method_pca.fit(img_laplace)
#获取方差比例
variance_ratio = method_pca.explained_variance_ratio_
ratio1 = np.cumsum(variance_ratio)
print(ratio1)
#获取贡献率大于百分之九十五的数量
components = np.argmax(ratio1 >= 0.95) + 1
print(components)

# 将n_components的数量作为主成分的个数
pca_comp=PCA(n_components=components)
img_laplace_pca = pca_comp.fit_transform(img_laplace)
print('大小为',img_laplace_pca.shape)


np.save('img_laplace_pca.npy',img_laplace_pca)
