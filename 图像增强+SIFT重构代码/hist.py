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

sex_label,_ = encode(labe_dict,'sex')

img_data = np.load("img.npy")  #读取图像文件，详见readme.txt


# bins 是一个长度为 257 的一维数组，表示 256 个像素值范围的边界，其中第一个元素为 0，最后一个元素为 256。可以用 bins 数组来绘制直方图。

# hist是所有图像的灰度出现情况
hist,bins = np.histogram(img_data.flatten(),256,[0,256])
# 计算累积分布图，
cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()
cdf_normalized = cdf / cdf.max()
# plt.plot(cdf_normalized, color = 'b')
plt.hist(img_data.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.ylim([0,1000000])
# plt.legend(('cfd','histogram'), loc = 'upper left')
plt.legend('histogram', loc = 'upper left')
plt.show()
print('1')



# from matplotlib import pyplot as plt
#
# #直方图均衡化,后面有保存，只需要执行一次，后面直接加载
# # 遍历每个图像，img_data.shape[0]
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

# 存储img_hist数组
# np.save('img_hist.npy',img_hist)


#