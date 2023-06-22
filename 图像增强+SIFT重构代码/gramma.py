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

img_data = np.load("img.npy")

# 遍历每个图像，img_data.shape[0]
r=0.5
for nnum1 in range(img_data.shape[0]):
	print('num=',nnum1)
	#变为浮点数
	img_ori=img_data[nnum1]
	img_float = np.float32(img_data[nnum1])
	img_arr_temp = np.zeros(len(img_float))
	for i in range(len(img_float)):
		a=img_float[i]

		img_arr_temp[i] = 255 * ((a / 255)** r)


	img_aug_gray_int = np.uint8(img_arr_temp)

	# # 将浮点数类型的图像转换为灰度图像
	# img_array = img_aug_gray_int.reshape(128, 128)
	# img_ori1=img_ori.reshape(128,128)
	# #显示测试
	# res = np.hstack((img_ori1, img_array))
	# # img1 = Image.fromarray(img_array)
	# # img_ori_o=Image.fromarray(img_ori1)
  	# #对比拼接,拼接完的图像为res
	# res1 = Image.fromarray(res)
	# # 显示图像
	# res1.show()


	# cv2.imshow('img', res)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	img_temp=img_aug_gray_int.flatten()
	if nnum1 ==0:
		img_gramma=copy.deepcopy(img_temp)
		continue
	img_gramma=np.vstack((img_gramma,img_temp))

# 存储img_hist数组
print(img_gramma.shape)
np.save('img_gramma.npy',img_gramma)
