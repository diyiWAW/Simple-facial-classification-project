import cv2
import numpy as np
from PIL import Image
import copy
from sklearn.decomposition import PCA


img_data = np.load("img.npy")

# 定义拉普拉斯算子
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])


for nnum1 in range(img_data.shape[0]):
    print('num=',nnum1)
    # 对图像进行卷积操作
    img_array = img_data[nnum1].reshape(128, 128)
    laplacian_img = cv2.filter2D(img_array, -1, laplacian_kernel)
    img_lapaug = img_array + laplacian_img
    img_temp = img_lapaug.flatten()
    if nnum1 == 0:
        img_laplace = copy.deepcopy(img_temp)
        continue
    img_laplace = np.vstack((img_laplace, img_temp))

# 存储img_hist数组
print(img_laplace.shape)
# np.save('img_laplace.npy', img_laplace)












# cv2.imshow('Original Image', img_array)
# print(np.shape(laplacian_img))
# cv2.imshow('Laplacian Enhanced Image', img_lapaug)
# cv2.waitKey(0)
# cv2.destroyAllWindows()