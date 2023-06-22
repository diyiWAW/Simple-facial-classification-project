# _*_ coding : utf-8_*_
# @Time  : 2023-06-20 14:55
# @Author : 广东工业大学 王红胜
# @File : keshihua 
# @Project : 人脸模式识别
import numpy as np
import matplotlib.pyplot as plt
# 从文件中读取npy格式的数据
data = np.load('pca_img.npy')
plt.imshow(data, cmap='gray')
plt.show()
# 打印数据
print(data)