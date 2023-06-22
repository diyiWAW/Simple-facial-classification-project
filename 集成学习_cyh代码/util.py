import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.manifold import Isomap
from PIL import Image
import math

# # 查看图像
# f = open("D:人脸图像识别/face/rawdata/5096","rb")
# x = np.frombuffer(f.read(),dtype=np.uint8)
# x = x.reshape(128,128)
# plt.imshow(x,cmap='gray')
# plt.show()

#-----------------------------------------------------------------
#功能：读取标签文件，并将结果保存为一个list
#输入：label_Path(标签文件路径)
#输出：label(由字典组成的list，详见readme.txt和label.txt)
#-----------------------------------------------------------------
def Get_label(label_Path) :
	f1 = open(label_Path+"faceDS","rb")
	f2 = open(label_Path+"faceDR","rb")
	label_1 = f1.read().decode()
	label_1 = label_1.split('\n')
	label_2 = f2.read().decode()
	label_2 = label_2.split('\n')
	label_3 = label_1 + label_2    #这个时候的标签是一个元素为字符串的list
	label = []
	for str in label_3 :
		dict = {'id':'','sex':'','age':'', 'race':'', 'face':'', 'prop':'', 'missing':'false'}
		l = re.findall("\((.*?)\)",str)       #从一堆字符串中提取想要的信息
		if len(l) == 0:
			continue
		if "_missing descriptor" in l :
			dict['id'] = str[1:5]
			dict['missing'] = 'true'
			label.append(dict)
			continue
		dict['sex'] = l[0][6:]
		dict['age'] = l[1][6:]
		dict['race'] = l[2][6:]
		dict['face'] = l[3][6:]
		dict['prop'] = l[4][8:]
		dict['id'] = str[1:5]
		label.append(dict)
	return label


#------------------------------------------------------------------------------
#功能：从所有标签中提取指定的标签并编码
#输入：label_list(包含所有标签的list)、label_name(想要提取的指定的标签，例如：'sex')
#输出：返回指定标签编码后的list(由0、1、2.....表示不同的类),和分类个数
#-------------------------------------------------------------------------------
def encode(label_list, label_name):
	label_encoded = []      #储存编码后的标签
	m = []                  #储存标签字符串,内容不能重复
	for dict in label_list :
		x = dict[label_name]   #取出标签字符串
		if x not in m:         
			m.append(x)
		label_encoded.append(m.index(x)) 
	return label_encoded, len(m)



#-----------------------------------------------------------------
#功能：将图片转化为一个numpy数组
#输入：img_path(图片路径)、label(标签数组)
#输出：无返回值，保存img_data(numpy储存的数组)的npy文件,shape为3993*16384
#-----------------------------------------------------------------
def Get_img(img_path, label) :
	id = label[0]['id']
	f = open(img_path + id,"rb")
	img_data = np.frombuffer(f.read(),dtype=np.uint8)
	for i in range (1,len(label)-1) :
		if label[i]['missing'] != 'true' :
			id = label[i]['id']
			print(id)
			path_img = img_path + id
			f = open(path_img,"rb")
			x = np.frombuffer(f.read(),dtype=np.uint8)
			if x.shape[0] != 16384 :  #有的图像不是128*128的，把它reshape
				size = x.shape
				print(size)
				size = int(math.sqrt(size[0]))
				x = x.reshape((size,size))
				x = Image.fromarray(x)
				x = x.resize((128,128))
				x = np.array(x)
				x = x.reshape(1,-1)
			img_data = np.vstack((img_data, x))
	print(img_data.shape)
	np.save('img.npy',img_data)

#-----------------------------------------------------------------
#功能：把人脸图像保存为jpg文件
#输入：img_data(图片数据)
#输出：无
#-----------------------------------------------------------------
def img2jpg(img_data) :
	label_Path = './人脸图像识别/face/'
	save_Path = './人脸图像识别/jpg/'
	label_dict = Get_label(label_Path)

	for i in range(img_data.shape[0]) :
		id = label_dict[i]['id']
		img = img_data[i]
		img = img.reshape(128,128)
		img = Image.fromarray(img)
		img.save(save_Path + id + '.jpg')

#-----------------------------------------------------------------
#功能：删除错误数据
#输入：img_data(图片数据)、label(标签数组)
#输出：返回 img_del 和 label_del，为删除错误数据及标签后的
#-----------------------------------------------------------------
def DeleteBadData(img_data, label) :
	img_mean = np.mean(img_data,axis=1) #计算图片均值(一张图片所有像素值的均值)

	index = np.where(img_mean < 1 )  
	index = index[0] #index是一个元组，[0]提取出数据
	id_abnormal = []
	for i in index :
		l = label[i]
		id = l['id']
		id_abnormal.append(id)
	print('数据错误图片对应的id为：',id_abnormal)
	img_del = np.delete(img_data,index,0)
	label_del = [n for i, n in enumerate(label) if i not in index]
	return img_del,label_del

