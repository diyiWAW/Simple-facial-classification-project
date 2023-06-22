from torchvision import datasets,models,transforms
from torch.utils.data import Dataset,DataLoader
from util import Get_label, encode
from PIL import Image
import torch
import numpy as np
import math
#---------------------------------------------------------------------------
#----------------------------自定义数据集------------------------------------
class FaceDataset(Dataset):
	def __init__(self,label_Path,img_path,label_name, transform=None, useAll = False):
		super().__init__()
		gt_list = Get_label(label_Path)
		label_all = []
		for dict in gt_list :            #对于标签缺失的数据直接删除
			if dict['missing'] != 'true' :
				label_all.append(dict)
			else :
				continue
		
		if useAll == True :
			sex,_ = encode(label_all,'sex')
			age,_ = encode(label_all,'age')
			face,_ = encode(label_all, 'face')
			race,_ = encode(label_all, 'race')
			self.label_multi = [sex,age,face,race]
			

		label_encoded, cls_num = encode(label_all,label_name)   #取出指定标签并编码
		self.label_encoded =label_encoded     #编码后的指定标签list
		self.label_all = label_all             #包含所有标签信息的list         
		self.img_path = img_path              #图片路径
		self.transfrom = transform            
		self.label_name = label_name          #标签的名字，如'sex'
		self.cls_num = cls_num                #分类的个数，如二分类
		self.useAll = useAll


	def __len__(self) :    #返回数据集的大小
		return len(self.label_encoded)

	def __getitem__(self,index):
		id = self.label_all[index]['id']       #取出图片id
		img_path = self.img_path + id      #获得图片路径
		f = open(img_path,"rb")
		img_data = np.frombuffer(f.read(),dtype=np.uint8)     #读取图片为numpy
		if img_data.shape[0] != 16384 :                  #将不是128*128的图片resize成128*128
			size = int(math.sqrt(img_data.shape[0]))
			img_data = img_data.reshape((size,size))
		else :
			img_data = img_data.reshape(128,128)
		img_data=Image.fromarray(img_data)             #将图片从numpy转化为PIL
		img_data = self.transfrom(img_data)           #转化为tensor并归一化

		if self.useAll == True: 
			sex = self.label_multi[0][index]
			age = self.label_multi[1][index]
			face = self.label_multi[2][index]
			race = self.label_multi[3][index]
			label = [[sex],[age],[face],[race]]
		else :
			label = self.label_encoded[index]
		label = torch.tensor(label)
		return img_data, label
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


# label_Path = "./人脸图像识别/face/"
# img_path = "./人脸图像识别/face/rawdata/"
# learning_rate = 1e-3
# batch_size = 128
# epochs = 100
# label_name = 'sex'         #对性别进行分类
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#                 transforms.Resize((128,128)),   #神经网络模型只能接受224*224输入
#                 transforms.ToTensor()           #转化为tensor并归一化
# ])
# full_dataset=FaceDataset(label_Path,img_path,label_name,useAll=True,transform=transform)
# train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# for batch, (X, y) in enumerate(train_loader):
# 	X,y = X.cuda(),y.cuda()
# 	batch_size = y.shape[0]
# 	y = torch.reshape(y,(batch_size,4))
# 	print(y[:,0])
# 	break
