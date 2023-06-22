import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets,models,transforms


resnet_18 = models.resnet18(pretrained=False)    #使用resnet18网络结构
resnet_18.fc = torch.nn.Linear(512, 2)           #修改全连接层的分类数目
resnet_18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #修改卷积核，因为图片单通道
resnet_18.load_state_dict(torch.load('./best_network.pth'))
resnet_18.fc = nn.Sequential() #删除全连接层
print(resnet_18)

img_data = np.load('./img.npy')

data = img_data[0].reshape(128,128)
data = torch.tensor(data,dtype=torch.float)
data = data.unsqueeze(0)
data = data.unsqueeze(0)
feature =resnet_18(data)
feature = feature.cpu().detach().numpy()

for i in range(1,img_data.shape[0]) :
	print(img_data[i].shape)
	data = img_data[i].reshape(128,128)
	data = torch.tensor(data,dtype=torch.float)
	data = data.unsqueeze(0)
	data = data.unsqueeze(0)
	output =resnet_18(data)
	output = output.cpu().detach().numpy()
	feature = np.vstack((feature, output))

print(feature.shape)
np.save('feature.npy',feature)