import numpy
import torch

correct = 0
pred = torch.tensor([[0.8,0.5,0.8,0.4],[0.9,0.8,0.1,0.2]])
y = torch.tensor([[0,1],[0,1]])

# print(pred[:,0:2].argmax(1) == y[:,0])
# print(pred[:,2:4].argmax(1) == y[:,1])
# print(((pred[:,0:2].argmax(1) == y[:,0]).numpy() & (pred[:,2:4].argmax(1) == y[:,1]).numpy()).sum())
# print(pred[:,2:4].argmax(1))
# # correct += (pred[:,0:2].argmax(1) == y[:,0] and
# # 						pred[:,2:4].argmax(1) == y[:,1] 
# # 						).type(torch.float).sum().item()

import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号