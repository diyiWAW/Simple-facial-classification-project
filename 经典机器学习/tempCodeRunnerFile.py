print(pred[:,0:2].argmax(1) == y[:,0])
# print(pred[:,2:4].argmax(1) == y[:,1])
# print(((pred[:,0:2].argmax(1) == y[:,0]).numpy() & (pred[:,2:4].argmax(1) == y[:,1]).numpy()).sum())
# print(pred[:,2:4].argmax(1))
# # correct += (pred[:,0:2].argmax(1) == y[:,0] and
# # 						pred[:,2:4].argmax(1) == y[:,1] 
# # 						).type(torch.float).sum().item()