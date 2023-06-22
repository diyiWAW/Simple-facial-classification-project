import copy

import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from util import Get_label, encode
from sklearn.model_selection import cross_val_score
import math
from FaceDataset import FaceDataset
import torch
from torch.utils.data import Dataset,DataLoader
from collections import Counter

#
# label_Path = r"D:\pattern_recognition\facedection\facedection\facedect\face\face"
# img_path = r"D:\pattern_recognition\facedection\facedection\facedect\face\face\rawdata"
# learning_rate = 1e-3
# batch_size = 128
# epochs = 100
# label_name = 'sex'         #对性别进行分类
#
# full_dataset=FaceDataset(label_Path,img_path,label_name)
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size]) #按照5:1划分测试集训练集
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


label_Path = r"D:\pattern_recognition\facedection\facedection\facedect\face\face"
label_all = Get_label(label_Path)
labe_dict = []
id_list=[]
num=3500   # 参与计算的有效标签的值
count=6   #最优的n个值
test_num=10  #测试用数据个数

for dict in label_all:  # 对于标签缺失的数据直接删除
    if dict['missing'] != 'true':
        labe_dict.append(dict)
        id_list.append(int(dict['id']))
    else:
        continue
print(np.shape(labe_dict))
sex_label, _ = encode(labe_dict, 'sex')
print(type(id_list))
id_list=np.array(id_list)
#打乱数组顺序
np.random.shuffle(id_list)

#选取前x作为训练，总数-x作为测试，保证不重复
id_list1=id_list[:num]
id_list2=id_list[num:]



# 提取人脸特征向量
def extract_features(image_path):
    with open(image_path, 'rb') as f:
        content = f.read()
    # 将文件内容转换为numpy数组
    img_array = np.frombuffer(content, dtype=np.uint8)
    if img_array[0] != 16384:  # 有的图像不是128*128的，把它reshape
        size = img_array.shape
        # print(size)
        size = int(math.sqrt(size[0]))
        img_array = img_array.reshape((size, size))
    else:
        img_array = img_array.reshape(128, 128)
    # 使用PIL库将numpy数组转换为图像对象
    # img1 = Image.fromarray(img_array)
    # gray_image = np.stack((img_array,) * 3, axis=-1)
    # gray_image = cv2.merge([img_array, img_array, img_array])
    # gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(img_array, None)
    print('keypoint大小',np.shape(keypoints),'原特征向量大小',np.shape(descriptors))
    if len(keypoints) == 0:
        descriptors = np.zeros((1, 128))
    # descriptors.flatten()
    return descriptors

# 加载人脸数据库并提取特征向量
def load_dataset(dataset_path,id_list):
    features = []
    features_idex=[]
    a=0
    on=0
    for root, dirs, files in os.walk(dataset_path):
        on=on+1
        print('files=',files)
        for file in files:
            if np.isin(int(file),id_list):   #判断id是否在要求的训练集中
                if on==1:
                    features_idex.append(file)
                image_path = os.path.join(root, file)
                # print('succ',a)
                features.append(extract_features(image_path))
                a=a+1
                # print('feature的shape是',np.shape(features))
                print('file=',file)

    print('feature的 shape',np.shape(features_idex))
    print('on=',on)

    # features = np.vstack([np.array(arr) for arr in features])
    # features = np.vstack([descriptor.reshape(-1, descriptor.shape[-1]) for descriptor in features])
    return features,features_idex



# 训练随机森林分类器
# def train_classifier(features, labels):
#     clf = RandomForestClassifier(n_estimators=50, max_depth=10)
#     clf.fit(features, labels)
#     return clf

# 加载人脸数据库并训练分类器
dataset_path = r'D:\pattern_recognition\facedection\facedection\facedect\face\face\rawdata'
feature_index=[]
features,feature_index= load_dataset(dataset_path,id_list1)
feature_index = list(map(int, feature_index))
feature_index = np.array(feature_index)



#Flann 算法
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

dir_path=r'D:\pattern_recognition\facedection\facedection\facedect\face\face\rawdata'
file_path_list =id_list2[:test_num]   # 测试图文件名
file_path_str_list=[str(num) for num in file_path_list]   # 转为字符串
file_path_str_list1=[]
file_path_str_list1=copy.deepcopy(file_path_str_list)

#检测测试图是否无特征
for name1 in file_path_str_list:
    path_temp = os.path.join(dir_path,name1)
    feature_temp=extract_features(path_temp)
    if feature_temp.shape[0]==1:
        file_path_str_list1.remove(name1)
        print('删除',name1)


#创建一个嵌套列表,用于存储权重和,次数
vect_arr=[]
size1=feature_index.size
for ii in range(size1):
    num_feature_tmp=features[ii].shape[0]
    a_temp=np.zeros(num_feature_tmp)
    vect_arr.append(a_temp)
# 遍历所有训练集,删除无特征的
x=0
del_arr=[]
for desc in features:
    if np.all(desc==0)==1:

        print('跳过无特征的图')
        #删除对应的数组,id列表
        del_arr.append(x)
        x = x + 1
        continue
    x=x+1

for i in sorted(del_arr, reverse=True):
    del vect_arr[i]
    feature_index = np.delete(feature_index, i)
    del features[i]
# #赋值测试
# vect_arr[2][4]=12


all_test_score_list=[]     # 去除ID的用于存储匹配结果
all_test_score_list1=[]   # 不去除ID的
 # 循环测试的图片集
for img_file_name in file_path_str_list1:
    name1 = os.path.join(dir_path, img_file_name)
    with open(name1, 'rb') as f:
        content = f.read()
    # 将文件内容转换为numpy数组
    img_array_test = np.frombuffer(content, dtype=np.uint8)
    img_array_test = img_array_test.reshape(128, 128)
    detector = cv2.SIFT_create()
    keypoints_test, descriptors_test = detector.detectAndCompute(img_array_test, None)

    matches = []
    x=0
    # 遍历所有训练集
    for desc in features:
        # if np.all(desc==0)==1:
        #
        #     print('跳过无特征的图')
        #     #删除对应的数组,id列表
        #     del vect_arr[x]
        #     feature_index = np.delete(feature_index,1)
        #     x = x + 1
        #     continue
        flann_matches = flann.knnMatch(descriptors_test,desc, k=2)
        matches.append(flann_matches)
        print('匹配数为',x,'大小为',np.shape(flann_matches))
        x=x+1

    score_list=[]   #得分表，用于选取前面的值
    best_match_score = 0
    best_match_label = ''
    for i, match in enumerate(matches):
        good_points = []
        print('match大小',np.shape(match),'i=',i)
        print(len(match))
        for m, n  in match:
            if m.distance < 0.7 * n.distance:

                print(f"queryIdx={m.queryIdx}, trainIdx={m.trainIdx}")  #看看索引
                print('i=',i,'feature_index[i]=',feature_index[i])
                vect_arr[i][m.trainIdx]=vect_arr[i][m.trainIdx]+1   # 统计出现次数
                good_points.append(m)
        score = len(good_points) / len(match)
        score_list.append(feature_index[i])#计算完插入Id号
        if score > best_match_score:
            best_match_score = score
            best_match_label = feature_index[i]
            if i >=1 :
                score_list=[score_list[-1]]+score_list[:-1] #最好的放在首位

    print('Best match: ', best_match_label)
    print('scorelist=',score_list[:count])

    #统计排名靠前的标签值

    id_str_list=[str(num) for num in score_list[:count]]   #转字符列表
    index_list=[]
    for k in id_str_list: #找index
        index_list.append(next((i for i,d in enumerate(labe_dict) if d["id"] == k), None))

    score_label=[]
    for g in index_list:  #对应index赋值
        score_label.append(labe_dict[int(g)])
    print(score_label)

    #统计各项的最大值，去除ID版
    final_infer = {k: Counter(d[k] for d in score_label).most_common(1)[0][0] for k in score_label[0].keys() if k != "id"}
    #不去除ID
    final_infer1={k: Counter(d[k] for d in score_label).most_common(1)[0][0] for k in score_label[0].keys() }
    print('final_infer',final_infer)
    all_test_score_list.append(final_infer)  # 加入结果到最终列表
    all_test_score_list1.append(final_infer1)

# 统计各项指标的准确率

#给出测试数据label
ori_index=[] #测试原数据index
ori_test_label=[]  #测试原数据标签，包含id
ori_test_label1=[] #不含id

for k2 in file_path_str_list:  # 找index
    ori_index.append(next((i for i, d in enumerate(labe_dict) if d["id"] == k2), None))

for g2 in ori_index:  # 对应index赋值
    ori_test_label.append(labe_dict[int(g2)])

# ori_test_label1=ori_test_label[:]
ori_test_label1=copy.deepcopy(ori_test_label)
#去除ID版   ori_test_label
key_to_remove = "id"
for d in ori_test_label1:
    del d[key_to_remove]

v1_test=[]
v2_ori=[]

#计算正确值
# for r1 in all_test_score_list:
#     v1_test.append([v for k, v in sorted(r1.items())])  # 匹配结果向量label
# for r2 in ori_test_label1:
#     v2_ori.append([v for k, v in sorted(r2.items())])     #原测试label
for r1 in all_test_score_list:
    v1_test.append(list(r1.values()))
for r2 in ori_test_label1:
    v2_ori.append(list(r2.values()))


acc1=[] # 准确率
acc_temp1=0
for ii in range(6):
    acc_temp1=0
    for jj in range(test_num):
        if v2_ori[jj][ii]==v1_test[jj][ii]:
            acc_temp1=acc_temp1+1
    acc1.append(acc_temp1/test_num)
print(acc1)

#保存参数
np.save('v2_ori.npy',v2_ori)
np.save('v1_test.npy',v1_test)
# vect_arr1=np.vstack(vect_arr) # 列表转数组
np.save('vect_arr.npy',vect_arr)

print(type(vect_arr[0][2]))
print(1)
