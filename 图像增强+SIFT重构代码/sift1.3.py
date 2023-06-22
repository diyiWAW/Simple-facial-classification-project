import copy
import random
import cv2
import os
import numpy as np
from util import Get_label, encode
import math
from collections import Counter


label_Path = r"D:\pattern_recognition\facedection\facedection\facedect\face\face"
label_all = Get_label(label_Path)
labe_dict = []
id_list=[]
num=3500   # 参与计算的有效标签的值

count=5   #最优的n个值

for dict in label_all:  # 对于标签缺失的数据直接删除
    if dict['missing'] != 'true':
        labe_dict.append(dict)
        id_list.append(int(dict['id']))
    else:
        continue
print(np.shape(labe_dict))
#预测类别，名字就不改了，age:senior,adult,teen,child;race:white,hispanic,Asian,other,black;face:serious,smiling,funny
# prop:hat,hat moustache beard ,moustache beard,glasses,
sex_label, _ = encode(labe_dict, 'face')
print(type(id_list))
id_list=np.array(id_list)
#打乱数组顺序
np.random.shuffle(id_list)

#选取前x作为训练，总数-x作为测试，保证不重复
id_list1=id_list[:num]
id_list2=id_list[num:]

##########################################################################################################
#准备训练和测试数据

#训练数据
train_num = 5   #训练数据个数,随机打乱,用id_list1里的
# random_integers = random.sample(range(len(id_list2)), train_num)
# file_path_list=[]
# for kl in random_integers:
#     file_path_list.append(id_list1[kl])    #后面须转成字符串数组

#不搞那么麻烦了
file_path_list=[]
file_path_list=copy.deepcopy(id_list2[0:train_num])

#测试数据
test_num1 = 5    #真正测试数据个数,随机打乱
true_test_name=[]
true_test_name=copy.deepcopy(id_list2[train_num:train_num+test_num1])
# random_integers1 = random.sample(range(len(id_list2)), test_num1)   #获取其Index
# #避免和训练相同
# while(1):
#     for se in random_integers:
#         if se in random_integers1:
#             random_integers1.remove(se)
#     if len(random_integers1)==test_num1:
#         break
#     else:
#         comp=test_num1-len(random_integers1)
#         for k in range(comp):
#             random_int = random.randint(0, len(id_list2))
#             random_integers1.append(random_int)
# #获取其ID
# true_test_name=[]    # 后面须转为字符串数组
# for mm in random_integers1:
#     true_test_name.append(id_list2[mm])


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

    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(img_array, None)
    print('keypoint大小',np.shape(keypoints),'原特征向量大小',np.shape(descriptors))
    if len(keypoints) == 0:
        descriptors = np.zeros((1, 128))
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

# 加载人脸数据库并训练分类器
dataset_path = r'D:\pattern_recognition\facedection\facedection\facedect\face\face\rawdata'
feature_index=[]
features,feature_index= load_dataset(dataset_path,id_list1)
feature_index = list(map(int, feature_index))
feature_index = np.array(feature_index)



#Flann 算法
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

dir_path=r'D:\pattern_recognition\facedection\facedection\facedect\face\face\rawdata'



# file_path_list =id_list2[:test_num]   # 测试图文件名
file_path_str_list=[str(num) for num in file_path_list]   # 转为字符串
file_path_str_list1=[]
file_path_str_list1=copy.deepcopy(file_path_str_list)

#检测训练图是否无特征,这可作为一个模块
for name1 in file_path_str_list:
    path_temp = os.path.join(dir_path,name1)
    feature_temp=extract_features(path_temp)
    if feature_temp.shape[0]==1:
        file_path_str_list1.remove(name1)
        train_num=train_num-1
        print('删除',name1)


#创建一个嵌套列表,用于存储权重和,次数
vect_arr=[]
size1=feature_index.size
for ii in range(size1):
    num_feature_tmp=features[ii].shape[0]
    a_temp=np.zeros(num_feature_tmp)
    vect_arr.append(a_temp)

# 遍历所有训练集,删除无特征的图对应的ID和features
x=0
del_arr=[]
for desc in features:
    if np.all(desc==0)==1:

        print('跳过无特征的图')
        #添加需要删除的位置index给del_arr
        del_arr.append(x)
        x = x + 1
        continue
    x=x+1
# 训练集实施删除
for i in sorted(del_arr, reverse=True):
    del vect_arr[i]
    feature_index = np.delete(feature_index, i)
    del features[i]
# #赋值测试
# vect_arr[2][4]=12


#给出测试数据label
ori_index=[] #测试原数据index
ori_test_label=[]  #测试原数据标签，包含id
ori_test_label1=[] #不含id


# 从ID找label
#先找index
for k2 in file_path_str_list1:  # 找index
    ori_index.append(next((i for i, d in enumerate(labe_dict) if d["id"] == k2), None))

for g2 in ori_index:  # 对应index赋值
    ori_test_label.append(labe_dict[int(g2)])

# ori_test_label1=ori_test_label[:]
ori_test_label1=copy.deepcopy(ori_test_label)
#去除ID版   ori_test_label
key_to_remove = "id"
for d in ori_test_label1:
    del d[key_to_remove]

# 建立单个label部分
#创建一个矩阵sex_vect,用于对应测试数据于训练数据的关系,假设训练数据有3400个,测试数据有400个,那这个矩阵就是(400*3400),对于训练数据与测试数据相同
#的ID,那么其权值设置为1,否则设置为0.这个矩阵将作为权重与次数据怎vect_arr相乘

sex_vect=np.zeros((len(ori_index),len(vect_arr)))
feature_index_str =[str(num) for num in feature_index]   # 创建一个str类型的训练集index表
#查找与测试数据相同的label值
tem=0   #用于循环sexvect
tem2=0
#激活矩阵
for list_temp in ori_test_label :
    #测试集ID所对应的Label dict 的index
    for idx_temp in feature_index_str:
        #搜索此ID的label并判断是否对应测试集的label
        label_temp_index=next((i for i, d in enumerate(labe_dict) if d["id"] == idx_temp), None)
        # 判断是否相同
        if labe_dict[int(label_temp_index)]['face']==list_temp['face']:
            sex_vect[tem][tem2]=1
        tem2=tem2+1
    tem2=0
    tem=tem+1

tem=0
tem2=0


all_test_score_list=[]     # 去除ID的用于存储匹配结果
all_test_score_list1=[]   # 不去除ID的

##################################################################################################################
###################################################################################################################
###############################################################################################################
#训练开始

 # 循环训练的图片集
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



v1_test=[]
v2_ori=[]


for r1 in all_test_score_list:
    v1_test.append(list(r1.values()))
for r2 in ori_test_label1:
    v2_ori.append(list(r2.values()))

acc1=[] # 准确率
acc_temp1=0
for ii in range(6):
    acc_temp1=0
    for jj in range(train_num):
        if v2_ori[jj][ii]==v1_test[jj][ii]:
            acc_temp1=acc_temp1+1
    acc1.append(acc_temp1/train_num)
print(acc1)

#保存参数
np.save('v2_ori.npy',v2_ori)
np.save('v1_test.npy',v1_test)
# vect_arr1=np.vstack(vect_arr) # 列表转数组
np.save('vect_arr.npy',vect_arr)

####################################################################################################
######################################################################################################
#训练完毕,准备测试数据
## 计算权值weight_vect
weight_vect=copy.deepcopy(vect_arr)
for kk in range(sex_vect.shape[0]):
    for jj in range(sex_vect.shape[1]):
        slice1_weight_vect=vect_arr[jj]*sex_vect[kk][jj]
        weight_vect[jj]=weight_vect[jj]+slice1_weight_vect

## 找出数组列表里所有不为0的值,以及他们的index,还有对应的向量

# 查找所有不为0的值
#非零权重矩阵:no_zero_weight,及其feature索引no_zero_weight_index
no_zero_weight=[]
no_zero_weight_index=[]
#创建一个ID的索引,[1223,1223,1223,1223,1224,1224,1224] 重复的个数为一个图像中有效的特征个数,及no_zero_weight[i]这个数组的长度
ID_index1=[]
#为了避免出现一组图像中没有显著特征向量,导致其被筛掉排序混乱,于是添加一个index记录数组
doc_index1=[]
doc=0
#选取有权重的向量
# 这里只是用0作为筛选的条件,理论上要根据具体条件进一步筛选
for i in range(len(weight_vect)):
    #数组全0则跳过
    if np.all(weight_vect[i]==0):
        doc=doc+1
        continue
    temp_weight_values1=weight_vect[i][weight_vect[i]!=0]
    no_zero_weight.append(temp_weight_values1)

    for l in range(len(temp_weight_values1)):
        ID_index1.append(feature_index[i])

    #非零元素的index
    temp_wieght_index=np.nonzero(weight_vect[i])[0]
    no_zero_weight_index.append(temp_wieght_index)
    doc_index1.append(i)   # 记录有效Index
    doc=doc+1

print('------------------no zero weight done-----------------------------')
# 将所有的feature都汇集到sex_feature 中,要考虑到可能出现没有显著特征的组
sex_feature=[]
iii=0
for b1 in doc_index1:
    for idx1 in no_zero_weight_index[iii]:
        print('b1=',b1,'idx1=',idx1)
        sex_feature.append(features[b1][idx1])
    iii=iii+1
print('------------------sex_feature compile done-----------------------------')

# 转换sex_feature 为数组
sex_feature1=np.array(sex_feature)

# 将no_zero_weight变成一维数组
no_zero_weight_arr=np.concatenate(no_zero_weight)

#求出ID_index对应的sex值,male=1,female=0,给到sex_feature_label
sex_feature_label=[]
sex_feature_label_index=[]
k2_pre=0
for k2 in ID_index1:  # 找index
    if k2_pre ==k2:
        sex_feature_label_index.append(a1_pre)
        continue
    a1=next((i for i, d in enumerate(labe_dict) if d["id"] == str(k2)), None)
    sex_feature_label_index.append(a1)
    print('k2=',k2)
    k2_pre=k2
    a1_pre=a1
print('------------------sex feature label index get done-----------------------------')
#race,face,age,sex
for g2 in sex_feature_label_index:  # 对应index赋值
    if labe_dict[int(g2)]['face']=='funny':
        sex_feature_label.append(3)
        continue
    if labe_dict[int(g2)]['face']=='smiling':
        sex_feature_label.append(2)
        continue
    else:
        sex_feature_label.append(1)

print('------------------sex feature label get done-----------------------------')

#############################################################################################
###############################################################################################
################################################################################################
##测试, 随机获取测试的id

#转为字符串
true_test_name_str=[str(num) for num in true_test_name]

#避免测试集出现无特征图,同时调整已设置的测试个数
for name1 in true_test_name_str:
    path_temp = os.path.join(dir_path,name1)
    feature_temp=extract_features(path_temp)
    if feature_temp.shape[0]==1:
        true_test_name_str.remove(name1)
        test_num1=test_num1-1
        print('删除',name1)

##获取测试的ID的sex值
testset_index=[]
testset_sex=[]

for k2 in true_test_name_str:  # 找index
    testset_index.append(next((i for i, d in enumerate(labe_dict) if d["id"] == k2), None))

for g2 in testset_index:  # 对应index赋值
    testset_sex.append(labe_dict[int(g2)]['face'])


sex_feature2=[]
#对sex_feature切片,避免过大
num_in_group=100   #设置100个向量一组

inter1=int(sex_feature1.shape[0])
remain1= int(sex_feature1.shape[0]%num_in_group)


for loc in range(0,inter1,num_in_group):
    if loc+num_in_group>= inter1:
        break
    sex_feature2.append(sex_feature1[loc:loc+num_in_group,:])

sex_feature2.append(sex_feature1[loc:-1,:])
print(np.shape(sex_feature2))
##重复上边的测试

result_test_list1=[]

for img_file_name in true_test_name_str:
    name1 = os.path.join(dir_path, img_file_name)
    with open(name1, 'rb') as f:
        content = f.read()
    # 将文件内容转换为numpy数组
    img_array_test = np.frombuffer(content, dtype=np.uint8)
    img_array_test = img_array_test.reshape(128, 128)
    detector = cv2.SIFT_create()
    keypoints_test, descriptors_test = detector.detectAndCompute(img_array_test, None)

    matches = []
    x=0   #
    # 遍历所有训练集
    for desc1 in sex_feature2:
        flann_matches = flann.knnMatch(descriptors_test,desc1, k=2)
        matches.append(flann_matches)
        print('匹配数为',x,'大小为',np.shape(flann_matches))
        x=x+1

    score_list1=[]   #得分表，用于选取前面的值
    valid_point_loc=[]
    best_match_score = 0
    best_match_label = ''
    for i, match in enumerate(matches):
        good_points = []
        print('match大小',np.shape(match),'i=',i)
        print(len(match))
        score_temp_best=0
        for m, n  in match:
            if m.distance < 0.7 * n.distance:
                score_temp=1-(m.distance/n.distance)   #1-直接相除,得到得分
                print(f"queryIdx={m.queryIdx}, trainIdx={m.trainIdx}")  #看看索引
                #插入有效的点
                valid_point_loc.append(num_in_group*i+m.trainIdx)
                print('i=',i,'feature_index[i]=',feature_index[i])
                good_points.append(m)
                score_list1.append(score_temp)

    # 获取前面count个score
    top_count = sorted(score_list1, reverse=True)[:count]
    # 查找这count个元素在原始score列表中的索引,放在top_count_index中
    top_count_index = []
    for i in top_count:
        top_count_index.append(score_list1.index(i))


    #再从这个index找到具体点的位置对应valid_point_loc中的weight,sex值的index,放在top_list_all,这是直接参与计算的
    top_list_all=[]
    for nj in top_count_index:
        top_list_all.append(valid_point_loc[nj])

    # 找到sex_feature_label 在这top对应的sex值,放入result1中
    result1=[]
    for mn in top_list_all:
        result1.append(sex_feature_label[mn])

    # 找到权重,放入result_weight中
    result_weight=[]
    for mn1 in top_list_all:
        result_weight.append(no_zero_weight_arr[mn1])

    # 计算权重得分
    result1=np.array(result1)
    result_weight=np.array(result_weight)
    result_weight=np.transpose(result_weight)

    result_test=np.dot(result1,result_weight)
    result_test=result_test/np.sum(result_weight)   # 计算得分

    #放到列表里
    result_test_list1.append(result_test)
    print('result=',result_test)
    print('-------------------------------------')

print(result_test_list1)

result_test_word=[]
# 转换数值为male or female(年龄则为senior,adult,teen),race,face,prop
for rd in result_test_list1:
    if rd >2:
        result_test_word.append('funny')
        continue
    if rd >1 and rd <= 2:
        result_test_word.append('smiling')
        continue

    else:
        result_test_word.append('serious')

# 计算准确率
count1 =0
for l in range(len(result_test_word)):
    if result_test_word[l]==testset_sex[l]:
        count1=count1+1

accu= count1/len(result_test_word)

print('accu=',accu)
print('testset=',testset_sex)
print('predict=',result_test_word)


        # if score > best_match_score:
        #     best_match_score = score
        #     best_match_label = feature_index[i]
        #     if i >=1 :
        #         score_list1=[score_list1[-1]]+score_list1[:-1] #最好的放在首位


#统计分别







print(1)
