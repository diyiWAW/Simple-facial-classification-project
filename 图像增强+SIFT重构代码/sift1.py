import cv2
import os
import numpy as np
from util import Get_label, encode
import math
import matplotlib.pyplot as plt
from PIL import Image


label_Path = r"D:\pattern_recognition\facedection\facedection\facedect\face\face"
label_all = Get_label(label_Path)
labe_dict = []
for dict in label_all:  # 对于标签缺失的数据直接删除
    if dict['missing'] != 'true':
        labe_dict.append(dict)
    else:
        continue
print(np.shape(labe_dict))
sex_label, _ = encode(labe_dict, 'sex')


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
    # descriptors.flatten()
    return descriptors

# 加载人脸数据库并提取特征向量
def load_dataset(dataset_path):
    features = []
    for root, dirs, files in os.walk(dataset_path):
        print('files=',files)
        for file in files:
            image_path = os.path.join(root, file)
            # print('succ',a)
            features.append(extract_features(image_path))
            # print('feature的shape是',np.shape(features))
            print('file=',file)

    return features


# 加载人脸数据库并训练分类器
dataset_path = r'D:\pattern_recognition\facedection\facedection\facedect\face\face\rawdata'
features= load_dataset(dataset_path)

#Flann 算法
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

with open(r'D:\pattern_recognition\facedection\facedection\facedect\face\face\rawdata\5217', 'rb') as f:
    content = f.read()
# 将文件内容转换为numpy数组
img_array_test = np.frombuffer(content, dtype=np.uint8)
img_array_test = img_array_test.reshape(128, 128)
detector = cv2.SIFT_create()
keypoints_test, descriptors_test = detector.detectAndCompute(img_array_test, None)

matches = []
x=0
for desc in features:
    if np.all(desc==0)==1:
        x=x+1
        print('跳过无特征的图')
        continue
    flann_matches = flann.knnMatch(desc, descriptors_test, k=2)
    matches.append(flann_matches)
    print('匹配数为',x,'大小为',np.shape(flann_matches))
    x=x+1


best_match_score = 0
best_match_label = ''
for i, match in enumerate(matches):
    good_points = []
    print('match大小',np.shape(match))
    print(len(match))
    for m, n  in match:
        if m.distance < 0.7 * n.distance:
            good_points.append(m)
    score = len(good_points) / len(match)
    if score > best_match_score:
        best_match_score = score
        best_match_label = labe_dict[i]

img1 = Image.fromarray(img_array_test)

name1=best_match_label['id']

path2=os.path.join(dataset_path,name1)
with open(path2, 'rb') as f:
    content = f.read()
    # 将文件内容转换为numpy数组
img_array2 = np.frombuffer(content, dtype=np.uint8)
if img_array2[0] != 16384:  # 有的图像不是128*128的，把它reshape
    size = img_array2.shape
    # print(size)
    size = int(math.sqrt(size[0]))
    img_array2 = img_array2.reshape((size, size))
else:
    img_array2 = img_array2.reshape(128, 128)

img2=Image.fromarray(img_array2)
# img1.show()
bf = cv2.BFMatcher()
sift = cv2.SIFT_create()
keypoints_2, descriptors_2 = detector.detectAndCompute(img_array2, None)

img3 = cv2.drawKeypoints(img_array_test,keypoints_test,img_array_test,color=(255,0,255)) #画出特征点，并显示为红色圆圈
#img3 = cv2.drawKeypoints(gray, kp, img) 在图中画出关键点   参数说明：gray表示输入图片, kp表示关键点，img表示输出的图片
#print(img3.size)
img4 = cv2.drawKeypoints(img_array2,keypoints_2,img_array2,color=(255,0,255)) #画出特征点，并显示为红色圆圈

hmerge = np.hstack((img3, img4)) #水平拼接
cv2.imshow("point", hmerge) #拼接显示为gray
cv2.waitKey(0)


img5 = cv2.drawMatchesKnn(img_array_test,keypoints_test,img_array2,keypoints_2,match,None,flags=2)
cv2.imshow("BFmatch", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()


print('Best match: ', best_match_label)


