import json
import cv2

# with open('/home/liuhao/seedland/dataset/coco/coco2014/annotations/TEST_objects.json','r') as f:
#     labels = json.load(f)
#
# with open('/home/liuhao/seedland/dataset/coco/coco2014/annotations/TEST_images.json','r') as f:
#     paths = json.load(f)
#
# for i,path in enumerate(paths):
#     img =cv2.imread(path)
#     label = labels[i]
#     for bbox in label['boxes']:
#         cv2.rectangle(img,pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]),thickness=2,color=(255,5,5))
#     cv2.imshow('a',img)
#     cv2.waitKey(1000)

txt_path = '/home/liuhao/seedland/dataset/crowhuman/train_crowhuman.txt'

f = open(txt_path, 'r')
lines = f.readlines()
isFirst = True
imgs_path = []
labels = []
words = []
for line in lines:
    line = line.rstrip()
    if line.startswith('#'):
        if isFirst is True:
            isFirst = False
        else:
            labels_copy = labels.copy()
            words.append(labels_copy)
            labels.clear()
        path = line[2:]
        path = txt_path.replace('train_crowhuman.txt', 'TRAIN/') + path
        imgs_path.append(path)
    else:
        line = line.split(' ')
        label = [float(x) for x in line]
        labels.append(label)
from tqdm import tqdm
import cv2
for i,path in tqdm(enumerate(imgs_path)):
    img = cv2.imread(path)
    for j in range(len(words[i])):
        xmin = int(words[i][j][0])
        ymin = int(words[i][j][1])
        xmax = int(words[i][j][2])
        ymax = int(words[i][j][3])
        cv2.rectangle(img ,pt1=(xmin,ymin),pt2=(xmax,ymax),color=(233,3,3),thickness=2)
    cv2.imshow('asf', img)
    cv2.waitKey(200)
