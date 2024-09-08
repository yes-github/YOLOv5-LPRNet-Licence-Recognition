"""
@Author: HuKai
@Date: 2022/5/29  10:44
@github: https://github.com/HuKai97
"""

import os
import random
import shutil

image_raw_dir = "Z:/CCPD/raw"
trainfiles = os.listdir(image_raw_dir)  # （图片文件夹）
num_train = len(trainfiles)
print("num_train: " + str(num_train))
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)  # 打乱顺序
num = 0

image_base_dir = "Z:/CCPD/det/images"

trainDir = os.path.join(image_base_dir, "train")
validDir = os.path.join(image_base_dir, "val")
testDir = os.path.join(image_base_dir, "test")

# 删除目录
shutil.rmtree(image_base_dir, ignore_errors=True)

# 新建目录
os.makedirs(image_base_dir, exist_ok=True)
os.makedirs(trainDir, exist_ok=True)
os.makedirs(validDir, exist_ok=True)
os.makedirs(testDir, exist_ok=True)

for i in index_list:
    src_file_path = os.path.join(image_raw_dir, trainfiles[i])

    if num < num_train * 0.7:  # 7:1:2
        print(str(src_file_path))
        shutil.copy2(src_file_path, os.path.join(trainDir, trainfiles[i]))
    elif num < num_train * 0.8:
        print(str(src_file_path))
        shutil.copy2(src_file_path, os.path.join(validDir, trainfiles[i]))
    else:
        print(str(src_file_path))
        shutil.copy2(src_file_path, os.path.join(testDir, trainfiles[i]))
    num += 1
