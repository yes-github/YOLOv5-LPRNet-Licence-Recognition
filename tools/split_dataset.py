"""
划分训练集、验证集和测试集
@Author: HuKai
@Date: 2022/5/29  10:44
@github: https://github.com/HuKai97
"""

import os
from loguru import logger
import random
from shutil import copy2

curr_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(curr_dir, "..")))

# 复制5w张图片
# find CCPD/raw/CCPD2019/ccpd_base -type f | head -n 50000 | xargs -I {} cp {} datasets/ccpd/raw/base/
base_dir = os.path.abspath(os.path.join(curr_dir,"..", "tmp/datasets/brazil/rec"))
trainfiles = os.listdir(os.path.join(base_dir, "base"))  # （图片文件夹）

num_train = len(trainfiles)
logger.info("num_train: " + str(num_train))
index_list = list(range(num_train))
# print(index_list)
random.shuffle(index_list)  # 打乱顺序
num = 0
trainDir = os.path.join(base_dir, "train")  # （将图片文件夹中的6份放在这个文件夹下）
validDir = os.path.join(base_dir, "val")  # （将图片文件夹中的2份放在这个文件夹下）
detectDir = os.path.join(base_dir, "test")  # （将图片文件夹中的2份放在这个文件夹下）
for i in index_list:
    fileName = os.path.join(
        base_dir, "base", trainfiles[i]
    )  # （图片文件夹）+图片名=图片地址
    if num < num_train * 0.7:  # 7:1:2
        # print(str(fileName))
        copy2(fileName, trainDir)
    elif num < num_train * 0.8:
        # print(str(fileName))
        copy2(fileName, validDir)
    else:
        # print(str(fileName))
        copy2(fileName, detectDir)
    num += 1

logger.info("== END ==")