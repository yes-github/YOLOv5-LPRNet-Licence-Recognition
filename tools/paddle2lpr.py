# -*- coding: utf-8 -*-
import os
import shutil
import cv2
import numpy as np
from PIL import Image

"""
将paddle标注的车牌图片数据转成LPRNet格式：
- 图片名即为车牌内容
- 大小为94*24
"""

#
if __name__ == "__main__":
    label_file_path = "/home/ye/CODE/LPAS/train-data/PaddleOCR/train_data/license-plate-recognition-v2/train_data/brazilian_license_plates/rec/val/val01.txt"
    image_dir_path = "/home/ye/CODE/LPAS/train-data/PaddleOCR/train_data/license-plate-recognition-v2/train_data/brazilian_license_plates/rec/val/val01"

    output_dir_path = "/home/ye/CODE/MY/YOLOv5-LPRNet-Licence-Recognition/tmp/datasets/brazil/rec/test_01"
    shutil.rmtree(output_dir_path, ignore_errors=True)
    os.makedirs(output_dir_path, exist_ok=True)

    with open(label_file_path, "r", encoding="utf-8") as label_file:
        label_file_lines = label_file.readlines()

    for line in label_file_lines:
        columns = line.strip().split("\t")
        src_file_path = os.path.join(image_dir_path, columns[0].split("/")[-1])
        dest_file_name = columns[1]
        dest_file_path = os.path.join(output_dir_path, "%s.jpg" % dest_file_name)

        img = cv2.imread(src_file_path)
        img = Image.fromarray(img)
        img = img.resize((94, 24), Image.LANCZOS)
        img = np.asarray(img)  # 转成array,变成24*94*3

        cv2.imencode(".jpg", img)[1].tofile(os.path.join(dest_file_path))
