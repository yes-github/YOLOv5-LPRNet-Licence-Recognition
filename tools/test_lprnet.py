# -*- coding: utf-8 -*-
# /usr/bin/env/python3

"""
test pretrained model.
Author: aiboy.wei@outlook.com .
"""
from torch.utils.data import DataLoader
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(curr_dir, "..")))

from models.LPRNet import CHARS, LPRNet
from utils.load_lpr_data import LPRDataLoader


def get_parser():
    parser = argparse.ArgumentParser(description="parameters to train net")
    parser.add_argument("--img_size", default=[94, 24], help="the image size")
    parser.add_argument(
        "--test_img_dirs",
        default="/home/ye/CODE/MY/YOLOv5-LPRNet-Licence-Recognition/tmp/datasets/brazil/rec/test_01",
        help="the test images path",
    )
    parser.add_argument("--dropout_rate", default=0, help="dropout rate.")
    parser.add_argument(
        "--lpr_max_len", default=7, help="license plate number max length."
    )
    parser.add_argument("--test_batch_size", default=100, help="testing batch size.")
    parser.add_argument(
        "--phase_train", default=False, type=bool, help="train or test phase flag."
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Number of workers used in dataloading",
    )
    parser.add_argument(
        "--cuda", default=True, type=bool, help="Use cuda to train model"
    )
    parser.add_argument(
        "--show",
        default=True,
        type=bool,
        help="show test image and its predict result or not.",
    )
    parser.add_argument(
        "--pretrained_model",
        default="/home/ye/CODE/MY/YOLOv5-LPRNet-Licence-Recognition/tmp/runs/brazil/lprnet-best.pth",
        help="pretrained base model",
    )

    args = parser.parse_args()

    return args


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


def test():
    args = get_parser()

    lprnet = LPRNet(
        lpr_max_len=args.lpr_max_len,
        phase=args.phase_train,
        class_num=len(CHARS),
        dropout_rate=args.dropout_rate,
    )
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)

    logger.info("== device:%s" % (device))
    logger.info("== Successful to build network! ==")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        logger.info("load pretrained model successful!")
    else:
        logger.error("Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(
        test_img_dirs.split(","), args.img_size, args.lpr_max_len
    )
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args, print_debug = True)
    finally:
        cv2.destroyAllWindows()


def Greedy_Decode_Eval(Net, datasets, args, print_debug=False):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size

    if epoch_size == 0:
        epoch_size = 1

    batch_iterator = iter(
        DataLoader(
            datasets,
            args.test_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    )

    Tp = 0
    Tn = 0
    # Tn_1 = 0
    # Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start : start + length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        # images: [bs, 3, 24, 94]
        # prebs:  [bs, 68, 18]
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        raw_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]  # 对每张图片 [68, 18]
            preb_label = list()

            # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))

            # 去除重复字符和空白字符'-'
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:  # 记录重复字符
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):  # 去除重复字符和空白字符'-'
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c

            # 得到最终的无重复字符和无空白字符的序列
            preb_labels.append(no_repeat_blank_label)

            #""" debug: 转成字符
            preb_lb = ""
            for i in preb_label:
                preb_lb += CHARS[i]
            raw_labels.append(preb_lb)
            
            # no_repeat_blank_lb = ""
            # for i in no_repeat_blank_label:
            #     no_repeat_blank_lb += CHARS[i]
            # print(preb_lb, no_repeat_blank_lb)
            
            #"""

        # 统计准确率
        for i, label in enumerate(preb_labels):
            target = targets[i]

            # 转成字符
            lb = ""
            for j in label:
                lb += CHARS[j]
            tg = ""
            for j in target.tolist():
                tg += CHARS[int(j)]

            # 判断对错
            if lb == tg:
                Tp += 1  # 正确
            else:
                Tn += 1  # 错误
                if print_debug:
                    print("target: ", tg, " ### F ### ", "predict: ", lb, "raw: " , raw_labels[i])

    Acc = Tp * 1.0 / (Tp + Tn)
    logger.info("Test Accuracy: {} [{}+{}={}]".format(Acc, Tp, Tn, (Tp + Tn)))
    t2 = time.time()
    logger.info(
        "Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets))
    )

    return Acc


@DeprecationWarning
def show(img, label, target):
    img = np.transpose(img, (1, 2, 0))
    img *= 128.0
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    else:
        # 只打印错误
        print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
    # print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)

    # img = cv2ImgAddText(img, lb, (0, 0))
    # cv2.imshow("test", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if isinstance(img, np.ndarray):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        os.path.join(curr_dir, "..", "fonts/simsun.ttc"), textSize, encoding="utf-8"
    )
    draw.text(pos, text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    test()
