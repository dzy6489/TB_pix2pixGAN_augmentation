import xml.etree.ElementTree as ET
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import json

import config
data_direction = config.Data_direction

data_pairs = {}



def overlay(img_canny,img_anno,img_lung_mask):
    # 确保图像 A 和图像 B 具有相同的尺寸
    img_anno = cv2.resize(img_anno, (img_canny.shape[1], img_canny.shape[0]))

    # 将图像 B 转换为灰度图像
    gray_b = cv2.cvtColor(img_anno, cv2.COLOR_BGR2GRAY)

    # 二值化处理，将非零像素设置为白色（255），零像素设置为黑色（0）
    _, mask = cv2.threshold(gray_b, 1, 255, cv2.THRESH_BINARY)

    # 反转遮罩，使得非零像素为黑色，零像素为白色
    mask_inv = cv2.bitwise_not(mask)

    # 将图像 B 中的非零像素区域复制到图像 A 中
    image_a_bg = cv2.bitwise_and(img_canny, img_canny, mask=mask_inv)
    image_b_fg = cv2.bitwise_and(img_anno, img_anno, mask=mask)
    result = cv2.add(image_a_bg, image_b_fg)
    masked_canny_anno_img = cv2.bitwise_and(result,img_lung_mask)

    return masked_canny_anno_img




def generate_singe_anno_img(image_id,t1,t2,train_val):

    tb_img_path = config.Data_direction+"TB/CHNCXR_"+image_id+"_1.png"   #image name
    tb_img = cv2.imread(tb_img_path)
    tb_img = cv2.resize(tb_img,(512,512))   ###原本的tb图，resize到512，512，否则edge图产生不了纹理，因为太细节了，没有骨骼轮廓
    if tb_img.shape.__len__()!=3:
        tb_img = cv2.cvtColor(tb_img,cv2.COLOR_GRAY2RGB)
    canny_img = cv2.Canny(tb_img, t1, t2)
    canny_img_color = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2RGB)  #edge图

    lung_anno_path = config.Data_direction+"lesion_with_lung_outline/masked_CHNCXR_"+image_id+"_1_mask.png"
    lung_anno_img = cv2.imread(lung_anno_path)
    lung_anno_img = cv2.resize(lung_anno_img,(512,512))  ###仅有病灶和肺外部轮廓的图片


    lung_mask_path = config.Data_direction+"lung_mask/CHNCXR_"+image_id+"_1_mask.png"
    lung_mask_img = cv2.imread(lung_mask_path)
    lung_mask_img = cv2.resize(lung_mask_img, (512, 512))

    masked_canny_image = overlay(canny_img_color,lung_anno_img,lung_mask_img)  ###将病灶图片和edge图片叠加起来，并且只保留肺部区域图片

    sub_dir = 'canny_crop_'+str(t1)+'_'+str(t2)+'/'+train_val

    sub_TB_dir = sub_dir + 'TB/'
    sub_canny_dir = sub_dir + 'canny/'
    sub_canny_crop_dir = sub_dir + 'lesion_with_edge_lung_outline/'
    sub_lung_anno_dir = sub_dir + 'lesion_with_lung_outline/'

    data_direction = config.Data_direction +'canny_crop_datasets/'
    os.makedirs(data_direction + sub_TB_dir, exist_ok=True)
    os.makedirs(data_direction + sub_canny_dir, exist_ok=True)
    os.makedirs(data_direction + sub_canny_crop_dir, exist_ok=True)
    os.makedirs(data_direction + sub_lung_anno_dir, exist_ok=True)

    cv2.imwrite(data_direction + sub_TB_dir + image_id + '.png', tb_img)

    cv2.imwrite(data_direction + sub_lung_anno_dir + image_id + '.png', lung_anno_img)

    cv2.imwrite(data_direction + sub_canny_dir + image_id + '_canny.png', canny_img_color)

    cv2.imwrite(data_direction + sub_canny_crop_dir + image_id + '_canny_crop.png', masked_canny_image)


def generate_directory_anno_img(directory,t1,t2):
    for root, dirs, files in os.walk(directory+"lesion_with_lung_outline/"):
        count = 0
        for file in files:
            img_id = file.split('_')[2]
            if count<(len(files)*config.train_test_ratio):
                    generate_singe_anno_img(img_id,t1,t2,'train/')
                    count = count + 1
            else:
                    generate_singe_anno_img(img_id, t1, t2, 'val/')
                    count = count + 1
    with open('data_pairs_256.json', 'w', encoding='utf-8') as f:
        json.dump(data_pairs, f, ensure_ascii=False, indent=4)

    print("字典已成功保存到本地文件。")

generate_directory_anno_img(data_direction,config.edge_t1,config.edge_t2)  ###产生所有带注释的图片

