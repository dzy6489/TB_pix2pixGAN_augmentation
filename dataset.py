import cv2
import numpy as np
import config
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import json






class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))

        input_image = image[:, :256, :]    #for edge2shoes
        target_image = image[:, 256:, :]    #for edge2shoes

        # input_image = image[:, 256:, :]  # for facades
        # target_image = image[:, :256, :]  # for facades

        # input_image = image[:, :600, :]   ###for maps
        # target_image = image[:, 600:, :]###for maps

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        # input_image = config.transform_only_input(image=target_image)["image0"]
        # target_image = config.transform_only_mask(image=input_image)["image0"]

        return input_image, target_image

class TB_shenzhen_Dataset_anno_TB(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tb_dir = os.path.join(root_dir, "TB")
        self.canny_crop_dir = os.path.join(root_dir, "lesion_with_lung_outline")
        self.tb_files = os.listdir(self.tb_dir)
        self.canny_crop_files = os.listdir(self.canny_crop_dir)

    def __len__(self):
        return len(self.canny_crop_files)

    def __getitem__(self, idx):

        canny_crop_img_name = os.path.join(self.canny_crop_dir, self.canny_crop_files[idx])  # Assuming filenames match
        tb_img_name = os.path.join(self.tb_dir, self.canny_crop_files[idx][:4]+'.png')

        # Open images
        # image = np.array(Image.open(img_path))
        tb_img = np.array(Image.open(tb_img_name))
        if tb_img.shape.__len__()==2:
            tb_img = cv2.cvtColor(tb_img, cv2.COLOR_GRAY2RGB)


        canny_crop_img = np.array(Image.open(canny_crop_img_name))

        augmentations = config.both_transform(image=canny_crop_img, image0=tb_img)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]


        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]


        return input_image, target_image


class TB_shenzhen_Dataset_canny_tb(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tb_dir = os.path.join(root_dir, "TB")
        self.canny_crop_dir = os.path.join(root_dir, "lesion_with_edge_lung_outline")
        self.tb_files = os.listdir(self.tb_dir)
        self.canny_crop_files = os.listdir(self.canny_crop_dir)

    def __len__(self):
        return len(self.canny_crop_files)

    def __getitem__(self, idx):

        canny_crop_img_name = os.path.join(self.canny_crop_dir, self.canny_crop_files[idx])  # Assuming filenames match
        tb_img_name = os.path.join(self.tb_dir, self.canny_crop_files[idx][:4]+'.png')

        # Open images
        # image = np.array(Image.open(img_path))
        tb_img = np.array(Image.open(tb_img_name))
        if tb_img.shape.__len__()==2:
            tb_img = cv2.cvtColor(tb_img, cv2.COLOR_GRAY2RGB)


        canny_crop_img = np.array(Image.open(canny_crop_img_name))

        augmentations = config.both_transform(image=canny_crop_img, image0=tb_img)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]


        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]


        return input_image, target_image

class TB_shenzhen_Dataset_anno_canny(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tb_anno_dir = os.path.join(root_dir, "lesion_with_lung_outline")
        self.canny_crop_dir = os.path.join(root_dir, "lesion_with_edge_lung_outline")
        self.tb_anno_files = os.listdir(self.tb_anno_dir)
        self.canny_crop_files = os.listdir(self.canny_crop_dir)

    def __len__(self):
        return len(self.canny_crop_files)

    def __getitem__(self, idx):

        canny_crop_img_name = os.path.join(self.canny_crop_dir, self.canny_crop_files[idx])  # Assuming filenames match
        tb_anno_img = os.path.join(self.tb_anno_dir, self.canny_crop_files[idx][:4]+'.png')

        # Open images
        # image = np.array(Image.open(img_path))
        tb_anno_img = np.array(Image.open(tb_anno_img))



        canny_crop_img = np.array(Image.open(canny_crop_img_name))

        augmentations = config.both_transform(image0=canny_crop_img, image=tb_anno_img)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]


        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]


        return input_image, target_image

# if __name__ == "__main__":
#     dataset = MapDataset("data/train/")
#     loader = DataLoader(dataset, batch_size=5)
#     for x, y in loader:
#         print(x.shape)
#         save_image(x, "x.png")
#         save_image(y, "y.png")
#         import sys
#
#         sys.exit()
