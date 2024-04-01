import torch
import config
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()



def showTB_gen_pic_edge_tb(gen, img_path, img_name, save_path, t1, t2, load_epoch):
    new_save_path = save_path +config.edge_tb_or_anno_edge+ '_edge' + str(t1) + '_' + str(t2) + '_epoch' +str(load_epoch)+'_LAMBDA'+str(config.L1_LAMBDA)+'/'
    os.makedirs(new_save_path, exist_ok=True)

    input_image = np.array(Image.open(img_path+'lesion_with_edge_lung_outline/'+img_name+'_canny_crop.png'))
    target_image = np.array(Image.open(img_path+'TB/'+img_name+'.png'))
    augmentations = config.both_transform(image=input_image, image0=target_image)
    input_image = augmentations["image"]
    target_image = augmentations["image0"]

    input_image = config.transform_only_input(image=input_image)["image"]
    target_image = config.transform_only_mask(image=target_image)["image"]
    input_image, target_image = input_image.to(config.DEVICE), target_image.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(input_image.unsqueeze(0))
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, new_save_path + f"/y_gen_{img_name}.png")
        save_image(input_image * 0.5 + 0.5, new_save_path + f"/input_{img_name}.png")
        save_image(target_image * 0.5 + 0.5, new_save_path + f"/label_{img_name}.png")
    gen.train()


def showTB_gen_pic_anno_tb(gen, img_path, img_name, save_path, t1, t2, load_epoch):
    new_save_path = save_path +config.edge_tb_or_anno_edge+ '_edge' + str(t1) + '_' + str(t2) + '_epoch' +str(load_epoch)+'_LAMBDA'+str(config.L1_LAMBDA)+'/'
    os.makedirs(new_save_path, exist_ok=True)

    input_image = np.array(Image.open(img_path+'lesion_with_lung_outline/'+img_name+'.png'))
    target_image = np.array(Image.open(img_path+'TB/'+img_name+'.png'))
    augmentations = config.both_transform(image=input_image, image0=target_image)
    input_image = augmentations["image"]
    target_image = augmentations["image0"]

    input_image = config.transform_only_input(image=input_image)["image"]
    target_image = config.transform_only_mask(image=target_image)["image"]
    input_image, target_image = input_image.to(config.DEVICE), target_image.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(input_image.unsqueeze(0))
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, new_save_path + f"/y_gen_{img_name}.png")
        save_image(input_image * 0.5 + 0.5, new_save_path + f"/input_{img_name}.png")
        save_image(target_image * 0.5 + 0.5, new_save_path + f"/label_{img_name}.png")
    gen.train()


def showTB_gen_pic_anno_edge(gen, img_path, img_name, save_path, t1, t2, load_epoch):
    new_save_path = save_path +config.edge_tb_or_anno_edge+ 'edge' + str(t1) + '_' + str(t2) + '_epoch' +str(load_epoch)+'_LAMBDA'+str(config.L1_LAMBDA)+'/'
    os.makedirs(new_save_path, exist_ok=True)

    input_image = np.array(Image.open(img_path+'lesion_with_lung_outline/'+img_name+'.png'))
    target_image = np.array(Image.open(img_path+'lesion_with_edge_lung_outline/'+img_name+'_canny_crop.png'))
    augmentations = config.both_transform(image=input_image, image0=target_image)
    input_image = augmentations["image"]
    target_image = augmentations["image0"]

    input_image = config.transform_only_input(image=input_image)["image"]
    target_image = config.transform_only_mask(image=target_image)["image"]
    input_image, target_image = input_image.to(config.DEVICE), target_image.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(input_image.unsqueeze(0))
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, new_save_path + f"/y_gen_{img_name}.png")
        save_image(input_image * 0.5 + 0.5, new_save_path + f"/input_{img_name}.png")
        save_image(target_image * 0.5 + 0.5, new_save_path + f"/label_{img_name}.png")
    gen.train()



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    torch.save(checkpoint, config.CHECKPOINT_PATH+filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


