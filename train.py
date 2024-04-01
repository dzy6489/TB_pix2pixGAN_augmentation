import os
import torch.nn.functional as F
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples,showTB_gen_pic_edge_tb,showTB_gen_pic_anno_tb,showTB_gen_pic_anno_edge
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset,TB_shenzhen_Dataset_canny_tb,TB_shenzhen_Dataset_anno_canny,TB_shenzhen_Dataset_anno_TB
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)  #简笔画
        y = y.to(config.DEVICE)  #真实图片



        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            print("D_loss:")
            print(D_loss.item())

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            print("  G_fake_loss:")
            print(G_fake_loss.item())

            L1 = l1_loss(y_fake , y ) * config.L1_LAMBDA

            # L1 = l1_loss(y_fake*mask_permuted, y*mask_permuted) * config.L1_LAMBDA
            print("   L1_loss:")
            print(L1.item())
            # L2 = l1_loss(y_fake * one_minus_mask, y * one_minus_mask) * config.L2_LAMBDA
            # print("   L2_loss:")
            # print(L2.item())
            G_loss = G_fake_loss + L1 #+ L2

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_PATH+config.load_epoch+"_"+config.edge_tb_or_anno_edge+"_"+config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_PATH+config.load_epoch+"_"+config.edge_tb_or_anno_edge+"_"+config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )
    if config.edge_tb_or_anno_edge == 'edge_tb':
        train_dataset = TB_shenzhen_Dataset_canny_tb(root_dir=config.TRAIN_DIR)
    if config.edge_tb_or_anno_edge == 'anno_edge':
        train_dataset = TB_shenzhen_Dataset_anno_canny(root_dir=config.TRAIN_DIR)
    if config.edge_tb_or_anno_edge == 'anno_tb':
        train_dataset = TB_shenzhen_Dataset_anno_TB(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    if config.edge_tb_or_anno_edge == 'edge_tb':
        val_dataset = TB_shenzhen_Dataset_canny_tb(root_dir=config.VAL_DIR)
    if config.edge_tb_or_anno_edge == 'anno_edge':
        val_dataset = TB_shenzhen_Dataset_anno_canny(root_dir=config.VAL_DIR)
    if config.edge_tb_or_anno_edge == 'anno_tb':
        val_dataset = TB_shenzhen_Dataset_anno_TB(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    if config.LOAD_MODEL==False:
        for epoch in range(config.NUM_EPOCHS):
            print("epoch:"+str(epoch)+"    。")
            train_fn(
                disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
            )

            if config.SAVE_MODEL and epoch % config.every_epoch_save == 0 and epoch >=350 :
            # if config.SAVE_MODEL and epoch % config.every_epoch_save == 0:
                save_checkpoint(gen, opt_gen, filename=str(epoch)+"_"+config.edge_tb_or_anno_edge+"_"+config.CHECKPOINT_GEN)
                save_checkpoint(disc, opt_disc, filename=str(epoch)+"_"+config.edge_tb_or_anno_edge+"_"+config.CHECKPOINT_DISC)
            else:
                if epoch % 2 == 0 and epoch >=485:
                    save_checkpoint(gen, opt_gen, filename=str(epoch)+"_"+config.edge_tb_or_anno_edge+"_"+config.CHECKPOINT_GEN)
                    save_checkpoint(disc, opt_disc, filename=str(epoch)+"_"+config.edge_tb_or_anno_edge+"_"+config.CHECKPOINT_DISC)

            save_some_examples(gen, val_loader, epoch, folder="evaluation")



# ##########################测试VAL目录中所有图片的效果
#     # 指定文件夹路径
#     folder_path = config.VAL_DIR+'lesion_with_edge_lung_outline/'
#
#     # 获取文件夹中所有文件的名称
#     file_names = os.listdir(folder_path)
#
#     # 筛选出png文件并去掉文件扩展名
#     png_files = [file_name[:-4] for file_name in file_names if file_name.endswith('.png')]
#
#     for img in png_files:
#         img_name = img[:4]
#         if config.edge_tb_or_anno_edge == 'edge_tb':
#             showTB_gen_pic_edge_tb(gen, config.VAL_DIR, img_name, './show_results/', config.edge_t1, config.edge_t2,
#                        config.load_epoch)  ###产生edge->tb
#         if config.edge_tb_or_anno_edge == 'anno_edge':
#             showTB_gen_pic_anno_edge(gen, config.VAL_DIR, img_name, './show_results/', config.edge_t1, config.edge_t2,
#                        config.load_epoch)###产生anno->edge
#         if config.edge_tb_or_anno_edge == 'anno_tb':
#             showTB_gen_pic_anno_tb(gen, config.VAL_DIR, img_name, './show_results/', config.edge_t1, config.edge_t2,
#                                      config.load_epoch)  ###产生anno->tb

#####################测试单张图片的效果
    # # ############测试edge->tb
    # showTB_gen_pic(gen,config.VAL_DIR,'0340','./show_results/',config.edge_t1,config.edge_t2,config.load_epoch)
    #
    # ###########测试anno->edge
    # showTB_edge_gen_pic(gen,config.VAL_DIR,'0417','./show_results/',config.edge_t1,config.edge_t2,config.load_epoch)

if __name__ == "__main__":
    main()
