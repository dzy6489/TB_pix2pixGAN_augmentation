（1）数据集预先准备工作：
    准备好shenzhen数据集，找到其中的TB数据，保存在.dataset/TB/TB目录下，https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/CXR_png/index.html
    准备好shenzhen的lung mask数据，保存在.dataset/TB/lung_mask目录下，https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen
     运行download_masks.py下载lesion的mask图片,保存在anno1_mask_pngs，https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/Annotations/masks/index.html
     运行generate_lesion_images.py产生lesion图片
     运行lung_outline.py产生肺部轮廓图
     运行lesion_with_lung_outline.py，将lesion图和肺部轮廓图相加，保存到lesion_with_lung_outline文件夹

 （2）运行generate_all_dataset.py产生数据集，保存在datasets/TB/canny_crop_datasets/中，
 分别有canny文件夹（tb图片产生的edge图，包括肺部以内和肺部以外的edge），
 canny_crop文件夹（肺部区域以内的lesion和edge图）
 lung_anno文件夹（肺部以内的lesion图）
 TB文件夹（原始TB图，都含有TB）。
 调节config.train_test_ratio可以控制train和test的比例


（3）config中的LOAD_MODEL在训练的时候为False，test的时候为True

 (4) config中的 edge_tb_or_anno_edge = 'edge_tb'  #从边缘图到tb图
               edge_tb_or_anno_edge = 'anno_edge'  #从病灶部位图到边缘骨骼图

 (5) config.L1_LAMBDA = 10   #L1的系数   可调节

（6）train.py 中val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        shuffle=True测试同样图片，false测试随机图片，在evaluation中显示
