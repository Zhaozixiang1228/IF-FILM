import os
from utils.img_read_save import image_read_cv2
import h5py
import numpy as np
from tqdm import tqdm
import cv2


img_text_path = 'VLFDataset'
h5_path = "VLFDataset_h5"
os.makedirs(h5_path, exist_ok=True)
task_name = 'IVF'
dataset_name = 'MSRS'
dataset_mode = 'train'
size = 'small'
h5_file_path = os.path.join(h5_path, dataset_name + '_' + dataset_mode +'.h5')
small_size_0 = (384, 288)
small_size_1 = (288, 384)

with h5py.File(h5_file_path, 'w') as h5_file:
    # 创建两个数据集组
    imageA = h5_file.create_group('imageA')
    imageB = h5_file.create_group('imageB')
    text = h5_file.create_group('text')


    text_txt = os.path.join(img_text_path, 'Image', task_name, dataset_name, dataset_mode + '.txt')
    with open(text_txt, 'r') as file:
        file_list = [line.strip() for line in file.readlines()]
    sample_names = []
    for name in file_list:
        name = name.split('.')[0]
        sample_names.append(name)
        # 获取所有样本的文件名（不包括文件扩展名）

    if task_name == 'MFF':
        # 遍历每个样本
        for sample_name in tqdm(sample_names):
            if dataset_mode == 'train' and size == 'small':
                img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'MRI', sample_name + '.png'), mode='GRAY')
                h, w = img_A.shape
                if h < w:
                    img_A = cv2.resize(img_A, small_size_0)[None, ...] / 255.0
                    img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'CT', sample_name + '.png'), mode='GRAY'), small_size_0)[None, ...] / 255.0
                else:
                    img_A = cv2.resize(img_A, small_size_1).T[None, ...] / 255.0
                    img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'CT', sample_name + '.png'), mode='GRAY'), small_size_1).T[None, ...] / 255.0
            else:
                img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'NEAR', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
                img_B = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'FAR', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
            text_feature = np.load(os.path.join(img_text_path, 'TextFeature', task_name, dataset_name, sample_name + '.npy'))

            # 将图像和文本保存到HDF5文件中
            imageA.create_dataset(sample_name, data=img_A)
            imageB.create_dataset(sample_name, data=img_B)
            text.create_dataset(sample_name, data=text_feature)
    
    elif task_name == 'MEF':
        for sample_name in tqdm(sample_names):
            if dataset_mode == 'train' and size == 'small':
                img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'OVER', sample_name + '.png'), mode='GRAY')
                h, w = img_A.shape
                if h < w:
                    img_A = cv2.resize(img_A, small_size_0)[None, ...] / 255.0
                    img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'UNDER', sample_name + '.png'), mode='GRAY'), small_size_0)[None, ...] / 255.0
                else:
                    img_A = cv2.resize(img_A, small_size_1).T[None, ...] / 255.0
                    img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'UNDER', sample_name + '.png'), mode='GRAY'), small_size_1).T[None, ...] / 255.0

            else:
                img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'OVER', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
                img_B = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'UNDER', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
            text_feature = np.load(os.path.join(img_text_path, 'TextFeature', task_name, dataset_name, sample_name + '.npy'))

            # 将图像和文本保存到HDF5文件中
            imageA.create_dataset(sample_name, data=img_A)
            imageB.create_dataset(sample_name, data=img_B)
            text.create_dataset(sample_name, data=text_feature)

    elif task_name == 'IVF':
        for sample_name in tqdm(sample_names):
            if dataset_mode == 'train' and size == 'small':
                img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'IR', sample_name + '.png'), mode='GRAY')
                h, w = img_A.shape
                if h < w:
                    img_A = cv2.resize(img_A, small_size_0)[None, ...] / 255.0
                    img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode='GRAY'), small_size_0)[None, ...] / 255.0
                else:
                    img_A = cv2.resize(img_A, small_size_1).T[None, ...] / 255.0
                    img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode='GRAY'), small_size_1).T[None, ...] / 255.0

            else:
                img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'IR', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
                img_B = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
            text_feature = np.load(os.path.join(img_text_path, 'TextFeature', task_name, dataset_name, sample_name + '.npy'))

            # 将图像和文本保存到HDF5文件中
            imageA.create_dataset(sample_name, data=img_A)
            imageB.create_dataset(sample_name, data=img_B)
            text.create_dataset(sample_name, data=text_feature)

    elif task_name == 'MIF':
        for sample_name in tqdm(sample_names):
            img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'MRI', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
            img_B = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'T', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
            text_feature = np.load(os.path.join(img_text_path, 'TextFeature', task_name, dataset_name, sample_name + '.npy'))

            # 将图像和文本保存到HDF5文件中
            imageA.create_dataset(sample_name, data=img_A)
            imageB.create_dataset(sample_name, data=img_B)
            text.create_dataset(sample_name, data=text_feature)

