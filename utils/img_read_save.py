import numpy as np
import cv2
import os
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  # 读出来不完全是整数，若需要整数则要round
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath):
    # assert image.dtype==np.float32,'输入数据格式不对,应为float32'
    #assert np.max(image)> 1,'输入数据范围不对,应为0-255'

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)), image)

def text_read(path):
    file = open(path, 'r', encoding='UTF-8')  # 创建的这个文件，也是一个可迭代对象
    try:
        text = file.read()  # 结果为str类型
    finally:
        file.close()
    return text