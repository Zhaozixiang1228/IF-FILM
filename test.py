import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from torch.utils.data import DataLoader
import argparse
from net.Film import Net
warnings.filterwarnings('ignore')  # 不显示warnings
import time
from tqdm import tqdm

task_name = 'IVF'
dataset_name = 'MSRS'

from utils.H5_read import H5ImageTextDataset
testloader = DataLoader(H5ImageTextDataset(os.path.join('VLFDataset_h5', dataset_name+'_test.h5')), batch_size=1, shuffle=True,
                        num_workers=0)


ckpt_path = os.path.join("models", task_name+'.pth')
save_path = os.path.join("test_output", dataset_name, "Gray")
os.makedirs(save_path, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(hidden_dim=256, image2text_dim=32).to(device)
model = nn.DataParallel(model)


pbar = tqdm(total=len(testloader))

model.load_state_dict(torch.load(ckpt_path)['model'])
model.eval()

with torch.no_grad():
    for i, (data_IR, data_VIS, text, index) in tqdm(enumerate(testloader)):
        text = text.squeeze(1).cuda()
        data_IR = torch.FloatTensor(data_IR)
        data_VIS = torch.FloatTensor(data_VIS)
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        data_Fuse = model(data_IR, data_VIS, text)[0]
        data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
        fi = np.squeeze((data_Fuse * 255).detach().cpu().numpy())
        fi = fi.astype('uint8')
        img_save(fi, index[0], save_path)
