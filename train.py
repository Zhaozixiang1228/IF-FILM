# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
import os
import matplotlib.pyplot as plt
import sys
import time
import datetime
import torch
from utils.Logger import Logger1
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.lossfun import Fusionloss, LpLssimLossweight
import numpy as np
from utils.H5_read import H5ImageTextDataset
import argparse
import warnings
from net.Film import Net
import logging
import shutil
import re
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.CRITICAL)

parser = argparse.ArgumentParser()


parser.add_argument('--i2t_dim', type=int, default=32, help='')
parser.add_argument('--hidden_dim', type=int, default=256, help='')
parser.add_argument('--numepochs', type=int, default=150, help='')
parser.add_argument('--lr', type=float, default=1e-5, help='')
parser.add_argument('--gamma', type=float, default=0.6, help='')
parser.add_argument('--step_size', type=int, default=50, help='')
parser.add_argument('--batch_size', type=int, default=2, help='')
parser.add_argument('--loss_grad_weight', type=int, default=20, help='')
parser.add_argument('--loss_ssim', type=int, default=0, help='')
parser.add_argument('--dataset_path', type=str, default="VLFDataset_h5\MSRS_train.h5", help='')
opt = parser.parse_args()

'''
------------------------------------------------------------------------------
Set the hyper-parameters for training
------------------------------------------------------------------------------
'''
pre_model = ""
num_epochs = opt.numepochs
lr = opt.lr
step_size = opt.step_size
gamma = opt.gamma
weight_decay = 0
batch_size = opt.batch_size
weight_ingrad = opt.loss_grad_weight
weight_ssim = opt.loss_ssim
hidden_dim = opt.hidden_dim
i2t_dim = opt.i2t_dim
dataset_path = opt.dataset_path
exp_name = ''

'''
------------------------------------------------------------------------------
model
------------------------------------------------------------------------------
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.nn.DataParallel(
    Net(hidden_dim=hidden_dim, image2text_dim=i2t_dim))
if pre_model != "":
    model.load_state_dict(torch.load(pre_model)['model'])
    print('load_pretrain_model')
model.to(device)
criterion = LpLssimLossweight().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

trainloader = DataLoader(H5ImageTextDataset(dataset_path), batch_size=batch_size,
                         shuffle=True, num_workers=0, drop_last=True)
time_begin = time.strftime("%y_%m_%d_%H_%M", time.localtime())
save_path = "exp/" + str(time_begin) + '_epochs_%s' % (
    str(opt.numepochs)) + '_lr_%s' % (str(opt.lr)) + '_stepsize_%s' % (str(opt.step_size)) + '_bs_%s' % (
                           opt.batch_size) + '_gradweight_%s' % (str(opt.loss_grad_weight)) + '_gamma_%s' % (
                           str(opt.gamma)) + exp_name
logger = Logger1(rootpath=save_path, timestamp=True)
params = {
    'epoch': num_epochs,
    'lr': lr,
    'batch_size': batch_size,
    'optim_step': step_size,
    'optim_gamma': gamma,
    'gradweight': weight_ingrad,
}
logger.save_param(params)
logger.new_subfolder('model')
writer = SummaryWriter(logger.logpath)
exp_folder = logger.get_timestamp_folder_name()
destination_folder = os.path.join(save_path, exp_folder, 'code')


def save_code_files(source_file, destination_folder):
    global model_file_path
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(source_file, 'r', encoding="utf-8") as file:
        content = file.read()
    match = re.search(r'from net\.(\w+) import Net', content)
    if match:
        model_name = match.group(1)
        model_file_path = os.path.join('net', f'{model_name}.py')
    dest_train_file_path = os.path.join(destination_folder, os.path.basename(__file__))

    shutil.copyfile(source_file, dest_train_file_path)
    shutil.copyfile(model_file_path, os.path.join(destination_folder, f'{model_name}.py'))

save_code_files(os.path.basename(__file__), destination_folder)
'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
start_time = time.time()
loss = Fusionloss(coeff_grad=weight_ingrad, device=device)

for epoch in range(num_epochs):
    ''' train '''
    s_temp = time.time()
    model.train()
    for i, (data_IR, data_VIS, text, index) in enumerate(trainloader):
        data_VIS, data_IR, text = data_VIS.to(device), data_IR.to(device), text.to(device)
        text = text.squeeze(1).to(device)
        F = model(data_IR, data_VIS, text)
        batchsize, channels, rows, columns = data_IR.shape
        weighttemp = int(np.sqrt(rows * columns))
        lplssimA, lpA, lssimA = criterion(image_in=data_IR, image_out=F, weight=weighttemp)
        lplssimB, lpB, lssimB = criterion(image_in=data_VIS, image_out=F, weight=weighttemp)
        loss_in_grad, _, _ = loss(data_IR, data_VIS, F)
        loss_ssim = lplssimA + lplssimB
        lossALL = loss_in_grad + weight_ssim * loss_ssim
        optimizer.zero_grad()
        lossALL.backward()
        optimizer.step()


        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        logger.log_and_print(
            "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [loss_in_grad: %f] [lplssimA: %f] [lplssimB: %f] ETA: %.10s"
            % (
                epoch + 1,
                num_epochs,
                i,
                len(trainloader),
                lossALL.item(),
                loss_in_grad.item(),
                lplssimA.item(),
                lplssimB.item(),
                time_left,
            )
        )


        writer.add_scalar('loss/01 Loss', lossALL.item(), step)
        writer.add_scalar('loss/01 loss_in_grad', loss_in_grad.item(), step)
        writer.add_scalar('loss/01 lplssimA', lplssimA.item(), step)
        writer.add_scalar('loss/01 lplssimB', lplssimB.item(), step)
        writer.add_scalar('loss/14 learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step += 1


        if (epoch + 1) % 1 == 0:
            if i <= 1:
                for j in range(data_IR.shape[0]):

                    temp = np.zeros((rows, 3 * columns))

                    temp[:rows, 0:columns] = np.squeeze(data_IR[j].detach().cpu().numpy()) * 255
                    temp[:rows, columns:columns * 2] = np.squeeze(data_VIS[j].detach().cpu().numpy()) * 255
                    temp[:rows, columns * 2:columns * 3] = np.squeeze(F[j].detach().cpu().numpy()) * 255

                    if not os.path.exists(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1))):
                        os.makedirs(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1)))
                    plt.imsave(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1),
                                            str(index[j]) + '.png'),
                               temp,
                               cmap="gray")

    scheduler.step()

    if (epoch + 1) % 1 == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer1': optimizer.state_dict(),
            'lr_schedule1': scheduler.state_dict(),
            "epoch": epoch,
            'step': step,
        }
        os.path.join(logger.logpath, 'model')
        torch.save(checkpoint, os.path.join(logger.logpath, 'model', 'ckpt_%s.pth' % (str(epoch + 1))))
    e_temp = time.time()
    print("This Epoch takes time: " + str(e_temp - s_temp))

end_time = time.time()
logger.log_and_print("total_time: " + str(end_time - start_time))
