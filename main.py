'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-08-08 20:55:54
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-02-27 09:26:19
FilePath: /OCT_Seg_Version2/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from setup.data_loader import get_dataloader, set_seed
from trainer import train_model_endo_depth
from model.depthpolyp import build_depthpolyp
from setup.losses import uncertainty_weighted_loss

set_seed()
DEVICE = 'cuda:0'
EPOCHS = 200

train_dataset_loader, val_dataset_loader, dataset_name = get_dataloader(dataset_name='KVASIR-v2', batch_size=batch_size, shuffle=True)
# train_dataloader2, val_dataloader2, dataset_name = get_dataloader(dataset_name='ClinicDB-v2', batch_size=8, shuffle=True)=
# train_dataloader3, val_dataloader3, dataset_name = get_dataloader(dataset_name='ColonDB-v2', batch_size=16, shuffle=True)

stepsize = EPOCHS / 5
learningrate=1e-4
decay_rate = learningrate / EPOCHS 

model = build_depthpolyp(
    encoder_name='b0',
    in_channels=3,
    num_classes=2,
    decoder_channels=256,
    activation=None,
)

def lambda_setup(epoch, warmup_epochs=EPOCHS//10, cosine_epochs=EPOCHS, eta_min=5e-7): 
    if epoch < warmup_epochs:
        return epoch / warmup_epochs  # Warm-up 阶段
    else:
        return 0.5 * (math.cos((epoch - warmup_epochs) / (cosine_epochs - warmup_epochs) * math.pi) + 1) * (1 - eta_min) + eta_min

uncertainty_weighted_loss = uncertainty_weighted_loss()

opt = Adam(
   list(model.parameters()),
   lr=learningrate,
   betas=(0.9, 0.999),
   eps=1e-8,
   weight_decay=decay_rate,
   amsgrad=None
)

opt = torch.optim.Adam([
    {'params': model.parameters(), 'lr': learningrate},
    {'params': uncertainty_weighted_loss.parameters(), 'lr': learningrate}  # 损失函数参数使用更高的学习率
], betas=(0.9, 0.999), eps=1e-8, weight_decay=decay_rate)

scheduler = LambdaLR(opt, lr_lambda=lambda_setup)


opt_name = opt.__class__.__name__
scheduler_name = scheduler.__class__.__name__

model_full_name = f"{dataset_name}_DepthPolpy_{learningrate:.0e}_{opt_name}_{scheduler_name}_woUncertainty"
print(model_full_name)

train_model_endo_depth(model_full_name, model, train_dataset_loader, val_dataset_loader, uncertainty_weighted_loss, opt, scheduler, num_ep, task=TASK, mode=MODE)

