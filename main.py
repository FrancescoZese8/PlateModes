
# import sys
import os
from torch.utils.data import DataLoader
import Data_Set
import torch
import numpy as np
import KirchoffLoss
import modules
import training
from diff_operators import gradient

'''sys.path.insert(1, '/Users/user/PycharmProjects/ReLoBraLoSirenP/siren')'''


# from meta_modules import NeuralProcessImplicit2DHypernet

# p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
# config_filepath = ??
logging_root = './logs'
experiment_name = 'exp_helm_test'

# General training options
batch_size = 64
lr = 2e-5
num_epochs = 2000
epochs_til_ckpt = 500
steps_til_summary = 100
opt_model = 'sine'
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
checkpoint_path = None
val = 'y'
print('new_case')
W = 10
H = 10
T = 0.2
E = 30000
nue = 0.2
p0 = 0.15
N = 10
nb = 5
D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate
total_length = 1

# dataset = Data_Set.KirchhoffPDE(p, u_val, T, nu, E, H, W)
# dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)
def load(x, y):
    return p0 * torch.sin(x * np.pi / W) * torch.sin(y * np.pi / H)

def u_val(x, y):
    return p0 / (np.pi ** 4 * D * (W ** -2 + H ** -2) ** 2) * torch.sin(x * np.pi / W) * torch.sin(y * np.pi / H)

plate = Data_Set.KirchhoffDataset(p=load, u_val=u_val, T=T, nue=nue, E=E, W=W, H=H, total_length=total_length)
# plate.visualise()
''' Creazione del DataLoader'''
data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)
model = modules.PINNet(out_features=1, type='tanh', mode=mode)

# Define the loss
#loss_fn = KirchoffLoss.KirchhoffLoss(plate)
loss_fn = KirchoffLoss.ReLoBRaLoKirchhoffLoss(plate)
summary_fn = None  # utils.write_helmholtz_summary

root_path = os.path.join(logging_root, experiment_name)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, lr=lr,
               steps_til_summary=steps_til_summary, epochs_til_checkpoint=epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=clip_grad,
               use_lbfgs=False)
plate.visualise(model)
