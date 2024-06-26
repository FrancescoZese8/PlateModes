# import sys
import os
from torch.utils.data import DataLoader
import Data_Set
import torch
import numpy as np
import KirchoffLoss
import modules
import training
import matplotlib.pyplot as plt

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # con questo il codice utilizza solo la cpu (non vede gpu)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

# from torchviz import make_dot
import pdb
# 1%/2-2:
#NMSE: 0.09404823830498811   loss: 0.00000090915
#NMSE: 0.09810258528924976   loss: 0.00000099067


# from IPython.display import Image
'''sys.path.insert(1, '/Users/user/PycharmProjects/ReLoBraLoSirenP/siren')'''

logging_root = './logs'
experiment_name = 'exp_helm_test'

# General training options
batch_size = 32
lr = 0.001
num_epochs = 400
epochs_til_ckpt = 50
steps_til_summary = 10
opt_model = 'silu'
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
checkpoint_path = None
val = 'y'
relo = True
W = 10
H = 10
T = 0.2
E = 30000
nue = 0.2
p0 = 0.15
den = 1000
n = 2
m = 2
batch_size_domain = 800
batch_size_boundary = 100
percentage_of_known_points = 1  # %

nkp = percentage_of_known_points * batch_size_domain // 100
known_points_x = torch.rand((nkp, 1)) * W
known_points_y = torch.rand((nkp, 1)) * H
print('kpx: ', known_points_x)
print('kpy: ', known_points_y)

D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate
omega = ((n * np.pi / W) ** 2 + (m * np.pi / H) ** 2) * np.sqrt(D / (den * T))
print('omega:', omega)

N = 10
nb = 5

total_length = 1
n_step = 50

max_epochs_without_improvement = 100
current_epochs_without_improvement = 0
best_loss = float('inf')


def load(x, y):
    return p0 * torch.sin(x * np.pi / W) * torch.sin(y * np.pi / H)


# def u_val(x, y):
# return p0 / (np.pi ** 4 * D * (W ** -2 + H ** -2) ** 2) * torch.sin(x * np.pi / W) * torch.sin(y * np.pi / H)
def u_val(x, y):
    return (torch.sin(n * np.pi * x / W) * torch.sin(m * np.pi * y / H)) / 100


plate = Data_Set.KirchhoffDataset(p=load, u_val=u_val, T=T, nue=nue, E=E, W=W, H=H, total_length=total_length, den=den,
                                  omega=omega,batch_size_domain=batch_size_domain, batch_size_boundary=
                                  batch_size_boundary, known_points_x=known_points_x, known_points_y=known_points_y, nkp=nkp)
# plate.visualise()
data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)
model = modules.PINNet(out_features=1, type=opt_model, mode=mode)

# model = model.half()
# model = model.to(device)
# print("Model is currently on device:", next(model.parameters()).device)
summary_fn = None
history_loss = {'L_f': [], 'L_b0': [], 'L_b2': [], 'L_u': [], 'L_t': []}
if not relo:
    loss_fn = KirchoffLoss.KirchhoffLoss(plate)
    kirchhoff_metric = KirchoffLoss.KirchhoffMetric(plate)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = KirchoffLoss.ReLoBRaLoKirchhoffLoss(plate, temperature=0.1, rho=0.99, alpha=0.999)
    kirchhoff_metric = KirchoffLoss.KirchhoffMetric(plate)
    history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': [], 'L_t_lambda': []}
    metric_lam = KirchoffLoss.ReLoBRaLoLambdaMetric(loss_fn)

''' 
#history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': []}
history_lambda = None
#pdb.set_trace() #serve per fermare il codice
# Define the loss
loss_fn = KirchoffLoss.KirchhoffLoss(plate)
#loss_fn = KirchoffLoss.ReLoBRaLoKirchhoffLoss(plate, temperature=0.1, rho=0.99, alpha=0.999)
summary_fn = None  # utils.write_helmholtz_summary

kirchhoff_metric = KirchoffLoss.KirchhoffMetric(plate)
#metric_lam = KirchoffLoss.ReLoBRaLoLambdaMetric(loss_fn)
metric_lam = None
'''

# pdb.set_trace() #serve per fermare il codice
root_path = os.path.join(logging_root, experiment_name)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step, lr=lr,
               steps_til_summary=steps_til_summary, epochs_til_checkpoint=epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, history_loss=history_loss, history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, summary_fn=summary_fn, clip_grad=clip_grad,
               use_lbfgs=False, max_epochs_without_improvement=max_epochs_without_improvement,
               current_epochs_without_improvement=current_epochs_without_improvement, best_loss=best_loss)
model.eval()
plate.visualise(model)

fig = plt.figure(figsize=(6, 4.5), dpi=100)
plt.plot(torch.log(torch.tensor(history_loss['L_f'])), label='$L_f$ governing equation')
plt.plot(torch.log(torch.tensor(history_loss['L_b0'])), label='$L_{b0}$ Dirichlet boundaries')
plt.plot(torch.log(torch.tensor(history_loss['L_b2'])), label='$L_{b2}$ Moment boundaries')
plt.plot(torch.log(torch.tensor(history_loss['L_t'])), label='$L_t$ Known points')
plt.plot(torch.log(torch.tensor(history_loss['L_u'])), label='$L_u$ analytical solution')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log-loss')
plt.title('Loss evolution Kirchhoff PDE')
plt.savefig('kirchhoff_loss_unscaled')
plt.show()

if metric_lam is not None:
    fig2 = plt.figure(figsize=(6, 4.5), dpi=100)
    plt.plot(history_lambda['L_f_lambda'], label='$\lambda_f$ governing equation')
    plt.plot(history_lambda['L_b0_lambda'], label='$\lambda_{b0}$ Dirichlet boundaries')
    plt.plot(history_lambda['L_b2_lambda'], label='$\lambda_{b2}$ Moment boundaries')
    plt.plot(history_lambda['L_t_lambda'], label='$\lambda_{t}$ Known points')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('scalings lambda')  # $\lambda$')
    plt.title('ReLoBRaLo weights on Kirchhoff PDE')
    plt.savefig('kirchhoff_lambdas_relobralo')
    plt.show()

'''
print('\n losses:')
print(history_loss['L_f'], end='\n')
print(history_loss['L_b0'], end='\n')
print(history_loss['L_b2'], end='\n')
print(history_loss['L_u'], end='\n')
print('\n ')
print(history_lambda['L_f_lambda'], end='\n')
print(history_lambda['L_b0_lambda'], end='\n')
print(history_lambda['L_b2_lambda'], end='\n')
'''

'''
sample_batch = next(iter(data_loader))
#batch_text = torch.cat([batch[0]['coords']], dim=0)
i = 0
for sample_dict in sample_batch:
    #for key, value in sample_dict.items():
    i+=1
    if i==1:
        m_i = {key: value for key, value in sample_dict.items()}


yhat = model(m_i)
yhat = yhat['model_out']
# Usa make_dot per visualizzare il grafo computazionale
make_dot(yhat, params=dict(list(model.named_parameters()))).render("/nas/home/rsebastiani/rnn_torchviz.png", format="png")

Image("rnn_torchviz.png")

print('dot_img')'''
