import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from diff_operators import gradient
import torch.nn.functional as F
from torch.autograd import detect_anomaly

EPS = 1e-7

def compute_derivatives(coords, u):
  with detect_anomaly():
    du = gradient(u,coords)
   # du = F.normalize(du, p=float('inf'), dim=1)
    dudx = du[:,:,0]
    dudy = du[:,:,1]

    ddux = gradient(dudx,coords)
    # ddux = F.normalize(ddux, p=float('inf'), dim=1)
    dudxx = ddux[:,:,0]

    dduy = gradient(dudy,coords)
    # dduy = F.normalize(dduy, p=float('inf'), dim=1)

    dudyy = dduy[:,:,1]


    dddux = gradient(dudxx,coords)
    # dddux = F.normalize(dddux, p=float('inf'), dim=1)
    dudxxx = dddux[:, :, 0]
    dudxxy = dddux[:, :, 1]

    ddduy = gradient(dudyy,coords)
    # ddduy = F.normalize(ddduy, p=float('inf'), dim=1)
    dudyyy = ddduy[:, :, 0]
    dudyyx = ddduy[:, :, 1]

    #ora fai l'ultimo pezzo

    ddddux = gradient(dudxxx,coords)
    #ddddux = F.normalize(ddddux, p=float('inf'), dim=1)
    #ddddux = F.normalize(ddddux, p=float('inf'), dim=0)
    #ddddux = F.normalize(ddddux, p=float('inf'), dim=-1)
    # print('\n la la la \n ddddux:', ddddux)
    dudxxxx = ddddux[:,:,1]
    dddduy = gradient(dudyyy,coords)
    # print('dddduuy:', dddduy)
    #dddduy = F.normalize(ddduy, p=float('inf'), dim=1)
    dudyyyy = dddduy[:,:,1]

    dduddu = gradient(dudxxy,coords)
    # print('dduddu:', dduddu)
    # dduddu = F.normalize(dduddu, p=float('inf'), dim=1)
    dudxxyy = dduddu[:,:,1]
    # print('ao jaa')
    # print('mins:',torch.min(dudxx), torch.min(dudyy), torch.min(dudxxxx), torch.min(dudyyyy), torch.min(dudxxyy))
    return dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy

'''
    print('shapes:', dudxx.shape, dudyx.shape, dudxy.shape, dudyx.shape)

    print('ddu:', ddu)
    print('ddu.shape:', ddu.shape)
#    ddu = gradient(du,coords)

#    print('ddu.shape:', ddu.shape)

    #ddu = gradient(dudx, c)
    #print(ddu.shape)
    #dudxx, dudyy = ddu[:,:,:], ddu[:,:,:]

    dudxxx = gradient(dudxx, x)
    dudxxy = gradient(dudxx, y)
    dudyyy = gradient(dudyy, y)

    dudxxxx = gradient(dudxxx, x)
    dudxxyy = gradient(dudxxy, y)
    dudyyyy = gradient(dudyyy, y)

    return dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy '''


def compute_moments(D, nue, dudxx, dudyy):
    mx = -D * (dudxx + nue * dudyy)
    my = -D * (nue * dudxx + dudyy)
    return mx, my


class KirchhoffDataset(Dataset):

    def __init__(self, p, u_val, T, nue, E, H, W, total_length):
        self.p = p
        self.u_val = u_val
        self.T = T
        self.nue = nue
        self.E = E
        self.D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffness of the plate
        self.H = H
        self.W = W
        self.num_terms = 3
        self.total_length = total_length

    def __getitem__(self, item):
        x, y = self.training_batch()
        l = x.shape[-1]
        xy = torch.cat([x,y],dim=-1) # torch.cat(self.training_batch(), dim=-1)
        return {'coords': xy, 'l': l}, {'gt': xy}
        #return {'x': x, 'y': y}, {'gt': xy}

    def __len__(self):
        return self.total_length

    def training_batch(self, batch_size_domain: int = 800, batch_size_boundary: int = 100) -> Tuple[
        torch.Tensor, torch.Tensor]:

        x_in = torch.rand((batch_size_domain, 1)) * self.W
        x_b1 = torch.zeros((batch_size_boundary, 1))
        x_b2 = torch.zeros((batch_size_boundary, 1)) + self.W
        x_b3 = torch.rand((batch_size_boundary, 1)) * self.W
        x_b4 = torch.rand((batch_size_boundary, 1)) * self.W
        x = torch.cat([x_in, x_b1, x_b2, x_b3, x_b4], dim=0)

        y_in = torch.rand((batch_size_domain, 1)) * self.H
        y_b1 = torch.rand((batch_size_boundary, 1)) * self.H
        y_b2 = torch.rand((batch_size_boundary, 1)) * self.H
        y_b3 = torch.zeros((batch_size_boundary, 1))
        y_b4 = torch.zeros((batch_size_boundary, 1)) + self.H
        y = torch.cat([y_in, y_b1, y_b2, y_b3, y_b4], dim=0)

        return x, y

    def validation_batch(self, grid_width=64, grid_height=64):

        x, y = np.mgrid[0:self.W:complex(0, grid_width), 0:self.H:complex(0, grid_height)]
        x = torch.tensor(x.reshape(grid_width * grid_height, 1), dtype=torch.float32)
        y = torch.tensor(y.reshape(grid_width * grid_height, 1), dtype=torch.float32)

        x = x[None, ...]
        y = y[None, ...]

        # Assuming self.u_val is a PyTorch function
        u = self.u_val(x, y)

        return x, y, u

    def compute_loss(self, x, y, preds, eval=False):

        c = torch.cat([x,y],dim=1)
        x, y = c[:,:,0], c[:,:,1]
        #print(preds[:,:,3:4], preds[:,:,4:5], preds[:,:,5:6])
        # governing equation loss
        f = preds[:,:,3:4] + 2 * preds[:,:,4:5] + preds[:,:,5:6] - self.p(x, y) / self.D
        L_f = f ** 2

        # determine which points are on the boundaries of the domain
        # if a point is on either of the boundaries, its value is 1 and 0 otherwise
        x_lower = torch.where(x <= EPS, torch.tensor(1.0), torch.tensor(0.0))
        x_upper = torch.where(x >= self.W - EPS, torch.tensor(1.0), torch.tensor(0.0))
        y_lower = torch.where(y <= EPS, torch.tensor(1.0), torch.tensor(0.0))
        y_upper = torch.where(y >= self.H - EPS, torch.tensor(1.0), torch.tensor(0.0))

        # compute 0th order boundary condition loss
        L_b0 = ((x_lower + x_upper + y_lower + y_upper) * preds[:, :, :1]) ** 2
        # compute 2nd order boundary condition loss
        mx, my = compute_moments(self.D, self.nue, preds[:, :, 1:2], preds[:, :, 2:3])
        L_b2 = ((x_lower + x_upper) * mx) ** 2 + ((y_lower + y_upper) * my) ** 2

        if eval:
            L_u = (self.u_val(x, y) - preds[:, :, 0:1]) ** 2
            return L_f, L_b0, L_b2, L_u

        return L_f, L_b0, L_b2

    def __validation_results(self, pinn, image_width=64, image_height=64):

        x, y, u_real = self.validation_batch(image_width, image_height)
        print('ao:',x.shape, y.shape, u_real.shape)
        c = {'coords': torch.cat([x, y], dim=-1).float()}
        pred = pinn(c)['model_out']
        u_pred, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = (
            pred[:,:,0:1], pred[:,:,1:2], pred[:,:,2:3], pred[:,:,3:4], pred[:,:,4:5], pred[:,:,5:6]
        )
        print(pred[:,:,2:3], pred[:,:,3:4], pred[:,:,4:5], pred[:,:,5:6])
        mx, my = compute_moments(self.D, self.nue, dudxx, dudyy)
        f = dudxxxx + 2 * dudxxyy + dudyyyy
        p = self.p(x, y)
        return u_real, u_pred, mx, my, f, p

    def visualise(self, pinn=None, image_width=64, image_height=64):

        if pinn is None:
            x, y, u_real = self.validation_batch(image_width, image_height)
            load = self.p(x, y).numpy().reshape(image_width, image_height)
            fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), dpi=100)
            self.__show_image(
                load,
                axis=axs[0],
                title='Load distribution on the plate',
                z_label='$\\left[\\frac{NM}{m^2}\\right]$'
            )
            self.__show_image(
                u_real.numpy().reshape(image_width, image_height),
                axis=axs[1],
                title='Deformation',
                z_label='[m]'
            )
            plt.tight_layout()
            plt.show()

        else:
            u_real, u_pred, mx, my, f, p = self.__validation_results(pinn, image_width, image_height)
            u_real = u_real.detach().numpy().reshape(image_width, image_height)
            u_pred = u_pred.detach().numpy().reshape(image_width, image_height)
            mx = mx.detach().numpy().reshape(image_width, image_height)
            my = my.detach().numpy().reshape(image_width, image_height)
            f = f.detach().numpy().reshape(image_width, image_height)
            p = p.detach().numpy().reshape(image_width, image_height)

            fig, axs = plt.subplots(3, 2, figsize=(9.5, 12))
            self.__show_image(u_pred, axs[0, 0], 'Predicted Displacement (m)')
            self.__show_image((u_pred - u_real) ** 2, axs[0, 1], 'Squared Error Displacement')
            self.__show_image(mx, axs[1, 0], 'Moments mx')
            self.__show_image(my, axs[1, 1], 'Moments my')
            self.__show_image(f, axs[2, 0], 'Governing Equation')
            self.__show_image((f - p) ** 2, axs[2, 1], 'Squared Error Governing Equation')

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()

            plt.tight_layout()
            plt.show()

    def __show_image(self, img, axis=None, title='', x_label='x [m]', y_label='y [m]', z_label=''):
        if axis is None:
            _, axis = plt.subplots(1, 1, figsize=(4, 3.2), dpi=100)
        im = axis.imshow(np.rot90(img, k=3), cmap='plasma', origin='lower', aspect='auto')
        cb = plt.colorbar(im, label=z_label, ax=axis)
        axis.set_xticks([0, img.shape[0] - 1])
        axis.set_xticklabels([0, self.W])
        axis.set_yticks([0, img.shape[1] - 1])
        axis.set_yticklabels([0, self.H])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        return im
