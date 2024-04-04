import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from diff_operators import gradient
import torch.nn.functional as F
from torch.autograd import detect_anomaly

EPS = 1e-6


def compute_derivatives(x, y, u):
    dudx = gradient(u, x)
    dudy = gradient(u, y)

    dudxx = gradient(dudx, x)
    dudyy = gradient(dudy, y)

    dudxxx = gradient(dudxx, x)
    dudxxy = gradient(dudxx, y)
    dudyyy = gradient(dudy, y)

    dudxxxx = gradient(dudxxx, x)
    dudxxyy = gradient(dudxxy, y)
    dudyyyy = gradient(dudyyy, y)

    return dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy


def compute_moments(D, nue, dudxx, dudyy):
    mx = -D * (dudxx + nue * dudyy)
    my = -D * (nue * dudxx + dudyy)

    return mx, my


def min_max_normalization(data, min_val, max_val):
    return 2 * (data - min_val) / (max_val - min_val) - 1


def inv(data):
    return (data + 1) * 10 / 2


class KirchhoffDataset(Dataset):

    def __init__(self, p, u_val, T, nue, E, H, W, total_length, den: float, omega: float,batch_size_domain,
                 batch_size_boundary, known_points_x, known_points_y, nkp):
        self.p = p
        self.u_val = u_val
        self.T = T
        self.nue = nue
        self.E = E
        self.D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffness of the plate
        self.H = H
        self.W = W
        self.num_terms = 4
        self.total_length = total_length
        self.den = den
        self.omega = omega
        self.batch_size_domain = batch_size_domain
        self.batch_size_boundary = batch_size_boundary
        self.known_points_x = known_points_x
        self.known_points_y = known_points_y
        self.nkp = nkp

    def __getitem__(self, item):
        x, y = self.training_batch()
        x.requires_grad_(True)
        y.requires_grad_(True)
        xy = torch.cat([x, y], dim=-1)
        return {'coords': xy}, {'gt': xy}

    def __len__(self):
        return self.total_length

    def training_batch(self) -> Tuple[
        torch.Tensor, torch.Tensor]:

        x_t = self.known_points_x
        x_in = torch.rand((self.batch_size_domain, 1)) * self.W
        x_b1 = torch.zeros((self.batch_size_boundary, 1))
        x_b2 = torch.zeros((self.batch_size_boundary, 1)) + self.W
        x_b3 = torch.rand((self.batch_size_boundary, 1)) * self.W
        x_b4 = torch.rand((self.batch_size_boundary, 1)) * self.W

        x = torch.cat([x_t, x_in, x_b1, x_b2, x_b3, x_b4], dim=0)  # .to(self.device)

        y_t = self.known_points_y
        y_in = torch.rand((self.batch_size_domain, 1)) * self.H
        y_b1 = torch.rand((self.batch_size_boundary, 1)) * self.H
        y_b2 = torch.rand((self.batch_size_boundary, 1)) * self.H
        y_b3 = torch.zeros((self.batch_size_boundary, 1))
        y_b4 = torch.zeros((self.batch_size_boundary, 1)) + self.H
        y = torch.cat([y_t, y_in, y_b1, y_b2, y_b3, y_b4], dim=0)  # .to(self.device)

        return x, y

    def validation_batch(self, grid_width=64, grid_height=64):
        x, y = np.mgrid[0:self.W:complex(0, grid_width), 0:self.H:complex(0, grid_height)]
        x = torch.tensor(x.reshape(grid_width * grid_height, 1), dtype=torch.float32)  # .to(self.device)
        y = torch.tensor(y.reshape(grid_width * grid_height, 1), dtype=torch.float32)  # .to(self.device)

        x = x[None, ...]
        y = y[None, ...]

        # Assuming self.u_val is a PyTorch function
        u = self.u_val(x, y)

        return x, y, u

    def compute_loss(self, x, y, preds, eval=False):
        # governing equation loss
        preds = np.squeeze(preds, axis=0)
        x = np.squeeze(x, axis=0)
        y = np.squeeze(y, axis=0)
        #x = x[self.nkp:]
        #y = y[self.nkp:]
        x_t = x[:self.nkp]
        y_t = y[:self.nkp]
        u_t = np.squeeze(preds[:self.nkp, 0:1])
        #u = np.squeeze(preds[self.nkp:, 0:1])
        u = np.squeeze(preds[:, 0:1])
        #dudxx = np.squeeze(preds[self.nkp:, 1:2])
        #dudyy = np.squeeze(preds[self.nkp:, 2:3])
        #dudxxxx = np.squeeze(preds[self.nkp:, 3:4])
        #dudyyyy = np.squeeze(preds[self.nkp:, 4:5])
        #dudxxyy = np.squeeze(preds[self.nkp:, 5:6])
        dudxx = np.squeeze(preds[:, 1:2])
        dudyy = np.squeeze(preds[:, 2:3])
        dudxxxx = np.squeeze(preds[:, 3:4])
        dudyyyy = np.squeeze(preds[:, 4:5])
        dudxxyy = np.squeeze(preds[:, 5:6])
        err_t = self.u_val(x_t, y_t) - u_t
        f = dudxxxx + 2 * dudxxyy + dudyyyy - (
                    self.den * self.T * (self.omega ** 2)) / self.D * u

        L_f = f ** 2
        L_t = err_t ** 2

        # determine which points are on the boundaries of the domain
        # if a point is on either of the boundaries, its value is 1 and 0 otherwise
        x_lower = torch.where(x <= EPS, torch.tensor(1.0), torch.tensor(0.0))
        x_upper = torch.where(x >= self.W - EPS, torch.tensor(1.0), torch.tensor(0.0))
        y_lower = torch.where(y <= EPS, torch.tensor(1.0), torch.tensor(0.0))
        y_upper = torch.where(y >= self.H - EPS, torch.tensor(1.0), torch.tensor(0.0))

        L_b0 = torch.mul((x_lower + x_upper + y_lower + y_upper), u) ** 2

        # compute 2nd order boundary condition loss
        mx, my = compute_moments(self.D, self.nue, dudxx, dudyy)
        L_b2 = torch.mul((x_lower + x_upper), mx) ** 2 + torch.mul((y_lower + y_upper), my) ** 2

        if eval:
            L_u = (self.u_val(x, y) - u) ** 2

            return L_f, L_b0, L_b2, L_u, L_t
        return L_f, L_b0, L_b2, L_t

    def __validation_results(self, pinn, image_width=64, image_height=64):
        x, y, u_real = self.validation_batch(image_width, image_height)
        c = {'coords': torch.cat([x, y], dim=-1).float()}
        pred = pinn(c)['model_out']
        u_pred, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = (
            pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:3], pred[:, :, 3:4], pred[:, :, 4:5], pred[:, :, 5:6]
        )
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

            # mx = mx.detach().numpy().reshape(image_width, image_height)
            # my = my.detach().numpy().reshape(image_width, image_height)
            # f = f.detach().numpy().reshape(image_width, image_height)
            # p = p.detach().numpy().reshape(image_width, image_height)

            self.__plot_3d(u_real, 'Real Displacement (m)')

            # Plot 3D for u_pred
            self.__plot_3d(u_pred, 'Predicted Displacement (m)')

            fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
            self.__show_image(u_pred, axs[0], 'Predicted Displacement (m)')
            self.__show_image((u_pred - u_real) ** 2, axs[1], 'Squared Error Displacement')
            NMSE = (np.linalg.norm(u_pred - u_real) ** 2) / (np.linalg.norm(u_pred) ** 2)
            print('NMSE: ', NMSE)
            # self.__show_image(mx, axs[1, 0], 'Moments mx')
            # self.__show_image(my, axs[1, 1], 'Moments my')
            # self.__show_image(f, axs[2, 0], 'Governing Equation')
            # self.__show_image((f-p)**2, axs[2, 1], 'Squared Error Governing Equation')

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

    def __plot_3d(self, data, title=''):

        X, Y = np.mgrid[0:self.W:complex(0, 64), 0:self.H:complex(0, 64)]
        Z = data

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.invert_xaxis()
        plt.show()
