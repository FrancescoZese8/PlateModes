import numpy as np
from Data_Set import KirchhoffDataset
import torch
import torch.nn as nn


class CustomVariable:
    def __init__(self, initial_value, trainable=True, dtype=torch.float32):
        self.data = torch.nn.Parameter(torch.tensor(initial_value, dtype=dtype), requires_grad=trainable)

    def assign(self, new_value):
        self.data.data = torch.tensor(new_value, dtype=self.data.dtype)

class KirchhoffLoss(torch.nn.Module):
    def __init__(self, plate:KirchhoffDataset):
        super(KirchhoffLoss, self).__init__()
        self.plate = plate

    def call(self, preds, xy):
      # super().call()
      xy = xy['gt']
      x, y = xy[:, :1], xy[:, 1:]
      preds = preds['model_out']
      L_f, L_b0, L_b2 = self.plate.compute_loss(x, y, preds)
      return {'L_f': L_f, 'L_b0': L_b0, 'L_b2': L_b2}

class ReLoBRaLoKirchhoffLoss(KirchhoffLoss):

    def __init__(self, plate: KirchhoffDataset, alpha: float = 0.999, temperature: float = 1., rho: float = 0.9999):

        super().__init__(plate)
        self.plate = plate
        self.alpha = torch.tensor(alpha)
        self.temperature = temperature
        self.rho = rho
        self.call_count = CustomVariable(0., trainable=False, dtype=torch.float32)
        
        self.lambdas = [CustomVariable(1., trainable=False) for _ in range(plate.num_terms)]
        self.last_losses = [CustomVariable(1., trainable=False) for _ in range(plate.num_terms)]
        self.init_losses = [CustomVariable(1., trainable=False) for _ in range(plate.num_terms)]
'''
    def call(self, preds, xy):
        xy = xy['gt']
        x, y = xy[:, :1], xy[:, 1:]
        preds = preds['model_out']
        EPS = 1e-7
        losses = [torch.mean(loss) for loss in self.plate.compute_loss(x, y, preds)]
        cond1 = torch.tensor(self.call_count == 0, dtype=torch.bool)
        cond2 = torch.tensor(self.call_count == 1,  dtype=torch.bool)
        alpha = torch.where(cond1, torch.tensor(1.0),
                    torch.where(cond2, torch.tensor(0.0),
                                self.alpha))
        rho = torch.where(cond1, torch.tensor(1.0),
                    torch.where(cond2, torch.tensor(1.0),
                                  (torch.rand(()) < self.rho).float()))

        # compute new lambdas w.r.t. the losses in the previous iteration
        lambdas_hat = [losses[i] / (self.last_losses[i].data * self.temperature + EPS) for i in range(len(losses))]
        lambdas_hat = torch.tensor(lambdas_hat)
        lambdas_hat = torch.nn.functional.softmax(lambdas_hat - torch.max(lambdas_hat), dim=0) * float(len(losses))

        # compute new lambdas w.r.t. the losses in the first iteration
        init_lambdas_hat = [losses[i] / (self.init_losses[i].data * self.temperature + EPS) for i in range(len(losses))]
        init_lambdas_hat = torch.tensor(init_lambdas_hat)
        init_lambdas_hat = torch.nn.functional.softmax(init_lambdas_hat - torch.max(init_lambdas_hat), dim=0) * float(len(losses))

        # use rho for deciding, whether a random lookback should be performed
        new_lambdas = [
            (rho * alpha * self.lambdas[i].data + (1 - rho) * alpha * init_lambdas_hat[i].data + (1 - alpha) * lambdas_hat[i].data) for
            i in range(len(losses))]
        # self.lambdas = [var.assign(tf.stop_gradient(lam)) for var, lam in zip(self.lambdas, new_lambdas)]
        self.lambdas = [var.detach().requires_grad_(False) for var in new_lambdas]

        # compute weighted loss
        loss = torch.sum(torch.stack([lam * loss for lam, loss in zip(self.lambdas, losses)]))

        # store current losses in self.last_losses to be accessed in the next iteration
        #self.last_losses = [var.assign(tf.stop_gradient(loss)) for var, loss in zip(self.last_losses, losses)]
        self.last_losses = [loss.detach() for loss in losses]
        # in first iteration, store losses in self.init_losses to be accessed in next iterations
        # first_iteration = (self.call_count < 1).float()
        first_iteration = torch.tensor(self.call_count.data < 1, dtype=torch.float32)
        #self.init_losses = [var.assign(tf.stop_gradient(loss * first_iteration + var * (1 - first_iteration))) for
                            var, loss in zip(self.init_losses, losses)]
        for i, (var, loss) in enumerate(zip(self.init_losses, losses)): self.init_losses[i].data = loss.data * first_iteration + var.data * (1 - first_iteration)


        self.call_count.data += 1
        print('loss_new:',loss)
        return loss'''


def call(self, preds, xy):
    xy = xy['gt']
    x, y = xy[:, :1], xy[:, 1:]
    preds = preds['model_out']
    EPS = 1e-7

    # Calcola le losses distinte
    L_f = torch.mean(self.plate.compute_loss_component(x, y, preds, component_index=0))
    L_b0 = torch.mean(self.plate.compute_loss_component(x, y, preds, component_index=1))
    L_b2 = torch.mean(self.plate.compute_loss_component(x, y, preds, component_index=2))

    losses = [L_f, L_b0, L_b2]

    cond1 = torch.tensor(self.call_count == 0, dtype=torch.bool)
    cond2 = torch.tensor(self.call_count == 1,  dtype=torch.bool)
    alpha = torch.where(cond1, torch.tensor(1.0),
                torch.where(cond2, torch.tensor(0.0),
                            self.alpha))
    rho = torch.where(cond1, torch.tensor(1.0),
                torch.where(cond2, torch.tensor(1.0),
                              (torch.rand(()) < self.rho).float()))

    # Calcola nuove lambdas w.r.t. le losses nella precedente iterazione
    lambdas_hat = [losses[i] / (self.last_losses[i].data * self.temperature + EPS) for i in range(len(losses))]
    lambdas_hat = torch.tensor(lambdas_hat)
    lambdas_hat = torch.nn.functional.softmax(lambdas_hat - torch.max(lambdas_hat), dim=0) * float(len(losses))

    # Calcola nuove lambdas w.r.t. le losses nella prima iterazione
    init_lambdas_hat = [losses[i] / (self.init_losses[i].data * self.temperature + EPS) for i in range(len(losses))]
    init_lambdas_hat = torch.tensor(init_lambdas_hat)
    init_lambdas_hat = torch.nn.functional.softmax(init_lambdas_hat - torch.max(init_lambdas_hat), dim=0) * float(len(losses))

    # Usa rho per decidere se eseguire uno sguardo casuale all'indietro
    new_lambdas = [
        (rho * alpha * self.lambdas[i].data + (1 - rho) * alpha * init_lambdas_hat[i].data + (1 - alpha) * lambdas_hat[i].data) for
        i in range(len(losses))]
    self.lambdas = [var.detach().requires_grad_(False) for var in new_lambdas]

    # Calcola la loss ponderata
    total_loss = torch.sum(torch.stack([lam * loss for lam, loss in zip(self.lambdas, losses)]))

    # Memorizza le losses correnti in self.last_losses per essere accedute nella prossima iterazione
    self.last_losses = [loss.detach() for loss in losses]
    # Nella prima iterazione, memorizza le losses in self.init_losses per essere accedute nelle iterazioni successive
    first_iteration = torch.tensor(self.call_count.data < 1, dtype=torch.float32)
    for i, (var, loss) in enumerate(zip(self.init_losses, losses)):
        self.init_losses[i].data = loss.data * first_iteration + var.data * (1 - first_iteration)

    self.call_count.data += 1

    # Restituisci un dizionario contenente le losses distinte
    return {'L_f': L_f, 'L_b0': L_b0, 'L_b2': L_b2}


class KirchhoffMetric(nn.Module):
    def __init__(self, plate, name='kirchhoff_metric'):
        super(KirchhoffMetric, self).__init__()
        self.plate = plate
        self.L_f_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_b0_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_b2_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_u_mean = nn.Parameter(torch.zeros(1), requires_grad=False)

    def update_state(self, xy, y_pred, sample_weight=None):
        x, y = xy[:, :1], xy[:, 1:]
        L_f, L_b0, L_b2, L_u = self.plate.compute_loss(x, y, y_pred, eval=True)
        self.L_f_mean.data = torch.mean(L_f[:, 0], dim=0)
        self.L_b0_mean.data = torch.mean(L_b0[:, 0], dim=0)
        self.L_b2_mean.data = torch.mean(L_b2[:, 0], dim=0)
        self.L_u_mean.data = torch.mean(L_u[:, 0], dim=0)

    def reset_state(self):
        self.L_f_mean.data = torch.zeros(1)
        self.L_b0_mean.data = torch.zeros(1)
        self.L_b2_mean.data = torch.zeros(1)
        self.L_u_mean.data = torch.zeros(1)

    def result(self):
        return {'L_f': self.L_f_mean.item(),
                'L_b0': self.L_b0_mean.item(),
                'L_b2': self.L_b2_mean.item(),
                'L_u': self.L_u_mean.item()}
