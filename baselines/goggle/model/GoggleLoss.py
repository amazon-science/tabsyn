import torch
from torch import nn


class GoggleLoss(nn.Module):
    def __init__(self, alpha=1, beta=0, graph_prior=None, device="cpu"):
        super(GoggleLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.device = device
        self.alpha = alpha
        self.beta = beta
        if graph_prior is not None:
            self.use_prior = True
            self.graph_prior = (
                torch.Tensor(graph_prior).requires_grad_(False).to(device)
            )
        else:
            self.use_prior = False

    def forward(self, x_recon, x, mu, logvar, graph):
        loss_mse = self.mse_loss(x_recon, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if self.use_prior:
            loss_graph = (graph - self.graph_prior).norm(p=1) / torch.numel(graph)
        else:
            loss_graph = graph.norm(p=1) / torch.numel(graph)

        loss = loss_mse + self.alpha * loss_kld + self.beta * loss_graph

        return loss, loss_mse, loss_kld, loss_graph