import torch
from torch import nn


class LearnedGraph(nn.Module):
    def __init__(self, input_dim, graph_prior, prior_mask, threshold, device):
        super(LearnedGraph, self).__init__()

        self.graph = nn.Parameter(
            torch.zeros(input_dim, input_dim, requires_grad=True, device=device)
        )

        if all(i is not None for i in [graph_prior, prior_mask]):
            self.graph_prior = (
                graph_prior.detach().clone().requires_grad_(False).to(device)
            )
            self.prior_mask = (
                prior_mask.detach().clone().requires_grad_(False).to(device)
            )
            self.use_prior = True
        else:
            self.use_prior = False

        self.act = nn.Sigmoid()
        self.threshold = nn.Threshold(threshold, 0)
        self.device = device

    def forward(self, iter):
        if self.use_prior:
            graph = (
                self.prior_mask * self.graph_prior + (1 - self.prior_mask) * self.graph
            )
        else:
            graph = self.graph

        graph = self.act(graph)
        graph = graph.clone()
        graph = graph * (
            torch.ones(graph.shape[0]).to(self.device)
            - torch.eye(graph.shape[0]).to(self.device)
        ) + torch.eye(graph.shape[0]).to(self.device)

        if iter > 50:
            graph = self.threshold(graph)
        else:
            graph = graph

        return graph