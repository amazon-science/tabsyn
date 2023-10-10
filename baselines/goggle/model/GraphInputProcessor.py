import dgl
import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse


class GraphInputProcessorHomo(nn.Module):
    def __init__(self, input_dim, decoder_dim, het_encoding, device):
        super(GraphInputProcessorHomo, self).__init__()
        self.device = device
        self.het_encoding = het_encoding

        if het_encoding:
            feat_dim = input_dim + 1
        else:
            feat_dim = 1

        self.embedding_functions = []
        for _ in range(input_dim):
            self.embedding_functions.append(
                nn.Sequential(nn.Linear(feat_dim, decoder_dim), nn.Tanh()).to(device)
            )

    def forward(self, z, adj):
        """
        Prepares embeddings for graph decoding
            Parameters:
                z (Tensor): feature embeddings
                adj (Tensor): adjacency matrix
                iter (int): training iteration
            Returns:
                b_z (Tensor): dense feature matrix, shape = (b_size*n_nodes, n_feats)
                b_adj (Tensor): batched adjacency matrix
                b_edge_weight (Sparse Tensor): sparse edge weights, shape = (n_edges)
        """
        b_z = z.unsqueeze(-1)
        b_size, n_nodes, _ = b_z.shape

        if self.het_encoding:
            one_hot_encoding = torch.eye(n_nodes).to(self.device)
            b_encoding = torch.stack([one_hot_encoding for _ in range(b_size)], dim=0)
            b_z = torch.cat([b_z, b_encoding], dim=-1)

        b_z = [f(b_z[:, i]) for i, f in enumerate(self.embedding_functions)]
        b_z = torch.stack(b_z, dim=1)
        b_z = torch.flatten(b_z, start_dim=0, end_dim=1)

        edge_index = adj.nonzero().t()
        row, col = edge_index
        edge_weight = adj[row, col]

        g = dgl.graph((edge_index[0], edge_index[1]))
        b_adj = dgl.batch([g] * b_size)
        b_edge_weight = edge_weight.repeat(b_size)

        return (b_z, b_adj, b_edge_weight)


class GraphInputProcessorHet(nn.Module):
    def __init__(self, input_dim, decoder_dim, n_edge_types, het_encoding, device):
        super(GraphInputProcessorHet, self).__init__()
        self.n_edge_types = n_edge_types
        self.device = device
        self.het_encoding = het_encoding

        if het_encoding:
            feat_dim = input_dim + 1
        else:
            feat_dim = 1

        self.embedding_functions = []
        for _ in range(input_dim):
            self.embedding_functions.append(
                nn.Sequential(nn.Linear(feat_dim, decoder_dim), nn.Tanh()).to(device)
            )

    def forward(self, z, adj):
        """
        Prepares embeddings for graph decoding
            Parameters:
                z (Tensor): feature embeddings
                adj (Tensor): adjacency matrix
                het_encoding (bool): use of heterogeneous encoding
            Returns:
                b_z (Tensor): dense feature matrix, shape = (b_size*n_nodes, n_feats)
                b_adj (Tensor): batched adjacency matrix, shape = (b_size, n_nodes, n_nodes)
                b_edge_index (Sparse Tensor): sparse edge index, shape = (2, n_edges)
                b_edge_weights (Sparse Tensor): sparse edge weights, shape = (n_edges)
                b_edge_types (Sparse Tensor): sparse edge type, shape = (n_edges)
        """
        b_size, n_nodes = z.shape

        b_z = z.unsqueeze(-1)

        if self.het_encoding:
            one_hot_encoding = torch.eye(n_nodes).to(self.device)
            b_encoding = torch.stack([one_hot_encoding for _ in range(b_size)], dim=0)
            b_z = torch.cat([b_z, b_encoding], dim=-1)

        b_z = [f(b_z[:, i]) for i, f in enumerate(self.embedding_functions)]
        b_z = torch.stack(b_z, dim=1)
        b_size, n_nodes, n_feats = b_z.shape

        n_edge_types = self.n_edge_types
        edge_types = torch.arange(1, n_edge_types + 1, 1).reshape(n_nodes, n_nodes)

        b_adj = torch.stack([adj for _ in range(b_size)], dim=0)

        b_edge_index, b_edge_weights = dense_to_sparse(b_adj)
        r, c = b_edge_index
        b_edge_types = edge_types[r % n_nodes, c % n_nodes]
        b_z = b_z.reshape(b_size * n_nodes, n_feats)

        return (b_z, b_edge_index, b_edge_weights, b_edge_types)