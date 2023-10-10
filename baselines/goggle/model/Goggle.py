# 3rd party
import torch
from torch import nn

# Goggle
from baselines.goggle.model.Encoder import Encoder
from baselines.goggle.model.GraphDecoder import GraphDecoderHet, GraphDecoderHomo
from baselines.goggle.model.GraphInputProcessor import (
    GraphInputProcessorHet,
    GraphInputProcessorHomo,
)
from baselines.goggle.model.LearnedGraph import LearnedGraph


class Goggle(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_dim=64,
        encoder_l=2,
        het_encoding=True,
        decoder_dim=64,
        decoder_l=2,
        threshold=0.1,
        decoder_arch="gcn",
        graph_prior=None,
        prior_mask=None,
        device="cpu",
    ):
        super(Goggle, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.learned_graph = LearnedGraph(
            input_dim, graph_prior, prior_mask, threshold, device
        )
        self.encoder = Encoder(input_dim, encoder_dim, encoder_l, device)
        if decoder_arch == "het":
            n_edge_types = input_dim * input_dim
            self.graph_processor = GraphInputProcessorHet(
                input_dim, decoder_dim, n_edge_types, het_encoding, device
            )
            self.decoder = GraphDecoderHet(decoder_dim, decoder_l, n_edge_types, device)
        else:
            self.graph_processor = GraphInputProcessorHomo(
                input_dim, decoder_dim, het_encoding, device
            )
            self.decoder = GraphDecoderHomo(
                decoder_dim, decoder_l, decoder_arch, device
            )

    def forward(self, x, iter):
        z, (mu_z, logvar_z) = self.encoder(x)
        b_size, _ = z.shape
        adj = self.learned_graph(iter)
        graph_input = self.graph_processor(z, adj)
        x_hat = self.decoder(graph_input, b_size)

        return x_hat, adj, mu_z, logvar_z

    def sample(self, count):
        with torch.no_grad():
            mu = torch.zeros(self.input_dim)
            sigma = torch.ones(self.input_dim)
            q = torch.distributions.Normal(mu, sigma)
            z = q.rsample(sample_shape=torch.Size([count])).squeeze().to(self.device)

            self.learned_graph.eval()
            self.graph_processor.eval()
            self.decoder.eval()

            adj = self.learned_graph(100)
            graph_input = self.graph_processor(z, adj)
            synth_x = self.decoder(graph_input, count)

        return synth_x